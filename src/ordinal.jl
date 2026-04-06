# ─── ordinal.jl ────────────────────────────────────────────────────────────────
# Ordinal variable support: polychoric correlations + threshold estimation.
# Mirrors lavaan's lav_uvord.R (thresholds) + lav_bvord.R (polychoric).
# ─────────────────────────────────────────────────────────────────────────────

using Distributions: Normal, cdf, pdf, quantile
using Optim

# ─── 20-point Gauss-Legendre nodes on [-1, 1] ────────────────────────────────
# 10 symmetric pairs (±xᵢ, same weight wᵢ)

const _GL20_NODES = Float64[
    -0.9931285991850949,  0.9931285991850949,
    -0.9639719272779138,  0.9639719272779138,
    -0.9122344282513259,  0.9122344282513259,
    -0.8391169718222188,  0.8391169718222188,
    -0.7463062256567498,  0.7463062256567498,
    -0.6360536807265150,  0.6360536807265150,
    -0.5108670019508271,  0.5108670019508271,
    -0.3737060887154196,  0.3737060887154196,
    -0.2277858511416451,  0.2277858511416451,
    -0.0765265211334973,  0.0765265211334973,
]

const _GL20_WEIGHTS = Float64[
    0.0176140071391521,  0.0176140071391521,
    0.0406014298003869,  0.0406014298003869,
    0.0626720483341091,  0.0626720483341091,
    0.0832767415767047,  0.0832767415767047,
    0.1019301198172404,  0.1019301198172404,
    0.1181945319615184,  0.1181945319615184,
    0.1316886384491766,  0.1316886384491766,
    0.1420961093183821,  0.1420961093183821,
    0.1491729864726038,  0.1491729864726038,
    0.1527533871307259,  0.1527533871307259,
]

const _NORM = Normal()

"""
    bivnorm_cdf(h, k, rho) → Float64

Bivariate standard normal CDF: P(X₁ ≤ h, X₂ ≤ k) where Corr(X₁,X₂) = ρ.

Uses 20-point Gauss-Legendre on the 1D integral representation:
    P = ∫₀^{Φ(h)} Φ((k − ρ·Φ⁻¹(u)) / √(1−ρ²)) du
"""
function bivnorm_cdf(h::Real, k::Real, rho::Real)::Float64
    h, k, rho = Float64(h), Float64(k), Float64(rho)

    # ── Boundary / degenerate cases ───────────────────────────────────────────
    (h == -Inf || k == -Inf) && return 0.0
    (h ==  Inf && k ==  Inf) && return 1.0
    h ==  Inf && return cdf(_NORM, k)
    k ==  Inf && return cdf(_NORM, h)
    rho ==  1.0 && return cdf(_NORM, min(h, k))
    rho == -1.0 && return max(cdf(_NORM, h) + cdf(_NORM, k) - 1.0, 0.0)
    rho ==  0.0 && return cdf(_NORM, h) * cdf(_NORM, k)

    rho = clamp(rho, -1.0 + 1e-10, 1.0 - 1e-10)

    # Upper limit of integration in u-space
    ub = cdf(_NORM, h)
    ub <= 0.0 && return 0.0

    sqrt1r2 = sqrt(1.0 - rho^2)

    # Gauss-Legendre on [0, ub]: ∫₀^{ub} f(u) du ≈ (ub/2) Σᵢ wᵢ f((ub/2)(1+xᵢ))
    half_ub = ub / 2.0
    result = 0.0
    @inbounds for i in eachindex(_GL20_NODES)
        u = half_ub * (1.0 + _GL20_NODES[i])
        u = clamp(u, 1e-15, 1.0 - 1e-15)
        x = quantile(_NORM, u)
        result += _GL20_WEIGHTS[i] * cdf(_NORM, (k - rho * x) / sqrt1r2)
    end
    return half_ub * result
end

"""
    _bivnorm_cdf_rho(h_f64, k_f64, rho)

Type-generic version of bivnorm_cdf that preserves the numeric type of `rho`.
Used internally so that ForwardDiff dual numbers flow through `rho` during
polychoric profile-likelihood optimization. `h` and `k` must be plain Float64.
"""
function _bivnorm_cdf_rho(h::Float64, k::Float64, rho::T) where {T<:Real}
    # Extract the primal (Float64) value of rho for boundary comparisons.
    # For plain Float64, this is a no-op. For ForwardDiff duals, this drops
    # the partial components (which is correct: boundary is a measure-zero event).
    rho_val = ForwardDiff.value(rho)

    # Boundary cases — return scalar constants promoted to type T so the
    # caller's type inference remains consistent.
    (h == -Inf || k == -Inf) && return zero(T)
    (h ==  Inf && k ==  Inf) && return one(T)
    h ==  Inf  && return oftype(rho, cdf(_NORM, k))
    k ==  Inf  && return oftype(rho, cdf(_NORM, h))
    rho_val ==  1.0 && return oftype(rho, cdf(_NORM, min(h, k)))
    rho_val == -1.0 && return oftype(rho, max(cdf(_NORM, h) + cdf(_NORM, k) - 1.0, 0.0))
    # Note: do NOT short-circuit rho==0 here; we need the gradient to flow through.

    rho = clamp(rho, oftype(rho, -1.0 + 1e-10), oftype(rho, 1.0 - 1e-10))

    ub = cdf(_NORM, h)          # Float64, depends only on fixed h
    ub <= 0.0 && return zero(T)

    sqrt1r2 = sqrt(one(T) - rho^2)

    # half_ub is a plain Float64 scaling factor; multiply at the end
    half_ub_f = ub / 2.0
    result = zero(T)
    @inbounds for i in eachindex(_GL20_NODES)
        u = half_ub_f * (1.0 + _GL20_NODES[i])
        u = clamp(u, 1e-15, 1.0 - 1e-15)
        x = quantile(_NORM, u)          # Float64 node, fixed
        result += _GL20_WEIGHTS[i] * cdf(_NORM, (k - rho * x) / sqrt1r2)
    end
    return oftype(rho, half_ub_f) * result
end

# ─── Univariate threshold estimation ─────────────────────────────────────────

"""
    OrdinalThresholds

Result of univariate ordinal probit fit.
"""
struct OrdinalThresholds
    theta::Vector{Float64}   # [τ₁, τ₂, ..., τ_{k-1}]
    nth::Int                 # number of thresholds = k - 1
    y_ncat::Int              # number of categories k
    converged::Bool
end

"""
    fit_thresholds(y) → OrdinalThresholds

Estimate thresholds for an ordinal variable using ML (probit link).
y must be an integer vector with values 1..k (recoded internally if needed).
"""
function fit_thresholds(y::Vector{Int})::OrdinalThresholds
    y = y .- minimum(y) .+ 1          # recode to 1..k
    y_ncat = maximum(y)
    nth = y_ncat - 1
    nobs = length(y)

    # Starting values: probit of cumulative proportions
    cumprops = cumsum([count(==(k), y) / nobs for k in 1:y_ncat])[1:end-1]
    theta_start = quantile.(_NORM, clamp.(cumprops, 1e-10, 1 - 1e-10))

    # Boundary offsets: prevents log(0) for top/bottom categories
    o_hi = [y[i] == y_ncat ?  100.0 : 0.0 for i in 1:nobs]
    o_lo = [y[i] == 1      ? -100.0 : 0.0 for i in 1:nobs]

    # Indicator matrices: Y1[i,j] = (y[i]==j+1), Y2[i,j] = (y[i]==j)
    Y1 = [(y[i] == j + 1) for i in 1:nobs, j in 1:nth]
    Y2 = [(y[i] == j)     for i in 1:nobs, j in 1:nth]

    function nll(tau)
        TH = vcat(0.0, tau, 0.0)
        z_hi = TH[y .+ 1] .+ o_hi
        z_lo = TH[y]       .+ o_lo
        pi_i = cdf.(_NORM, z_hi) .- cdf.(_NORM, z_lo)
        pi_i = max.(pi_i, 1e-300)
        return -sum(log.(pi_i)) / nobs
    end

    result = optimize(nll, theta_start, LBFGS(),
                      Optim.Options(iterations=10_000, g_tol=1e-8);
                      autodiff=:forward)

    return OrdinalThresholds(
        Optim.minimizer(result),
        nth,
        y_ncat,
        Optim.converged(result),
    )
end

# ─── Polychoric correlation ───────────────────────────────────────────────────

"""
    polychoric_cor(y1, y2) → Float64

Estimate the polychoric correlation between two ordinal variables.
Two-stage: thresholds from univariate probit (stage 1), then profile
likelihood over ρ = tanh(z) holding thresholds fixed (stage 2).
"""
function polychoric_cor(y1::Vector{Int}, y2::Vector{Int})::Float64
    # Recode to 1..k
    y1 = y1 .- minimum(y1) .+ 1
    y2 = y2 .- minimum(y2) .+ 1

    # Stage 1: thresholds from marginals
    fit1 = fit_thresholds(y1)
    fit2 = fit_thresholds(y2)

    τ1 = vcat(-Inf, fit1.theta, Inf)
    τ2 = vcat(-Inf, fit2.theta, Inf)

    n = length(y1)

    # Stage 2: profile likelihood over ρ = tanh(z), z ∈ ℝ
    # Uses _bivnorm_cdf_rho so ForwardDiff dual numbers flow through rho
    function neg_profile_ll(z)
        rho = tanh(z[1])
        ll = zero(eltype(z))
        for i in 1:n
            s, t = y1[i], y2[i]
            p = _bivnorm_cdf_rho(τ1[s+1], τ2[t+1], rho) -
                _bivnorm_cdf_rho(τ1[s],   τ2[t+1], rho) -
                _bivnorm_cdf_rho(τ1[s+1], τ2[t],   rho) +
                _bivnorm_cdf_rho(τ1[s],   τ2[t],   rho)
            ll += log(max(p, eltype(z)(1e-300)))
        end
        return -ll
    end

    result = optimize(neg_profile_ll, [0.0], LBFGS(),
                      Optim.Options(iterations=1000, g_tol=1e-7);
                      autodiff=:forward)

    return tanh(Optim.minimizer(result)[1])
end

# ─── Polychoric correlation matrix + asymptotic Gamma ─────────────────────────

"""
    vech_pos(p, i, j) → Int

1-based position of element (i,j) with i≥j in vech(A) (column-major lower-triangular).
"""
function vech_pos(p::Int, i::Int, j::Int)::Int
    @assert i >= j
    offset = (j - 1) * p - (j - 1) * (j - 2) ÷ 2
    return offset + (i - j + 1)
end

"""
    _polychoric_var(y1, y2, rho_hat, n) → Float64

Asymptotic variance of ρ̂ via numerical second derivative of profile NLL.
Var(ρ̂) ≈ 1 / I_obs(ρ̂) where I_obs = -∂²ℓ/∂ρ² at ρ̂.
"""
function _polychoric_var(y1::Vector{Int}, y2::Vector{Int},
                          rho_hat::Float64, n::Int)::Float64
    y1 = y1 .- minimum(y1) .+ 1
    y2 = y2 .- minimum(y2) .+ 1

    fit1 = fit_thresholds(y1)
    fit2 = fit_thresholds(y2)
    τ1 = vcat(-Inf, fit1.theta, Inf)
    τ2 = vcat(-Inf, fit2.theta, Inf)

    # Build nll function — note: use rho_eval (not rho_c) to avoid Julia closure
    # variable capture issues (if rho_c is assigned in the outer scope, Julia boxes it
    # and the inner assignment would be invisible).
    function nll_rho(rho_in)
        rho_eval = clamp(rho_in, -1.0 + 1e-8, 1.0 - 1e-8)
        ll = 0.0
        for i in 1:n
            s, t = y1[i], y2[i]
            p = bivnorm_cdf(τ1[s+1], τ2[t+1], rho_eval) -
                bivnorm_cdf(τ1[s],   τ2[t+1], rho_eval) -
                bivnorm_cdf(τ1[s+1], τ2[t],   rho_eval) +
                bivnorm_cdf(τ1[s],   τ2[t],   rho_eval)
            ll += log(max(p, 1e-300))
        end
        return -ll
    end

    # Re-optimize rho using these thresholds to ensure we evaluate the second
    # derivative at the true minimum (rho_hat from polychoric_cor may have used
    # slightly different threshold values due to optimization variability).
    # Use Brent's method (1D, no autodiff needed, no Dual-number issues).
    res = optimize(z -> nll_rho(tanh(z)),
                   -6.0, 6.0,
                   Optim.Brent();
                   rel_tol=1e-8)
    rho_local = tanh(Optim.minimizer(res))

    # Numerical second derivative at local optimum
    ε = 1e-4
    rho_at = clamp(rho_local, -1.0 + 2ε, 1.0 - 2ε)
    d2 = (nll_rho(rho_at - ε) - 2.0 * nll_rho(rho_at) + nll_rho(rho_at + ε)) / ε^2
    d2 = max(d2, 1e-10)
    return 1.0 / d2   # Var(ρ̂) = 1/I_obs (NLL is sum not mean)
end

"""
    compute_polychoric_matrix(X, ov_names, ordered) → (R, Gamma_diag)

Compute the polychoric correlation matrix for variables in `ordered`
(Pearson for any continuous-continuous pairs) and the diagonal of the
asymptotic weight matrix Gamma_poly (in vech order).

Returns:
- `R`          : p×p correlation matrix
- `Gamma_diag` : m-vector (m = p(p+1)/2) of diagonal Gamma entries
"""
function compute_polychoric_matrix(X::Matrix{Float64},
                                   ov_names::Vector{String},
                                   ordered::Vector{String})::Tuple{Matrix{Float64},Vector{Float64}}
    p = length(ov_names)
    n = size(X, 1)
    m = p * (p + 1) ÷ 2

    R = Matrix{Float64}(I, p, p)
    Gamma_diag = fill(2.0, m)   # default: normal-theory var of sample variance

    ord_set = Set(ordered)

    for j in 1:p
        for i in (j+1):p
            k = vech_pos(p, i, j)
            name_i = ov_names[i]
            name_j = ov_names[j]

            if name_i in ord_set && name_j in ord_set
                yi = round.(Int, X[:, i])
                yj = round.(Int, X[:, j])
                rho = polychoric_cor(yi, yj)
                R[i, j] = R[j, i] = rho
                var_rho = _polychoric_var(yi, yj, rho, n)
                Gamma_diag[k] = n * var_rho
            else
                r = cor(X[:, i], X[:, j])
                R[i, j] = R[j, i] = r
                Gamma_diag[k] = (1.0 - r^2)^2
            end
        end
    end

    return R, Gamma_diag
end
