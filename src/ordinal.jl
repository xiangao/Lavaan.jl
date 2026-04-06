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
