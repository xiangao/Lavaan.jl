# ─── efa.jl ───────────────────────────────────────────────────────────────────
# Exploratory Factor Analysis.
# Mirrors lavaan's efa() function.
#
# Two-stage pipeline:
#   1. ML extraction: minimize F_ML = log|Λ Λᵀ + Θ| + tr((Λ Λᵀ + Θ)⁻¹ S) − log|S| − p
#      with Θ diagonal and a lower-triangular identification constraint on Λ.
#   2. Rotation to the target criterion:
#      :varimax   — orthogonal, maximize variance of squared loadings (Kaiser 1958)
#      :quartimax — orthogonal, maximize kurtosis of loadings
#      :geomin    — oblique, minimize geometric mean of squared loadings (lavaan default)
#      :oblimin   — oblique, quartimin (gamma=0) by default
#      :none      — no rotation (raw extraction)
#
# Reference: Bernaards & Jennrich (2005), J. Comput. Graph. Stat.
# ─────────────────────────────────────────────────────────────────────────────

"""
    EFAResult

Result object from `efa()`. Contains rotated loadings, communalities,
uniquenesses, factor correlations, and basic goodness-of-fit statistics.
"""
struct EFAResult
    loadings::DataFrame             # p × k rotated pattern matrix
    communalities::Vector{Float64}  # h² = Σ_k λ_ik² (per variable)
    uniquenesses::Vector{Float64}   # 1 − h²
    factor_corr::Matrix{Float64}    # Φ: k × k factor correlation (I for orthogonal)
    rotation_matrix::Matrix{Float64}
    ov_names::Vector{String}
    n_obs::Int
    n_factors::Int
    rotation::Symbol
    chisq::Float64                  # χ² goodness-of-fit
    df::Int
    pvalue::Float64
    converged::Bool
end

# ─── ML Extraction ────────────────────────────────────────────────────────────

"""
    _extract_ml_efa(S, k; maxit, tol) → (Lambda, Theta_diag, converged)

Minimum-residual ML extraction for k factors.

Parameterization:
  - Λ is p × k; lower triangular identification (Λ[i,j] = 0 for i < j)
  - Θ = diag(exp(t)) ensures positivity
  - Objective: F_ML = log|Λ Λᵀ + Θ| + tr((Λ Λᵀ + Θ)⁻¹ S) − log|S| − p
"""
function _extract_ml_efa(S::Matrix{Float64}, k::Int;
                          maxit::Int = 2000,
                          tol::Float64 = 1e-7)
    p = size(S, 1)

    # Starting values via eigendecomposition
    eig   = eigen(Symmetric(S))
    order = sortperm(eig.values, rev=true)
    vals  = max.(eig.values[order][1:k], 0.0)
    vecs  = eig.vectors[:, order[1:k]]
    L0    = vecs .* sqrt.(vals)'

    # Apply lower-triangular constraint
    for j in 1:k, i in 1:(j-1)
        L0[i, j] = 0.0
    end

    theta0_diag = max.(diag(S) .- diag(L0 * L0'), 0.005)

    # Pack: [vec(Λ); log(diag(Θ))]
    function pack(L, t)
        [vec(L); log.(t)]
    end

    function unpack_apply_constraint(par)
        L = reshape(par[1:p*k], p, k)
        # Lower triangular identification: zero upper triangle
        for j in 1:k, i in 1:(j-1)
            L[i, j] = 0.0
        end
        t = exp.(par[p*k+1:end])
        return L, t
    end

    logdet_S = logdet(Symmetric(S + 1e-12 * I))

    function objective(par)
        L, t = unpack_apply_constraint(par)
        Sigma = Symmetric(L * L' + Diagonal(t))
        chol  = cholesky(Sigma + 1e-10 * I(p))
        logdet_sigma = 2 * sum(log.(diag(chol.L)))
        tr_term      = tr(chol \ S)
        return logdet_sigma + tr_term - logdet_S - p
    end

    par0   = pack(L0, theta0_diag)
    result = Optim.optimize(objective, par0, Optim.LBFGS(),
                            Optim.Options(g_tol=tol, iterations=maxit);
                            autodiff=:forward)

    L_opt, t_opt = unpack_apply_constraint(Optim.minimizer(result))
    return L_opt, t_opt, Optim.converged(result)
end

# ─── Rotation algorithms ──────────────────────────────────────────────────────

"""
    _rotate_varimax(L; normalize, maxit, tol) → (L_rot, T)

Orthogonal varimax rotation via pairwise Jacobi sweeps (Kaiser 1958).
`normalize=true` applies Kaiser row normalization before rotation.
Returns rotated loading matrix and orthogonal rotation matrix T (k × k).
"""
function _rotate_varimax(L::Matrix{Float64};
                          normalize::Bool = true,
                          maxit::Int = 1000,
                          tol::Float64 = 1e-6)
    p, k = size(L)
    k == 1 && return copy(L), Matrix{Float64}(I, k, k)

    h = normalize ? sqrt.(sum(L.^2, dims=2)) : ones(p, 1)
    h[h .< 1e-12] .= 1.0
    L_n = L ./ h

    T = Matrix{Float64}(I, k, k)

    for _iter in 1:maxit
        T_old = copy(T)
        for i in 1:k, j in (i+1):k
            Li = L_n * T[:, i]
            Lj = L_n * T[:, j]
            u  = Li.^2 .- Lj.^2
            v  = 2.0 .* Li .* Lj
            A  = sum(u)
            B  = sum(v)
            C  = sum(u.^2 .- v.^2)
            D  = sum(u .* v)
            phi = atan(D - A*B/p, C - (A^2 - B^2)/(2p)) / 4.0
            s, c = sincos(phi)
            rot  = [c -s; s c]
            T[:, [i, j]] = T[:, [i, j]] * rot
        end
        norm(T - T_old) < tol && break
    end

    L_rot = (L_n * T) .* h
    return L_rot, T
end

"""
    _rotate_quartimax(L; normalize, maxit, tol) → (L_rot, T)

Orthogonal quartimax rotation. Maximizes sum of fourth powers of loadings
(simplest structure within each variable). Uses pairwise Jacobi sweeps.
"""
function _rotate_quartimax(L::Matrix{Float64};
                            normalize::Bool = true,
                            maxit::Int = 1000,
                            tol::Float64 = 1e-6)
    p, k = size(L)
    k == 1 && return copy(L), Matrix{Float64}(I, k, k)

    h = normalize ? sqrt.(sum(L.^2, dims=2)) : ones(p, 1)
    h[h .< 1e-12] .= 1.0
    L_n = L ./ h

    T = Matrix{Float64}(I, k, k)

    for _iter in 1:maxit
        T_old = copy(T)
        for i in 1:k, j in (i+1):k
            Li  = L_n * T[:, i]
            Lj  = L_n * T[:, j]
            u   = Li.^2 .- Lj.^2
            v   = 2.0 .* Li .* Lj
            A   = sum(u)
            B   = sum(v)
            C   = sum(u.^2 .- v.^2)
            D   = sum(u .* v)
            # quartimax: gamma = 0 in the Kaiser family
            phi = atan(D, C - (A^2 - B^2)/2) / 4.0
            s, c = sincos(phi)
            rot  = [c -s; s c]
            T[:, [i, j]] = T[:, [i, j]] * rot
        end
        norm(T - T_old) < tol && break
    end

    L_rot = (L_n * T) .* h
    return L_rot, T
end

"""
    _rotate_oblique(L, criterion; epsilon, maxit, tol) → (P, T, Phi)

General oblique rotation via gradient descent on the rotation matrix T
(an invertible k × k matrix with columns normalized to unit length).

The pattern matrix is P = L * (T')⁻¹ and factor correlations Φ = T' T.

`criterion` must be a concrete callable (use `where {C}` dispatch so
ForwardDiff can specialize on the concrete function type).
"""
function _rotate_oblique(L::Matrix{Float64},
                          criterion::C;
                          epsilon::Float64 = 0.01,
                          maxit::Int = 2000,
                          tol::Float64 = 1e-7) where {C}
    p, k = size(L)
    k == 1 && return copy(L), Matrix{Float64}(I, k, k), Matrix{Float64}(I, k, k)

    # Criterion as a function of Float64 T_vec only (no autodiff through inv)
    function obj(T_vec::Vector{Float64})
        T = reshape(T_vec, k, k)
        col_norms = sqrt.(max.(vec(sum(T.^2, dims=1)), 1e-12))
        T_n = T ./ col_norms'
        P = L / T_n'        # L * inv(T_n')
        return criterion(P, k, p, epsilon)
    end

    # Use finite differences — no ForwardDiff compilation needed;
    # for k≤6 (k² ≤ 36 variables) this is fast and numerically stable.
    T0     = vec(Matrix{Float64}(I, k, k))
    result = Optim.optimize(obj, T0, Optim.LBFGS(),
                            Optim.Options(g_tol=tol, iterations=maxit);
                            autodiff=:finite)

    T_vec     = Optim.minimizer(result)
    T         = reshape(T_vec, k, k)
    col_norms = sqrt.(max.(vec(sum(T.^2, dims=1)), 1e-12))
    T_n       = T ./ col_norms'
    P         = L / T_n'
    Phi       = Matrix(Symmetric(inv(T_n' * T_n)))

    return P, T_n, Phi
end

# Geomin criterion: Σ_i (∏_j (λ_ij² + ε))^(1/k)
function _geomin_crit(P, k, p, epsilon)
    Q = zero(eltype(P))
    inv_k = 1.0 / k
    for i in 1:p
        prod_i = one(eltype(P))
        for j in 1:k
            prod_i *= P[i,j]^2 + epsilon
        end
        Q += prod_i^inv_k
    end
    return Q
end

# Quartimin (oblimin γ=0) criterion: (1/4) Σ_i Σ_{j≠l} λ_ij² λ_il²
function _oblimin_crit(P, k, p, epsilon)
    Q = zero(eltype(P))
    for i in 1:p, j in 1:k, l in 1:k
        j == l && continue
        Q += P[i,j]^2 * P[i,l]^2
    end
    return Q / 4
end

# ─── Main EFA entry point ─────────────────────────────────────────────────────

"""
    efa(data, nfactors; rotation=:geomin, standardize=true, ov_names=nothing,
        maxit=2000, tol=1e-7, seed=nothing) → EFAResult

Exploratory Factor Analysis with ML extraction and optional rotation.

# Arguments
- `data`       : DataFrame (or Matrix) of observed variables
- `nfactors`   : number of factors to extract
- `rotation`   : `:geomin` (default), `:varimax`, `:quartimax`, `:oblimin`, `:none`
- `standardize`: if true, work on correlation matrix (default `true`)
- `ov_names`   : variable names (inferred from DataFrame if not provided)
- `maxit`      : max iterations for extraction optimizer
- `tol`        : convergence tolerance

# Returns
An `EFAResult` struct. Use `show(result)` for a formatted table.

# Example
```julia
HS = holzinger_swineford()
result = efa(HS[:, [:x1,:x2,:x3,:x4,:x5,:x6,:x7,:x8,:x9]], 3)
show(result)
```
"""
function efa(data::DataFrame,
             nfactors::Int;
             rotation::Symbol   = :geomin,
             standardize::Bool  = true,
             ov_names::Union{Nothing,Vector{String}} = nothing,
             maxit::Int         = 2000,
             tol::Float64       = 1e-7,
             seed               = nothing)::EFAResult

    seed !== nothing && Random.seed!(seed)

    # Variable names
    names_v = ov_names !== nothing ? ov_names : String.(names(data))
    p = length(names_v)

    # Data matrix
    X = Matrix{Float64}(data[:, names_v])
    n = size(X, 1)

    # Sample covariance / correlation
    S = cov(X)
    if standardize
        D = Diagonal(1.0 ./ sqrt.(diag(S)))
        S = D * S * D
    end

    @assert nfactors >= 1 "nfactors must be ≥ 1"
    @assert nfactors <= p "nfactors must be ≤ p (number of variables)"
    # Identification requires at least k(k-1)/2 zero constraints
    min_indicators = nfactors + div(nfactors * (nfactors - 1), 2)
    @assert p >= min_indicators "Too many factors for $p variables (need ≥ $min_indicators indicators)"

    # ── 1. ML Extraction ──────────────────────────────────────────────────────
    L0, Theta_diag, converged = _extract_ml_efa(S, nfactors; maxit=maxit, tol=tol)

    # ── 2. Rotation ───────────────────────────────────────────────────────────
    Phi = Matrix{Float64}(I, nfactors, nfactors)
    T   = Matrix{Float64}(I, nfactors, nfactors)

    if rotation == :none || nfactors == 1
        L_rot = L0
    elseif rotation == :varimax
        L_rot, T = _rotate_varimax(L0)
    elseif rotation == :quartimax
        L_rot, T = _rotate_quartimax(L0)
    elseif rotation == :geomin
        L_rot, T, Phi = _rotate_oblique(L0, _geomin_crit)
    elseif rotation == :oblimin
        L_rot, T, Phi = _rotate_oblique(L0, _oblimin_crit)
    else
        @warn "efa: unknown rotation :$rotation — using :geomin"
        L_rot, T, Phi = _rotate_oblique(L0, _geomin_crit)
    end

    # ── 3. Communalities and uniquenesses ─────────────────────────────────────
    h2 = vec(sum(L_rot.^2, dims=2))
    u2 = max.(1.0 .- h2, 0.0)

    # ── 4. Chi-square goodness-of-fit ─────────────────────────────────────────
    # df = [(p - k)² - (p + k)] / 2
    df  = max(div((p - nfactors)^2 - (p + nfactors), 2), 0)
    Sigma_hat = Symmetric(L0 * L0' + Diagonal(Theta_diag))
    logdet_S   = logdet(Symmetric(S + 1e-12 * I))
    logdet_sig = logdet(Sigma_hat + 1e-12 * I)
    F_obj  = logdet_sig + tr(Sigma_hat \ S) - logdet_S - p
    chisq  = max((n - 1 - (2p + 5)/6 - 2*nfactors/3) * F_obj, 0.0)
    pvalue = df > 0 ? 1.0 - Distributions.cdf(Distributions.Chisq(df), chisq) : NaN

    # ── 5. Factor names and loading table ─────────────────────────────────────
    factor_names = ["F$(j)" for j in 1:nfactors]
    loadings_df  = DataFrame(L_rot, factor_names)
    insertcols!(loadings_df, 1, :variable => names_v)

    return EFAResult(
        loadings_df, h2, u2, Phi, T,
        names_v, n, nfactors, rotation,
        chisq, df, pvalue, converged,
    )
end

# Convenience: accept a Matrix
function efa(data::Matrix{Float64},
             nfactors::Int;
             ov_names::Union{Nothing,Vector{String}} = nothing,
             kwargs...)::EFAResult
    names_v = ov_names !== nothing ? ov_names : ["V$(i)" for i in 1:size(data, 2)]
    return efa(DataFrame(data, names_v), nfactors; ov_names=names_v, kwargs...)
end

# ─── Show method ──────────────────────────────────────────────────────────────

function Base.show(io::IO, r::EFAResult)
    println(io, "")
    println(io, "Exploratory Factor Analysis")
    println(io, "  Rotation  : $(r.rotation)")
    println(io, "  Factors   : $(r.n_factors)")
    println(io, "  N obs     : $(r.n_obs)")
    println(io, "  χ²($(r.df)) = $(round(r.chisq, digits=3))  p = $(round(r.pvalue, digits=4))")
    println(io, "  Converged : $(r.converged)")
    println(io, "")

    # Loading table
    p  = length(r.ov_names)
    kk = r.n_factors
    header = rpad("Variable", 12) * join(rpad("F$(j)", 9) for j in 1:kk) * "  Communality"
    println(io, header)
    println(io, "─"^length(header))

    for i in 1:p
        row = rpad(r.ov_names[i], 12)
        for j in 1:kk
            row *= rpad(@sprintf("% .3f", r.loadings[i, j+1]), 9)
        end
        row *= "  $(round(r.communalities[i], digits=3))"
        println(io, row)
    end

    if kk > 1 && !isapprox(r.factor_corr, I, atol=1e-6)
        println(io, "")
        println(io, "Factor correlations:")
        for i in 1:kk
            row = "  "
            for j in 1:kk
                row *= rpad(@sprintf("% .3f", r.factor_corr[i, j]), 8)
            end
            println(io, row)
        end
    end
end
