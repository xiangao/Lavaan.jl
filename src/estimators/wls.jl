# ─── estimators/wls.jl ────────────────────────────────────────────────────────
# Weighted / Diagonally-Weighted Least Squares estimators.
# Mirrors lavaan's lav_model_objective.R (WLS/DWLS) +
#         lav_samplestats_gamma.R (asymptotic weight matrix Gamma)
#
# WLS objective:
#   F_WLS(θ) = (s - σ(θ))ᵀ W⁻¹ (s - σ(θ))
#
# where:
#   s    = vech(S)                              (sample moments, lower triangle)
#   σ(θ) = vech(Σ(θ))                          (model-implied moments)
#   W    = Γ                                    (full asymptotic weight matrix)
#
# DWLS uses only the diagonal of Γ:
#   F_DWLS(θ) = (s - σ(θ))ᵀ diag(Γ)⁻¹ (s - σ(θ))
#
# Asymptotic covariance of vech(S) under normality (Browne, 1974):
#   Γ[ij,kl] = (n-1)/n * [σ_ik σ_jl + σ_il σ_jk]
#             = 2 * s_ij,kl  (fourth-order cumulants; zero for normal data)
#
# Under normality Γ reduces to:
#   Γ_ij,kl = σ_ik σ_jl + σ_il σ_jk  (normal-theory ADF weight matrix)
# ─────────────────────────────────────────────────────────────────────────────

# ─── vech helpers ─────────────────────────────────────────────────────────────

"""
    vech(A) → Vector

Lower-triangular vectorization of a symmetric matrix (column-major order).
For a p×p matrix, returns p(p+1)/2 elements.
"""
function vech(A::AbstractMatrix{T})::Vector{T} where T
    p = size(A, 1)
    v = Vector{T}(undef, p * (p + 1) ÷ 2)
    k = 0
    for j in 1:p, i in j:p
        k += 1
        v[k] = A[i, j]
    end
    return v
end

"""
    vech_indices(p) → Vector{Tuple{Int,Int}}

Return the (i,j) indices corresponding to vech positions.
"""
function vech_indices(p::Int)::Vector{Tuple{Int,Int}}
    [(i, j) for j in 1:p for i in j:p]
end

# ─── Gamma: asymptotic covariance of vech(S) ─────────────────────────────────

"""
    compute_gamma_normal(S, n) → Matrix{Float64}

Normal-theory asymptotic covariance matrix of vech(S) (Browne, 1974):

    Γ[(i,j),(k,l)] = (σ_ik σ_jl + σ_il σ_jk)

Dimension: m × m where m = p(p+1)/2.
"""
function compute_gamma_normal(S::Matrix{Float64}, n::Int)::Matrix{Float64}
    p = size(S, 1)
    m = p * (p + 1) ÷ 2
    Γ = Matrix{Float64}(undef, m, m)
    idx = vech_indices(p)

    for b in 1:m, a in 1:m
        i, j = idx[a]
        k, l = idx[b]
        Γ[a, b] = S[i,k] * S[j,l] + S[i,l] * S[j,k]
    end
    return Γ
end

"""
    compute_gamma(stats, g) → Matrix{Float64}

Compute or retrieve cached Gamma for group g.
Uses normal-theory formula (empirical 4th-order extension in Phase 3).
"""
function compute_gamma(stats::SampleStats, g::Int)::Matrix{Float64}
    if stats.Gamma !== nothing
        return stats.Gamma[g]
    end
    return compute_gamma_normal(stats.S[g], stats.nobs[g])
end

# ─── WLS objective ────────────────────────────────────────────────────────────

"""
    wls_objective(theta, model, stats) → scalar

Full WLS: F = (s - σ(θ))ᵀ Γ⁻¹ (s - σ(θ))
"""
function wls_objective(theta::AbstractVector{T},
                       model::LavaanModel,
                       stats::SampleStats)::T where T <: Real
    Sigma_list, _ = compute_implied_typed(model, theta)
    F = zero(T)
    for g in 1:model.nblocks
        s     = T.(vech(stats.S[g]))
        sigma = vech(Sigma_list[g])
        resid = s - sigma

        Γ = compute_gamma(stats, g)
        # Solve Γ r = resid for r, then F += resid' * r
        # Use Cholesky for efficiency
        chol_Γ = try
            cholesky(Symmetric(Γ))
        catch
            return T(1e20)
        end
        # chol_Γ is Float64; resid is T (may be Dual for ForwardDiff).
        # Julia's triangular solve promotes types automatically.
        r = chol_Γ \ resid
        F += dot(resid, r)
    end
    return F
end

"""
    dwls_objective(theta, model, stats) → scalar

Diagonal WLS: F = (s - σ(θ))ᵀ diag(Γ)⁻¹ (s - σ(θ))
"""
function dwls_objective(theta::AbstractVector{T},
                        model::LavaanModel,
                        stats::SampleStats)::T where T <: Real
    Sigma_list, _ = compute_implied_typed(model, theta)
    F = zero(T)
    for g in 1:model.nblocks
        s     = T.(vech(stats.S[g]))
        sigma = vech(Sigma_list[g])
        resid = s - sigma

        Γ = compute_gamma(stats, g)
        diag_Γ = T.(diag(Γ))
        # F += Σᵢ resid[i]² / Γ[i,i]
        F += sum(resid[i]^2 / diag_Γ[i] for i in eachindex(resid))
    end
    return F
end
