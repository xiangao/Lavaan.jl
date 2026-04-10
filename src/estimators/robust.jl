# ─── estimators/robust.jl ────────────────────────────────────────────────────
# Satorra-Bentler mean-adjusted scaled test statistic (MLM estimator).
#
# For continuous data with non-normal distributions, the ML chi-square
# T_ML = N * F_ML(θ̂) is inflated. The Satorra-Bentler (1994) correction
# replaces it with T_MLM = T_ML / c where:
#
#   c = tr(U_proj · Γ̂) / df
#
# Γ̂  = asymptotic covariance of √N · vech(S), estimated from 4th-order moments
# U_proj = Ω - Ω Δ (ΔᵀΩΔ)⁻¹ ΔᵀΩ   (projected saturated weight matrix)
# Ω   = expected Fisher information for saturated model
# Δ   = ∂vech(Σ)/∂θ   (Jacobian of model-implied covariance)
#
# Reference: Satorra & Bentler (1994), "Corrections to test statistics and
# standard errors in covariance structure analysis", in A. von Eye &
# C.C. Clogg (eds), Latent variables analysis, pp. 399–419.
# ─────────────────────────────────────────────────────────────────────────────

"""
    _vech_indices(p) → Vector{Tuple{Int,Int}}

Column-major lower-triangle index pairs for a p×p symmetric matrix.
r-th pair (a,b) satisfies a ≥ b; used for vech(·) operations throughout
the robust/sandwich estimator machinery.
"""
function _vech_indices(p::Int)::Vector{Tuple{Int,Int}}
    pairs = Tuple{Int,Int}[]
    for j in 1:p, i in j:p
        push!(pairs, (i, j))
    end
    return pairs
end

"""
    _compute_Gamma_hat(X, S) → Matrix{Float64}

Asymptotic covariance matrix of √N · vech(S) estimated from raw data.

Γ̂[r,s] = (N/(N-1)) * [(1/N) Σᵢ vech(zᵢzᵢᵀ)ᵣ · vech(zᵢzᵢᵀ)ₛ − vech(S)ᵣ · vech(S)ₛ]

where zᵢ = xᵢ − x̄ (mean-centered observations).
"""
function _compute_Gamma_hat(X::Matrix{Float64}, S::Matrix{Float64})::Matrix{Float64}
    n, p = size(X)
    vech_idx = _vech_indices(p)
    d = length(vech_idx)

    # Center observations
    xbar = vec(mean(X, dims=1))
    Z    = X .- xbar'          # n × p

    # Build V[i, r] = vech(z_i z_i')[r]
    V = Matrix{Float64}(undef, n, d)
    for i in 1:n
        z = Z[i, :]
        for (r, (a, b)) in enumerate(vech_idx)
            V[i, r] = z[a] * z[b]
        end
    end

    # vech(S) vector
    s = [S[a, b] for (a, b) in vech_idx]

    # Γ̂ = N/(N-1) * [(1/N) VᵀV − s sᵀ]
    return (n / (n - 1)) * (V'V / n - s * s')
end

"""
    _compute_Omega(Sigma_inv, vech_idx) → Matrix{Float64}

Expected Fisher information for the saturated normal model (per observation).

Ω[r,s] = Σ⁻¹[a,c] · Σ⁻¹[b,d] + Σ⁻¹[a,d] · Σ⁻¹[b,c]

where r → (a,b) and s → (c,d) are lower-triangle index pairs.
"""
function _compute_Omega(Sigma_inv::Matrix{Float64},
                         vech_idx::Vector{Tuple{Int,Int}})::Matrix{Float64}
    d = length(vech_idx)
    Omega = Matrix{Float64}(undef, d, d)
    for (r, (a, b)) in enumerate(vech_idx)
        for (s, (c, dd)) in enumerate(vech_idx)
            Omega[r, s] = Sigma_inv[a, c] * Sigma_inv[b, dd] +
                          Sigma_inv[a, dd] * Sigma_inv[b, c]
        end
    end
    return Omega
end

"""
    _compute_Delta_g(model, theta_hat, vech_idx, g) → Matrix{Float64}

Jacobian of vech(Σ_g(θ)) with respect to θ (d × nθ).
Computed via ForwardDiff.
"""
function _compute_Delta_g(model::LavaanModel,
                           theta_hat::Vector{Float64},
                           vech_idx::Vector{Tuple{Int,Int}},
                           g::Int)::Matrix{Float64}
    jac_f = theta -> begin
        Sl, _ = compute_implied_typed(model, theta)
        [Sl[g][a, b] for (a, b) in vech_idx]
    end
    return ForwardDiff.jacobian(jac_f, theta_hat)   # d × nθ
end

"""
    satorra_bentler_c(model, stats, lavdata, implied, theta_hat, df) → Float64

Compute the Satorra-Bentler scaling correction factor c.

  c = Σ_g (n_g/N) · tr(U_proj_g · Γ̂_g) / df

Returns NaN if computation fails (e.g. df=0, singular matrix).
"""
function satorra_bentler_c(model::LavaanModel,
                            stats::SampleStats,
                            lavdata::LavaanData,
                            implied::ImpliedMoments,
                            theta_hat::Vector{Float64},
                            df::Int)::Float64

    df <= 0 && return NaN

    p        = stats.p
    N        = Float64(stats.ntotal)
    vech_idx = _vech_indices(p)

    numerator = 0.0

    for g in 1:model.nblocks
        Sigma_g   = implied.Sigma[g]
        Sigma_inv = inv(Sigma_g)
        X_g       = lavdata.X[g]
        n_g       = Float64(size(X_g, 1))
        w_g       = n_g / N

        # 1. Γ̂_g from 4th-order moments
        Gamma_g = _compute_Gamma_hat(X_g, stats.S[g])

        # 2. Ω_g: expected information for saturated model
        Omega_g = _compute_Omega(Sigma_inv, vech_idx)

        # 3. Δ_g: Jacobian of vech(Σ_g) w.r.t. θ
        Delta_g = try
            _compute_Delta_g(model, theta_hat, vech_idx, g)
        catch e
            @warn "Satorra-Bentler: Jacobian failed for group $g: $e" maxlog=1
            return NaN
        end

        # 4. U_proj = Ω - Ω Δ (ΔᵀΩΔ)⁻¹ ΔᵀΩ
        OmegaDelta = Omega_g * Delta_g            # d × nθ
        A = Delta_g' * OmegaDelta                 # nθ × nθ
        A_inv = try
            inv(Symmetric(A))
        catch
            return NaN
        end
        U_proj = Omega_g - OmegaDelta * A_inv * OmegaDelta'

        # 5. Contribution to numerator
        numerator += w_g * tr(U_proj * Gamma_g)
    end

    return numerator / df
end
