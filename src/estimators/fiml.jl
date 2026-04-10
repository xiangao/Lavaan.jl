# ─── estimators/fiml.jl ───────────────────────────────────────────────────────
# Full-Information Maximum Likelihood for incomplete (missing) data.
# Mirrors lavaan's lav_model_objective.R (estimator="FIML") +
#         lav_samplestats.R (missing="fiml" path)
#
# Objective (casewise log-likelihood, summed):
#
#   F_FIML = -2 Σᵢ ℓᵢ(θ)
#
# where each casewise contribution is:
#
#   ℓᵢ(θ) = -½ [ log|Σᵢ(θ)| + (xᵢ - μᵢ(θ))ᵀ Σᵢ(θ)⁻¹ (xᵢ - μᵢ(θ)) ]
#
# and Σᵢ(θ), μᵢ(θ) are the model-implied moments restricted to the
# observed indices of observation i.
#
# Implementation: group observations by missing pattern to share Cholesky
# decompositions across observations with the same pattern. O(K × p³) where
# K = number of distinct patterns (usually K << n).
# ─────────────────────────────────────────────────────────────────────────────

# FIMLPattern is defined in types.jl (before samplestats.jl)
# build_fiml_patterns is defined in samplestats.jl

# ─── Per-group FIML objective ──────────────────────────────────────────────────

"""
    fiml_group_objective(Sigma_g, Mu_g, patterns) → scalar

Compute the FIML objective for one group given typed model-implied moments.
Returns F_FIML = -2 Σᵢ ℓᵢ  (minimized at θ̂).
"""
function fiml_group_objective(Sigma_g::Matrix{T},
                               Mu_g::Vector{T},
                               patterns::Vector{FIMLPattern})::T where T <: Real
    F = zero(T)
    p = length(Mu_g)

    for pat in patterns
        idx = pat.obs_idx
        p_i = length(idx)
        n_i = length(pat.rows)

        # Extract submatrix and submean
        Sigma_i = Sigma_g[idx, idx]          # p_i × p_i
        mu_i    = Mu_g[idx]                  # p_i

        # Cholesky of Σᵢ
        chol_i = try
            cholesky(Sigma_i)
        catch
            return T(1e20)
        end

        # log|Σᵢ| = 2 * sum(log(diag(U)))
        logdet_i = 2 * sum(log(chol_i.U[j,j]) for j in 1:p_i)

        # Sum casewise Mahalanobis distances for this pattern
        # (xᵢ - μᵢ)ᵀ Σᵢ⁻¹ (xᵢ - μᵢ)  for each row
        mahal_sum = zero(T)
        for r in 1:n_i
            # pat.X_obs is Float64; convert residual to T for ForwardDiff
            x_r   = T.(pat.X_obs[r, :])
            resid = x_r - mu_i
            # Σᵢ⁻¹ resid via Cholesky solve
            mahal_sum += dot(resid, chol_i \ resid)
        end

        # Contribution: n_i * log|Σᵢ| + Σ mahal distances
        F += n_i * logdet_i + mahal_sum
    end

    return F    # = -2 * Σᵢ ℓᵢ  (since ℓᵢ = -½(log|Σᵢ| + mahal))
end

# ─── Main FIML objective ───────────────────────────────────────────────────────

"""
    fiml_objective(theta, model, stats) → scalar

Full-information ML discrepancy function for incomplete data.
Patterns are read from stats.fiml_patterns (precomputed in compute_samplestats).

When meanstructure=false (default), uses sample means for residuals — same as
lavaan's behavior, which concentrates the likelihood over the mean and gives
the same estimate as ML for complete data.
"""
function fiml_objective(theta::AbstractVector{T},
                        model::LavaanModel,
                        stats::SampleStats)::T where T <: Real
    fiml_data = stats.fiml_patterns
    if fiml_data === nothing
        @warn "FIML patterns not precomputed; falling back to ML" maxlog=1
        return ml_objective(theta, model, stats)
    end

    Sigma_list, Mu_list = compute_implied_typed(model, theta)
    F = zero(T)
    for g in 1:model.nblocks
        # Use model-implied μ(θ) only when meanstructure is modeled explicitly.
        # Otherwise use sample means (concentrated likelihood, same as ML).
        mu_g = model.options.meanstructure ? Mu_list[g] : T.(stats.mu[g])
        F += fiml_group_objective(Sigma_list[g], mu_g, fiml_data[g])
    end

    N_total = T(stats.ntotal)
    return F / (2 * N_total)
end

