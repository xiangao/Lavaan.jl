# ─── estimators/ml.jl ─────────────────────────────────────────────────────────
# Maximum Likelihood estimator for SEM with continuous data.
# Mirrors lavaan's lav_model_objective.R (estimator="ML") +
#         lav_model_vcov.R (se="standard")
#
# ML objective (normal-theory discrepancy function):
#   F_ML(θ) = Σ_g n_g/N * [log|Σ_g(θ)| + tr(Σ_g(θ)⁻¹ S_g) - log|S_g| - p]
#
# Gradient: via ForwardDiff.gradient() on this function.
# Hessian:  via ForwardDiff.hessian() at θ̂ (for standard SEs).
#
# CRITICAL: All operations must be generic over T <: Real.
# Use T = Float64 at runtime, T = ForwardDiff.Dual during differentiation.
# ─────────────────────────────────────────────────────────────────────────────

"""
    ml_objective(theta, model, stats) → scalar

Normal-theory ML discrepancy function.
Returns F_ML(θ) — minimized at θ̂.
"""
function ml_objective(theta::AbstractVector{T},
                      model::LavaanModel,
                      stats::SampleStats)::T where T <: Real

    F = zero(T)
    N_total = T(stats.ntotal)

    Sigma_list, Mu_list = compute_implied_typed(model, theta)

    is_multilevel = !isempty(model.options.clusters)
    is_crossed = length(model.options.clusters) > 1

    if !is_multilevel
        # Standard ML ...
        for g in 1:model.nblocks
            Sigma_g = Sigma_list[g]
            S_g     = stats.S[g]
            n_g     = T(stats.nobs[g])
            p       = stats.p

            chol_g = try
                cholesky(Sigma_g)
            catch
                return T(1e20)
            end

            logdet_Sigma = 2 * sum(log(chol_g.U[j,j]) for j in 1:p)
            tr_term = tr(chol_g \ S_g)

            w_g = n_g / N_total
            F  += w_g * (logdet_Sigma + tr_term - stats.logdet_S[g] - T(p))
        end
    elseif !is_crossed
        # Nested Multilevel (MUML) ...
        ngroups = model.nblocks ÷ (1 + length(model.options.clusters))
        nlevels = length(model.options.clusters) + 1
        
        for g in 1:ngroups
            b_w = (g - 1) * nlevels + 1
            Sigma_W = Sigma_list[b_w]
            S_W = stats.S_W[g]
            
            n = T(stats.nobs[g])
            p = stats.p
            N_total = T(stats.ntotal)

            chol_W = try cholesky(Sigma_W) catch; return T(1e20) end
            logdet_W = 2 * sum(log(chol_W.U[j,j]) for j in 1:p)
            tr_W = tr(chol_W \ S_W)
            
            n_clusters_1 = T(size(stats.YLp[g][1], 1))
            w_W = (n - n_clusters_1) / N_total
            F += w_W * (logdet_W + tr_W)

            for c in 1:(nlevels - 1)
                b_b = (g - 1) * nlevels + 1 + c
                Sigma_B = Sigma_list[b_b]
                S_B = stats.S_B[g][c]
                s = T(stats.s[g][c])
                n_clusters = T(size(stats.YLp[g][c], 1))

                Sigma_B_implied = Sigma_B + s * Sigma_W
                chol_B = try cholesky(Sigma_B_implied) catch; return T(1e20) end
                logdet_B = 2 * sum(log(chol_B.U[j,j]) for j in 1:p)
                tr_B = tr(chol_B \ S_B)

                w_B = (n_clusters - (c == 1 ? 1 : 0)) / N_total
                F += w_B * (logdet_B + tr_B)
            end
        end
    else
        # Crossed Random Effects (Global FIML via Sparse Matrices)
        # This is a general implementation for any number of non-nested clusters.
        # It uses the full data Likelihood.
        F = ml_objective_crossed(theta, model, stats)
    end

    return F
end

function ml_objective_crossed(theta::AbstractVector{T},
                              model::LavaanModel,
                              stats::SampleStats)::T where T <: Real
    F = zero(T)
    N_total = T(stats.ntotal)
    p = stats.p
    Sigma_list, Mu_list = compute_implied_typed(model, theta)
    nlevels = length(model.options.clusters) + 1
    ngroups = model.nblocks ÷ nlevels

    for g in 1:ngroups
        X = stats.X[g]
        n = size(X, 1)
        
        # 1. Build Sigma_total (sparse)
        # Sigma_W is block 1
        Sigma_W = Sigma_list[(g-1)*nlevels + 1]
        
        # Initial sparse matrix: diagonal blocks of Sigma_W
        # We can use a Block Diagonal representation or just a sum of terms
        # Constructing the full sparse matrix:
        rows = Int[]
        cols = Int[]
        vals = T[]
        
        # Add Sigma_W blocks
        for i in 1:n
            for r in 1:p, c in 1:p
                push!(rows, (i-1)*p + r)
                push!(cols, (i-1)*p + c)
                push!(vals, Sigma_W[r,c])
            end
        end
        
        # Add Cluster level blocks
        for k in 1:(nlevels-1)
            Sigma_k = Sigma_list[(g-1)*nlevels + 1 + k]
            c_idx = stats.cluster_idx[g][k] 
            
            # This is slow, but we'll optimize if it works
            for i in 1:n, j in 1:n
                if c_idx[i] == c_idx[j]
                    for r in 1:p, c in 1:p
                        push!(rows, (i-1)*p + r)
                        push!(cols, (j-1)*p + c)
                        push!(vals, Sigma_k[r,c])
                    end
                end
            end
        end
        
        Sigma_total = sparse(rows, cols, vals, n*p, n*p)
        
        # 2. Compute Likelihood
        y = Vector(vec(X')) # Flatten data: row1[v1,v2...], row2[v1,v2...] ...
        # Standardize: y - mu_total
        mu_g = stats.mu[g]
        mu_total = repeat(mu_g, n)
        y_cent = y - mu_total
        
        chol = try
            cholesky(Sigma_total)
        catch
            return T(1e20)
        end
        
        logdet_Sigma = logdet(chol)
        quad = dot(y_cent, chol \ y_cent) 
        
        F += (logdet_Sigma + quad) / N_total
    end

    return F
end

"""
    ml_loglikelihood(theta, model, stats) → scalar

Log-likelihood value at θ (for AIC/BIC computation).
ℓ(θ) = -N/2 * [p*log(2π) + F_ML(θ) + log|S|]
"""
function ml_loglikelihood(theta::Vector{Float64},
                          model::LavaanModel,
                          stats::SampleStats)::Float64
    F = ml_objective(theta, model, stats)
    p = stats.p
    N = stats.ntotal
    logdet_S_sum = sum(stats.logdet_S[g] * stats.nobs[g] / N
                       for g in 1:length(stats.nobs))
    return -N/2 * (p * log(2π) + F + logdet_S_sum)
end

# ─── Dispatch on estimator symbol ─────────────────────────────────────────────

"""
    estimator_objective(theta, model, stats, estimator) → scalar

Route to the correct estimator's objective function.
Currently implemented: :ML, :MLR, :MLM (all use the normal-theory objective).
"""
function estimator_objective(theta::AbstractVector{T},
                              model::LavaanModel,
                              stats::SampleStats,
                              estimator::Symbol)::T where T <: Real
    if estimator in (:ML, :MLR, :MLM)
        return ml_objective(theta, model, stats)
    elseif estimator == :GLS
        return gls_objective(theta, model, stats)
    elseif estimator == :ULS
        return uls_objective(theta, model, stats)
    elseif estimator == :WLS
        return wls_objective(theta, model, stats)
    elseif estimator in (:DWLS, :WLSMV)
        return dwls_objective(theta, model, stats)
    elseif estimator == :FIML
        return fiml_objective(theta, model, stats)
    elseif estimator == :GSEM
        return gsem_objective(theta, model, stats)
    else
        @warn "Estimator :$estimator not yet implemented; using ML" maxlog=1
        return ml_objective(theta, model, stats)
    end
end

# ─── GLS and ULS stubs (Phase 2 will flesh these out) ─────────────────────────

function gls_objective(theta::AbstractVector{T},
                       model::LavaanModel,
                       stats::SampleStats)::T where T <: Real
    # F_GLS = ½ tr[(S S⁻¹ - Σ(θ) S⁻¹)²] = ½ tr[(I - Σ(θ) S⁻¹)²]
    F = zero(T)
    Sigma_list, _ = compute_implied_typed(model, theta)
    for g in 1:model.nblocks
        D = I - Sigma_list[g] * stats.S_inv[g]
        F += T(stats.nobs[g]) / T(stats.ntotal) * (tr(D * D) / 2)
    end
    return F
end

function uls_objective(theta::AbstractVector{T},
                       model::LavaanModel,
                       stats::SampleStats)::T where T <: Real
    # F_ULS = ½ tr[(S - Σ(θ))²]
    F = zero(T)
    Sigma_list, _ = compute_implied_typed(model, theta)
    for g in 1:model.nblocks
        D = stats.S[g] - Sigma_list[g]
        F += T(stats.nobs[g]) / T(stats.ntotal) * (tr(D * D) / 2)
    end
    return F
end
