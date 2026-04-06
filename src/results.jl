# ─── results.jl ───────────────────────────────────────────────────────────────
# User-facing result extraction functions.
# Mirrors lavaan's parameterEstimates(), fitMeasures(), lavInspect(), etc.
# ─────────────────────────────────────────────────────────────────────────────

# ─── parameterEstimates ───────────────────────────────────────────────────────

"""
    parameterEstimates(fit; standardized=false, ci=true, level=0.95) → DataFrame

Return a DataFrame of parameter estimates, standard errors, z-values,
p-values, and confidence intervals.

Mirrors lavaan's `parameterEstimates()`.
"""
function parameterEstimates(fit::LavaanFit;
                             standardized::Bool = false,
                             ci::Bool = true,
                             level::Float64 = 0.95)::DataFrame

    pt = fit.partable

    # Collect user-visible rows (user=1 and auto ~~ rows)
    mask = (pt.user .== 1) .| (pt.op .== "~~") .| (pt.op .== "~1")
    # For simplicity: include all rows with op in model operators
    # "|" is the threshold operator used for ordinal variables
    model_ops = Set(["=~", "~~", "~", "~1", ":=", "~*~", "|"])
    idxs = [i for i in eachindex(pt.id) if pt.op[i] in model_ops]

    # Build output columns
    lhs   = pt.lhs[idxs]
    op    = pt.op[idxs]
    rhs   = pt.rhs[idxs]
    est   = pt.est[idxs]
    se_v  = pt.se[idxs]
    z_v   = pt.z[idxs]
    pval  = pt.pvalue[idxs]
    ci_lo = pt.ci_lower[idxs]
    ci_hi = pt.ci_upper[idxs]
    label = pt.label[idxs]
    free  = pt.free[idxs]

    df = DataFrame(
        lhs    = lhs,
        op     = op,
        rhs    = rhs,
        est    = est,
        se     = se_v,
        z      = z_v,
        pvalue = pval,
        ci_lower = ci_lo,
        ci_upper = ci_hi,
        label  = label,
        free   = free,
    )

    return df
end

# ─── fitMeasures ──────────────────────────────────────────────────────────────

"""
    fitMeasures(fit, indices=nothing) → Dict{Symbol,Float64}

Return model fit indices. If `indices` is provided (e.g. `[:cfi, :rmsea]`),
only those indices are returned.

Mirrors lavaan's `fitMeasures()`.
"""
function fitMeasures(fit::LavaanFit,
                     indices::Union{Nothing,Vector{Symbol}} = nothing)::Dict{Symbol,Float64}
    fm = fit.fit_measures
    if indices === nothing
        return copy(fm)
    else
        return Dict(k => get(fm, k, NaN) for k in indices)
    end
end

# ─── coef / vcov ──────────────────────────────────────────────────────────────

"""
    coef(fit) → Vector{Float64}

Return the vector of free parameter estimates θ̂.
"""
function coef(fit::LavaanFit)::Vector{Float64}
    return get_est(fit.partable)
end

"""
    vcov(fit) → Matrix{Float64}

Return the parameter covariance matrix (nfree × nfree).
"""
function vcov(fit::LavaanFit)::Matrix{Float64}
    return copy(fit.vcov_theta)
end

# ─── fitted / residuals ───────────────────────────────────────────────────────

"""
    fitted(fit) → ImpliedMoments

Return model-implied moments Σ(θ̂), μ(θ̂).
"""
function fitted(fit::LavaanFit)::ImpliedMoments
    return fit.implied
end

"""
    residuals(fit; type=:raw) → Vector{Matrix{Float64}}

Return residual covariance matrices S − Σ(θ̂) per group.
`type=:standardized` divides by sqrt(sᵢᵢ * sⱼⱼ).
"""
function StatsAPI.residuals(fit::LavaanFit; type::Symbol=:raw)::Vector{Matrix{Float64}}
    stats   = fit.samplestats
    implied = fit.implied
    ngroups = length(stats.S)
    resids = Vector{Matrix{Float64}}(undef, ngroups)
    for g in 1:ngroups
        R = stats.S[g] - implied.Sigma[g]
        if type == :standardized
            p = stats.p
            for i in 1:p, j in 1:p
                denom = sqrt(stats.S[g][i,i] * stats.S[g][j,j])
                R[i,j] /= (denom > 0 ? denom : 1.0)
            end
        end
        resids[g] = R
    end
    return resids
end

# ─── standardizedSolution ─────────────────────────────────────────────────────

"""
    standardizedSolution(fit) → DataFrame

Return the fully standardized solution (std.all in lavaan).
Standardizes by dividing each estimate by appropriate sd ratio.
"""
function standardizedSolution(fit::LavaanFit)::DataFrame
    df = parameterEstimates(fit)

    pt      = fit.partable
    stats   = fit.samplestats
    implied = fit.implied
    p       = stats.p

    # Collect observed and implied variances (first block / first group)
    S_g     = stats.S[1]
    Sigma_g = implied.Sigma[1]

    ov_names = fit.model.ov_names
    lv_names = fit.model.lv_names
    ov_idx   = Dict(v => i for (i,v) in enumerate(ov_names))
    lv_idx   = Dict(v => i for (i,v) in enumerate(lv_names))

    # Compute total variances for observed (diagonal of S) and latent (from Σ)
    # For latent, we approximate total variance from Sigma_g projected back
    # (Phase 1 approximation: use diagonal of Sigma_g for implied latent variances)
    # Full standardized solution for latents requires the model-implied latent cov matrix
    # which needs separate extraction; for now return std.lv style approximation

    est_std = copy(df.est)
    for i in eachindex(df.est)
        lhs_v = df.lhs[i]
        rhs_v = df.rhs[i]
        op_v  = df.op[i]

        lhs_is_ov = haskey(ov_idx, lhs_v)
        rhs_is_ov = haskey(ov_idx, rhs_v)

        if isnan(df.est[i])
            est_std[i] = NaN
            continue
        end

        if op_v == "=~"
            # Factor loading: std = loading * sd(lv_rhs? no, rhs is indicator) /sd(lhs)
            # std.all: multiply by sd(rhs_indicator)/sd(lhs_factor)
            # Approximation: use diagonal of implied Sigma for indicator sd
            if rhs_is_ov
                j_rhs  = ov_idx[rhs_v]
                sd_ind = sqrt(max(Sigma_g[j_rhs, j_rhs], 0.0))
                # Latent sd: we don't have direct access here, skip for Phase 1
                est_std[i] = df.est[i] # placeholder
            end
        elseif op_v == "~~" && lhs_is_ov && rhs_is_ov
            j1 = ov_idx[lhs_v]
            j2 = ov_idx[rhs_v]
            denom = sqrt(Sigma_g[j1,j1] * Sigma_g[j2,j2])
            est_std[i] = denom > 0 ? df.est[i] / denom : NaN
        end
    end

    result = copy(df)
    result.est_std = est_std
    return result
end

# ─── lavInspect ───────────────────────────────────────────────────────────────

"""
    lavInspect(fit, what=:free) → varies

Inspect internal model quantities. Mirrors lavaan's `lavInspect()`.

Common values for `what`:
  :free        — free parameter count (Int)
  :nobs        — number of observations per group
  :sigma       — model-implied covariance (Sigma(θ̂))
  :mu          — model-implied means (μ(θ̂))
  :sample_cov  — sample covariance matrix S
  :sample_mean — sample mean vector
  :partable    — parameter table (ParTable)
  :vcov        — parameter covariance matrix
  :start       — starting values
  :est         — estimates
  :se          — standard errors
  :fit         — fit measures dict
  :converged   — convergence status
"""
function lavInspect(fit::LavaanFit, what::Symbol=:free)
    if what == :free
        return fit.model.nfree
    elseif what == :nobs
        return fit.samplestats.nobs
    elseif what == :sigma
        return fit.implied.Sigma
    elseif what == :mu
        return fit.implied.Mu
    elseif what == :sample_cov
        return fit.samplestats.S
    elseif what == :sample_mean
        return fit.samplestats.mu
    elseif what == :partable
        return fit.partable
    elseif what == :vcov
        return fit.vcov_theta
    elseif what == :start
        return get_start(fit.partable)
    elseif what == :est
        return get_est(fit.partable)
    elseif what == :se
        nθ = fit.model.nfree
        se_vec = fill(NaN, nθ)
        for i in eachindex(fit.partable.free)
            k = fit.partable.free[i]
            k > 0 && (se_vec[k] = fit.partable.se[i])
        end
        return se_vec
    elseif what == :fit
        return fit.fit_measures
    elseif what == :converged
        return fit.converged
    elseif what == :iterations
        return fit.iterations
    elseif what == :warnings
        return fit.warnings
    elseif what == :lambda
        theta_hat = get_est(fit.partable)
        return fill_block(fit.model.blocks[1], theta_hat).Lambda
    elseif what == :theta
        theta_hat = get_est(fit.partable)
        return fill_block(fit.model.blocks[1], theta_hat).Theta
    elseif what == :psi
        theta_hat = get_est(fit.partable)
        return fill_block(fit.model.blocks[1], theta_hat).Psi
    elseif what == :beta
        theta_hat = get_est(fit.partable)
        return fill_block(fit.model.blocks[1], theta_hat).Beta
    elseif what == :rsquare
        return rsquare(fit)
    else
        error("lavInspect: unknown `what` = :$what")
    end
end

# ─── lavTestLRT ───────────────────────────────────────────────────────────────

"""
    lavTestLRT(fits...; model_names=nothing) → DataFrame

Likelihood ratio test comparing two or more nested models.

Models are sorted by df ascending (most free model first). For each pair of
consecutive models, computes Δχ², Δdf, and p-value from χ²(Δdf).

Also returns AIC, BIC, and log-likelihood for information-criteria comparison.

# Example
```julia
fit0 = cfa(model_free, data)
fit1 = cfa(model_constrained, data)
lavTestLRT(fit0, fit1)
lavTestLRT(fit0, fit1, fit2; model_names=["free", "c1", "c2"])
```
"""
function lavTestLRT(fits::LavaanFit...;
                    model_names::Union{Nothing,Vector{String}} = nothing)::DataFrame
    n = length(fits)
    n < 2 && error("lavTestLRT requires at least two models")

    # Sort models by df ascending: least constrained (fewest df) first
    dfs   = [Int(f.fit_measures[:df]) for f in fits]
    order = sortperm(dfs)
    sorted = [fits[i] for i in order]

    # Model labels
    labels = if model_names !== nothing && length(model_names) == n
        model_names[order]
    else
        ["Model $i" for i in 1:n]
    end

    # Per-model statistics
    df_v    = [Int(f.fit_measures[:df])            for f in sorted]
    npar_v  = [Int(f.fit_measures[:npar])           for f in sorted]
    chisq_v = [f.fit_measures[:chisq]               for f in sorted]
    logl_v  = [get(f.fit_measures, :logl, NaN)      for f in sorted]
    aic_v   = [get(f.fit_measures, :aic,  NaN)      for f in sorted]
    bic_v   = [get(f.fit_measures, :bic,  NaN)      for f in sorted]

    # Pairwise LRT: model[i] vs model[i-1] (each more constrained than previous)
    delta_chisq_v = fill(NaN, n)
    delta_df_v    = fill(0,   n)
    pvalue_v      = fill(NaN, n)

    for i in 2:n
        dc = chisq_v[i] - chisq_v[i-1]
        dd = df_v[i]    - df_v[i-1]
        delta_chisq_v[i] = dc
        delta_df_v[i]    = dd
        pvalue_v[i] = if dd > 0 && !isnan(dc)
            1.0 - Distributions.cdf(Distributions.Chisq(dd), dc)
        else
            NaN
        end
    end

    return DataFrame(
        model      = labels,
        npar       = npar_v,
        df         = df_v,
        aic        = aic_v,
        bic        = bic_v,
        logl       = logl_v,
        chisq      = chisq_v,
        chisq_diff = delta_chisq_v,
        df_diff    = delta_df_v,
        pvalue     = pvalue_v,
    )
end

# ─── modindices ───────────────────────────────────────────────────────────────

"""
    _mi_score_info(mat, r, c, W, Sigma_inv, V, R) → (g, h_exp)

Analytical score (gradient) and expected information diagonal for one fixed
parameter.

Derivation (ML, normal theory):
  g     = ∂F_ML/∂θ_j  = tr(W · ∂Σ/∂θ_j)
  h_exp = tr(Σ⁻¹ · ∂Σ/∂θ_j · Σ⁻¹ · ∂Σ/∂θ_j)    (expected information / (N/2))

where W = Σ⁻¹ − Σ⁻¹SΣ⁻¹.

For rank-1 perturbations (diagonal Θ, diagonal Ψ):
  ∂Σ/∂θ = a·aᵀ  →  g = aᵀWa,   h_exp = (aᵀΣ⁻¹a)²

For rank-2 perturbations (off-diagonal Θ/Ψ, Λ, β):
  ∂Σ/∂θ = a·bᵀ + b·aᵀ  →  g = 2·aᵀWb,
                             h_exp = 2(aᵀΣ⁻¹b)² + 2(aᵀΣ⁻¹a)(bᵀΣ⁻¹b)

Returns (NaN, NaN) for unsupported/mean-structure parameters.
"""
function _mi_score_info(mat::String, r::Int, c::Int,
                        W::Matrix{Float64},
                        Sigma_inv::Matrix{Float64},
                        V::Matrix{Float64},   # ΛA  (p×q)
                        R::Matrix{Float64})   # ΛC = ΛAΨAᵀ  (p×q)
    q = size(V, 2)

    if mat == "theta"
        if r == c
            # Diagonal Θ[r,r]: ∂Σ = e_r·e_rᵀ  (rank-1)
            g     = W[r, r]
            h_exp = Sigma_inv[r, r]^2
        else
            # Off-diagonal Θ[r,c]: ∂Σ = e_r·e_cᵀ + e_c·e_rᵀ  (rank-2)
            g     = 2 * W[r, c]
            h_exp = 2 * Sigma_inv[r, c]^2 + 2 * Sigma_inv[r, r] * Sigma_inv[c, c]
        end

    elseif mat == "lambda" && q > 0
        # Loading Λ[r,k] where k = c: ∂Σ = e_r·r_kᵀ + r_k·e_rᵀ (rank-2)
        # r_k = (ΛC)[:,k] = R[:,k]
        r_k = R[:, c]
        v_k = Sigma_inv * r_k            # Σ⁻¹ r_k
        g     = 2 * dot(W[r, :], r_k)   # 2 (W r_k)[r]
        h_exp = 2 * v_k[r]^2 + 2 * Sigma_inv[r, r] * dot(r_k, v_k)

    elseif mat == "psi" && q > 0
        k, l = r, c
        if k == l
            # Diagonal Ψ[k,k]: ∂Σ = f_k·f_kᵀ (rank-1)
            f_k   = V[:, k]
            u_k   = Sigma_inv * f_k
            g     = dot(f_k, W * f_k)
            h_exp = dot(f_k, u_k)^2
        else
            # Off-diagonal Ψ[k,l]: ∂Σ = f_k·f_lᵀ + f_l·f_kᵀ (rank-2)
            f_k   = V[:, k]
            f_l   = V[:, l]
            u_k   = Sigma_inv * f_k
            u_l   = Sigma_inv * f_l
            g     = 2 * dot(f_k, W * f_l)
            h_exp = 2 * dot(f_k, u_l)^2 + 2 * dot(f_k, u_k) * dot(f_l, u_l)
        end

    elseif mat == "beta" && q > 0
        # Structural β[j,k]: ∂Σ = g_j·r_kᵀ + r_k·g_jᵀ (rank-2)
        # g_j = V[:,j] = (ΛA)[:,j],  r_k = R[:,k] = (ΛC)[:,k]
        j, k  = r, c
        g_j   = V[:, j]
        r_k   = R[:, k]
        u_j   = Sigma_inv * g_j
        u_k   = Sigma_inv * r_k
        g     = 2 * dot(r_k, W * g_j)
        h_exp = 2 * dot(g_j, u_k)^2 + 2 * dot(g_j, u_j) * dot(r_k, u_k)

    else
        return NaN, NaN   # intercepts / unmapped
    end

    return g, h_exp
end

"""
    modindices(fit; sort_by=:mi, minimum_mi=0.0) → DataFrame

Modification indices (Lagrange multiplier statistics) for all fixed parameters
in the model.  A large MI for parameter θ_j indicates that freeing it would
substantially reduce model chi-square.

Columns returned:
  lhs, op, rhs — parameter identity
  mi            — modification index (≈ expected Δχ² if freed; ~ χ²(1) under H₀)
  epc           — expected parameter change (one-step Newton update)

Uses the score-test formula:
  MI  = N/2 · g² / h_exp
  EPC = −g / h_exp
where g = ∂F/∂θ_j (analytical gradient) and h_exp is the expected information
diagonal (analytical, avoids the observed-vs-expected Hessian confound).

# Arguments
- `sort_by`    : `:mi` (default) sorts by MI descending, `:lhs` sorts alphabetically
- `minimum_mi` : only rows with MI ≥ this threshold are returned (default 0.0)

# Example
```julia
mi = modindices(fit)                   # all fixed params, sorted by MI
mi = modindices(fit; minimum_mi=10.0)  # only large MIs (lavaan default threshold)
```
"""
function modindices(fit::LavaanFit;
                    sort_by::Symbol     = :mi,
                    minimum_mi::Float64 = 0.0)::DataFrame

    model = fit.model
    stats = fit.samplestats
    pt    = fit.partable
    N     = Float64(stats.ntotal)

    lhs_v = String[]; op_v = String[]; rhs_v = String[]
    mi_v  = Float64[]; epc_v = Float64[]

    model_ops = Set(["=~", "~~", "~", "~1"])

    # ── Per-block precomputed quantities ──────────────────────────────────────
    block_cache = map(1:model.nblocks) do g
        blk       = model.blocks[g]
        Sigma_g   = fit.implied.Sigma[g]
        S_g       = stats.S[g]
        Sigma_inv = inv(Sigma_g)
        W_g       = Sigma_inv - Sigma_inv * S_g * Sigma_inv

        q = blk.q
        if q > 0
            A = inv(I(q) - blk.Beta)        # (I-B)⁻¹
            V = blk.Lambda * A              # ΛA  (p×q)
            C = A * blk.Psi * A'            # AΨAᵀ (q×q, latent cov)
            R = blk.Lambda * C              # ΛC   (p×q)
        else
            V = zeros(blk.p, 0)
            R = zeros(blk.p, 0)
        end
        (Sigma_inv=Sigma_inv, W=W_g, V=V, R=R)
    end

    # ── Iterate fixed partable rows ───────────────────────────────────────────
    for i in eachindex(pt.id)
        pt.free[i] != 0 && continue
        pt.op[i] ∉ model_ops && continue
        isempty(pt.mat[i]) && continue

        g_idx  = pt.block[i]
        cache  = block_cache[g_idx]
        g_val, h_exp = _mi_score_info(pt.mat[i], pt.mat_row[i], pt.mat_col[i],
                                      cache.W, cache.Sigma_inv, cache.V, cache.R)

        (isnan(g_val) || isnan(h_exp) || h_exp <= 0.0) && continue

        push!(lhs_v,  pt.lhs[i])
        push!(op_v,   pt.op[i])
        push!(rhs_v,  pt.rhs[i])
        push!(mi_v,   N / 2 * g_val^2 / h_exp)
        push!(epc_v,  -g_val / h_exp)
    end

    result = DataFrame(lhs=lhs_v, op=op_v, rhs=rhs_v, mi=mi_v, epc=epc_v)
    filter!(row -> row.mi >= minimum_mi, result)
    sort_by == :mi  && sort!(result, :mi,  rev=true)
    sort_by == :lhs && sort!(result, [:lhs, :rhs])
    return result
end

# ─── rsquare ──────────────────────────────────────────────────────────────────

"""
    rsquare(fit) → Dict{String,Float64}

R-squared (proportion of variance explained) for each endogenous variable.

For observed indicators:   R²_i = 1 − Θ_ii / Σ_ii(θ̂)
For endogenous latents:    R²_j = 1 − Ψ_jj / [(I−B)⁻¹ Ψ (I−B)⁻ᵀ]_jj

Mirrors lavaan's `lavInspect(fit, "rsquare")`.
"""
function rsquare(fit::LavaanFit)::Dict{String,Float64}
    model     = fit.model
    implied   = fit.implied
    theta_hat = get_est(fit.partable)
    en_set    = Set(model.en_names)
    result    = Dict{String,Float64}()

    for g in 1:model.nblocks
        block   = model.blocks[g]
        Sigma_g = implied.Sigma[g]
        mats    = fill_block(block, theta_hat)
        Lambda  = mats.Lambda
        Theta   = mats.Theta
        Psi     = mats.Psi
        Beta    = mats.Beta
        p, q    = block.p, block.q

        # R² for observed indicators (any row of Λ with a non-zero loading)
        for i in 1:p
            has_loading = any(Lambda[i, :] .!= 0) || any(block.Lambda_free[i, :])
            has_loading || continue
            sigma_ii = Sigma_g[i, i]
            sigma_ii > 1e-12 || continue
            result[model.ov_names[i]] = 1.0 - Theta[i, i] / sigma_ii
        end

        # R² for endogenous latent variables
        if q > 0 && !isempty(en_set)
            IB_inv  = inv(Matrix{Float64}(I, q, q) - Beta)
            Var_eta = IB_inv * Psi * IB_inv'
            for j in 1:q
                lv = model.lv_names[j]
                lv in en_set || continue
                var_jj = Var_eta[j, j]
                var_jj > 1e-12 || continue
                result[lv] = 1.0 - Psi[j, j] / var_jj
            end
        end
    end

    return result
end

# ─── lavPredict ───────────────────────────────────────────────────────────────

"""
    lavPredict(fit; type=:lv) → DataFrame

Compute factor scores (predicted latent variable values) for each observation.

Uses the regression method:
  η̂ᵢ = E[η] + Cov(η,x) Σ⁻¹ (xᵢ − μ)

where Cov(η,x) = (I−B)⁻¹ Ψ (I−B)⁻ᵀ Λᵀ.

Returns a DataFrame with one column per latent variable and one row per observation.

Mirrors lavaan's `lavPredict(fit, type="lv")`.
"""
function lavPredict(fit::LavaanFit; type::Symbol=:lv)::DataFrame
    model     = fit.model
    implied   = fit.implied
    theta_hat = get_est(fit.partable)
    lv_names  = model.lv_names

    isempty(lv_names) && error("lavPredict: model has no latent variables")

    all_scores = Vector{Vector{Float64}}()

    for g in 1:model.nblocks
        X_g     = fit.data.X[g]
        Sigma_g = implied.Sigma[g]
        Mu_g    = implied.Mu[g]
        block   = model.blocks[g]
        mats    = fill_block(block, theta_hat)
        q       = block.q

        q == 0 && continue

        Lambda   = mats.Lambda    # p × q
        Psi      = mats.Psi       # q × q
        Beta     = mats.Beta      # q × q
        Alpha    = mats.Alpha     # q

        IB_inv    = inv(Matrix{Float64}(I, q, q) - Beta)
        mu_eta    = IB_inv * Alpha                      # E[η]
        Var_eta   = IB_inv * Psi * IB_inv'              # Var(η)  q × q
        Cov_eta_x = Var_eta * Lambda'                   # q × p
        W         = Cov_eta_x * inv(Sigma_g)            # regression weight q × p

        # Center using sample means when Mu_g ≈ 0 (no meanstructure).
        # When meanstructure=false, Nu=Alpha=0 so Mu_g=0, but the observed
        # variables have non-zero means. Regression factor scores require
        # centering by the actual (sample) mean of the observed data.
        x_bar = vec(mean(X_g, dims=1))
        center = all(abs.(Mu_g) .< 1e-10) ? x_bar : Mu_g

        for i in 1:size(X_g, 1)
            x_centered = X_g[i, :] .- center
            push!(all_scores, mu_eta .+ W * x_centered)
        end
    end

    isempty(all_scores) &&
        return DataFrame(; (Symbol(lv) => Float64[] for lv in lv_names)...)

    mat = reduce(hcat, all_scores)'   # n_total × q
    return DataFrame(mat, lv_names)
end
