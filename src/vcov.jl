# ─── vcov.jl ──────────────────────────────────────────────────────────────────
# Parameter covariance matrix estimation.
# Mirrors lavaan's lav_model_vcov.R
#
# Standard ML (se = :standard):
#   vcov(θ̂) = (N/2)⁻¹ · H(θ̂)⁻¹
#   where H = ForwardDiff.hessian of F_ML at θ̂
#   (equivalently, the observed Fisher information)
#
# MLR Sandwich (se = :robust, :mlr, :robust_sem):
#   vcov(θ̂) = H⁻¹ · B̂ · H⁻¹   (Huber-White sandwich)
#   where B̂ = (1/N²) Σᵢ sᵢ sᵢᵀ  (outer products of casewise scores)
#   and sᵢ = ∂ℓᵢ/∂θ  (gradient of case i's log-likelihood contribution)
#
# Casewise ML score for observation i:
#   sᵢ = ∂/∂θ [-½ log|Σ| - ½ (xᵢ-μ)ᵀ Σ⁻¹ (xᵢ-μ)]
#      = ½ vec(Σ⁻¹ [(xᵢ-μ)(xᵢ-μ)ᵀ - Σ] Σ⁻¹)ᵀ ∂vech(Σ)/∂θ
#        + (xᵢ-μ)ᵀ Σ⁻¹ ∂μ/∂θ
#   Computed via ForwardDiff on the casewise log-likelihood.
#
# Non-convergence contract (Gap G2 extension):
#   - NEVER throw on vcov failure
#   - Return NaN matrix and push warning to fit.warnings
# ─────────────────────────────────────────────────────────────────────────────

"""
    compute_vcov(model, stats, opts, theta_hat, converged, warnings) → Matrix{Float64}

Compute the parameter covariance matrix (nfree × nfree) at θ̂.

Returns a NaN-filled matrix if not converged or if the Hessian is singular.
Appends to `warnings` on any failure.
"""
function compute_vcov(model::LavaanModel,
                      stats::SampleStats,
                      opts::LavaanOptions,
                      lavdata::LavaanData,
                      theta_hat::Vector{Float64},
                      converged::Bool,
                      warnings::Vector{String})::Matrix{Float64}

    nθ = length(theta_hat)
    nan_mat = fill(NaN, nθ, nθ)

    if !converged || nθ == 0
        return nan_mat
    end

    # ── Bootstrap vcov ────────────────────────────────────────────────────────
    if opts.se == :bootstrap
        V = _bootstrap_vcov(model, opts, lavdata, theta_hat, warnings)
        return V !== nothing ? V : nan_mat
    end

    # ── Compute observed Hessian via ForwardDiff ──────────────────────────────
    obj = theta -> estimator_objective(theta, model, stats, opts.estimator)

    H = try
        ForwardDiff.hessian(obj, theta_hat)
    catch e
        push!(warnings, "Hessian computation failed: $e. SEs will be NaN.")
        return nan_mat
    end

    # ── Standard ML vcov: (N/2)⁻¹ H⁻¹ ──────────────────────────────────────
    # H is the Hessian of F_ML; the information matrix is I = (N/2) H
    # so vcov = I⁻¹ = (2/N) H⁻¹
    N = Float64(stats.ntotal)

    # Scale H to information scale: I = (N/2) * H
    I_mat = (N / 2) .* H

    # Invert via Cholesky if PD, else via pinv with warning
    vcov_mat = try
        chol = cholesky(Symmetric(I_mat))
        inv(chol)
    catch
        # Information matrix not PD — model may be non-identified
        try
            V = pinv(I_mat)
            push!(warnings,
                "Information matrix not positive definite. " *
                "Using Moore-Penrose pseudoinverse for SEs. " *
                "Model may be empirically under-identified.")
            V
        catch e2
            push!(warnings, "vcov inversion failed entirely: $e2. SEs will be NaN.")
            return nan_mat
        end
    end

    # ── MLR sandwich: H⁻¹ B̂ H⁻¹  ────────────────────────────────────────────
    if opts.se in (:robust, :robust_sem, :robust_huber_white, :mlr)
        B = _compute_score_outer(model, lavdata, theta_hat, warnings)
        if B !== nothing
            vcov_mat = vcov_mat * B * vcov_mat
        else
            push!(warnings, "MLR score computation failed; returning standard SEs.")
        end
    end

    # Ensure symmetry (small numerical drift)
    return Matrix{Float64}((vcov_mat + vcov_mat') / 2)
end

# ─── Casewise score outer product (meat of sandwich) ──────────────────────────

"""
    _compute_score_outer(model, lavdata, theta_hat, warnings) → Matrix or nothing

Compute the sandwich meat B̂ = (1/N²) Σᵢ sᵢ sᵢᵀ where sᵢ is the gradient
of the casewise ML log-likelihood contribution for observation i.

Computes the Jacobian analytically:
  sᵢ = (1/2) Δᵀ vech(Wᵢ)  where Wᵢ = Σ⁻¹(xᵢ-μ)(xᵢ-μ)ᵀΣ⁻¹ - Σ⁻¹
  Δ = ∂vech(Σ)/∂θ  (d × nθ, computed once via ForwardDiff)
"""
function _compute_score_outer(model::LavaanModel,
                               lavdata::LavaanData,
                               theta_hat::Vector{Float64},
                               warnings::Vector{String})::Union{Matrix{Float64}, Nothing}

    nθ = length(theta_hat)
    N  = sum(lavdata.nobs)
    B  = zeros(nθ, nθ)
    p  = size(lavdata.X[1], 2)

    vech_idx = _vech_indices(p)
    d = length(vech_idx)

    Sigma_list, Mu_list = compute_implied_typed(model, theta_hat)

    for g in 1:model.nblocks
        Sigma_g   = Matrix{Float64}(Sigma_list[g])
        Mu_g      = Vector{Float64}(Mu_list[g])
        Sigma_inv = inv(Sigma_g)
        X_g       = lavdata.X[g]
        n_g       = size(X_g, 1)

        # Jacobian Δ = ∂vech(Σ_g)/∂θ  (d × nθ)
        jac_f = theta -> begin
            Sl, _ = compute_implied_typed(model, theta)
            [Sl[g][a, b] for (a, b) in vech_idx]
        end
        Delta = try
            ForwardDiff.jacobian(jac_f, theta_hat)
        catch e
            push!(warnings, "MLR Jacobian failed: $e. Returning standard SEs.")
            return nothing
        end

        # Per-observation score contribution
        for i in 1:n_g
            r_i   = X_g[i, :] - Mu_g
            W_i   = Sigma_inv * (r_i * r_i') * Sigma_inv - Sigma_inv
            w_vec = [W_i[a, b] for (a, b) in vech_idx]
            s_i   = (0.5 / N) * Delta' * w_vec
            B    .+= s_i * s_i'
        end
    end

    return B
end

# ─── Bootstrap vcov ───────────────────────────────────────────────────────────
# _vech_indices is defined in estimators/robust.jl (included before vcov.jl)

"""
    _bootstrap_vcov(model, opts, lavdata, theta_hat, warnings) → Matrix or nothing

Compute parameter covariance via nonparametric bootstrap.
Resamples rows (within groups) R times, re-optimizes each sample, and
returns the sample covariance of the R parameter vectors.

Uses Julia threads for parallelism.
"""
function _bootstrap_vcov(model::LavaanModel,
                          opts::LavaanOptions,
                          lavdata::LavaanData,
                          theta_hat::Vector{Float64},
                          warnings::Vector{String})::Union{Matrix{Float64}, Nothing}

    R  = opts.nboot
    nθ = length(theta_hat)

    if lavdata.original_df === nothing
        push!(warnings, "Bootstrap requires a raw DataFrame. Returning NaN SEs.")
        return nothing
    end

    df_orig    = convert(DataFrame, lavdata.original_df)
    ngroups    = lavdata.ngroups
    group_col  = lavdata.group_col
    ov_names   = lavdata.ov_names
    lv_names   = lavdata.lv_names
    group_lbls = lavdata.group_labels

    boot_thetas = Vector{Union{Vector{Float64},Nothing}}(nothing, R)

    Threads.@threads for b in 1:R
        rng = Random.MersenneTwister(opts.nboot + b)  # reproducible per iteration
        try
            # ── Resample within each group ────────────────────────────────
            if ngroups == 1
                n = nrow(df_orig)
                df_b = df_orig[rand(rng, 1:n, n), :]
            else
                parts = Vector{DataFrame}(undef, ngroups)
                for g in 1:ngroups
                    mask  = df_orig[!, group_col] .== group_lbls[g]
                    rows  = findall(mask)
                    ng    = length(rows)
                    parts[g] = df_orig[rows[rand(rng, 1:ng, ng)], :]
                end
                df_b = vcat(parts...)
            end

            # ── Re-fit on bootstrap sample ────────────────────────────────
            lavdata_b = prepare_data(df_b, ov_names, lv_names;
                                     group=group_col,
                                     missing_method=opts.missing_method)
            stats_b = compute_samplestats(lavdata_b, opts)
            theta_b, conv_b, _, _ = optimize_model(model, stats_b, opts;
                                                    theta0=theta_hat)

            if conv_b && !any(isnan, theta_b)
                boot_thetas[b] = theta_b
            end
        catch
            # silently skip non-converging bootstrap samples
        end
    end

    successful = filter(!isnothing, boot_thetas)
    n_ok = length(successful)

    if n_ok < max(2, R ÷ 2)
        push!(warnings,
            "Only $n_ok / $R bootstrap samples converged. Bootstrap SEs unreliable.")
        n_ok < 2 && return nothing
    end

    # vcov = sample covariance of bootstrap θ̂ vectors
    boot_mat = reduce(hcat, successful)   # nθ × n_ok
    return cov(boot_mat')                 # nθ × nθ covariance matrix
end


"""
    fill_ses!(pt, theta_hat, vcov_mat)

Write SEs, z-values, p-values, and 95% CIs back into the ParTable.
"""
function fill_ses!(pt::ParTable,
                   theta_hat::Vector{Float64},
                   vcov_mat::Matrix{Float64})

    for i in eachindex(pt.free)
        k = pt.free[i]
        k == 0 && continue

        est_k = theta_hat[k]
        se_k  = sqrt(max(vcov_mat[k, k], 0.0))

        pt.se[i]       = se_k
        pt.z[i]        = se_k > 0 ? est_k / se_k : NaN
        pt.pvalue[i]   = se_k > 0 ? 2 * (1 - Distributions.cdf(Distributions.Normal(), abs(est_k / se_k))) : NaN
        pt.ci_lower[i] = est_k - 1.96 * se_k
        pt.ci_upper[i] = est_k + 1.96 * se_k
    end
end

# ─── Defined parameters (:=) ─────────────────────────────────────────────────

"""
    _eval_defined_expr(expr_str, ctx) → Float64

Evaluate a lavaan `:=` expression (e.g. `"a * b"`, `"indirect + c"`) given a
Dict mapping label names to their current values.
"""
function _eval_defined_expr(expr_str::String, ctx::Dict{String,Float64})::Float64
    m = Module(:_LavaanDefinedEval)
    for (k, v) in ctx
        Core.eval(m, :($(Symbol(k)) = $v))
    end
    return Float64(Core.eval(m, Meta.parse(expr_str)))
end

"""
    compute_defined_params!(pt, vcov_mat)

Evaluate all `:=` defined parameters in the ParTable and fill in their
estimates and delta-method standard errors.

Algorithm (delta method):
  If f = f(a, b, ...) where a, b, ... are labeled free parameters,
  then SE²(f) = ∇f' · V · ∇f  where ∇f[k] = ∂f/∂θ_k for free index k.
"""
function compute_defined_params!(pt::ParTable, vcov_mat::Matrix{Float64})
    # Build label → (estimate, free_index) maps from free params with labels
    label_est = Dict{String, Float64}()
    label_idx = Dict{String, Int}()
    for i in eachindex(pt.free)
        isempty(pt.label[i]) && continue
        label_est[pt.label[i]] = pt.est[i]
        pt.free[i] > 0 && (label_idx[pt.label[i]] = pt.free[i])
    end

    n_free = size(vcov_mat, 1)
    defined_est = Dict{String, Float64}()  # accumulate as we go (for chained := )

    for i in eachindex(pt.op)
        pt.op[i] == ":=" || continue
        lhs      = pt.lhs[i]
        expr_str = pt.rhs[i]

        # Evaluation context: labeled free params + previously-defined params
        ctx = merge(label_est, defined_est)

        val = try
            _eval_defined_expr(expr_str, ctx)
        catch
            NaN
        end
        pt.est[i] = val
        defined_est[lhs] = val

        isnan(val) && continue
        n_free == 0 && continue

        # Delta method: gradient w.r.t. free parameter vector θ
        grad_theta = zeros(n_free)
        h = 1e-6
        for (lbl, idx) in label_idx
            ctx_hi = copy(ctx); ctx_hi[lbl] = ctx[lbl] + h
            ctx_lo = copy(ctx); ctx_lo[lbl] = ctx[lbl] - h
            dfdlbl = try
                (_eval_defined_expr(expr_str, ctx_hi) - _eval_defined_expr(expr_str, ctx_lo)) / (2h)
            catch
                0.0
            end
            grad_theta[idx] += dfdlbl
        end

        se_sq = dot(grad_theta, vcov_mat * grad_theta)
        se    = sqrt(max(se_sq, 0.0))
        pt.se[i]       = se
        pt.z[i]        = se > 0 ? val / se : NaN
        pt.pvalue[i]   = se > 0 ? 2*(1 - Distributions.cdf(Distributions.Normal(), abs(val/se))) : NaN
        pt.ci_lower[i] = val - 1.96 * se
        pt.ci_upper[i] = val + 1.96 * se
    end
end
