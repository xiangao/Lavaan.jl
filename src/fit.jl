# ─── fit.jl ───────────────────────────────────────────────────────────────────
# Model fit index computation.
# Mirrors lavaan's lav_fit.R + lav_test.R + lav_model_fit.R
#
# Indices computed here:
#   chisq, df, pvalue           — likelihood ratio test statistic
#   cfi, tli                    — incremental fit (requires baseline)
#   rmsea, rmsea_ci_lower/upper — parsimony-adjusted residual
#   srmr                        — standardized RMR
#   aic, bic, sabic             — information criteria
#   logl, npar                  — log-likelihood and parameter count
#
# Baseline model (Gap G6):
#   Independence model: only variances free, all covariances fixed to 0.
#   Fitted once and cached in LavaanFit.baseline.
# ─────────────────────────────────────────────────────────────────────────────

# ─── Degrees of freedom ───────────────────────────────────────────────────────

"""
    model_df(stats, model) → Int

Degrees of freedom for the target model.
df = [p(p+1)/2 * ngroups] − nfree   (when meanstructure=false)
   = [p(p+1)/2 + p] * ngroups − nfree  (when meanstructure=true)
"""
function model_df(stats::SampleStats, model::LavaanModel)::Int
    p    = stats.p
    nθ   = model.nfree
    has_mean = model.options.meanstructure
    # Number of non-redundant moments per group
    moments_per_group = p * (p + 1) ÷ 2 + (has_mean ? p : 0)
    total_moments = moments_per_group * model.nblocks
    return max(total_moments - nθ, 0)
end

"""
    baseline_df(stats) → Int

Degrees of freedom for the baseline (independence) model.
Only variances free (p per group), no covariances.
df_baseline = p(p+1)/2 * ngroups − p * ngroups  =  p(p-1)/2 * ngroups
"""
function baseline_df(stats::SampleStats, nblocks::Int)::Int
    p = stats.p
    return p * (p - 1) ÷ 2 * nblocks
end

# ─── Baseline model construction ──────────────────────────────────────────────

"""
    build_baseline_partable(model) → ParTable

Construct a ParTable for the independence (baseline) model.
Only observed variances are free; all covariances = 0.
"""
function build_baseline_partable(model::LavaanModel, stats::SampleStats)::ParTable
    p = stats.p
    ov = model.ov_names

    # Each group contributes p variance rows
    nblocks = model.nblocks
    nrows = p * nblocks
    pt = ParTable(nrows)

    row = 0
    free_k = 0
    for g in 1:nblocks
        for j in 1:p
            row += 1
            free_k += 1
            pt.id[row]    = row
            pt.lhs[row]   = ov[j]
            pt.op[row]    = "~~"
            pt.rhs[row]   = ov[j]
            pt.user[row]  = 0
            pt.block[row] = g
            pt.free[row]  = free_k
            pt.lower[row] = 0.0            # variances must be non-negative
            pt.upper[row] = Inf
            # Starting value: sample variance
            pt.start[row] = stats.S[min(g, length(stats.S))][j, j]
            pt.est[row]   = pt.start[row]
        end
    end
    return pt
end

"""
    fit_baseline_model(model, stats, opts) → (F_baseline, df_baseline)

Fit the independence model and return (objective value, degrees of freedom).
We reuse the existing optimization pipeline but with a simplified model.
"""
function fit_baseline_model(model::LavaanModel,
                             stats::SampleStats,
                             opts::LavaanOptions)

    pt_bl = build_baseline_partable(model, stats)

    # Build a baseline LavaanModel: no latent variables, no structural paths
    # The baseline only has Theta (diagonal only)
    p = stats.p
    nblocks = model.nblocks

    blocks_bl = Vector{LISRELBlock}(undef, nblocks)
    for g in 1:nblocks
        q = 0  # no latent variables
        Lambda   = zeros(p, 0)
        Theta    = zeros(p, p)
        Psi      = zeros(0, 0)
        Beta     = zeros(0, 0)
        Nu       = zeros(p)
        Alpha    = zeros(0)
        Lf = fill(false, p, 0)
        Tf = fill(false, p, p)
        Pf = fill(false, 0, 0)
        Bf = fill(false, 0, 0)
        Nf = fill(false, p)
        Af = fill(false, 0)

        free_map  = Tuple{Symbol,Int,Int}[]
        free_fidx = Int[]

        # Map each variance row in pt_bl to Theta[j,j]
        for j in 1:p
            row = (g - 1) * p + j
            Theta[j, j]      = pt_bl.start[row]
            Tf[j, j]         = true
            push!(free_map,  (:theta, j, j))
            push!(free_fidx, pt_bl.free[row])
        end

        blocks_bl[g] = LISRELBlock(
            Lambda, Theta, Psi, Beta, Nu, Alpha,
            Lf, Tf, Pf, Bf, Nf, Af,
            free_map, free_fidx, p, q,
        )
    end

    model_bl = LavaanModel(
        blocks_bl, nblocks, nfree(pt_bl),
        model.ov_names, String[],          # no latent vars
        String[], String[],
        pt_bl, opts,
    )

    # Minimize independence model objective
    theta0_bl = get_start(pt_bl)
    baseline_est = opts.estimator in (:DWLS, :WLSMV, :WLS, :GSEM) ? opts.estimator : :ML
    obj_bl = theta -> estimator_objective(theta, model_bl, stats, baseline_est)

    result_bl = try
        Optim.optimize(obj_bl, theta0_bl, Optim.LBFGS(),
                       Optim.Options(g_tol=1e-10, iterations=5000);
                       autodiff=:forward)
    catch
        return nothing
    end

    if !Optim.converged(result_bl)
        return nothing
    end

    F_bl = Optim.minimum(result_bl)
    df_bl = baseline_df(stats, nblocks)
    return (F_bl, df_bl)
end

# ─── SRMR ─────────────────────────────────────────────────────────────────────

"""
    compute_srmr(implied, stats) → Float64

Standardized Root Mean Square Residual.
SRMR = sqrt[ (2 / p(p+1)) * Σᵢⱼ ((sᵢⱼ - σᵢⱼ(θ̂)) / (sᵢᵢ * sⱼⱼ))² ]
"""
function compute_srmr(implied::ImpliedMoments, stats::SampleStats)::Float64
    p = stats.p
    total = 0.0
    count = 0
    for g in 1:length(stats.S)
        S_g     = stats.S[g]
        Sigma_g = implied.Sigma[g]
        for i in 1:p, j in 1:i
            resid = (S_g[i,j] - Sigma_g[i,j]) / sqrt(S_g[i,i] * S_g[j,j])
            total += resid^2
            count += 1
        end
    end
    return count > 0 ? sqrt(total / count) : NaN
end

# ─── RMSEA confidence interval ────────────────────────────────────────────────

"""
    rmsea_ci(chisq, df, N; alpha=0.10) → (lower, upper)

90% confidence interval for RMSEA using the non-central chi-square inversion.
"""
function rmsea_ci(chisq::Float64, df::Int, N::Int; alpha::Float64=0.10)
    df == 0 && return (0.0, 0.0)
    N == 0  && return (NaN, NaN)
    (isnan(chisq) || isinf(chisq) || chisq < 0) && return (NaN, NaN)

    # Lower bound: find λ such that P(χ²(df, λ) ≥ chisq) = 1 - alpha/2
    # Upper bound: find λ such that P(χ²(df, λ) ≥ chisq) = alpha/2
    #
    # lavaan uses uniroot; we use a simple bisection

    function ncp_pvalue(ncp, tail)
        # P(χ²(df, ncp) ≥ chisq)
        return robust_ncp_ccdf(df, max(ncp, 0.0), chisq)
    end

    # Lower RMSEA CI: find λ such that P(χ²(df,λ) ≥ chisq) = alpha/2
    # f is increasing in ncp; smaller ncp → smaller RMSEA lower bound
    target_lo = alpha / 2
    if ncp_pvalue(0.0, :x) > target_lo
        # chisq is small (model fits well); RMSEA lower bound = 0
        rmsea_lo = 0.0
    else
        ncp_lo = _bisect_ncp(ncp_pvalue, target_lo, chisq, df)
        rmsea_lo = sqrt(max(ncp_lo, 0.0) / (df * N))
    end

    # Upper RMSEA CI: find λ such that P(χ²(df,λ) ≥ chisq) = 1 - alpha/2
    # larger ncp → larger RMSEA upper bound
    target_hi = 1.0 - alpha / 2
    ncp_hi = _bisect_ncp(ncp_pvalue, target_hi, chisq, df)
    rmsea_hi = sqrt(max(ncp_hi, 0.0) / (df * N))

    return (rmsea_lo, rmsea_hi)
end

"""
    robust_ncp_ccdf(df, ncp, x)

P(χ²(df, ncp) ≥ x) with clamping to avoid non-convergence in SpecialFunctions.
"""
function robust_ncp_ccdf(df::Int, ncp::Float64, x::Float64)
    (isnan(x) || isinf(x) || x < 0) && return 0.0
    ncp_clamped = max(ncp, 0.0)
    if x > 1e12 || ncp_clamped > 1e12
        return x < ncp_clamped ? 1.0 : 0.0
    end
    try
        d = Distributions.NoncentralChisq(df, ncp_clamped)
        return 1.0 - Distributions.cdf(d, x)
    catch
        return x < ncp_clamped ? 1.0 : 0.0
    end
end

function _bisect_ncp(f, target, chisq, df; maxiter=200, tol=1e-6)
    (isnan(chisq) || isinf(chisq)) && return NaN
    lo, hi = 0.0, max(chisq * 10.0, 100.0)
    # Expand hi until f(hi) > target (f is increasing in ncp)
    while f(hi, :x) < target && hi < 1e8
        hi *= 10.0
    end
    for _ in 1:maxiter
        mid = (lo + hi) / 2
        if f(mid, :x) > target
            hi = mid   # f too large → ncp too large
        else
            lo = mid   # f too small → ncp too small
        end
        abs(hi - lo) < tol && break
    end
    return (lo + hi) / 2
end

# ─── Main fit index computation ───────────────────────────────────────────────

"""
    compute_fit_measures(fit) → Dict{Symbol,Float64}

Compute all model fit indices and store in fit.fit_measures.
Also fits and caches the baseline model in fit.baseline.

Called after optimization and vcov are complete.
"""
function compute_fit_measures!(fit::LavaanFit)::Dict{Symbol,Float64}
    model   = fit.model
    stats   = fit.samplestats
    opts    = fit.options
    pt      = fit.partable
    implied = fit.implied

    nθ = model.nfree
    N  = stats.ntotal
    p  = stats.p

    fm = Dict{Symbol,Float64}()

    # ── Basic info ────────────────────────────────────────────────────────────
    fm[:npar] = Float64(nθ)
    fm[:df]   = Float64(model_df(stats, model))
    fm[:N]    = Float64(N)

    # ── Chi-square test statistic: T_ML = N * F_ML(θ̂) ───────────────────────
    # lavaan convention: multiply by N (not N-1) for the Wishart-based test
    theta_hat = get_est(pt)
    F_val = estimator_objective(theta_hat, model, stats, opts.estimator)
    T_ML  = Float64(N) * F_val

    fm[:chisq] = T_ML
    fm[:fmin]  = F_val

    df_int = Int(fm[:df])
    if df_int > 0
        d = Distributions.Chisq(df_int)
        fm[:pvalue] = 1.0 - Distributions.cdf(d, T_ML)
    else
        fm[:pvalue] = NaN
    end

    # ── Satorra-Bentler scaled test statistic (MLM estimator) ────────────────
    if opts.estimator == :MLM
        c = satorra_bentler_c(model, stats, fit.data, implied, theta_hat, df_int)
        fm[:scaling_factor] = c
        if isfinite(c) && c > 0 && df_int > 0
            T_scaled = T_ML / c
            fm[:chisq_scaled]  = T_scaled
            fm[:pvalue_scaled] = 1.0 - Distributions.cdf(Distributions.Chisq(df_int), T_scaled)
        else
            fm[:chisq_scaled]  = NaN
            fm[:pvalue_scaled] = NaN
        end
    end

    # ── Log-likelihood and information criteria ───────────────────────────────
    logl = ml_loglikelihood(theta_hat, model, stats)
    fm[:logl] = logl
    fm[:aic]  = -2 * logl + 2 * nθ
    fm[:bic]  = -2 * logl + log(N) * nθ
    # SABIC (sample-size adjusted BIC)
    fm[:sabic] = -2 * logl + log((N + 2) / 24) * nθ

    # ── Baseline model (Gap G6) ───────────────────────────────────────────────
    bl_result = fit_baseline_model(model, stats, opts)

    if bl_result !== nothing
        F_bl, df_bl = bl_result

        T_BL = Float64(N) * F_bl
        fm[:baseline_chisq] = T_BL
        fm[:baseline_df]    = Float64(df_bl)

        if df_bl > 0
            d_bl = Distributions.Chisq(df_bl)
            fm[:baseline_pvalue] = 1.0 - Distributions.cdf(d_bl, T_BL)
        else
            fm[:baseline_pvalue] = NaN
        end

        # ── CFI: comparative fit index ─────────────────────────────────────
        # CFI = 1 − max(T_M − df_M, 0) / max(T_B − df_B, T_M − df_M, 0)
        d_model    = max(T_ML - fm[:df], 0.0)
        d_baseline = max(T_BL - Float64(df_bl), 0.0)
        denom_cfi  = max(d_baseline, d_model)
        fm[:cfi] = denom_cfi > 0 ? 1.0 - d_model / denom_cfi : 1.0

        # ── TLI (NNFI): Tucker-Lewis index ────────────────────────────────
        # TLI = (T_B/df_B − T_M/df_M) / (T_B/df_B − 1)
        if df_int > 0 && df_bl > 0
            ratio_bl = T_BL / Float64(df_bl)
            ratio_m  = T_ML / fm[:df]
            denom_tli = ratio_bl - 1.0
            fm[:tli] = denom_tli > 0 ? (ratio_bl - ratio_m) / denom_tli : NaN
        else
            fm[:tli] = NaN
        end
    else
        # Baseline fit failed — CFI/TLI unavailable
        fm[:baseline_chisq]  = NaN
        fm[:baseline_df]     = NaN
        fm[:baseline_pvalue] = NaN
        fm[:cfi]             = NaN
        fm[:tli]             = NaN
        push!(fit.warnings, "Baseline model did not converge. CFI/TLI are NaN.")
    end

    # ── RMSEA ─────────────────────────────────────────────────────────────────
    if df_int > 0 && N > 0
        fm[:rmsea] = sqrt(max(T_ML - fm[:df], 0.0) / (fm[:df] * Float64(N)))
        (rmsea_lo, rmsea_hi) = rmsea_ci(T_ML, df_int, N)
        fm[:rmsea_ci_lower] = rmsea_lo
        fm[:rmsea_ci_upper] = rmsea_hi
        # pclose: P(RMSEA ≤ 0.05)
        ncp_close = max(T_ML - fm[:df], 0.0)
        if df_int > 0 && ncp_close >= 0
            threshold = 0.05^2 * df_int * Float64(N)
            fm[:rmsea_pvalue] = robust_ncp_ccdf(df_int, ncp_close, threshold + df_int)
        else
            fm[:rmsea_pvalue] = NaN
        end
    else
        fm[:rmsea]           = NaN
        fm[:rmsea_ci_lower]  = NaN
        fm[:rmsea_ci_upper]  = NaN
        fm[:rmsea_pvalue]    = NaN
    end

    # ── SRMR ──────────────────────────────────────────────────────────────────
    fm[:srmr] = compute_srmr(implied, stats)

    fit.fit_measures = fm
    return fm
end
