# ─── print.jl ─────────────────────────────────────────────────────────────────
# Pretty-printing for LavaanFit objects.
# Mirrors lavaan's summary() output format.
# ─────────────────────────────────────────────────────────────────────────────

function Base.show(io::IO, fit::LavaanFit)
    println(io, "lavaan $(pkgversion_str()) fit object")
    println(io, "  Estimator:   $(fit.options.estimator)")
    println(io, "  Converged:   $(fit.converged ? "Yes" : "No (⚠ check results)")")
    if haskey(fit.fit_measures, :chisq)
        @printf(io, "  χ²(%.0f) = %.3f  [p = %.3f]\n",
                fit.fit_measures[:df], fit.fit_measures[:chisq], fit.fit_measures[:pvalue])
    end
    println(io, "  Call summary() for full output.")
end

"""
    summary(fit; fit_measures=false, standardized=false, rsquare=false)

Print a summary of the fitted lavaan model, mirroring R lavaan's output.
"""
function Base.summary(fit::LavaanFit;
                       fit_measures::Bool = false,
                       standardized::Bool = false,
                       rsquare::Bool = false)

    io = stdout
    fm = fit.fit_measures
    pt = fit.partable

    _hline(io, 72)

    # ── Header ────────────────────────────────────────────────────────────────
    println(io, "lavaan $(pkgversion_str()) -- $(uppercase(string(fit.options.model_type))) model")
    println(io)

    # ── Convergence info ──────────────────────────────────────────────────────
    if fit.converged
        println(io, "  ✓  Model converged normally")
    else
        println(io, "  ✗  WARNING: Model did NOT converge")
    end
    @printf(io, "     Number of observations:          %6d\n", fit.samplestats.ntotal)
    @printf(io, "     Estimator:                       %6s\n", string(fit.options.estimator))
    @printf(io, "     Number of model parameters:      %6d\n", fit.model.nfree)
    println(io)

    # ── Model test ────────────────────────────────────────────────────────────
    println(io, "Model Test User Model:")
    println(io, "  Test statistic:               $(fmt_f(get(fm, :chisq, NaN), 3))")
    println(io, "  Degrees of freedom:           $(fmt_i(get(fm, :df, NaN)))")
    println(io, "  P-value (Chi-square):         $(fmt_f(get(fm, :pvalue, NaN), 3))")
    println(io)

    # ── Fit measures (optional) ───────────────────────────────────────────────
    if fit_measures
        println(io, "Model Test Baseline Model:")
        @printf(io, "  Test statistic:               %s\n", fmt_f(get(fm, :baseline_chisq, NaN), 3))
        @printf(io, "  Degrees of freedom:           %s\n", fmt_i(get(fm, :baseline_df, NaN)))
        @printf(io, "  P-value:                      %s\n", fmt_f(get(fm, :baseline_pvalue, NaN), 3))
        println(io)

        println(io, "User Model versus Baseline Model:")
        @printf(io, "  Comparative Fit Index (CFI):  %s\n", fmt_f(get(fm, :cfi, NaN), 3))
        @printf(io, "  Tucker-Lewis Index (TLI):     %s\n", fmt_f(get(fm, :tli, NaN), 3))
        println(io)

        println(io, "Root Mean Square Error of Approximation:")
        @printf(io, "  RMSEA:                        %s\n", fmt_f(get(fm, :rmsea, NaN), 3))
        @printf(io, "  90%% Confidence Interval:      [%s, %s]\n",
                fmt_f(get(fm, :rmsea_ci_lower, NaN), 3),
                fmt_f(get(fm, :rmsea_ci_upper, NaN), 3))
        @printf(io, "  P-value H₀: RMSEA ≤ 0.050:   %s\n", fmt_f(get(fm, :rmsea_pvalue, NaN), 3))
        println(io)

        println(io, "Standardized Root Mean Square Residual:")
        @printf(io, "  SRMR:                         %s\n", fmt_f(get(fm, :srmr, NaN), 3))
        println(io)

        println(io, "Information Criteria:")
        @printf(io, "  Log-likelihood user model (H0): %s\n", fmt_f(get(fm, :logl, NaN), 3))
        @printf(io, "  AIC:                          %s\n", fmt_f(get(fm, :aic, NaN), 3))
        @printf(io, "  BIC:                          %s\n", fmt_f(get(fm, :bic, NaN), 3))
        @printf(io, "  BIC2 (Sample-size adjusted):  %s\n", fmt_f(get(fm, :sabic, NaN), 3))
        println(io)
    end

    # ── Parameter estimates ───────────────────────────────────────────────────
    println(io, "Parameter Estimates:")
    @printf(io, "  Standard errors:              %s\n", string(fit.options.se))
    @printf(io, "  Information:                  %s\n", string(fit.options.information))
    println(io)

    # Group parameter table by operator section
    _print_section(io, pt, "=~", "Latent Variable Definitions")
    _print_section(io, pt, "~",  "Regressions")
    _print_section(io, pt, "~~", "Variances and Covariances")
    _print_section(io, pt, "~1", "Intercepts")

    # ── Warnings ──────────────────────────────────────────────────────────────
    if !isempty(fit.warnings)
        println(io)
        println(io, "Warnings:")
        for w in fit.warnings
            println(io, "  ⚠ ", w)
        end
    end

    _hline(io, 72)
    println(io)
end

# ─── Section printer ──────────────────────────────────────────────────────────

function _print_section(io::IO, pt::ParTable, op_filter::String, title::String)
    # Collect relevant rows
    idxs = [i for i in eachindex(pt.id) if pt.op[i] == op_filter]
    isempty(idxs) && return

    println(io, title * ":")
    println(io)

    # Column header
    @printf(io, "  %-12s %-4s %-12s %8s %8s %6s %8s\n",
            "Observed", "", "Variable", "Estimate", "Std.Err", "z-value", "P(>|z|)")
    println(io, "  " * "─"^60)

    prev_lhs = ""
    for i in idxs
        lhs_v = pt.lhs[i]
        rhs_v = pt.rhs[i]

        # Print lhs only when it changes (grouped layout)
        lhs_str = lhs_v == prev_lhs ? "" : lhs_v
        prev_lhs = lhs_v

        # Format numeric fields
        est_s  = isnan(pt.est[i])    ? "    NA" : @sprintf("%8.3f", pt.est[i])
        se_s   = isnan(pt.se[i])     ? "    NA" : @sprintf("%8.3f", pt.se[i])
        z_s    = isnan(pt.z[i])      ? "      " : @sprintf("%6.3f", pt.z[i])
        p_s    = isnan(pt.pvalue[i]) ? "      NA" : @sprintf("%8.3f", pt.pvalue[i])

        # Fixed parameters: show "fixed" label instead of SE/z/p
        if pt.free[i] == 0
            @printf(io, "  %-12s %-4s %-12s %8s   (fixed)\n",
                    lhs_str, pt.op[i], rhs_v, est_s)
        else
            @printf(io, "  %-12s %-4s %-12s %s %s %s %s\n",
                    lhs_str, pt.op[i], rhs_v, est_s, se_s, z_s, p_s)
        end
    end
    println(io)
end

# ─── Helpers ──────────────────────────────────────────────────────────────────

fmt_f(x::Float64, digits::Int) = isnan(x) ? "   NA" : @sprintf("%.*f", digits, x)
fmt_i(x::Float64) = isnan(x) ? " NA" : string(Int(x))
fmt_i(x::Int) = string(x)

_hline(io, n) = println(io, "─"^n)

pkgversion_str() = "0.1.0-dev"
