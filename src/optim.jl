# ─── optim.jl ─────────────────────────────────────────────────────────────────
# Optimization engine: minimize the discrepancy function over free parameters θ.
# Uses Optim.jl with ForwardDiff gradients.
# Mirrors lavaan's lav_lavaan_step11_optim.R + lav_model_optim.R
#
# Non-convergence contract (Gap G2):
#   - NEVER throw on non-convergence
#   - Return (theta_hat, converged, iterations, optim_result)
# ─────────────────────────────────────────────────────────────────────────────

"""
    optimize_model(model, stats, opts) → (theta_hat, converged, iters, result)

Minimize the discrepancy function F(θ) and return the estimated parameter vector.

Returns:
  theta_hat  : estimated free parameters (NaN-filled if not converged)
  converged  : Bool
  iters      : number of optimizer iterations
  result     : raw Optim.jl result object
"""
function optimize_model(model::LavaanModel,
                        stats::SampleStats,
                        opts::LavaanOptions;
                        theta0::Union{Nothing,Vector{Float64}} = nothing)

    theta0 = theta0 !== nothing ? theta0 : get_start(model.partable)
    nθ     = length(theta0)

    if nθ == 0
        # Saturated or zero-df model — nothing to optimize
        return (theta0, true, 0, nothing)
    end

    # ── Build bounds ─────────────────────────────────────────────────────────
    lb = fill(-Inf, nθ)
    ub = fill(+Inf, nθ)
    pt = model.partable
    for i in eachindex(pt.free)
        pt.free[i] == 0 && continue
        k = pt.free[i]
        pt.lower[i] > -Inf && (lb[k] = pt.lower[i])
        pt.upper[i] < +Inf && (ub[k] = pt.upper[i])
    end
    has_bounds = any(isfinite, lb) || any(isfinite, ub)

    # ── Objective closure ─────────────────────────────────────────────────────
    if opts.estimator == :GSEM
        obj = theta -> gsem_objective(theta, model, stats)
    else
        obj = theta -> estimator_objective(theta, model, stats, opts.estimator)
    end

    # ── Run optimizer ─────────────────────────────────────────────────────────
    optim_opts = Optim.Options(
        g_tol      = opts.optim_tol,
        iterations = opts.optim_iter,
        show_trace = opts.debug,
    )

    result = try
        if has_bounds
            optimize(obj, lb, ub, theta0, Fminbox(LBFGS()),
                     optim_opts; autodiff=:forward)
        else
            optimize(obj, theta0, LBFGS(), optim_opts; autodiff=:forward)
        end
    catch e
        @warn "Optimizer threw an error: $e"
        return (fill(NaN, nθ), false, 0, nothing)
    end

    converged  = Optim.converged(result)
    theta_hat  = Optim.minimizer(result)
    iters      = Optim.iterations(result)

    if !converged
        @warn "Model did not converge after $iters iterations. " *
              "Parameter estimates may be unreliable. " *
              "Try different starting values or check model specification."
    end

    return (theta_hat, converged, iters, result)
end
