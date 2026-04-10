# ─── start.jl ─────────────────────────────────────────────────────────────────
# Compute starting values for all free parameters.
# Mirrors lavaan's lav_lavaan_step08_start.R + lav_start.R
#
# Strategy (lavaan "default" method):
#   =~  loadings        → 1.0 (free first indicators were already fixed to 1)
#   ~~  observed vars   → 0.5 × sample variance
#   ~~  latent vars     → 0.05 (small positive)
#   ~~  covariances     → 0.0
#   ~   regressions     → 0.0
#   ~1  observed        → sample mean
#   ~1  latent          → 0.0
# ─────────────────────────────────────────────────────────────────────────────

"""
    set_starting_values!(pt, stats, ov_names, lv_names, opts)

Fill in the `start` column of `pt` using sample statistics.
User-specified `ustart` values take priority.
"""
function set_starting_values!(pt::ParTable,
                               stats::SampleStats,
                               ov_names::Vector{String},
                               lv_names::Vector{String},
                               opts::LavaanOptions)

    lv_set = Set(lv_names)
    ov_idx = Dict(v => i for (i,v) in enumerate(ov_names))

    for i in eachindex(pt.id)
        # User-specified starting value takes priority
        if !ismissing(pt.ustart[i])
            pt.start[i] = Float64(pt.ustart[i])
            continue
        end

        # Skip fixed (free=0) rows that already have ustart
        pt.free[i] == 0 && continue

        op  = pt.op[i]
        lhs = pt.lhs[i]
        rhs = pt.rhs[i]
        g   = pt.block[i]
        g   = max(1, g)     # safety: block 0 → block 1

        # Limit block index to available stats blocks
        gs  = min(g, length(stats.S))

        if op == "=~"
            # Loadings: use 1.0 as default
            pt.start[i] = 1.0

        elseif op == "~~"
            if lhs == rhs
                if lhs in lv_set
                    # Latent variance: start small
                    pt.start[i] = 0.05
                elseif haskey(ov_idx, lhs)
                    # Observed residual variance: 50% of sample variance
                    j = ov_idx[lhs]
                    pt.start[i] = 0.5 * stats.S[gs][j, j]
                else
                    pt.start[i] = 0.05
                end
            else
                # Covariance: start at 0
                pt.start[i] = 0.0
            end

        elseif op == "~"
            # Regression coefficient: start at small nonzero value.
            # Starting at exactly 0 creates a saddle point where
            # cross-factor covariances don't enter the ML gradient,
            # causing L-BFGS to falsely "converge" without moving Beta.
            pt.start[i] = 0.1

        elseif op == "~1"
            if haskey(ov_idx, lhs)
                # Observed intercept: sample mean
                j = ov_idx[lhs]
                pt.start[i] = stats.mu[gs][j]
            else
                # Latent intercept: 0
                pt.start[i] = 0.0
            end

        elseif op == "~*~"
            pt.start[i] = 1.0

        else
            pt.start[i] = 0.0
        end
    end
end
