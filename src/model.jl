# ─── model.jl ─────────────────────────────────────────────────────────────────
# Build the LavaanModel (LISREL matrix templates) from a completed ParTable.
# Mirrors lavaan's lav_lavaan_step09_model.R + lav_representation_lisrel.R
#
# LISREL matrix mapping (Phase 1: continuous data, no conditional.x):
#   lhs =~ rhs  (rhs ∈ ov)  → Lambda[ov_idx(rhs), lv_idx(lhs)]
#   lhs =~ rhs  (rhs ∈ lv)  → Beta[lv_idx(rhs), lv_idx(lhs)]  (higher-order)
#   lhs ~ rhs   (both ∈ lv) → Beta[lv_idx(lhs), lv_idx(rhs)]
#   lhs ~~ rhs  (both ∈ ov) → Theta[ov_idx(lhs), ov_idx(rhs)]
#   lhs ~~ rhs  (either ∈lv)→ Psi[lv_idx(lhs), lv_idx(rhs)]
#   lhs ~1      (lhs ∈ ov)  → Nu[ov_idx(lhs)]
#   lhs ~1      (lhs ∈ lv)  → Alpha[lv_idx(lhs)]
# ─────────────────────────────────────────────────────────────────────────────

"""
    build_model(pt, ov_names, lv_names, opts) → LavaanModel
"""
function build_model(pt::ParTable,
                     ov_names::Vector{String},
                     lv_names::Vector{String},
                     opts::LavaanOptions)::LavaanModel

    lv_set = Set(lv_names)
    ov_set = Set(ov_names)
    ov_idx = Dict(v => i for (i,v) in enumerate(ov_names))
    lv_idx = Dict(v => i for (i,v) in enumerate(lv_names))

    p = length(ov_names)
    q = length(lv_names)

    # Determine endogenous and exogenous latent variables
    en_lv = endo_lv_names(pt, lv_names)
    ex_lv = exo_lv_names(pt, lv_names)

    # Single-block for Phase 1 (Phase 4 extends to multiple blocks)
    nblocks = maximum(pt.block; init=1)

    blocks = Vector{LISRELBlock}(undef, nblocks)
    for g in 1:nblocks
        blocks[g] = _build_block(pt, g, ov_names, lv_names, ov_idx, lv_idx, p, q, opts)
    end

    total_nfree = nfree(pt)

    return LavaanModel(
        blocks, nblocks, total_nfree,
        ov_names, lv_names, en_lv, ex_lv,
        pt, opts,
    )
end

function _build_block(pt::ParTable, g::Int,
                      ov_names, lv_names,
                      ov_idx, lv_idx,
                      p, q, opts)::LISRELBlock

    lv_set = Set(lv_names)
    ov_set = Set(ov_names)

    # ── Initialize matrix templates ────────────────────────────────────────
    Lambda = zeros(p, q)
    Theta  = zeros(p, p)
    Psi    = zeros(q, q)
    Beta   = zeros(q, q)
    Nu     = zeros(p)
    Alpha  = zeros(q)

    Lambda_free = fill(false, p, q)
    Theta_free  = fill(false, p, p)
    Psi_free    = fill(false, q, q)
    Beta_free   = fill(false, q, q)
    Nu_free     = fill(false, p)
    Alpha_free  = fill(false, q)

    # free_map: (matrix_sym, row, col) indexed by parTable row index
    free_map  = Tuple{Symbol,Int,Int}[]
    free_fidx = Int[]   # which θ index

    # ── Iterate parTable rows for this block ──────────────────────────────
    for i in eachindex(pt.id)
        pt.block[i] != g && continue
        pt.op[i] in ("==", ":=", "<", ">", ":") && continue

        lhs = pt.lhs[i]
        op  = pt.op[i]
        rhs = pt.rhs[i]

        # Get scalar value for this row (ustart or 0)
        val = ismissing(pt.ustart[i]) ? 0.0 : Float64(pt.ustart[i])
        is_free = pt.free[i] > 0

        if op == "=~"
            if haskey(ov_idx, rhs)
                # Standard measurement: Lambda[ov, lv]
                r, c = ov_idx[rhs], lv_idx[lhs]
                Lambda[r, c]      = val
                Lambda_free[r, c] = is_free
                if is_free
                    push!(free_map, (:lambda, r, c))
                    push!(free_fidx, pt.free[i])
                end
                # Store matrix location back into parTable
                pt.mat[i]     = "lambda"
                pt.mat_row[i] = r
                pt.mat_col[i] = c
            elseif haskey(lv_idx, rhs)
                # Higher-order CFA: Beta[lv_rhs, lv_lhs]
                r, c = lv_idx[rhs], lv_idx[lhs]
                Beta[r, c]      = val
                Beta_free[r, c] = is_free
                if is_free
                    push!(free_map, (:beta, r, c))
                    push!(free_fidx, pt.free[i])
                end
                pt.mat[i]     = "beta"
                pt.mat_row[i] = r
                pt.mat_col[i] = c
            end

        elseif op == "~"
            if haskey(lv_idx, lhs) && haskey(lv_idx, rhs)
                # Latent regression: Beta[lhs, rhs]
                r, c = lv_idx[lhs], lv_idx[rhs]
                Beta[r, c]      = val
                Beta_free[r, c] = is_free
                if is_free
                    push!(free_map, (:beta, r, c))
                    push!(free_fidx, pt.free[i])
                end
                pt.mat[i]     = "beta"
                pt.mat_row[i] = r
                pt.mat_col[i] = c
            end
            # Note: observed regression (lhs ∈ ov) requires dummy LV expansion
            # This is deferred to Phase 4 full SEM support

        elseif op == "~~"
            if haskey(lv_idx, lhs) && haskey(lv_idx, rhs)
                # Latent (co)variance: Psi
                r, c = lv_idx[lhs], lv_idx[rhs]
                Psi[r, c]      = val
                Psi[c, r]      = val  # symmetric
                Psi_free[r, c] = is_free
                Psi_free[c, r] = is_free
                if is_free
                    push!(free_map, (:psi, r, c))
                    push!(free_fidx, pt.free[i])
                    if r != c
                        push!(free_map, (:psi, c, r))
                        push!(free_fidx, pt.free[i])
                    end
                end
                pt.mat[i]     = "psi"
                pt.mat_row[i] = r
                pt.mat_col[i] = c
            elseif haskey(ov_idx, lhs) && haskey(ov_idx, rhs)
                # Observed residual (co)variance: Theta
                r, c = ov_idx[lhs], ov_idx[rhs]
                Theta[r, c]      = val
                Theta[c, r]      = val
                Theta_free[r, c] = is_free
                Theta_free[c, r] = is_free
                if is_free
                    push!(free_map, (:theta, r, c))
                    push!(free_fidx, pt.free[i])
                    if r != c
                        push!(free_map, (:theta, c, r))
                        push!(free_fidx, pt.free[i])
                    end
                end
                pt.mat[i]     = "theta"
                pt.mat_row[i] = r
                pt.mat_col[i] = c
            end

        elseif op == "~1"
            if haskey(ov_idx, lhs)
                r = ov_idx[lhs]
                Nu[r]      = val
                Nu_free[r] = is_free
                if is_free
                    push!(free_map, (:nu, r, 1))
                    push!(free_fidx, pt.free[i])
                end
                pt.mat[i]     = "nu"
                pt.mat_row[i] = r
                pt.mat_col[i] = 1
            elseif haskey(lv_idx, lhs)
                r = lv_idx[lhs]
                Alpha[r]      = val
                Alpha_free[r] = is_free
                if is_free
                    push!(free_map, (:alpha, r, 1))
                    push!(free_fidx, pt.free[i])
                end
                pt.mat[i]     = "alpha"
                pt.mat_row[i] = r
                pt.mat_col[i] = 1
            end
        end
    end

    return LISRELBlock(
        Lambda, Theta, Psi, Beta, Nu, Alpha,
        Lambda_free, Theta_free, Psi_free, Beta_free, Nu_free, Alpha_free,
        free_map, free_fidx, p, q,
    )
end

# ─── Fill LISREL matrices from θ vector ───────────────────────────────────────

"""
    fill_block!(block, theta)

Update LISREL matrix templates from the free parameter vector θ.
This is called inside the optimizer on every function evaluation.

The type parameter T allows ForwardDiff Dual numbers to flow through.
"""
function fill_block(block::LISRELBlock, theta::AbstractVector{T}) where T
    # Create mutable matrix views with type T
    Lambda = convert(Matrix{T}, block.Lambda)
    Theta  = convert(Matrix{T}, block.Theta)
    Psi    = convert(Matrix{T}, block.Psi)
    Beta   = convert(Matrix{T}, block.Beta)
    Nu     = convert(Vector{T}, block.Nu)
    Alpha  = convert(Vector{T}, block.Alpha)

    # Apply free parameters
    for k in eachindex(block.free_map)
        mat_sym, r, c = block.free_map[k]
        θ_idx = block.free_idx[k]
        val = theta[θ_idx]

        if mat_sym == :lambda
            Lambda[r, c] = val
        elseif mat_sym == :theta
            Theta[r, c] = val
        elseif mat_sym == :psi
            Psi[r, c] = val
        elseif mat_sym == :beta
            Beta[r, c] = val
        elseif mat_sym == :nu
            Nu[r] = val
        elseif mat_sym == :alpha
            Alpha[r] = val
        end
    end

    return (Lambda=Lambda, Theta=Theta, Psi=Psi, Beta=Beta, Nu=Nu, Alpha=Alpha)
end
