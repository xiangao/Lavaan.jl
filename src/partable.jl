# ─── partable.jl ──────────────────────────────────────────────────────────────
# Build a complete ParTable from parsed syntax rows + data variable names + options.
#
# Mirrors lavaan's lav_partable.R + lav_partable_flat.R pipeline.
#
# Steps:
#   1. Copy user-specified rows from ParsedRow vector
#   2. Auto-add variances (residual and latent) if auto_var=true
#   3. Auto-add latent covariances if auto_cov_lv_x=true
#   4. Auto-add observed covariances (auto_cov_y) if set
#   5. Auto-add intercepts if meanstructure=true
#   6. Fix first loading per factor if auto_fix_first=true
#   7. Fix single-indicator residuals to 0 if auto_fix_single=true
#   8. Assign free parameter indices (globally unique across blocks)
#   9. Assign internal plabels (.p1., .p2., ...)
#  10. Handle equality constraints (shared free index)
# ─────────────────────────────────────────────────────────────────────────────

"""
    build_partable(rows, ov_names, lv_names, opts; ngroups=1) → ParTable

Construct a complete ParTable from parsed model rows.

# Arguments
- `rows`     : output of `parse_model_string`
- `ov_names` : observed variable names (from data)
- `lv_names` : latent variable names (from model, in order)
- `opts`     : LavaanOptions
- `ngroups`  : number of groups (1 by default)
"""
function build_partable(rows::Vector{ParsedRow},
                        ov_names::Vector{String},
                        lv_names::Vector{String},
                        opts::LavaanOptions;
                        ngroups::Int = 1)::ParTable

    all_names = union(ov_names, lv_names)
    
    nlevels = 1 + length(opts.clusters)
    nblocks = ngroups * nlevels

    # ── Step 1: collect user-specified rows ───────────────────────────────────
    # For multi-group/multi-level: replicate rows or assign blocks
    user_rows = _collect_user_rows(rows, ov_names, lv_names, opts, ngroups, nlevels)

    # ── Step 2: auto-add residual variances for observed endogenous vars ──────
    if opts.auto_var
        user_rows = _add_residual_variances(user_rows, ov_names, lv_names, opts, nblocks)
    end

    # ── Step 3: auto-add latent variable variances ────────────────────────────
    user_rows = _add_latent_variances(user_rows, lv_names, opts, nblocks)

    # ── Step 4: auto-add latent covariances (exogenous latents) ──────────────
    if opts.auto_cov_lv_x
        user_rows = _add_latent_covariances(user_rows, lv_names, opts, nblocks)
    end

    # ── Step 5: auto-add endogenous observed covariances ─────────────────────
    if opts.auto_cov_y
        user_rows = _add_observed_covariances(user_rows, ov_names, lv_names, opts, nblocks)
    end

    # ── Step 6: auto-add mean structure ──────────────────────────────────────
    if opts.meanstructure
        user_rows = _add_intercepts(user_rows, ov_names, lv_names, opts, nblocks)
    end

    # ── Step 7: Fix first loading per factor ──────────────────────────────────
    if opts.auto_fix_first && !opts.std_lv
        _fix_first_loadings!(user_rows, lv_names)
    end
    # std_lv: fix latent variances to 1 instead
    if opts.std_lv
        _fix_latent_variances!(user_rows, lv_names)
    end

    # ── Step 8: Fix single-indicator residuals to 0 ───────────────────────────
    if opts.auto_var && opts.auto_fix_single
        _fix_single_indicators!(user_rows, lv_names)
    end

    # ── Step 9: Handle equality constraints ───────────────────────────────────
    label_map = _collect_label_map(user_rows)  # label → eq_id
    _assign_eq_ids!(user_rows, label_map)

    # ── Step 10: Assign free parameter indices ────────────────────────────────
    _assign_free_indices!(user_rows, label_map)

    # ── Step 11: Assign internal plabels ─────────────────────────────────────
    _assign_plabels!(user_rows)

    # ── Build ParTable struct ─────────────────────────────────────────────────
    n = length(user_rows)
    pt = ParTable(n)
    for (i, r) in enumerate(user_rows)
        pt.id[i]       = i
        pt.lhs[i]      = r.lhs
        pt.op[i]       = r.op
        pt.rhs[i]      = r.rhs
        pt.user[i]     = r.user
        pt.block[i]    = r.block
        pt.group[i]    = r.group
        pt.level[i]    = r.level
        pt.free[i]     = r.free
        pt.ustart[i]   = r.ustart
        pt.exo[i]      = r.exo
        pt.label[i]    = r.label
        pt.plabel[i]   = r.plabel
        pt.eq_id[i]    = r.eq_id
        pt.lower[i]    = r.lower
        pt.upper[i]    = r.upper
    end

    return pt
end

# ─── Internal mutable row (used during construction) ──────────────────────────

mutable struct PTRow
    lhs::String
    op::String
    rhs::String
    user::Int       # 1=user, 0=auto
    block::Int
    group::String
    level::String
    free::Int       # 0=fixed, >0=free index
    ustart::Union{Float64,Missing}
    exo::Bool
    label::String
    plabel::String
    eq_id::Int
    lower::Float64
    upper::Float64
end

function PTRow(lhs, op, rhs;
               user=1, block=1, group="", level="",
               free=1,          # will be reassigned
               ustart=missing, exo=false,
               label="", plabel="", eq_id=0,
               lower=-Inf, upper=Inf)
    PTRow(string(lhs), string(op), string(rhs),
          user, block, group, level, free, ustart, exo,
          label, plabel, eq_id, lower, upper)
end

# Check if a row already exists (to avoid duplicates)
function _row_exists(rows::Vector{PTRow}, lhs, op, rhs, block)
    for r in rows
        r.lhs == lhs && r.op == op && r.rhs == rhs && r.block == block && return true
    end
    return false
end

# ─── Step 1: collect user rows ────────────────────────────────────────────────

function _collect_user_rows(parsed::Vector{ParsedRow},
                             ov_names, lv_names, opts, ngroups, nlevels)::Vector{PTRow}
    rows = PTRow[]
    for pr in parsed
        # Constraint rows (==, :=, <, >)
        if pr.op in ("==", ":=", "<", ">")
            push!(rows, PTRow(pr.lhs, pr.op, pr.rhs;
                              user=1, free=0,
                              block=pr.block, group=pr.group, level=pr.level))
            continue
        end

        # Determine fixed vs free
        is_fixed = pr.fixed !== nothing
        ustart_val = is_fixed ? pr.fixed : pr.start

        # Determine lower/upper bounds
        lb = pr.lower
        ub = pr.upper

        # Set free=0 for fixed, 1 placeholder (will be reassigned) for free
        free_val = is_fixed ? 0 : 1
        
        # If user did not specify level in a multilevel setting, R lavaan puts it in block 1
        b = max(1, pr.block)

        push!(rows, PTRow(
            pr.lhs, pr.op, pr.rhs;
            user   = 1,
            block  = b,
            group  = pr.group,
            level  = pr.level,
            free   = free_val,
            ustart = is_fixed ? pr.fixed : (pr.start !== nothing ? pr.start : missing),
            label  = pr.label,
            lower  = lb,
            upper  = ub,
        ))
    end
    return rows
end

# ─── Step 2: auto-add residual variances for endogenous observed vars ─────────

function _add_residual_variances(rows::Vector{PTRow}, ov_names, lv_names, opts, nblocks)
    # Endogenous observed: appear as LHS of ~ or as indicators (RHS of =~)
    endo_ov = Set{String}()
    for r in rows
        r.op == "=~" && r.rhs != "1" && !(r.rhs in lv_names) && push!(endo_ov, r.rhs)
        r.op == "~"  && r.lhs in ov_names && push!(endo_ov, r.lhs)
    end
    # If auto_cov_y: all observed are endogenous
    if opts.auto_cov_y
        union!(endo_ov, ov_names)
    end

    for g in 1:nblocks
        for v in sort(collect(endo_ov))
            _row_exists(rows, v, "~~", v, g) && continue
            push!(rows, PTRow(v, "~~", v; user=0, block=g, free=1, lower=0.0))
        end
    end
    return rows
end

# ─── Step 3: auto-add latent variable variances ───────────────────────────────

function _add_latent_variances(rows::Vector{PTRow}, lv_names, opts, ngroups)
    for g in 1:ngroups
        for lv in lv_names
            _row_exists(rows, lv, "~~", lv, g) && continue
            push!(rows, PTRow(lv, "~~", lv; user=0, block=g, free=1, lower=0.0))
        end
    end
    return rows
end

# ─── Step 4: auto-add latent covariances for exogenous latents ────────────────

function _add_latent_covariances(rows::Vector{PTRow}, lv_names, opts, ngroups)
    # Exogenous latents: appear as LHS of =~ but NOT as LHS of ~ (not regressed upon)
    endo_lv = Set{String}()
    for r in rows
        r.op == "~" && r.lhs in lv_names && push!(endo_lv, r.lhs)
    end
    exo_lv = [lv for lv in lv_names if !(lv in endo_lv)]

    for g in 1:ngroups
        for i in 1:length(exo_lv)
            for j in (i+1):length(exo_lv)
                a, b = exo_lv[i], exo_lv[j]
                _row_exists(rows, a, "~~", b, g) && continue
                _row_exists(rows, b, "~~", a, g) && continue
                push!(rows, PTRow(a, "~~", b; user=0, block=g, free=1))
            end
        end
    end
    return rows
end

# ─── Step 5: auto-add observed covariances ────────────────────────────────────
function _add_observed_covariances(rows::Vector{PTRow}, ov_names, lv_names, opts, nblocks)
    # Only for endogenous observed vars
    endo_ov = [v for v in ov_names if any(r.op == "~" && r.lhs == v for r in rows)]
    for g in 1:nblocks
        for i in 1:length(endo_ov)
            for j in (i+1):length(endo_ov)
                a, b = endo_ov[i], endo_ov[j]
                _row_exists(rows, a, "~~", b, g) && continue
                _row_exists(rows, b, "~~", a, g) && continue
                push!(rows, PTRow(a, "~~", b; user=0, block=g, free=1))
            end
        end
    end
    return rows
end

# ─── Step 6: auto-add intercepts ─────────────────────────────────────────────

function _add_intercepts(rows::Vector{PTRow}, ov_names, lv_names, opts, ngroups)
    for g in 1:ngroups
        for v in ov_names
            opts.int_ov_free || continue  # only if intercepts are free
            _row_exists(rows, v, "~1", "1", g) && continue
            push!(rows, PTRow(v, "~1", "1"; user=0, block=g, free=1))
        end
        for lv in lv_names
            opts.int_lv_free || continue
            _row_exists(rows, lv, "~1", "1", g) && continue
            push!(rows, PTRow(lv, "~1", "1"; user=0, block=g, free=1))
        end
    end
    return rows
end

# ─── Step 7: Fix first loading per factor ─────────────────────────────────────

function _fix_first_loadings!(rows::Vector{PTRow}, lv_names)
    seen = Set{Tuple{String,Int}}()  # (lv_name, block) already processed
    for r in rows
        r.op != "=~" && continue
        r.user != 1 && continue  # only user-specified loadings
        key = (r.lhs, r.block)
        key in seen && continue
        push!(seen, key)
        # Fix this loading to 1.0 unless user already fixed it
        if r.free == 1  # still marked as free
            r.free   = 0
            r.ustart = 1.0
        end
    end
end

function _fix_latent_variances!(rows::Vector{PTRow}, lv_names)
    for r in rows
        r.op != "~~" && continue
        r.lhs != r.rhs && continue
        !(r.lhs in lv_names) && continue
        r.free   = 0
        r.ustart = 1.0
    end
end

# ─── Step 8: Fix single-indicator residuals ───────────────────────────────────

function _fix_single_indicators!(rows::Vector{PTRow}, lv_names)
    # Count indicators per latent variable per block
    ind_count = Dict{Tuple{String,Int},Int}()
    ind_name  = Dict{Tuple{String,Int},String}()
    for r in rows
        r.op != "=~" && continue
        key = (r.lhs, r.block)
        ind_count[key] = get(ind_count, key, 0) + 1
        ind_name[key]  = r.rhs  # last one wins, but if count==1 it's the only
    end
    single_inds = Set{Tuple{String,Int}}()
    for (key, cnt) in ind_count
        cnt == 1 && push!(single_inds, (ind_name[key], key[2]))
    end
    # Fix residual variance of single indicators to 0
    for r in rows
        r.op != "~~" && continue
        r.lhs != r.rhs && continue
        r.user != 0 && continue   # only auto-added residuals
        (r.lhs, r.block) in single_inds || continue
        r.free   = 0
        r.ustart = 0.0
    end
end

# ─── Step 9-10: Equality constraints, free indices, plabels ───────────────────

"""
Build a map from user label → equality constraint ID.
All rows sharing the same label will share the same free parameter index.
"""
function _collect_label_map(rows::Vector{PTRow})::Dict{String,Int}
    label_map = Dict{String,Int}()
    eq_id = 0
    for r in rows
        isempty(r.label) && continue
        haskey(label_map, r.label) && continue
        eq_id += 1
        label_map[r.label] = eq_id
    end
    return label_map
end

function _assign_eq_ids!(rows::Vector{PTRow}, label_map::Dict{String,Int})
    for r in rows
        isempty(r.label) && continue
        r.eq_id = get(label_map, r.label, 0)
    end
end

"""
Assign globally unique free parameter indices.
Rows with the same label share the same free index (equality constraint).
"""
function _assign_free_indices!(rows::Vector{PTRow}, label_map::Dict{String,Int})
    label_to_free = Dict{String,Int}()  # label → assigned free index
    next_free = 1

    for r in rows
        r.free == 0 && continue  # fixed parameter: skip

        if !isempty(r.label)
            if haskey(label_to_free, r.label)
                r.free = label_to_free[r.label]
            else
                r.free = next_free
                label_to_free[r.label] = next_free
                next_free += 1
            end
        else
            r.free = next_free
            next_free += 1
        end
    end
end

function _assign_plabels!(rows::Vector{PTRow})
    for (i, r) in enumerate(rows)
        r.plabel = ".p$(i)."
    end
end

# ─── Utility: variable name extraction from parTable ──────────────────────────

"""Return latent variable names in order of first appearance."""
function lv_names(pt::ParTable)::Vector{String}
    seen = String[]
    seen_set = Set{String}()
    for i in eachindex(pt.op)
        if pt.op[i] in ("=~", "<~") && !(pt.lhs[i] in seen_set)
            push!(seen, pt.lhs[i])
            push!(seen_set, pt.lhs[i])
        end
    end
    return seen
end

"""Return observed variable names (variables not in lv_names)."""
function ov_names(pt::ParTable, lv::Vector{String})::Vector{String}
    seen = Set{String}()
    lv_set = Set(lv)
    for i in eachindex(pt.op)
        pt.op[i] in ("==", ":=", "<", ">", ":") && continue
        v = pt.rhs[i]
        v != "1" && !(v in lv_set) && push!(seen, v)
        v = pt.lhs[i]
        !(v in lv_set) && push!(seen, v)
    end
    return sort(collect(seen))
end

"""
Endogenous latent variables: LHS of ~ with latent variable.
Exogenous latent variables: latent variables NOT appearing as LHS of ~.
"""
function endo_lv_names(pt::ParTable, lv::Vector{String})::Vector{String}
    lv_set = Set(lv)
    endo = Set{String}()
    for i in eachindex(pt.op)
        pt.op[i] == "~" && pt.lhs[i] in lv_set && push!(endo, pt.lhs[i])
    end
    return [v for v in lv if v in endo]
end

function exo_lv_names(pt::ParTable, lv::Vector{String})::Vector{String}
    endo = Set(endo_lv_names(pt, lv))
    return [v for v in lv if !(v in endo)]
end

"""
    _inject_threshold_rows!(pt, thresholds, threshold_ses)

Append threshold parameter rows (op="|") to the ParTable for display in
parameterEstimates(). Thresholds are NOT free structural parameters (free=0).
"""
function _inject_threshold_rows!(pt::ParTable,
                                  thresholds::Dict{String,Vector{Float64}},
                                  threshold_ses::Union{Nothing,Dict{String,Vector{Float64}}})
    for (var, th) in sort(collect(thresholds), by=first)
        for (k, tau) in enumerate(th)
            n = length(pt.id) + 1
            push!(pt.id,       n)
            push!(pt.lhs,      var)
            push!(pt.op,       "|")
            push!(pt.rhs,      "t$k")
            push!(pt.user,     0)
            push!(pt.block,    1)
            push!(pt.group,    "")
            push!(pt.level,    "")
            push!(pt.free,     0)
            push!(pt.ustart,   missing)
            push!(pt.exo,      false)
            push!(pt.label,    "")
            push!(pt.plabel,   "")
            push!(pt.eq_id,    0)
            push!(pt.lower,    -Inf)
            push!(pt.upper,    Inf)
            push!(pt.start,    tau)
            push!(pt.est,      tau)
            se_val = (threshold_ses !== nothing && haskey(threshold_ses, var)) ?
                     threshold_ses[var][k] : NaN
            push!(pt.se,       se_val)
            push!(pt.z,        isnan(se_val) ? NaN : tau / se_val)
            push!(pt.pvalue,   NaN)
            push!(pt.ci_lower, isnan(se_val) ? NaN : tau - 1.96 * se_val)
            push!(pt.ci_upper, isnan(se_val) ? NaN : tau + 1.96 * se_val)
            push!(pt.mat,      "")
            push!(pt.mat_row,  0)
            push!(pt.mat_col,  0)
        end
    end
end

"""
    _fix_poisson_residuals!(pt, family)

For any variable declared as :poisson in `family`, fix its residual variance
row (op="~~", lhs==rhs) to 0 and mark as not-free.

This is required because Poisson has no free dispersion parameter — the variance
equals the mean and is determined by λ, not a separate Θ_jj entry.
"""
function _fix_poisson_residuals!(pt::ParTable, family::Dict{String,Symbol})
    isempty(family) && return
    for k in eachindex(pt.op)
        pt.op[k] == "~~" || continue
        pt.lhs[k] == pt.rhs[k] || continue   # must be a variance (not covariance)
        varname = pt.lhs[k]
        get(family, varname, :gaussian) == :poisson || continue
        # Fix residual variance to 0 (not estimated)
        pt.ustart[k] = 0.0
        pt.free[k]   = 0
        pt.label[k]  = ""
    end
    # Re-number free parameters (fill in the gaps)
    free_ctr = 0
    for k in eachindex(pt.free)
        if pt.free[k] > 0
            free_ctr += 1
            pt.free[k] = free_ctr
        end
    end
end
