# ─── data.jl ──────────────────────────────────────────────────────────────────
# Data preparation: DataFrame → LavaanData
# Mirrors lavaan's lav_lavaan_step03_data.R + lav_data.R
# ─────────────────────────────────────────────────────────────────────────────

"""
    prepare_data(df, ov_names; kwargs...) → LavaanData

Prepare a DataFrame for model fitting.

# Arguments
- `df`       : the data DataFrame
- `ov_names` : observed variable names from the model
- `lv_names` : latent variable names (for reference)
- `group`    : column name for grouping variable (optional)
- `weights`  : column name for sampling weights (optional)
- `missing_method` : :listwise (default) or :fiml
"""
function prepare_data(df::DataFrame,
                      ov_names::Vector{String},
                      lv_names::Vector{String};
                      group::Union{Symbol,String,Nothing} = nothing,
                      cluster::Union{Symbol,String,Nothing,Vector{String}} = nothing,
                      weights::Union{Symbol,String,Nothing} = nothing,
                      missing_method::Symbol = :listwise,
                      ordered::Vector{String} = String[])::LavaanData

    group_col = group === nothing ? nothing : string(group)
    cluster_cols = if cluster === nothing
        String[]
    elseif cluster isa Vector
        string.(cluster)
    else
        [string(cluster)]
    end

    if group_col !== nothing
        # Multi-group: split by group
        group_vals = sort(unique(df[!, group_col]))
        group_labels = string.(group_vals)
        ngroups = length(group_labels)

        Xlist = Vector{Matrix{Float64}}(undef, ngroups)
        wlist = Vector{Union{Vector{Float64},Nothing}}(undef, ngroups)
        nobs_list = Vector{Int}(undef, ngroups)
        cluster_idx_list = Vector{Vector{Vector{Int}}}(undef, ngroups)
        cluster_sizes_list = Vector{Vector{Vector{Int}}}(undef, ngroups)

        for (g, gval) in enumerate(group_vals)
            sub = df[df[!, group_col] .== gval, :]
            Xlist[g], wlist[g], nobs_list[g], row_mask = _prepare_group(
                sub, ov_names; weights=weights, missing_method=missing_method)
            
            cluster_idx_list[g] = Vector{Int}[]
            cluster_sizes_list[g] = Vector{Int}[]
            for c_col in cluster_cols
                c_idx, c_sizes = _compute_cluster_info(sub[row_mask, c_col])
                push!(cluster_idx_list[g], c_idx)
                push!(cluster_sizes_list[g], c_sizes)
            end
        end
    else
        # Single group
        ngroups = 1
        group_labels = [""]
        X1, w1, n1, row_mask = _prepare_group(df, ov_names;
                                     weights=weights,
                                     missing_method=missing_method)
        Xlist = Matrix{Float64}[X1]
        wlist = Union{Vector{Float64},Nothing}[w1]
        nobs_list = Int[n1]
        
        cluster_idx_list = [Vector{Int}[]]
        cluster_sizes_list = [Vector{Int}[]]
        for c_col in cluster_cols
            c_idx, c_sizes = _compute_cluster_info(df[row_mask, c_col])
            push!(cluster_idx_list[1], c_idx)
            push!(cluster_sizes_list[1], c_sizes)
        end
    end

    return LavaanData(
        Xlist,
        ov_names,
        lv_names,
        ngroups,
        nobs_list,
        group_labels,
        wlist,
        df,           # keep original for bootstrap
        group_col,
        isempty(cluster_cols) ? nothing : cluster_cols[1],
        cluster_cols,
        cluster_idx_list,
        cluster_sizes_list,
        ordered,               # NEW
        Dict{String,Int}(),    # NEW: num_thresholds, filled later
    )
end

function _compute_cluster_info(cluster_vals)
    unique_clusters = unique(cluster_vals)
    val_to_idx = Dict(v => i for (i, v) in enumerate(unique_clusters))
    c_idx = [val_to_idx[v] for v in cluster_vals]
    c_sizes = zeros(Int, length(unique_clusters))
    for idx in c_idx
        c_sizes[idx] += 1
    end
    return c_idx, c_sizes
end

function _prepare_group(df::DataFrame, ov_names::Vector{String};
                         weights=nothing, missing_method=:listwise)
    # Extract columns
    X_raw = Matrix{Union{Float64,Missing}}(undef, nrow(df), length(ov_names))
    for (j, name) in enumerate(ov_names)
        col = df[!, name]
        X_raw[:, j] = Float64.(coalesce.(col, missing))
    end

    # Handle missing data
    if missing_method == :listwise
        # Drop rows with any missing value
        row_mask = [all(!ismissing, row) for row in eachrow(X_raw)]
        X = Float64.(X_raw[row_mask, :])
    else
        # FIML: keep all rows, replace missing with NaN (handled downstream)
        row_mask = trues(nrow(df))
        X = map(x -> ismissing(x) ? NaN : Float64(x), X_raw)
    end

    nobs = size(X, 1)

    # Sampling weights
    w = if weights !== nothing
        Float64.(df[row_mask, string(weights)])
    else
        nothing
    end

    return X, w, nobs, row_mask
end

# ─── Bundled datasets ─────────────────────────────────────────────────────────

const _DATA_DIR = joinpath(@__DIR__, "..", "data")

"""
    holzinger_swineford() → DataFrame

The Holzinger & Swineford (1939) dataset. Classic CFA benchmark.
301 children, 9 cognitive tests measuring 3 factors (visual, textual, speed).
"""
function holzinger_swineford()::DataFrame
    path = joinpath(_DATA_DIR, "HolzingerSwineford1939.csv")
    isfile(path) || error("Data file not found: $path\nRun the data export script first.")
    CSV.read(path, DataFrame)
end

"""
    political_democracy() → DataFrame

The Industrialization and Political Democracy dataset (Bollen, 1989).
75 developing countries. Classic SEM benchmark.
"""
function political_democracy()::DataFrame
    path = joinpath(_DATA_DIR, "PoliticalDemocracy.csv")
    isfile(path) || error("Data file not found: $path")
    CSV.read(path, DataFrame)
end
