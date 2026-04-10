# ─── validate.jl ──────────────────────────────────────────────────────────────
# Input validation: check model/data consistency before optimization.
# Raised errors mirror lavaan's informative messages.
# ─────────────────────────────────────────────────────────────────────────────

"""
    validate_model_data(pt, data, opts)

Validate that the model and data are consistent before fitting.
Raises informative errors for common problems.
"""
function validate_model_data(pt::ParTable, data::LavaanData, opts::LavaanOptions)
    warnings = String[]

    for g in 1:data.ngroups
        X = data.X[g]
        n, p = size(X)
        grp = data.ngroups > 1 ? " (group $(data.group_labels[g]))" : ""

        # ── Variables present in data ───────────────────────────────────────
        missing_vars = setdiff(data.ov_names, names(data.original_df isa DataFrame ? data.original_df : DataFrame()))
        # (already checked in prepare_data; belt-and-suspenders)

        # ── Zero-variance columns ────────────────────────────────────────────
        for (j, vname) in enumerate(data.ov_names)
            v = var(X[:, j])
            if v < 1e-10
                push!(warnings, "Variable '$vname'$grp has near-zero variance ($v). Results may be unreliable.")
            end
        end

        # ── N > p check ──────────────────────────────────────────────────────
        nfree_params = nfree(pt)
        if n < nfree_params
            push!(warnings, "N=$n < nfree=$nfree_params$grp. Model is likely not identified.")
        end

        # ── Positive-definite sample covariance ─────────────────────────────
        S = cov(X)
        try
            cholesky(S)
        catch e
            if e isa PosDefException || e isa LinearAlgebra.PosDefException
                push!(warnings, "Sample covariance matrix$grp is not positive definite. Check for perfect multicollinearity.")
            end
        end
    end

    # ── Warn but don't error (matches lavaan behavior) ───────────────────────
    for w in warnings
        @warn w
    end

    return warnings
end

"""
    check_ov_names_in_data(ov_names, df)

Check all model-referenced observed variables are present in the DataFrame.
"""
function check_ov_names_in_data(ov_names::Vector{String}, df::DataFrame)
    df_cols = string.(names(df))
    missing_vars = setdiff(ov_names, df_cols)
    if !isempty(missing_vars)
        mv_str = join(missing_vars, ", ")
        av_str = join(df_cols, ", ")
        error("Variable(s) in model not found in data: $mv_str\nAvailable variables: $av_str")
    end
end
