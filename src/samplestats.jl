# ─── samplestats.jl ───────────────────────────────────────────────────────────
# Compute sample statistics from LavaanData.
# Mirrors lavaan's lav_lavaan_step05_samplestats.R + lav_samplestats.R
# ─────────────────────────────────────────────────────────────────────────────

"""
    compute_samplestats(data, opts) → SampleStats

Compute sample covariance matrices, means, and (optionally) asymptotic
weight matrices from prepared data.
"""
function compute_samplestats(data::LavaanData, opts::LavaanOptions)::SampleStats
    ngroups = data.ngroups
    S_list        = Vector{Matrix{Float64}}(undef, ngroups)
    S_inv_list    = Vector{Matrix{Float64}}(undef, ngroups)
    logdet_S_list = Vector{Float64}(undef, ngroups)
    mu_list       = Vector{Vector{Float64}}(undef, ngroups)
    nobs_list     = Vector{Int}(undef, ngroups)

    for g in 1:ngroups
        X = data.X[g]
        n, p = size(X)
        nobs_list[g] = n

        # Sample mean
        mu_list[g] = vec(mean(X, dims=1))

        # Sample covariance (N-1 denominator, matching R's cov())
        # lavaan uses (n-1)/n scaling internally (sample.cov.rescale)
        # but stores the unscaled version; we follow suit
        S = cov(X)    # (N-1) denominator

        # Ridge regularization if requested
        if opts.ridge
            S = S + opts.ridge_constant * I(p)
        end

        S_list[g] = S

        # Inverse and log-determinant via Cholesky
        chol = cholesky(Symmetric(S))
        S_inv_list[g]    = inv(chol)
        logdet_S_list[g] = 2.0 * sum(log.(diag(chol.U)))
    end

    p = size(data.X[1], 2)

    # FIML patterns: built here if needed, before NaN-rows are stripped
    # (data.X still has NaN for missing values at this point)
    fiml_pats = if opts.estimator == :FIML
        [build_fiml_patterns(data.X[g]) for g in 1:ngroups]
    else
        nothing
    end

    YLp_list = nothing
    S_W_list = nothing
    S_B_list = nothing
    s_list   = nothing

    if !isempty(data.cluster_cols)
        YLp_list = [Vector{Matrix{Float64}}(undef, length(data.cluster_cols)) for _ in 1:ngroups]
        S_B_list = [Vector{Matrix{Float64}}(undef, length(data.cluster_cols)) for _ in 1:ngroups]
        s_list   = [Vector{Float64}(undef, length(data.cluster_cols)) for _ in 1:ngroups]
        S_W_list = Vector{Matrix{Float64}}(undef, ngroups)

        for g in 1:ngroups
            X = data.X[g]
            n, p = size(X)
            
            # Compute pooled within-cluster covariance (S_W)
            # For multiple clusters, this is complex; we start by using the first cluster
            # or a general residual variance approach. 
            # For now, we compute S_W relative to ALL clusters combined (if nested)
            # or just relative to the first clustering factor.
            
            for (c, c_col) in enumerate(data.cluster_cols)
                c_idx = data.cluster_idx[g][c]
                c_sizes = data.cluster_sizes[g][c]
                n_clusters = length(c_sizes)

                # Compute cluster means
                Y_bar = zeros(n_clusters, p)
                for i in 1:n
                    Y_bar[c_idx[i], :] .+= X[i, :]
                end
                for j in 1:n_clusters
                    Y_bar[j, :] ./= c_sizes[j]
                end
                YLp_list[g][c] = Y_bar

                # Compute Between-cluster covariance (S_B)
                SS_B = zeros(p, p)
                mu = mu_list[g]
                for j in 1:n_clusters
                    dev = Y_bar[j, :] .- mu
                    SS_B .+= c_sizes[j] .* (dev * dev')
                end
                S_B_list[g][c] = SS_B ./ (n_clusters - 1)

                # Compute scaling factor s
                sum_nj2 = sum(c_sizes .^ 2)
                s_list[g][c] = (n^2 - sum_nj2) / (n * (n_clusters - 1))
                
                # If this is the first (or only) cluster, use it for S_W
                if c == 1
                    SS_W = zeros(p, p)
                    for i in 1:n
                        dev = X[i, :] .- Y_bar[c_idx[i], :]
                        SS_W .+= dev * dev'
                    end
                    S_W_list[g] = SS_W ./ (n - n_clusters)
                end
            end
        end
    end

    return SampleStats(
        S_list,
        S_inv_list,
        logdet_S_list,
        mu_list,
        nobs_list,
        sum(nobs_list),
        p,
        nothing,      # Gamma (computed lazily for WLS)
        nothing,      # NACOV
        nothing,      # thresholds
        nothing,      # threshold_ses
        fiml_pats,    # FIML patterns (nothing unless estimator=:FIML)
        YLp_list,
        S_W_list,
        S_B_list,
        s_list,
        data.X,
        data.cluster_idx,
    )
end

# ─── FIML pattern builder ─────────────────────────────────────────────────────

"""
    build_fiml_patterns(X) → Vector{FIMLPattern}

Group rows of X (with NaN for missing) by missing-data pattern.
Each distinct pattern of observed/missing produces one FIMLPattern.
"""
function build_fiml_patterns(X::Matrix{Float64})::Vector{FIMLPattern}
    n, p = size(X)
    pattern_map = Dict{Vector{Bool}, Vector{Int}}()
    for i in 1:n
        key = [!isnan(X[i, j]) for j in 1:p]
        push!(get!(pattern_map, key, Int[]), i)
    end

    patterns = FIMLPattern[]
    for (key, rows) in pattern_map
        obs_idx = findall(key)
        isempty(obs_idx) && continue   # all-missing row → skip
        X_obs = X[rows, obs_idx]
        push!(patterns, FIMLPattern(obs_idx, rows, X_obs))
    end
    return patterns
end

# ─── Cholesky helper ──────────────────────────────────────────────────────────
# Julia's cholesky() needs PositiveDefinite flag for graceful fallback

"""
    safe_cholesky(A) → Cholesky factorization or error with context

Wraps cholesky() with a more informative error for non-PD matrices.
"""
function safe_cholesky(A::AbstractMatrix)
    try
        return cholesky(A)
    catch e
        if e isa PosDefException
            error("Matrix is not positive definite. This can happen when:\n" *
                  "  - The model is not identified\n" *
                  "  - Parameters are at a boundary (Heywood case)\n" *
                  "  - Starting values are poor\n" *
                  "Try: opts = LavaanOptions(ridge=true) or check model specification.")
        end
        rethrow(e)
    end
end
