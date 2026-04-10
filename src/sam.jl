# ─── sam.jl ───────────────────────────────────────────────────────────────────
# Structural After Measurement (SAM) approach.
# Rosseel & Loh (2022).
# ─────────────────────────────────────────────────────────────────────────────

"""
    run_sam_pipeline(model_string, data, opts; group=nothing)

The SAM orchestrator.
"""
function run_sam_pipeline(model_string::String,
                          data::DataFrame,
                          opts::LavaanOptions;
                          group::Union{String,Nothing} = nothing)::LavaanFit

    # 1. Parse full model
    parsed_rows = parse_model_string(model_string)
    lv_names = lv_names_from_rows(parsed_rows)
    ov_names = ov_names_from_rows(parsed_rows, lv_names)
    
    # Identify measurement blocks
    blocks = Dict{String, Vector{String}}()
    for lv in lv_names
        indicators = String[]
        for row in parsed_rows
            if row.op == "=~" && row.lhs == lv
                push!(indicators, row.rhs)
            end
        end
        blocks[lv] = indicators
    end

    # 2. Step 1: Estimate measurement models (Local SAM)
    # We estimate each factor's loadings and residual variances separately.
    mm_results = Dict{String, LavaanFit}()
    for lv in lv_names
        inds = blocks[lv]
        mm_syntax = "$lv =~ " * join(inds, " + ")
        # Fit small CFA
        mm_fit = cfa(mm_syntax, data; group=group, std_lv=true)
        mm_results[lv] = mm_fit
    end

    # 3. Build global Lambda and Theta from local results
    p = length(ov_names)
    q = length(lv_names)
    Lambda = zeros(p, q)
    Theta  = zeros(p, p)
    
    ov_map = Dict(name => i for (i, name) in enumerate(ov_names))
    lv_map = Dict(name => i for (i, name) in enumerate(lv_names))
    
    for (lv, fit) in mm_results
        j = lv_map[lv]
        pt = fit.partable
        for i in 1:length(pt)
            if pt.op[i] == "=~" && pt.lhs[i] == lv
                k = ov_map[pt.rhs[i]]
                Lambda[k, j] = pt.est[i]
            elseif pt.op[i] == "~~" && pt.lhs[i] == pt.rhs[i] && pt.lhs[i] in ov_names
                k = ov_map[pt.lhs[i]]
                Theta[k, k] = pt.est[i]
            end
        end
    end

    # 4. Compute Mapping Matrix M (Bartlett weights)
    # M = (Lambda' Theta^-1 Lambda)^-1 Lambda' Theta^-1
    # For Local SAM, since factors are estimated separately, 
    # Lambda is block-diagonal (after reordering) and Theta is diagonal.
    # So M is also block-diagonal.
    
    # Get observed covariance S
    full_lavdata = prepare_data(data, ov_names, lv_names; group=group)
    full_stats = compute_samplestats(full_lavdata, opts)
    S = full_stats.S[1] # Single group support for now
    
    # Compute M
    # Use a small epsilon for Theta if variances are zero (shouldn't happen in CFA)
    invTheta = inv(Theta + 1e-9 * I)
    M = inv(Lambda' * invTheta * Lambda) * Lambda' * invTheta
    
    # 5. Latent Sample Covariance Matrix
    # S_eta = M S M'
    S_eta = M * S * M'
    
    # Bias correction (SAM correction)
    # S_eta_corrected = S_eta - M * Theta * M'
    # According to Rosseel (2022), the correction is M * Theta * M'
    correction = M * Theta * M'
    S_eta_corrected = S_eta - correction
    
    # Ensure S_eta is symmetric and positive definite (small epsilon on diag)
    S_eta_corrected = (S_eta_corrected + S_eta_corrected') / 2
    for i in 1:q
        if S_eta_corrected[i,i] < 1e-6
            S_eta_corrected[i,i] = 1e-6
        end
    end

    # 6. Step 2: Fit Structural Model
    # Deriving variable names for the structural model
    # We treat the original latents as observed variables in Step 2.
    structural_syntax = ""
    for row in parsed_rows
        if (row.op == "~" || row.op == "~~") && row.lhs in lv_names && row.rhs in lv_names
            # Regressions among latents
            # To use the existing SEM engine, we create dummy latents for each Step 2 observed var.
            # e.g. f_dem60 =~ 1*dem60; f_dem60 ~ f_ind60
            structural_syntax *= "f_$(row.lhs) $(row.op) f_$(row.rhs)\n"
        end
    end
    
    # Add dummy indicators for Step 2
    for name in lv_names
        structural_syntax *= "f_$name =~ 1*$name\n"
        structural_syntax *= "$name ~~ 0*$name\n"
    end
    
    # Deriving names for Step 2
    # In Step 2, the 'observed' variables are the original latents (lv_names)
    # The 'latent' variables are the new f_* names.
    
    # Reorder S_eta_corrected to match the order in structural model
    # (lavaan will derive ov_names from the syntax)
    # Actually, we can just ensure S_eta is passed with correct names if we supported it.
    # For now, let's keep the order consistent.
    
    # Fit the structural model
    # Generate synthetic data from S_eta_corrected (preserves n for SEs)
    n_total = full_stats.ntotal
    S_sym = (S_eta_corrected + S_eta_corrected') / 2
    L_chol = cholesky(S_sym + 1e-8*I).L
    rng = Random.MersenneTwister(0)
    Z = randn(rng, n_total, q)
    X_eta = Z * L_chol'
    eta_df = DataFrame(X_eta, lv_names)

    struct_fit = lavaan(structural_syntax, eta_df; estimator=:ML)

    # 7. Final Fit Object
    # We combine the measurement estimates and structural estimates into a single ParTable.
    # For now, let's just return the structural fit as a proof-of-concept.
    # In a full implementation, we would merge the ParTables.
    
    return struct_fit
end
