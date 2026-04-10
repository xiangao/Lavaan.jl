# ─── gsem.jl ─────────────────────────────────────────────────────────────────
# Generalized SEM with non-Gaussian outcomes via Gauss-Hermite quadrature.
# Mirrors Stata's gsem with family(poisson) link(log).
#
# The marginal likelihood integrates over the latent variable distribution:
#   L_i = ∫ [Π_j f_j(y_ij|η)] · N(η; μ_η, Σ_η) dη
#
# Solved by product Gauss-Hermite quadrature with Q nodes per latent dimension.
# Cost: O(N · Q^q · p) where q = #latent factors, p = #indicators.
# ─────────────────────────────────────────────────────────────────────────────

"""
    _gh_nodes_weights(Q) → (nodes, weights)

Compute Q-point Gauss-Hermite quadrature nodes and weights via the symmetric
tridiagonal Jacobi matrix eigenproblem.

Satisfies: ∫_{-∞}^{∞} f(x) exp(-x²) dx ≈ Σ_m w_m f(x_m)
Normalisation: Σ w_m = √π
"""
function _gh_nodes_weights(Q::Int)
    # Off-diagonal of Hermite Jacobi matrix: β_k = sqrt(k/2)
    beta = sqrt.(collect(1:Q-1) ./ 2.0)
    # Symmetric tridiagonal with zeros on diagonal
    J = SymTridiagonal(zeros(Q), beta)
    ev = eigen(J)
    # Sort nodes ascending
    perm = sortperm(ev.values)
    nodes   = ev.values[perm]
    # Weights: w_m = √π · (first component of normalised eigenvector)²
    weights = sqrt(π) .* (ev.vectors[1, perm]).^2
    return nodes, weights
end

"""
    gsem_objective(theta, model, data, opts) → scalar

GSEM marginal log-likelihood via product Gauss-Hermite quadrature.

Each variable's contribution to the integrand is determined by opts.family:
  :gaussian → N(y_ij; ν_j + Λ_j'η, Θ_jj)
  :poisson  → Poisson(y_ij; exp(ν_j + Λ_j'η))   [log link]

Returns F = -2/N · Σ_i log(L_i)  (minimised at θ̂).
"""
function gsem_objective(theta::AbstractVector{T},
                        model::LavaanModel,
                        stats::SampleStats)::T where T <: Real

    opts = model.options
    if stats.X === nothing
        error("GSEM estimator requires raw data in SampleStats")
    end
    X_data = stats.X

    Q = opts.n_quad_points
    gh_nodes, gh_weights = _gh_nodes_weights(Q)

    F = zero(T)
    # Correctly identify total N from SampleStats
    N_total = T(stats.ntotal)

    for g in 1:model.nblocks
        X_g = X_data[g]           # n × p raw observations
        n, p = size(X_g)
        block = model.blocks[g]
        q = block.q               # number of latent factors

        # ── Fill LISREL matrices at current θ ─────────────────────────────
        mats   = fill_block(block, theta) # Using fill_block for AD
        Lambda = mats.Lambda      # p × q
        Theta  = mats.Theta       # p × p (residual covariances)
        Psi    = mats.Psi         # q × q (latent covariances)
        Beta   = mats.Beta        # q × q (structural paths among latents)
        Nu     = mats.Nu          # p    (observed intercepts)
        Alpha  = mats.Alpha       # q    (latent intercepts)

        # ── Reduced-form latent distribution ─────────────────────────────
        # η_i ~ N(μ_η, Σ_η)
        # μ_η = (I-B)⁻¹ α
        # Σ_η = (I-B)⁻¹ Ψ (I-B)⁻ᵀ
        IB     = Matrix{T}(I, q, q) - Beta
        IB_inv = try inv(IB) catch; return T(1e20) end
        mu_eta = IB_inv * Alpha                        # q
        Sigma_eta = IB_inv * Psi * IB_inv'            # q × q
        # Symmetrise to avoid Cholesky failure from floating-point asymmetry
        Sigma_eta = (Sigma_eta + Sigma_eta') / 2

        L_eta = try cholesky(Sigma_eta).L catch; return T(1e20) end  # q × q lower

        # ── Family for each observed variable ─────────────────────────────
        # fam_poisson[j] = true if variable j is :poisson
        fam_poisson = [get(opts.family, model.ov_names[j], :gaussian) == :poisson
                       for j in 1:p]
        theta_diag  = [Theta[j,j] for j in 1:p]   # residual variances (Gaussian only)

        # ── Build product GH quadrature grid ─────────────────────────────
        # For q dimensions: Iterators.product(fill(1:Q, q)...)
        # η_m = mu_eta + L_eta * √2 * ξ_m
        # w_m = π^(-q/2) * Π_k w_{m_k}
        sqrt2 = T(sqrt(2.0))
        pi_q  = T(π)^(-q/2)
        gh_nodes_T   = T.(gh_nodes)
        gh_weights_T = T.(gh_weights)

        # ── Sum log-likelihood over observations ──────────────────────────
        loglik_g = zero(T)

        for i in 1:n
            y_i = X_g[i, :]    # length-p observation vector

            # Accumulate L_i = Σ_m [Π_k w_{m_k}] · Π_j f_j(y_ij | η)
            L_i = zero(T)

            # NOTE: For q latent variables, this is an expensive product loop.
            # q=1: Q evals
            # q=2: Q^2 evals (225)
            # q=3: Q^3 evals (3375)
            for idx in Iterators.product(fill(1:Q, q)...)
                # GH node vector and product weight
                xi     = [gh_nodes_T[idx[k]] for k in 1:q]
                w_prod = prod(gh_weights_T[idx[k]] for k in 1:q)

                # Transform to η space: η = μ_η + L_η · √2 · ξ
                eta_val = mu_eta + L_eta * (sqrt2 .* xi)

                # Linear predictor: ν + Λ η  (p-vector)
                lin_pred = Nu .+ Lambda * eta_val

                # Log-integrand: Σ_j log f_j(y_ij | η)
                log_f = zero(T)
                valid = true
                for j in 1:p
                    yj = y_i[j]
                    lp = lin_pred[j]
                    if fam_poisson[j]
                        # Poisson log-PMF: y*log(λ) - λ - log(y!)
                        lambda_j = exp(lp)
                        k_val = round(Int, yj)
                        log_f += T(k_val) * log(lambda_j) - lambda_j - T(logfactorial(k_val))
                    else
                        # Gaussian log-PDF: -½[log(2πσ²) + (y-μ)²/σ²]
                        sigma2_j = theta_diag[j]
                        if sigma2_j <= 1e-10
                            valid = false; break
                        end
                        log_f += T(-0.5) * (log(T(2π) * sigma2_j) + (T(yj) - lp)^2 / sigma2_j)
                    end
                end

                if valid
                    L_i += w_prod * exp(log_f)
                end
            end

            # Normalise by π^(-q/2) and accumulate log-likelihood
            L_i_norm = pi_q * L_i
            if L_i_norm > 1e-300 && !isnan(L_i_norm)
                loglik_g += log(L_i_norm)
            else
                loglik_g += T(-1e10)
            end
        end

        F += loglik_g
    end

    return T(-2.0) * F / N_total
end
