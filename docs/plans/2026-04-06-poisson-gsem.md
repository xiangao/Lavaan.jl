# Poisson GSEM (Generalized SEM) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement GSEM with Poisson (count) outcomes via non-adaptive Gauss-Hermite quadrature, mirroring Stata's `gsem` with `family(poisson) link(log)`.

**Architecture:** Each outcome variable is assigned a family (`:gaussian` default, `:poisson`) via an `opts.family` dict. The GSEM objective integrates the conditional likelihood over the latent variable distribution using product Gauss-Hermite quadrature. The LISREL matrix structure is unchanged — only the likelihood changes. Auto-switch from `:ML` → `:GSEM` when any family entry is Poisson.

**Tech Stack:** Julia 1.9+, Optim.jl (LBFGS), ForwardDiff.jl, Distributions.jl (logfactorial), existing `fill_block`/`LISRELBlock` infrastructure.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/types.jl` | Modify | Add `family`, `n_quad_points` to `LavaanOptions` |
| `src/gsem.jl` | Create | GH nodes/weights, `gsem_objective` |
| `src/optim.jl` | Modify | Pass `data` kwarg; GSEM objective closure |
| `src/api.jl` | Modify | `family` kwarg; auto-switch; fix Poisson residuals; force meanstructure |
| `src/partable.jl` | Modify | `_fix_poisson_residuals!` helper |
| `src/Lavaan.jl` | Modify | `include("gsem.jl")` after `include("implied.jl")` |
| `test/gsem.jl` | Create | GH unit tests + Poisson CFA integration tests |
| `test/runtests.jl` | Modify | `include("gsem.jl")` |

---

## Statistical Background

For a CFA/SEM with Poisson indicators, the marginal likelihood for observation i is:

```
L_i(θ) = ∫ [Π_j f_j(y_ij | η)] · N(η; μ_η, Σ_η) dη
```

where:
- η is the q-dimensional latent variable vector (reduced form)
- `μ_η = (I-B)⁻¹ α`, `Σ_η = (I-B)⁻¹ Ψ (I-B)⁻ᵀ`
- For Gaussian j: `f_j(y_ij|η) = N(y_ij; ν_j + Λ_j'η, Θ_jj)`
- For Poisson j: `f_j(y_ij|η) = Poisson(y_ij; exp(ν_j + Λ_j'η))`

Gauss-Hermite product quadrature (Q nodes per latent dimension):
```
L_i ≈ π^(-q/2) · Σ_{m∈{1..Q}^q} [Π_k w_{m_k}] · Π_j f_j(y_ij | μ_η + L_η·√2·ξ_m)
```
where `L_η = chol(Σ_η)` (lower), `ξ_m = [x_{m_1}, ..., x_{m_q}]'` are GH nodes.

The objective to minimize: `F = -2/N · Σ_i log(L_i)`

---

## Task 1: Types — add `family` and `n_quad_points` to LavaanOptions

**Files:**
- Modify: `src/types.jl`

- [ ] **Step 1: Write failing test**

In `test/gsem.jl` (create new file):

```julia
# ─── Test Poisson GSEM ────────────────────────────────────────────────────────
using Test
using Lavaan
using DataFrames
using Random

@testset "GSEM types: family and n_quad_points" begin
    opts = LavaanOptions(family=Dict("y1"=>:poisson, "y2"=>:poisson))
    @test opts.family["y1"] == :poisson
    @test opts.family["y2"] == :poisson
    @test opts.n_quad_points == 15   # default

    opts2 = LavaanOptions(n_quad_points=7)
    @test opts2.n_quad_points == 7
end
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd ~/projects/software/Lavaan.jl && julia --project=. -e 'include("test/gsem.jl")'
```

Expected: `UndefVarError: family not defined` or similar.

- [ ] **Step 3: Add fields to LavaanOptions in `src/types.jl`**

After the `ordered::Vector{String}` line (around line 40), add:

```julia
    # GSEM: per-variable family/link (empty = all :gaussian)
    family::Dict{String,Symbol}  = Dict{String,Symbol}()
    n_quad_points::Int           = 15   # GH quadrature points per latent dim
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
cd ~/projects/software/Lavaan.jl && julia --project=. -e 'include("test/gsem.jl")'
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd ~/projects/software/Lavaan.jl
git add src/types.jl test/gsem.jl
git commit -m "feat(gsem): add family and n_quad_points to LavaanOptions"
```

---

## Task 2: GH nodes/weights + `src/gsem.jl` skeleton

**Files:**
- Create: `src/gsem.jl`
- Modify: `src/Lavaan.jl`

- [ ] **Step 1: Write failing test**

Append to `test/gsem.jl`:

```julia
@testset "GH quadrature: nodes and weights" begin
    nodes, weights = Lavaan._gh_nodes_weights(15)
    @test length(nodes) == 15
    @test length(weights) == 15
    # Weights sum to √π (normalisation for exp(-x²) weight)
    @test isapprox(sum(weights), sqrt(π); atol=1e-10)
    # Nodes are symmetric
    @test isapprox(nodes[1], -nodes[end]; atol=1e-12)
    # Centre node is 0
    @test isapprox(nodes[8], 0.0; atol=1e-12)

    nodes7, weights7 = Lavaan._gh_nodes_weights(7)
    @test length(nodes7) == 7
    @test isapprox(sum(weights7), sqrt(π); atol=1e-10)
end
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd ~/projects/software/Lavaan.jl && julia --project=. -e 'include("test/gsem.jl")'
```

Expected: `UndefVarError: _gh_nodes_weights not defined`.

- [ ] **Step 3: Create `src/gsem.jl`**

```julia
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

using LinearAlgebra, SpecialFunctions

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
                        data::LavaanData,
                        opts::LavaanOptions)::T where T <: Real

    Q = opts.n_quad_points
    gh_nodes, gh_weights = _gh_nodes_weights(Q)

    F = zero(T)
    N_total = T(sum(data.nobs))

    for g in 1:model.nblocks
        X_g = data.X[g]           # n × p raw observations
        n, p = size(X_g)
        block = model.blocks[g]
        q = block.q               # number of latent factors

        # ── Fill LISREL matrices at current θ ─────────────────────────────
        mats   = fill_block(block, theta)
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

            # Accumulate L_i = Σ_m [Π_k w_{m_k}] · Π_j f_j(y_ij | η_m)
            L_i = zero(T)

            for idx in Iterators.product(fill(1:Q, q)...)
                # GH node vector and product weight
                xi    = [gh_nodes_T[idx[k]] for k in 1:q]
                w_prod = prod(gh_weights_T[idx[k]] for k in 1:q)

                # Transform to η space: η = μ_η + L_η · √2 · ξ
                eta = mu_eta + L_eta * (sqrt2 .* xi)

                # Linear predictor: ν + Λ η  (p-vector)
                lin_pred = Nu .+ Lambda * eta

                # Log-integrand: Σ_j log f_j(y_ij | η)
                log_f = zero(T)
                valid = true
                for j in 1:p
                    yj = y_i[j]
                    lp = lin_pred[j]
                    if fam_poisson[j]
                        # Poisson log-PMF: y*log(λ) - λ - log(y!)
                        lambda_j = exp(lp)
                        k = round(Int, yj)
                        log_f += T(k) * log(lambda_j) - lambda_j - T(logfactorial(k))
                    else
                        # Gaussian log-PDF: -½[log(2πσ²) + (y-μ)²/σ²]
                        sigma2_j = theta_diag[j]
                        if sigma2_j <= 0
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
            if L_i_norm > 0 && !isnan(L_i_norm)
                loglik_g += log(L_i_norm)
            else
                loglik_g += T(-1e10)
            end
        end

        F += loglik_g
    end

    return T(-2.0) * F / N_total
end
```

- [ ] **Step 4: Add include to `src/Lavaan.jl`**

After `include("implied.jl")` (line 31), add:
```julia
include("gsem.jl")
```

So the block reads:
```julia
include("implied.jl")
include("gsem.jl")
include("estimators/ml.jl")
```

- [ ] **Step 5: Run test to confirm it passes**

```bash
cd ~/projects/software/Lavaan.jl && julia --project=. -e 'include("test/gsem.jl")'
```

Expected: PASS for both testsets.

- [ ] **Step 6: Commit**

```bash
cd ~/projects/software/Lavaan.jl
git add src/gsem.jl src/Lavaan.jl test/gsem.jl
git commit -m "feat(gsem): GH quadrature nodes/weights + gsem_objective skeleton"
```

---

## Task 3: Wire `gsem_objective` into `optimize_model` and `estimator_objective`

**Files:**
- Modify: `src/optim.jl`

- [ ] **Step 1: Write failing test**

Append to `test/gsem.jl`:

```julia
@testset "GSEM: objective callable with :GSEM estimator" begin
    # Pure Poisson CFA: 3 indicators of 1 latent factor
    Random.seed!(42)
    n = 200
    eta = randn(n)            # latent factor
    y1 = [rand(Poisson(exp(0.5 + 0.8*eta[i]))) for i in 1:n]
    y2 = [rand(Poisson(exp(0.3 + 1.0*eta[i]))) for i in 1:n]
    y3 = [rand(Poisson(exp(0.1 + 0.6*eta[i]))) for i in 1:n]
    df = DataFrame(y1=Float64.(y1), y2=Float64.(y2), y3=Float64.(y3))

    model_str = """
        F =~ y1 + y2 + y3
    """
    # fit with :GSEM family — should not throw
    fit = cfa(model_str, df;
              family=Dict("y1"=>:poisson,"y2"=>:poisson,"y3"=>:poisson),
              estimator=:GSEM)
    @test fit !== nothing
    @test !all(isnan.(coef(fit)))
end
```

Note: this test requires `Distributions` (for `Poisson`). Add `using Distributions` at the top of the test file.

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd ~/projects/software/Lavaan.jl && julia --project=. -e 'include("test/gsem.jl")'
```

Expected: error about `family` kwarg not defined in `cfa`, or `:GSEM` not handled.

- [ ] **Step 3: Modify `src/optim.jl` to accept `data` and dispatch GSEM**

Change `optimize_model` signature and objective closure:

```julia
function optimize_model(model::LavaanModel,
                        stats::SampleStats,
                        opts::LavaanOptions;
                        data::Union{LavaanData,Nothing} = nothing,
                        theta0::Union{Nothing,Vector{Float64}} = nothing)

    theta0 = theta0 !== nothing ? theta0 : get_start(model.partable)
    nθ     = length(theta0)

    if nθ == 0
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
        if data === nothing
            error("GSEM estimator requires raw data — pass data= to optimize_model")
        end
        obj = theta -> gsem_objective(theta, model, data, opts)
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
```

- [ ] **Step 4: Run test to confirm it still fails (missing api.jl wiring)**

```bash
cd ~/projects/software/Lavaan.jl && julia --project=. -e 'include("test/gsem.jl")'
```

Expected: error about `family` kwarg in `cfa`.

- [ ] **Step 5: Commit the optim.jl change**

```bash
cd ~/projects/software/Lavaan.jl
git add src/optim.jl
git commit -m "feat(gsem): add data= kwarg to optimize_model, dispatch :GSEM objective"
```

---

## Task 4: API wiring — `family` kwarg, auto-switch, fix Poisson residuals

**Files:**
- Modify: `src/api.jl`
- Modify: `src/partable.jl`

- [ ] **Step 1: Add `_fix_poisson_residuals!` to `src/partable.jl`**

Append at the end of `src/partable.jl`:

```julia
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
```

- [ ] **Step 2: Add `family` kwarg and auto-switch logic to `src/api.jl`**

In the `lavaan()` function signature, add `family` and `n_quad_points`:

```julia
function lavaan(model_string::String,
                data::DataFrame;
                group::Union{String,Nothing} = nothing,
                cluster::Union{String,Symbol,Nothing,Vector{String}} = nothing,
                ordered::Vector{String} = String[],
                family::Dict{String,Symbol} = Dict{String,Symbol}(),
                n_quad_points::Int = 15,
                estimator::Symbol  = :ML,
                se::Symbol         = :standard,
                # ... (rest unchanged)
                )::LavaanFit
```

In the `LavaanOptions(...)` constructor call, add:

```julia
    opts = LavaanOptions(
        estimator       = estimator,
        se              = se,
        ordered         = ordered,
        family          = family,
        n_quad_points   = n_quad_points,
        # ... (rest unchanged)
    )
```

In `_lavaan_pipeline`, replace the existing auto-switch block (currently only for ordinal) with:

```julia
    # Auto-switch estimator:
    # - ordinal data → :DWLS
    # - any Poisson family → :GSEM (with meanstructure forced on)
    if !isempty(opts.ordered) && opts.estimator == :ML
        opts = LavaanOptions(; [f => getfield(opts, f) for f in fieldnames(LavaanOptions)]...,
                             estimator=:DWLS)
    end
    has_poisson = any(v == :poisson for v in values(opts.family))
    if has_poisson && opts.estimator ∉ (:GSEM,)
        opts = LavaanOptions(; [f => getfield(opts, f) for f in fieldnames(LavaanOptions)]...,
                             estimator=:GSEM,
                             meanstructure=true,
                             int_ov_free=true)
    end
```

After `build_partable` (step 04 in pipeline), add:

```julia
    # Fix residual variances to 0 for Poisson variables (no free dispersion)
    _fix_poisson_residuals!(pt, opts.family)
```

In step 09 (optimization), pass `data=lavdata`:

```julia
    if opts.do_fit && model.nfree > 0
        theta_hat, converged, iters, optim_result =
            optimize_model(model, stats, opts; data=lavdata)
    else
        converged = true
    end
```

- [ ] **Step 3: Run test to confirm it passes**

```bash
cd ~/projects/software/Lavaan.jl && julia --project=. -e 'include("test/gsem.jl")'
```

Expected: PASS for all 3 testsets (including GSEM callable test).

- [ ] **Step 4: Commit**

```bash
cd ~/projects/software/Lavaan.jl
git add src/api.jl src/partable.jl
git commit -m "feat(gsem): add family kwarg to lavaan/cfa/sem, auto-switch to :GSEM, fix Poisson residuals"
```

---

## Task 5: Full integration tests

**Files:**
- Modify: `test/gsem.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Write integration tests**

Append to `test/gsem.jl`:

```julia
@testset "GSEM: pure Poisson CFA — parameter recovery" begin
    # Simulate from: F =~ y1 + y2 + y3  (Poisson indicators)
    # True params: nu = [0.5, 0.3, 0.1], lambda = [1.0, 0.8, 0.6], psi = 1.0
    Random.seed!(123)
    n = 500
    eta = randn(n)
    y1 = [rand(Poisson(exp(0.5 + 1.0*eta[i]))) for i in 1:n]
    y2 = [rand(Poisson(exp(0.3 + 0.8*eta[i]))) for i in 1:n]
    y3 = [rand(Poisson(exp(0.1 + 0.6*eta[i]))) for i in 1:n]
    df = DataFrame(y1=Float64.(y1), y2=Float64.(y2), y3=Float64.(y3))

    model_str = """
        F =~ y1 + y2 + y3
    """
    fit = cfa(model_str, df;
              family=Dict("y1"=>:poisson, "y2"=>:poisson, "y3"=>:poisson))

    pe = parameterEstimates(fit)
    lambdas = pe[pe.op .== "=~", :]

    # y1 loading fixed to 1.0 (identification)
    y1_load = lambdas[lambdas.rhs .== "y1", :est]
    @test !isempty(y1_load)
    @test isapprox(y1_load[1], 1.0; atol=1e-6)

    # y2 loading ≈ 0.8 (true value), allow ±0.20 tolerance (finite sample)
    y2_load = lambdas[lambdas.rhs .== "y2", :est]
    @test !isempty(y2_load)
    @test isapprox(y2_load[1], 0.8; atol=0.20)

    # y3 loading ≈ 0.6
    y3_load = lambdas[lambdas.rhs .== "y3", :est]
    @test !isempty(y3_load)
    @test isapprox(y3_load[1], 0.6; atol=0.20)

    # No NaN in coefficients
    @test !any(isnan.(coef(fit)))
end

@testset "GSEM: mixed Gaussian + Poisson indicators" begin
    # Simulate: F =~ x1 + x2 (Gaussian) + y1 + y2 (Poisson)
    Random.seed!(456)
    n = 400
    eta = randn(n)
    x1 = 0.2 .+ 0.9*eta .+ 0.3*randn(n)
    x2 = 0.1 .+ 0.7*eta .+ 0.3*randn(n)
    y1 = [rand(Poisson(exp(0.4 + 0.8*eta[i]))) for i in 1:n]
    y2 = [rand(Poisson(exp(0.2 + 0.6*eta[i]))) for i in 1:n]
    df = DataFrame(x1=x1, x2=x2, y1=Float64.(y1), y2=Float64.(y2))

    model_str = """
        F =~ x1 + x2 + y1 + y2
    """
    fit = cfa(model_str, df;
              family=Dict("y1"=>:poisson, "y2"=>:poisson))

    @test !any(isnan.(coef(fit)))

    pe = parameterEstimates(fit)
    @test nrow(pe) > 0

    # Gaussian residual variances should be estimated (non-zero)
    resid_x1 = pe[pe.lhs .== "x1" .& pe.op .== "~~" .& pe.rhs .== "x1", :est]
    @test !isempty(resid_x1)
    @test resid_x1[1] > 0

    # Poisson residual variances should be fixed to 0
    resid_y1 = pe[pe.lhs .== "y1" .& pe.op .== "~~" .& pe.rhs .== "y1", :est]
    # Poisson vars have no residual variance row (fixed and removed)
    # Either not in output or fixed to 0
    if !isempty(resid_y1)
        @test isapprox(resid_y1[1], 0.0; atol=1e-10)
    end
end

@testset "GSEM: cfa() and sem() pass family through" begin
    Random.seed!(789)
    n = 100
    eta = randn(n)
    y1 = [rand(Poisson(exp(0.3*eta[i]))) for i in 1:n]
    y2 = [rand(Poisson(exp(0.5*eta[i]))) for i in 1:n]
    df = DataFrame(y1=Float64.(y1), y2=Float64.(y2))

    model_str = "F =~ y1 + y2"
    fit_cfa = cfa(model_str, df; family=Dict("y1"=>:poisson,"y2"=>:poisson))
    fit_sem = sem(model_str, df; family=Dict("y1"=>:poisson,"y2"=>:poisson))

    @test !any(isnan.(coef(fit_cfa)))
    @test !any(isnan.(coef(fit_sem)))
end
```

- [ ] **Step 2: Run full test file to confirm all pass**

```bash
cd ~/projects/software/Lavaan.jl && julia --project=. -e 'include("test/gsem.jl")'
```

Expected: All testsets PASS (parameter recovery may be slow ~20-60s due to quadrature).

- [ ] **Step 3: Add to `test/runtests.jl`**

Append `include("gsem.jl")` at the end of `test/runtests.jl`.

- [ ] **Step 4: Run full suite to confirm nothing regressed**

```bash
cd ~/projects/software/Lavaan.jl && julia --project=. test/runtests.jl 2>&1 | tail -30
```

Expected: All existing tests still PASS. New GSEM tests PASS.

- [ ] **Step 5: Commit**

```bash
cd ~/projects/software/Lavaan.jl
git add test/gsem.jl test/runtests.jl
git commit -m "test(gsem): integration tests for pure Poisson, mixed, and cfa/sem wrappers"
```

---

## Task 6: Polish — docs and CLAUDE.md update

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

- [ ] **Step 1: Update `CLAUDE.md` overview line**

Change:
```
Julia port of R's lavaan SEM package. Phase 7 complete: 294/294 tests pass (ordinal/categorical WLSMV added).
```
To (update test count after running full suite):
```
Julia port of R's lavaan SEM package. Phase 8 complete: XXX/XXX tests pass (Poisson GSEM via GH quadrature added).
```

- [ ] **Step 2: Add to "Completed features" in `CLAUDE.md`**

```
**Poisson GSEM (Phase 8)**: `family=Dict("y1"=>:poisson)` kwarg to `cfa()`/`sem()`. Auto-switches to `:GSEM`. Non-adaptive product Gauss-Hermite quadrature (Q=15 per latent dim). Mixed Gaussian+Poisson outcomes supported. Poisson residual variances fixed to 0. Intercepts auto-enabled (meanstructure=true, int_ov_free=true).
```

Remove from "Known limitations":
```
- Ordinal support uses DWLS (not full WLSMV mean/variance adjustment)...
```
(Keep it, it's still true. Just update the Poisson line if present.)

- [ ] **Step 3: Update `README.md`**

Add Poisson example to the "Ordinal / categorical data" section (rename section to "Non-Gaussian outcomes"):

```julia
## Non-Gaussian outcomes

# Ordinal/categorical (auto-selects DWLS)
fit = cfa(model, data; ordered=["y1", "y2"])

# Poisson count outcomes (auto-selects GSEM via Gauss-Hermite quadrature)
fit = cfa(model, data; family=Dict("y1"=>:poisson, "y2"=>:poisson))

# Mixed: some Gaussian, some Poisson indicators on the same factor
fit = cfa(model, data; family=Dict("y3"=>:poisson, "y4"=>:poisson))
```

- [ ] **Step 4: Commit**

```bash
cd ~/projects/software/Lavaan.jl
git add CLAUDE.md README.md
git commit -m "docs: update for Phase 8 Poisson GSEM"
```

---

## Self-Review

**Spec coverage checklist:**
- [x] Pure Poisson CFA (Task 5, testset 1)
- [x] Mixed Gaussian+Poisson (Task 5, testset 2)
- [x] Poisson SEM (structural paths): Task 3 handles B≠0 via reduced-form latent distribution
- [x] `family=Dict(...)` kwarg on `cfa()`/`sem()` (Task 4)
- [x] Auto-switch to `:GSEM` (Task 4)
- [x] Poisson residual variances fixed to 0 (Task 4, `_fix_poisson_residuals!`)
- [x] Intercepts auto-enabled for GSEM (Task 4)
- [x] Q=15 default, configurable via `n_quad_points` (Task 1)
- [x] ForwardDiff-compatible (all operations type-T throughout)
- [x] GH nodes/weights via eigenproblem (generalises to any Q) (Task 2)

**Placeholder scan:** None found.

**Type consistency check:**
- `_gh_nodes_weights(Q::Int)` → `(Vector{Float64}, Vector{Float64})` used consistently in Task 2 and Task 3 ✓
- `gsem_objective(theta, model, data, opts)` signature used consistently in Task 3 and Task 4 ✓
- `_fix_poisson_residuals!(pt::ParTable, family::Dict{String,Symbol})` consistent in Task 4 ✓
- `optimize_model(...; data::Union{LavaanData,Nothing})` consistent in Task 3 and Task 4 ✓

**Performance note:**
- For q=1 latent factor, Q=15: 15 quadrature evaluations per observation — fast
- For q=2: 225 evaluations per observation — still fast (~100ms for n=500)
- For q=3: 3375 evaluations — slow (consider reducing n_quad_points=7 for q≥3)
- Consider warning user if q ≥ 3 and n_quad_points > 7
