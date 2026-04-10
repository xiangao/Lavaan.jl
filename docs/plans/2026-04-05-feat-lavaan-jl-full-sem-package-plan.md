---
title: "feat: Lavaan.jl — Full SEM Package in Julia"
type: feat
date: 2026-04-05
---

# Lavaan.jl — Full SEM Package in Julia

## Overview

Complete Julia port of R's `lavaan` (Latent Variable Analysis) package for Structural Equation Modeling. Provides CFA, SEM, EFA, Multilevel SEM, and SAM with identical string-based model syntax to R lavaan for drop-in migration. Replaces lavaan's R internals with Julia-native performance: `ForwardDiff.jl` autodiff, multithreaded bootstrap, and optional GPU via `CUDA.jl`.

**Target**: Researchers using R lavaan who want Julia performance, and Julia users needing a full-featured SEM toolkit.

## Problem Statement

Julia has no production-ready SEM package. The R `lavaan` package is the gold standard (200K+ monthly downloads), but R is single-threaded and slow for large models and bootstrap inference. Julia offers 10–40× speedups for matrix-heavy workloads, native multithreading, and composability with other Julia statistical packages. A faithful port preserves the lavaan API so users need zero relearning.

## Technical Architecture

### Internal Data Flow (mirrors lavaan's 17-step pipeline)

```
model string + DataFrame
    ↓  syntax.jl       (step04) → parTable
    ↓  partable.jl     (step04) → completed parTable with auto parameters
    ↓  data.jl         (step03) → LavaanData (matrix X, missing patterns)
    ↓  samplestats.jl  (step05) → S (sample cov), μ̄, Γ (NACOV)
    ↓  model.jl        (step09) → LISREL matrices (Λ, Θ, Ψ, B, ν, α)
    ↓  start.jl        (step08) → starting values θ₀
    ↓  optim.jl        (step11) → θ̂ (estimated parameters)
    ↓  implied.jl      (step12) → Σ(θ̂), μ(θ̂)
    ↓  vcov.jl         (step13) → Σ_θ (parameter covariance)
    ↓  test.jl         (step14) → chi-sq, SB, YB
    ↓  fit.jl          (step15) → CFI, TLI, RMSEA, SRMR, AIC, BIC
    ↓  api.jl          (step17) → LavaanFit object
```

### LISREL Matrix System

For a structural model with p observed and q latent variables:

| Matrix | Dims | Role |
|--------|------|------|
| Λ (Lambda) | p × q | Factor loadings |
| Θ (Theta) | p × p | Residual covariances of observed |
| Ψ (Psi) | q × q | Latent (co)variances |
| B (Beta) | q × q | Structural regression coefficients |
| ν (Nu) | p × 1 | Observed intercepts |
| α (Alpha) | q × 1 | Latent intercepts |

**Model-implied moments:**
```
Σ(θ) = Λ (I-B)⁻¹ Ψ (I-B)⁻ᵀ Λᵀ + Θ
μ(θ) = ν + Λ (I-B)⁻¹ α
```

### ML Objective Function

```
F_ML(θ) = log|Σ(θ)| + tr(Σ(θ)⁻¹ S) - log|S| - p
```

Minimized via `Optim.jl` L-BFGS-B with gradients from `ForwardDiff.jl`.

### Parameter Table (parTable) Key Columns

| Column | Type | Role |
|--------|------|------|
| `id` | Int | Row index |
| `lhs`, `op`, `rhs` | String | Model equation element |
| `block` | Int | Group/level block number |
| `free` | Int | 0=fixed; >0=index in θ vector |
| `ustart` | Float/NA | User-specified starting value |
| `start`, `est`, `se` | Float | Optimization outputs |
| `lower`, `upper` | Float | Parameter bounds |
| `label`, `plabel` | String | User/internal labels |
| `mat`, `row`, `col` | String/Int | LISREL matrix location |

### Model Syntax Operators

| Operator | Meaning | Example |
|----------|---------|---------|
| `=~` | Measurement (factor → indicators) | `f =~ x1 + x2 + x3` |
| `~` | Regression | `y ~ x + f` |
| `~~` | (Co)variance | `f1 ~~ f2` |
| `~1` | Intercept | `y ~ 1` |
| `\|` | Threshold (ordinal) | `x1 \| t1 + t2` |
| `%~%` | Unit variance scaling | `f %~% f` |
| `<~` | Composite | `c <~ x1 + x2` |
| `:=` | User-defined parameter | `indirect := a*b` |
| `==` | Equality constraint | `b1 == b2` |

---

## Implementation Phases

### Phase 1 — Core Pipeline

**Goal**: `cfa()` and `sem()` with ML estimation working on continuous data.

**Validation target**: HolzingerSwineford1939 CFA reproduces R lavaan output within 1e-5 (estimates) and 1e-4 (SEs).

#### 1.1 Package Scaffolding

**Files to create:**
- `Project.toml` — package metadata and dependencies
- `src/Lavaan.jl` — module root with `include()` chain and `export` list
- `data/HolzingerSwineford1939.csv` — CFA benchmark dataset
- `data/PoliticalDemocracy.csv` — SEM benchmark dataset
- `test/runtests.jl` — test suite skeleton

**Project.toml dependencies:**
```toml
[deps]
CSV = "336ed68f-..."
DataFrames = "a93c6f00-..."
Distributions = "31c24e10-..."
ForwardDiff = "f6369f11-..."
LinearAlgebra = "37e2e46d-..."  # stdlib
Optim = "429524aa-..."
Random = "9a3f8284-..."  # stdlib
RecipesBase = "3cdcf5f2-..."
Statistics = "10745b16-..."  # stdlib

[compat]
julia = "1.10"
ForwardDiff = "0.10"
Optim = "1"
DataFrames = "1"
Distributions = "0.25"

[extras]
Test = "8dfed614-..."
[targets]
test = ["Test"]
```

#### 1.2 Options (`options.jl`)

Port `lav_lavaan_step02_options.R`. Julia struct holding all lavaan options:

```julia
# src/options.jl
Base.@kwdef struct LavaanOptions
    estimator::Symbol = :ML       # :ML, :MLR, :MLM, :FIML, :GLS, :WLS, :DWLS, :WLSMV, :PML, :ULS
    se::Symbol = :standard        # :standard, :robust.sem, :robust.huber.white, :bootstrap, :none
    test::Vector{Symbol} = [:standard]
    missing::Symbol = :listwise   # :listwise, :fiml, :pairwise
    meanstructure::Bool = false
    fixed_x::Bool = true
    conditional_x::Bool = false
    std_lv::Bool = false
    orthogonal::Bool = false
    representation::Symbol = :LISREL   # :LISREL, :RAM
    parameterization::Symbol = :delta  # :delta, :theta
    int_ov_free::Bool = false
    int_lv_free::Bool = false
    auto_fix_first::Bool = true
    auto_fix_single::Bool = true
    auto_var::Bool = true
    auto_cov_lv_x::Bool = true
    auto_cov_y::Bool = true
    auto_th::Bool = true
    auto_delta::Bool = true
    do_fit::Bool = true
    bounds::Symbol = :none        # :none, :default, :wide, :wide.zerovar
    ridge::Bool = false
    ridge_constant::Float64 = 1e-5
    nboot::Int = 1000
    verbose::Bool = false
    rotation::Symbol = :geomin    # for EFA
    rotation_args::NamedTuple = (;)
end
```

#### 1.3 Model Syntax Parser (`syntax.jl`)

Port `lav_syntax_parser.R` (1090 lines) and `lav_syntax.R`.

**Input**: model string
**Output**: flat list of formula elements with `lhs`, `op`, `rhs`, `modifiers`

**Parsing stages:**
1. Tokenize: split on operators (`=~`, `~~`, `~1`, `~`, `|`, `:=`, `==`, `<~`, `%~%`, `+`, `*`)
2. Parse formulas: group tokens into LHS op RHS tuples
3. Expand: `f =~ x1 + x2 + x3` → three rows
4. Extract modifiers: `f =~ x1*1 + x2` → fixed loading for x1

**Key function signature:**
```julia
# src/syntax.jl
function parse_model_string(model::String) :: Vector{NamedTuple}
    # Returns flat list of (lhs, op, rhs, modifier) named tuples
end
```

#### 1.4 Parameter Table (`partable.jl`)

Port `lav_partable.R` + 19 auxiliary `lav_partable_*.R` files.

**Input**: parsed syntax + data variable names + LavaanOptions
**Output**: `ParTable` struct (essentially a columnar DataFrame)

**Key operations:**
- `auto_add_variances()` — add `~~` rows for all endogenous variables
- `auto_add_covariances()` — add `~~` for exogenous latent correlations
- `auto_fix_loadings()` — fix first loading to 1.0 per factor
- `assign_free_indices()` — assign sequential integers to free parameters
- `set_starting_values()` — port `lav_lavaan_step08_start.R`
- `set_bounds()` — port `lav_lavaan_step07_bounds.R`

```julia
# src/partable.jl
struct ParTable
    id::Vector{Int}
    lhs::Vector{String}
    op::Vector{String}
    rhs::Vector{String}
    user::Vector{Int}
    block::Vector{Int}
    group::Vector{String}
    level::Vector{String}
    free::Vector{Int}
    ustart::Vector{Union{Float64, Missing}}
    start::Vector{Float64}
    lower::Vector{Float64}
    upper::Vector{Float64}
    label::Vector{String}
    plabel::Vector{String}
    exo::Vector{Bool}
    eq_id::Vector{Int}
    unco::Vector{Int}
    # After optimization:
    est::Vector{Float64}
    se::Vector{Float64}
    z::Vector{Float64}
    pvalue::Vector{Float64}
    ci_lower::Vector{Float64}
    ci_upper::Vector{Float64}
    # Matrix location (set in model.jl):
    mat::Vector{String}
    row::Vector{Int}
    col::Vector{Int}
end
```

**Key query functions:**
```julia
free_params(pt::ParTable) = pt[pt.free .> 0, :]
nfree(pt::ParTable) = maximum(pt.free)
get_param(pt::ParTable, lhs, op, rhs, block=1) :: Float64
```

#### 1.5 Data Handling (`data.jl`)

Port `lav_lavaan_step03_data.R` + `lav_data.R`.

```julia
# src/data.jl
struct LavaanData
    X::Vector{Matrix{Float64}}      # one matrix per group, rows=obs, cols=vars
    ov_names::Vector{String}        # observed variable names
    lv_names::Vector{String}        # latent variable names
    ngroups::Int
    nobs::Vector{Int}               # obs per group
    missing_patterns::Vector{Any}   # for FIML
    weights::Vector{Union{Vector{Float64}, Nothing}}
end

function prepare_data(df::DataFrame, ov_names::Vector{String};
                      group::Union{Symbol,Nothing}=nothing,
                      cluster::Union{Symbol,Nothing}=nothing,
                      sampling_weights::Union{Symbol,Nothing}=nothing,
                      missing::Symbol=:listwise) :: LavaanData
```

#### 1.6 Sample Statistics (`samplestats.jl`)

Port `lav_lavaan_step05_samplestats.R` + `lav_samplestats.R`.

**Computes:**
- `S` — sample covariance matrix (per group)
- `μ̄` — sample means (per group)
- `S⁻¹` — inverse of sample covariance
- `Γ` — asymptotic covariance of vech(S) [for WLS/GLS]

```julia
# src/samplestats.jl
struct SampleStats
    S::Vector{Matrix{Float64}}         # sample cov per group
    S_inv::Vector{Matrix{Float64}}     # inverse
    mu::Vector{Vector{Float64}}        # sample means per group
    nobs::Vector{Int}
    p::Int                             # number of observed vars
    Gamma::Union{Nothing, Vector{Matrix{Float64}}}  # asymp. cov for WLS
    NACOV::Union{Nothing, Vector{Matrix{Float64}}}  # user-supplied NACOV
end
```

#### 1.7 LISREL Model Construction (`model.jl`)

Port `lav_lavaan_step09_model.R` + `lav_model.R` + `lav_representation_lisrel.R`.

**Input**: completed parTable + LavaanOptions
**Output**: `LavaanModel` with LISREL matrix templates and param → matrix mapping

```julia
# src/model.jl
struct LISRELBlock
    Lambda::Matrix{Float64}     # p × q factor loadings
    Theta::Matrix{Float64}      # p × p residual covs (lower tri)
    Psi::Matrix{Float64}        # q × q latent covs (lower tri)
    Beta::Matrix{Float64}       # q × q structural coefficients
    Nu::Vector{Float64}         # p intercepts of observed
    Alpha::Vector{Float64}      # q intercepts of latent
    # Index vectors mapping θ → matrix elements:
    free_map::Vector{Tuple{Symbol,Int,Int}}  # (matrix, row, col) for each free param
end

struct LavaanModel
    blocks::Vector{LISRELBlock}
    nblocks::Int
    nfree::Int
    partable::ParTable
    options::LavaanOptions
end
```

**Key function:**
```julia
function fill_matrices!(block::LISRELBlock, theta::Vector{Float64})
    # Map free parameter vector θ back into LISREL matrices
    # This is called inside the optimizer on every iteration
end
```

#### 1.8 Model-Implied Moments (`implied.jl`)

Port `lav_lavaan_step12_implied.R`.

```julia
# src/implied.jl
struct ImpliedMoments
    Sigma::Vector{Matrix{Float64}}   # model-implied cov per block
    Mu::Vector{Vector{Float64}}      # model-implied means per block
end

function compute_implied(model::LavaanModel, theta::Vector{Float64}) :: ImpliedMoments
    # For each block:
    # Σ(θ) = Λ (I-B)⁻¹ Ψ (I-B)⁻ᵀ Λᵀ + Θ
    # μ(θ) = ν + Λ (I-B)⁻¹ α
end
```

**Critical**: `compute_implied` must be fully differentiable w.r.t. `theta` for `ForwardDiff` to work. Use only `LinearAlgebra` operations (no special-casing).

#### 1.9 ML Estimator (`estimators/ml.jl`)

```julia
# src/estimators/ml.jl
function ml_objective(theta::Vector{T}, model::LavaanModel,
                      stats::SampleStats) :: T where T
    # Compute F_ML = Σ_g [log|Σg(θ)| + tr(Σg(θ)⁻¹ Sg) - log|Sg| - p]
    # T is a number type — Float64 normally, ForwardDiff.Dual during autodiff
    implied = compute_implied(model, theta)
    F = zero(T)
    for g in 1:model.nblocks
        Sigma_g = implied.Sigma[g]
        S_g = stats.S[g]
        n_g = stats.nobs[g]
        # Cholesky for log-det and solve
        chol = cholesky(Sigma_g)
        logdet_Sigma = 2 * sum(log.(diag(chol.L)))
        tr_term = tr(chol \ S_g)  # tr(Σ⁻¹ S) via Cholesky solve
        F += n_g * (logdet_Sigma + tr_term - stats.logdet_S[g] - stats.p)
    end
    return F / 2  # lavaan uses N/2 * F_ML
end
```

**Note on type generics**: The `T` type parameter is critical — `ForwardDiff` passes `Dual` numbers during differentiation; the objective must work for any `<:Real`.

#### 1.10 Optimization Engine (`optim.jl`)

Port `lav_lavaan_step11_optim.R`.

```julia
# src/optim.jl
using Optim, ForwardDiff

function optimize_model(model::LavaanModel, stats::SampleStats,
                        opts::LavaanOptions) :: OptimResult
    theta0 = get_starting_values(model)
    lb, ub = get_bounds(model)

    obj = theta -> estimator_objective(theta, model, stats, opts.estimator)

    # Use gradient via ForwardDiff, L-BFGS-B for bounded optimization
    if any(isfinite, lb) || any(isfinite, ub)
        result = optimize(obj, lb, ub, theta0, Fminbox(LBFGS()),
                          Optim.Options(g_tol=1e-7, iterations=10000);
                          autodiff=:forward)
    else
        result = optimize(obj, theta0, LBFGS(),
                          Optim.Options(g_tol=1e-7, iterations=10000);
                          autodiff=:forward)
    end
    return result
end
```

#### 1.11 Variance-Covariance of Estimates (`vcov.jl`)

Port `lav_lavaan_step13_vcov.R`.

**Standard ML vcov** (information matrix inverse):
```
Σ_θ = (1/N) * H⁻¹
```
where `H` is the Hessian of `F_ML` at `θ̂`, computed via `ForwardDiff.hessian`.

**Sandwich (MLR) vcov**:
```
Σ_θ = H⁻¹ Γ H⁻¹    (where Γ = outer product of scores)
```

```julia
# src/vcov.jl
function compute_vcov(model::LavaanModel, theta_hat::Vector{Float64},
                      stats::SampleStats, opts::LavaanOptions) :: Matrix{Float64}
    obj = theta -> estimator_objective(theta, model, stats, opts.estimator)
    H = ForwardDiff.hessian(obj, theta_hat)
    H_inv = inv(H)
    if opts.se == :standard
        return (2 / sum(stats.nobs)) * H_inv
    elseif opts.se in (:robust_sem, :robust_huber_white)
        Gamma = compute_Gamma(model, theta_hat, stats)
        return H_inv * Gamma * H_inv / sum(stats.nobs)
    end
end
```

#### 1.12 Fit Indices (`fit.jl`)

Port `lav_fit_measures.R` + component files (`lav_fit_cfi.R`, `lav_fit_rmsea.R`, `lav_fit_srmr.R`, etc.).

Full set of indices to implement:

| Index | Formula | Notes |
|-------|---------|-------|
| chi-sq | `N * F_ML` | `df = p(p+1)/2 - nfree` |
| CFI | `1 - (F_M - df_M) / (F_B - df_B)` | vs. baseline model |
| TLI (NNFI) | `(F_B/df_B - F_M/df_M) / (F_B/df_B - 1)` | |
| RMSEA | `sqrt(max(chi_sq/df - 1, 0) / N)` | with 90% CI |
| SRMR | `sqrt(mean((s_ij - σ̂_ij)² / (s_ii * s_jj)))` | |
| GFI | `1 - tr((Σ⁻¹S - I)²) / tr((Σ⁻¹S)²)` | |
| AIC | `chi_sq - 2*df` | |
| BIC | `chi_sq + log(N)*nfree` | |
| SABIC | `chi_sq + log((N+2)/24)*nfree` | |

Baseline (independence) model computed in `step15_baseline.R` logic — needed for CFI/TLI.

#### 1.13 User API (`api.jl`)

```julia
# src/api.jl

struct LavaanFit
    partable::ParTable
    data::LavaanData
    samplestats::SampleStats
    model::LavaanModel
    implied::ImpliedMoments
    vcov_theta::Matrix{Float64}
    optim_result::Any
    fit_measures::Dict{Symbol, Float64}
    options::LavaanOptions
    timing::Dict{Symbol, Float64}
end

function lavaan(model::String, data::DataFrame;
                ordered=nothing, sampling_weights=nothing,
                sample_cov=nothing, sample_mean=nothing, sample_nobs=nothing,
                group=nothing, cluster=nothing,
                constraints="", kwargs...) :: LavaanFit

function cfa(model::String, data::DataFrame; kwargs...) :: LavaanFit
    # Sets: auto_fix_first=true, auto_var=true, auto_cov_lv_x=true
    lavaan(model, data; model_type=:cfa, kwargs...)
end

function sem(model::String, data::DataFrame; kwargs...) :: LavaanFit
    # Sets: auto_fix_first=true, auto_var=true, auto_cov_y=true
    lavaan(model, data; model_type=:sem, kwargs...)
end

function growth(model::String, data::DataFrame; kwargs...) :: LavaanFit
    lavaan(model, data; model_type=:growth, meanstructure=true, kwargs...)
end
```

#### 1.14 Results Extraction (`results.jl`)

```julia
# src/results.jl — port lav_print.R, lav_standardize.R, lav_residuals.R

function parameterEstimates(fit::LavaanFit;
                             standardized=false, ci=true, level=0.95,
                             remove_nonfree=true) :: DataFrame

function fitMeasures(fit::LavaanFit,
                     fit_measures=:all) :: Dict{Symbol,Float64}

function coef(fit::LavaanFit) :: Vector{Float64}
function vcov(fit::LavaanFit) :: Matrix{Float64}
function fitted(fit::LavaanFit) :: NamedTuple
function residuals(fit::LavaanFit; type=:raw) :: NamedTuple
function standardizedSolution(fit::LavaanFit; type=:std_lv) :: DataFrame
function inspect(fit::LavaanFit, what::Symbol) :: Any
```

#### 1.15 Print Methods (`print.jl`)

Port `lav_print.R`. Implement `Base.show` for `LavaanFit`:

```julia
# src/print.jl
function Base.show(io::IO, fit::LavaanFit)
    # Output matching lavaan's summary():
    # - Header: estimator, optimizer, nobs, model test
    # - Parameter estimates table: lhs op rhs est se z pvalue ci.lower ci.upper
    # - Model fit: chi-sq, df, p-value, CFI, TLI, RMSEA, SRMR
end

function summary(fit::LavaanFit; fit_measures=false, standardized=false,
                 ci=false, rsquare=false, modindices=false)
```

#### 1.16 Phase 1 Tests (`test/runtests.jl`)

```julia
@testset "Lavaan.jl Phase 1 — Core CFA/SEM" begin

    @testset "Syntax Parser" begin
        # Test all operators: =~, ~, ~~, ~1, |, :=, ==
        model = "visual =~ x1 + x2 + x3\nspeed  =~ x4 + x5 + x6"
        pt = parse_model_string(model)
        @test length(pt) == 6
        @test all(r.op == "=~" for r in pt)
    end

    @testset "HolzingerSwineford1939 CFA" begin
        # Reproduce R lavaan output exactly
        df = holzinger_swineford()
        model = """
            visual  =~ x1 + x2 + x3
            textual =~ x4 + x5 + x6
            speed   =~ x7 + x8 + x9
        """
        fit = cfa(model, df)
        pe = parameterEstimates(fit)

        # Compare against known R lavaan values
        # x1 loading on visual = 1.0 (fixed)
        @test isapprox(get_est(pe, "x1", "=~", "visual"), 1.0, atol=1e-10)
        # x2 loading ≈ 0.5536
        @test isapprox(get_est(pe, "x2", "=~", "visual"), 0.5536, atol=1e-4)
        # CFI ≈ 0.930
        @test isapprox(fitMeasures(fit)[:cfi], 0.930, atol=0.001)
        # RMSEA ≈ 0.092
        @test isapprox(fitMeasures(fit)[:rmsea], 0.092, atol=0.001)
    end

    @testset "PoliticalDemocracy SEM" begin
        df = political_democracy()
        model = """
            ind60 =~ x1 + x2 + x3
            dem60 =~ y1 + y2 + y3 + y4
            dem65 =~ y5 + y6 + y7 + y8
            dem60 ~ ind60
            dem65 ~ ind60 + dem60
        """
        fit = sem(model, df)
        # Chi-sq ≈ 38.1, df = 35
        @test isapprox(fitMeasures(fit)[:chisq], 38.12, atol=0.5)
        @test fitMeasures(fit)[:df] == 35
    end

end
```

---

### Phase 2 — All Estimators

**Files**: `src/estimators/fiml.jl`, `src/estimators/wls.jl`, `src/estimators/gls.jl`, `src/estimators/pml.jl`, `src/estimators/uls.jl`

#### 2.1 FIML (`estimators/fiml.jl`)

Port `lav_samplestats.R` missing=`fiml` path + `lav_model_missing.R`.

Full-information ML for incomplete data. Each missing-data pattern has its own contribution to the likelihood:

```
F_FIML = -2 Σᵢ [-½ log|Σᵢ(θ)| - ½ (xᵢ - μᵢ(θ))ᵀ Σᵢ(θ)⁻¹ (xᵢ - μᵢ(θ))]
```

where Σᵢ and μᵢ are model moments restricted to observed variables for observation i.

Implementation: group observations by missing pattern, compute likelihood contribution per pattern via vectorized Cholesky solve.

#### 2.2 WLS / DWLS / WLSMV (`estimators/wls.jl`)

Port `lav_samplestats_gamma.R`, `lav_samplestats_icov.R`.

```
F_WLS(θ) = (s - σ(θ))ᵀ W⁻¹ (s - σ(θ))
```

where `s = vech(S)`, `σ(θ) = vech(Σ(θ))`, and W is the weight matrix.

- **WLS**: W = Γ (full asymptotic weight matrix)
- **DWLS**: W = diag(Γ) (diagonal only, robust test)
- **WLSMV**: DWLS with mean-and-variance adjusted test statistic

For ordinal data: Σ̂ is polychoric correlation matrix (computed via bivariate normal integration).

#### 2.3 GLS (`estimators/gls.jl`)

```
F_GLS(θ) = ½ tr[(I - Σ(θ) S⁻¹)²]
         = ½ tr[(S⁻¹ Σ(θ) - I)²]
```

#### 2.4 PML (`estimators/pml.jl`)

Port `lav_pml_utils.R`, `lav_pml_plrt.R`. Pairwise composite likelihood for ordinal + missing data:
```
F_PML = -2 Σ_{j<k} log L_{jk}(θ)
```
where each bivariate term is a bivariate normal/ordinal likelihood. Gradient via ForwardDiff (or finite differences for speed).

#### 2.5 ULS (`estimators/uls.jl`)

```
F_ULS(θ) = ½ tr[(S - Σ(θ))²]
```

Unweighted; rarely used in practice but needed for completeness.

#### 2.6 Robust SEs for MLR/MLM

Port `lav_model_vcov.R` sandwich estimator.

**MLR** (Huber-White sandwich):
```
Σ_θ^MLR = N * H⁻¹ Γ̂ H⁻¹
```
where Γ̂ = outer product of case-wise scores (port `lav_scores.R`).

**MLM** (Satorra-Bentler scaled chi-square):
Test statistic scaled by `tr(UΓ) / df` where U = Ψ⁻¹ ⊗ Ψ⁻¹.

---

### Phase 3 — Inference

#### 3.1 Multithreaded Bootstrap (`bootstrap.jl`)

Port `lav_bootstrap.R`.

```julia
# src/bootstrap.jl
function bootstrap(fit::LavaanFit; R=1000, type=:ordinary) :: BootstrapResult
    n = sum(fit.samplestats.nobs)
    results = Vector{Vector{Float64}}(undef, R)
    failed = Atomic{Int}(0)

    Threads.@threads for b in 1:R
        try
            idx = rand(1:n, n)  # resample rows
            df_b = fit.data.original[idx, :]
            fit_b = lavaan(fit.model_string, df_b; do_fit=true, se=:none)
            results[b] = coef(fit_b)
        catch
            atomic_add!(failed, 1)
            results[b] = fill(NaN, nfree(fit))
        end
    end
    return BootstrapResult(results, R, failed[])
end
```

Types: `:ordinary` (nonparametric), `:bollen_stine` (model-based), `:parametric`.

#### 3.2 Model Tests (`test.jl`)

Port `lav_test.R`, `lav_test_satorra_bentler.R`, `lav_test_yuan_bentler.R`, `lav_test_diff.R`.

- **Standard chi-sq**: `T = N * F_ML`, df = `p*(p+1)/2 - nfree`
- **Satorra-Bentler**: scaled and shifted chi-sq for non-normal data
- **Yuan-Bentler**: scaled test for MLM
- **LRT**: `lavTestLRT(fit1, fit2)` for nested model comparison
- **Score test**: `lavTestScore(fit)` — Lagrange multiplier test
- **Wald test**: `lavTestWald(fit, constraints)` — user-defined constraints

#### 3.3 Modification Indices (`modindices.jl`)

Port `lav_modindices.R`. Expected chi-sq reduction if a fixed parameter were freed:

```julia
function modindices(fit::LavaanFit; sort=true, minimum_value=10.0) :: DataFrame
    # Expected parameter change (EPC) and MI for each fixed parameter
    # MI = (∂F/∂θ)² / (∂²F/∂θ²)  [diagonal of information matrix for fixed params]
end
```

---

### Phase 4 — Advanced Models

#### 4.1 Multiple Groups (`groups.jl`)

Port `lav_partable.R` group handling + `lav_lavaan_step03_data.R` group splitting.

**Key change**: parTable gets a `block` column; each group = one block. LISREL matrices replicated per group with equality constraints (group.equal) handled via shared free indices in parTable.

```julia
# Parallel group estimation
function estimate_groups(model::LavaanModel, stats::SampleStats) :: Vector{Float64}
    # Groups estimated jointly (single optimizer, blocked objective)
    # Within-group matrices are independent → jacobian is block-diagonal
end
```

**Measurement invariance tests** (configural → metric → scalar):
```julia
lavTestLRT(fit_configural, fit_metric, fit_scalar)
```

#### 4.2 Multilevel SEM (`multilevel.jl`)

Port `lav_lavaan_step03_data.R` cluster handling + multilevel LISREL.

**Two-level decomposition**:
- Within-level model: variation within clusters
- Between-level model: variation of cluster means

Model syntax:
```r
model <- '
  level: 1
    fw =~ y1 + y2 + y3
    fw ~ x
  level: 2
    fb =~ y1 + y2 + y3
    fb ~ w
'
```

Both within/between LISREL blocks estimated jointly.

#### 4.3 EFA (`efa.jl`)

Port `lav_efa_extraction.R`, `lav_efa_utils.R`, `lav_lavaan_step16_rotation.R`.

**Extraction methods**: ML (default), ULS, DWLS
**Rotations** (port `lav_rotation.R`):
- Orthogonal: varimax, quartimax, equamax, geominT, bentlerT, bifactorT
- Oblique: oblimin, quartimin, geomin, CF-quartimax, bentlerQ, bifactorQ, promax
- 20+ methods total via rotation parameter

```julia
function efa(model::String, data::DataFrame; rotation=:geomin,
             nfactors::Int=1, kwargs...) :: LavaanFit
```

**EFA syntax** (from lavaan): uses `efa("block")*` notation:
```r
model <- '
  efa("efa1")*f1 =~ x1 + x2 + x3 + x4 + x5
  efa("efa1")*f2 =~ x1 + x2 + x3 + x4 + x5
'
```

#### 4.4 SAM (`sam.jl`)

Port `lav_sam_step0.R`, `lav_sam_step1.R`, `lav_sam_step1_local.R`, `lav_sam_step2.R`.

Structural After Measurement: first estimate measurement models (CFA) per block, then use factor scores as inputs for structural model. More robust to measurement model misspecification.

```julia
function sam(model::String, data::DataFrame;
             sam_method=:local,   # :local or :global
             kwargs...) :: LavaanFit
```

Both local SAM (block-by-block CFAs) and global SAM (simultaneous CFA) in Phase 4.

---

### Phase 5 — Polish & Extensions

#### 5.1 Standardized Solutions (`standardize.jl`)

Port `lav_standardize.R`. Three types:
- `std.lv`: standardize latent variables only
- `std.all`: standardize all (including observed)
- `std.nox`: standardize except exogenous observed

#### 5.2 Simulation (`simulate.jl`)

Port `lav_simulate.R`. Generate data from a specified model:

```julia
function simulateData(model::String; n=500, seed=nothing,
                      empirical=false, kwargs...) :: DataFrame
```

#### 5.3 Factor Scores & Predictions (`predict.jl`)

Port `lav_predict.R`.

```julia
function lavPredict(fit::LavaanFit; type=:lv, newdata=nothing) :: Matrix{Float64}
    # type=:lv → factor scores (regression, Bartlett, EBayes)
    # type=:ov → predicted observed values
    # type=:yhat → model-fitted values
end
```

#### 5.4 R-Square (`rsquare.jl`)

Port `lav_fit_other.R` R² section. For all endogenous variables.

#### 5.5 GPU Extension (`ext/LavaanCUDA/`)

Optional CUDA extension (Julia 1.9+ package extension API):

```
ext/
  LavaanCUDA.jl    # loaded when CUDA.jl is available
```

Accelerate: matrix multiply and invert in `compute_implied()`, Γ computation in `samplestats.jl` for large p.

Auto-detect and use GPU when available:
```julia
function compute_implied(model, theta, device=CPU())
    # dispatch to GPU implementation if device isa GPU
end
```

#### 5.6 Full Vignette Suite

```
vignettes/
  introduction.qmd          # CFA/SEM basics (HolzingerSwineford1939)
  multiple-groups.qmd       # measurement invariance testing
  ordinal-data.qmd          # DWLS with ordinal indicators
  multilevel.qmd            # two-level SEM
  efa.qmd                   # exploratory factor analysis
  paper-results.qmd         # reproduce published SEM analyses
  Project.toml              # vignette environment
```

---

## Acceptance Criteria

### Functional

- [x] `cfa()` and `sem()` produce parameter estimates matching R lavaan within `atol=1e-4`
- [x] `fitMeasures()` produces CFI, TLI, RMSEA, SRMR within `atol=0.001` of R lavaan
- [x] ML, FIML, WLS, DWLS, GLS, ULS estimators functional (PML and WLSMV Phase 3)
- [ ] Multiple groups with `group=` argument and `group.equal=` constraints
- [ ] `lavTestLRT()` for nested model comparison
- [ ] `modindices()` returns expected chi-sq improvements
- [ ] `efa()` with ≥10 rotation methods
- [ ] `sam()` both local and global
- [ ] Two-level multilevel SEM with `cluster=` argument
- [ ] Bootstrap inference multithreaded via `Threads.@threads`
- [ ] `simulateData()` for power analysis
- [ ] `lavPredict()` for factor scores

### Performance

- [ ] Bootstrap (R=1000) on HolzingerSwineford1939 runs in <10s on 8 threads
- [ ] GPU extension reduces matrix ops time by >5× for p>100 models
- [ ] L-BFGS-B optimization converges in <200 iterations for standard models

### API Compatibility

- [x] Model syntax strings from R lavaan run unchanged
- [x] `parameterEstimates()`, `fitMeasures()`, `coef()`, `vcov()`, `fitted()`, `residuals()` all return expected types
- [x] `summary(fit)` output visually matches R lavaan `summary(fit)`
- [x] `inspect(fit, "free")` returns the parTable rows with `free > 0`

### Quality

- [ ] Test suite covers all Phase 1–4 functionality (≥100 test cases)
- [x] All numerical tests validated against R lavaan reference values
- [ ] Vignettes render cleanly with `quarto render`
- [ ] No type instability (`@code_warntype` clean on hot paths)

---

## Dependencies & Prerequisites

### Julia Packages

| Package | Version | Role |
|---------|---------|------|
| `ForwardDiff.jl` | `0.10` | Automatic differentiation for gradients/Hessian |
| `Optim.jl` | `1.0` | L-BFGS-B optimization engine |
| `DataFrames.jl` | `1.0` | Data handling and output tables |
| `CSV.jl` | `0.10` | Loading benchmark datasets |
| `Distributions.jl` | `0.25` | Normal, MvNormal distributions |
| `LinearAlgebra` | stdlib | Cholesky, matrix ops |
| `Statistics` | stdlib | mean, cov |
| `Random` | stdlib | Bootstrap RNG |
| `RecipesBase.jl` | `1.0` | Plotting recipes |
| `CUDA.jl` | `5.0` | GPU extension (Phase 5, optional) |

### Reference Implementation

- R lavaan source: `~/projects/repo_cloned/lavaan/R/` (211 files, reference for all algorithms)
- R lavaan must be installed for validation: `install.packages("lavaan")`

---

## Critical Gaps from SpecFlow Analysis

The SpecFlow review identified 45 gaps; the 8 critical ones that affect Phase 1 architecture are resolved here.

### G1 — Define `LavaanFit` before writing any code

**Resolution**: Add `src/types.jl` as the **first file written** (before `syntax.jl`). Every other component indexes into this struct. Changing it retroactively touches all 15+ source files.

```julia
# src/types.jl
mutable struct LavaanFit
    # Inputs
    model_string::String
    options::LavaanOptions
    # Pipeline stages (all accessible via lavInspect)
    partable::ParTable
    data::LavaanData
    samplestats::SampleStats
    model::LavaanModel          # LISREL matrix templates
    implied::ImpliedMoments     # Σ(θ̂), μ(θ̂)
    vcov_theta::Matrix{Float64} # parameter covariance
    # Results
    fit_measures::Dict{Symbol,Float64}
    baseline::Union{LavaanFit,Nothing}  # independence model for CFI/TLI
    optim_result::Any
    # Status
    converged::Bool
    iterations::Int
    warnings::Vector{String}
    timing::Dict{Symbol,Float64}
end
```

### G2 — Non-convergence: never throw, always return

**Resolution**: `sem()`/`cfa()` returns a `LavaanFit` with `converged=false` and all estimates `NaN`. Never throws by default. Users check `fit.converged`. Bootstrap loops rely on this contract.

### G3 — Multiple groups: unified parameter vector, not per-group threads

**Resolution**: Multi-group estimation uses **one global parameter vector** with group-level index maps. The parTable's `block` column already encodes this — free parameter indices are globally unique across groups. Cross-group equality constraints (`group.equal=`) share the same `free` index. Parallelism moves to within-group matrix operations, not across groups.

```julia
# The block-aware objective:
function ml_objective_multigroup(theta, model, stats)
    F = 0.0
    for g in 1:model.nblocks
        F += ml_block_objective(theta, model.blocks[g], stats.blocks[g])
    end
    return F
end
```

### G4 — Labels and equality constraints in Phase 1

**Resolution**: `syntax.jl` must handle `b1*x1` labels and `b1 == b2` constraints in Phase 1. These are used in virtually every nontrivial SEM application. Inequality constraints (`b1 > 0`) deferred to Phase 3.

### G5 — Starting values in Phase 1 (not Phase 5)

**Resolution**: Port `lav_lavaan_step08_start.R` (starting value algorithm) as a **Phase 1 required task**. Add `src/start.jl`. Default strategy: unit loadings, observed variances for residuals/latent variances, zero regression paths. Poor starting values cause saddle-point convergence.

```julia
# src/start.jl
function set_starting_values!(pt::ParTable, stats::SampleStats, opts::LavaanOptions)
    # For =~ rows with fixed value: use that value
    # For =~ rows (free): use 1.0 for first loading, else empirical correlation / sqrt(obs_var)
    # For ~~ rows (variances, free): use observed variance * 0.05
    # For ~~ rows (covariances): use 0.0
    # For ~ rows: use 0.0
    # For ~1 rows: use observed mean
end
```

### G6 — Baseline model cached at fit time

**Resolution**: The independence model is **always fitted inside `sem()`/`cfa()`** and stored as `fit.baseline`. `fitMeasures()` uses the cached baseline. This adds ~5% runtime overhead but avoids silent `NA` for CFI/TLI.

### G7 — Input validation layer before optimization

**Resolution**: Add `src/validate.jl` called between `data.jl` and `model.jl`. Checks: all model variables in DataFrame, no zero-variance columns, N > p, covariance matrix positive definite, ordered encoding matches estimator. Raises named errors with R lavaan–style messages.

```julia
# src/validate.jl
function validate_model_data(pt::ParTable, data::LavaanData, opts::LavaanOptions)
    # G7a: all ov.names in DataFrame columns
    missing_vars = setdiff(ov_names(pt), data.ov_names)
    !isempty(missing_vars) && error("Variables in model not in data: $(join(missing_vars, \", \"))")
    # G7b: zero-variance check
    for (g, X) in enumerate(data.X)
        zero_var = findall(var(X, dims=1)[:] .< 1e-10)
        !isempty(zero_var) && @warn "Zero-variance columns in group $g: $(data.ov_names[zero_var])"
    end
    # G7c: N > p
    p = length(data.ov_names)
    all(n < p for n in data.nobs) && error("N < p: model is not identified")
end
```

### G8 — `lavInspect()` in Phase 1 API

**Resolution**: Add `lavInspect(fit, what::Symbol)` to `api.jl`. Dispatches on a symbol key to extract named components. Essential for debugging and for users migrating from R.

```julia
function lavInspect(fit::LavaanFit, what::Symbol)
    what == :lambda    && return [b.Lambda for b in fit.model.blocks]
    what == :theta     && return [b.Theta for b in fit.model.blocks]
    what == :psi       && return [b.Psi for b in fit.model.blocks]
    what == :beta      && return [b.Beta for b in fit.model.blocks]
    what == :implied   && return fit.implied
    what == :residuals && return compute_residuals(fit)
    what == :cov_obs   && return [s.S for s in fit.samplestats.blocks]
    what == :free      && return filter(r -> r.free > 0, fit.partable)
    what == :converged && return fit.converged
    what == :nobs      && return fit.data.nobs
    error("lavInspect: unknown 'what' = :$what")
end
```

---

## Risk Analysis

| Risk | Mitigation |
|------|-----------|
| ForwardDiff fails for non-smooth objectives (PML, bivariate integrals) | Fall back to finite differences for PML; use analytic gradient for ordinal likelihoods |
| Type instability breaks ForwardDiff (e.g., branching on Float64) | Ensure all hot-path code is type-generic; use `@code_warntype` during development |
| LISREL matrix singularity during optimization | Add ridge regularization option (already in lavaan: `ridge=1e-5`); catch `PosDefException` |
| Polychoric correlation computation numerical issues | Port lavaan's `lav_bvord.R` Gaussian CDF integration exactly; use `Distributions.jl` for bivariate normal |
| Multithreading race conditions in bootstrap | Use thread-local RNG instances; pre-allocate result vectors |

---

## Validation References

All numerical outputs validated against R lavaan:

```r
library(lavaan)

# Phase 1 CFA reference
fit_r <- cfa("visual =~ x1 + x2 + x3\ntextual =~ x4+x5+x6\nspeed =~ x7+x8+x9",
             data=HolzingerSwineford1939)
parameterEstimates(fit_r)
fitMeasures(fit_r, c("cfi", "tli", "rmsea", "srmr", "chisq", "df"))

# Phase 1 SEM reference
fit_sem_r <- sem("ind60 =~ x1+x2+x3\ndem60 =~ y1+y2+y3+y4\ndem65 =~ y5+y6+y7+y8\n
                  dem60 ~ ind60\ndem65 ~ ind60 + dem60",
                 data=PoliticalDemocracy)
```

---

## File Checklist

### Phase 1 (Core) — creation order matters ✓ COMPLETE (61/61 tests pass)
- [x] `src/types.jl`              ← FIRST: LavaanFit struct (G1)
- [x] `Project.toml`
- [x] `src/Lavaan.jl`             ← module root + include() chain
- [x] `src/options.jl`
- [x] `src/syntax.jl`             ← parser incl. labels + == constraints (G4)
- [x] `src/partable.jl`
- [x] `src/data.jl`
- [x] `src/validate.jl`           ← input validation before optimization (G7)
- [x] `src/samplestats.jl`
- [x] `src/model.jl`              ← LISREL matrices, unified param vector (G3)
- [x] `src/start.jl`              ← starting values algorithm (G5)
- [x] `src/implied.jl`
- [x] `src/estimators/ml.jl`
- [x] `src/optim.jl`              ← non-convergence returns LavaanFit (G2)
- [x] `src/vcov.jl`
- [x] `src/fit.jl`                ← baseline fitted inside sem() (G6)
- [x] `src/results.jl`
- [x] `src/print.jl`
- [x] `src/api.jl`                ← sem, cfa, lavInspect (G8)
- [x] `data/HolzingerSwineford1939.csv`
- [x] `data/PoliticalDemocracy.csv`
- [x] `test/runtests.jl`
- [x] `README.md`
- [x] `CLAUDE.md`

### Phase 2 (Estimators)
- [x] `src/estimators/fiml.jl`    ← FIML for missing data (pattern-based)
- [x] `src/estimators/wls.jl`     ← WLS + DWLS (normal-theory Gamma)
- [x] `src/estimators/gls.jl`     ← GLS (implemented inline in ml.jl)
- [ ] `src/estimators/pml.jl`     ← pairwise ML (deferred to Phase 3)
- [x] `src/estimators/uls.jl`     ← ULS (implemented inline in ml.jl)
- [x] Update `src/vcov.jl` (MLR sandwich stub with clear warning)

### Phase 3 (Inference)
- [ ] `src/bootstrap.jl`
- [ ] `src/test.jl`
- [ ] `src/modindices.jl`

### Phase 4 (Advanced)
- [ ] `src/groups.jl`
- [ ] `src/multilevel.jl`
- [ ] `src/efa.jl`
- [ ] `src/sam.jl`

### Phase 5 (Polish)
- [ ] `src/standardize.jl`
- [ ] `src/simulate.jl`
- [ ] `src/predict.jl`
- [ ] `src/rsquare.jl`
- [ ] `ext/LavaanCUDA.jl`
- [ ] `vignettes/` (6 Quarto files)

---

## References

### Internal
- Brainstorm: `docs/brainstorms/2026-04-05-lavaan-jl-brainstorm.md`
- lavaan R source: `~/projects/repo_cloned/lavaan/R/`
- SynthDiD.jl conventions: `~/projects/software/SynthDiD.jl/` (test patterns, Project.toml structure)
- Endid.jl threading: `~/projects/software/Endid.jl/src/Endid.jl`

### External
- Rosseel (2012). lavaan: An R Package for Structural Equation Modeling. *JSS*, 48(2).
- Arkhangelsky et al. (2021). Synthetic DiD. *AER* (not directly relevant but same team).
- Bollen (1989). *Structural Equations with Latent Variables*. Wiley. (LISREL reference)
- Satorra & Bentler (1994). Corrections to test statistics and standard errors in covariance structure analysis.
- Yuan & Bentler (2000). Three likelihood-based methods for mean and covariance structure analysis.
- Vandenberg & Lance (2000). A review and synthesis of the measurement invariance literature.
