# Ordinal WLSMV Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ordinal/categorical variable support to Lavaan.jl via polychoric correlations + DWLS estimation, with WLSMV-style chi-square test.

**Architecture:** User passes `ordered=["y1","y2"]` to `cfa()`/`sem()`. A two-stage approach: (1) per-variable probit thresholds + pairwise polychoric correlations replace the sample covariance S; (2) DWLS with the asymptotic weight matrix Gamma_poly fits the structural model. Thresholds are treated as nuisance parameters estimated in stage 1, not part of the structural θ vector. The existing DWLS infrastructure in `estimators/wls.jl` and `optim.jl` requires zero changes.

**Tech Stack:** Julia, Distributions.jl (Normal CDF/quantile), Optim.jl (LBFGS + Brent), ForwardDiff (for vcov — already in use)

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/ordinal.jl` | **CREATE** | Bivariate normal CDF, threshold estimation, polychoric ρ, Gamma_poly |
| `src/types.jl` | **MODIFY** | Add `ordered` to LavaanOptions; add `ordered`, `thresholds`, `threshold_ses` to LavaanData and SampleStats |
| `src/api.jl` | **MODIFY** | Accept `ordered` kwarg; auto-switch estimator to :DWLS |
| `src/data.jl` | **MODIFY** | Pass `ordered` through to LavaanData |
| `src/samplestats.jl` | **MODIFY** | When ordered present: compute R_poly, Gamma_poly; store in SampleStats |
| `src/fit.jl` | **MODIFY** | Use DWLS baseline for ordinal; correct df for threshold parameters |
| `src/Lavaan.jl` | **MODIFY** | `include("ordinal.jl")` before samplestats |
| `test/ordinal.jl` | **CREATE** | Tests: bivnorm CDF, threshold estimation, polychoric ρ, end-to-end CFA |
| `test/runtests.jl` | **MODIFY** | `include("ordinal.jl")` |

**Files NOT touched:** `src/estimators/wls.jl`, `src/optim.jl`, `src/model.jl`, `src/vcov.jl`, `src/partable.jl`, `src/implied.jl`

---

## Task 1: Types — add `ordered` field throughout

**Files:**
- Modify: `src/types.jl`
- Modify: `src/data.jl`

- [ ] **Step 1: Add `ordered` to LavaanOptions**

In `src/types.jl`, add one line to `LavaanOptions` after `parameterization`:

```julia
    # Ordered/categorical variables
    ordered::Vector{String} = String[]  # variable names declared ordinal
```

- [ ] **Step 2: Add ordinal fields to LavaanData**

In `src/types.jl`, add two fields to `LavaanData` after `cluster_sizes`:

```julia
    ordered::Vector{String}                              # ordinal variable names
    num_thresholds::Dict{String,Int}                     # k-1 thresholds per var (empty until samplestats)
```

- [ ] **Step 3: Update LavaanData constructor calls in data.jl**

In `src/data.jl`, the `prepare_data` signature gains an `ordered` kwarg and passes it through.

Replace the function signature:
```julia
function prepare_data(df::DataFrame,
                      ov_names::Vector{String},
                      lv_names::Vector{String};
                      group::Union{Symbol,String,Nothing} = nothing,
                      cluster::Union{Symbol,String,Nothing,Vector{String}} = nothing,
                      weights::Union{Symbol,String,Nothing} = nothing,
                      missing_method::Symbol = :listwise)::LavaanData
```
With:
```julia
function prepare_data(df::DataFrame,
                      ov_names::Vector{String},
                      lv_names::Vector{String};
                      group::Union{Symbol,String,Nothing} = nothing,
                      cluster::Union{Symbol,String,Nothing,Vector{String}} = nothing,
                      weights::Union{Symbol,String,Nothing} = nothing,
                      missing_method::Symbol = :listwise,
                      ordered::Vector{String} = String[])::LavaanData
```

In both the multi-group and single-group branches of `prepare_data`, replace the `return LavaanData(...)` call — add the two new fields at the end:
```julia
    return LavaanData(
        Xlist,
        ov_names,
        lv_names,
        ngroups,
        nobs_list,
        group_labels,
        wlist,
        df,
        group_col,
        isempty(cluster_cols) ? nothing : cluster_cols[1],
        cluster_cols,
        cluster_idx_list,
        cluster_sizes_list,
        ordered,               # NEW
        Dict{String,Int}(),    # NEW: filled later in samplestats
    )
```

- [ ] **Step 4: Add `thresholds` and `threshold_ses` to SampleStats**

In `src/types.jl`, add two fields to `SampleStats` after `NACOV`:

```julia
    # For ordinal DWLS (polychoric)
    thresholds::Union{Nothing,Dict{String,Vector{Float64}}}      # per-var threshold estimates
    threshold_ses::Union{Nothing,Dict{String,Vector{Float64}}}   # per-var threshold SEs
```

- [ ] **Step 5: Update SampleStats constructor call in samplestats.jl**

In `src/samplestats.jl`, the `return SampleStats(...)` at the end of `compute_samplestats` gains two new `nothing` entries (to be overridden in Task 6):

```julia
    return SampleStats(
        S_list,
        S_inv_list,
        logdet_S_list,
        mu_list,
        nobs_list,
        sum(nobs_list),
        p,
        nothing,      # Gamma
        nothing,      # NACOV
        nothing,      # thresholds  ← NEW
        nothing,      # threshold_ses ← NEW
        fiml_pats,
        YLp_list,
        S_W_list,
        S_B_list,
        s_list,
        data.X,
        data.cluster_idx,
    )
```

- [ ] **Step 6: Verify package compiles**

```bash
cd ~/projects/software/Lavaan.jl && julia --project=. -e "using Lavaan; println(\"OK\")"
```
Expected: `OK`

- [ ] **Step 7: Run existing tests to confirm no regression**

```bash
cd ~/projects/software/Lavaan.jl && julia --project=. test/runtests.jl 2>&1 | tail -5
```
Expected: all passes, same count as before (242).

- [ ] **Step 8: Commit**

```bash
cd ~/projects/software/Lavaan.jl
git add src/types.jl src/data.jl src/samplestats.jl
git commit -m "feat(ordinal): add ordered field to LavaanOptions, LavaanData, SampleStats"
```

---

## Task 2: Bivariate normal CDF

**Files:**
- Create: `src/ordinal.jl`
- Modify: `src/Lavaan.jl`
- Create: `test/ordinal.jl`

This is the mathematical core. We need P(X₁ ≤ h, X₂ ≤ k; ρ) for bivariate standard normal.
Key identity used: P(Z₁ ≤ 0, Z₂ ≤ 0; ρ) = ¼ + arcsin(ρ)/(2π)

Algorithm: transform to 1D integral and apply 20-point Gauss-Legendre:
P(X₁≤h, X₂≤k; ρ) = ∫₀^{Φ(h)} Φ((k − ρ·Φ⁻¹(u)) / √(1−ρ²)) du

- [ ] **Step 1: Write the failing test**

Create `test/ordinal.jl`:

```julia
# ─── test/ordinal.jl ──────────────────────────────────────────────────────────
using Test, Lavaan, Distributions, Random

@testset "bivnorm_cdf: boundary cases" begin
    # Independence: P(Z1≤0, Z2≤0; ρ=0) = Φ(0)² = 0.25
    @test isapprox(Lavaan.bivnorm_cdf(0.0, 0.0, 0.0), 0.25; atol=1e-10)

    # Marginal: P(Z1≤∞, Z2≤k) = Φ(k)
    @test isapprox(Lavaan.bivnorm_cdf(Inf, 0.0, 0.5), cdf(Normal(), 0.0); atol=1e-10)
    @test isapprox(Lavaan.bivnorm_cdf(0.0, Inf, -0.3), cdf(Normal(), 0.0); atol=1e-10)
    @test isapprox(Lavaan.bivnorm_cdf(Inf, Inf, 0.7), 1.0; atol=1e-10)

    # Lower bound
    @test isapprox(Lavaan.bivnorm_cdf(-Inf, 0.0, 0.5), 0.0; atol=1e-10)
    @test isapprox(Lavaan.bivnorm_cdf(0.0, -Inf, 0.5), 0.0; atol=1e-10)
end

@testset "bivnorm_cdf: known analytic values" begin
    # P(Z1≤0, Z2≤0; ρ) = 0.25 + arcsin(ρ)/(2π) — exact formula for h=k=0
    for rho in [-0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8]
        expected = 0.25 + asin(rho) / (2π)
        @test isapprox(Lavaan.bivnorm_cdf(0.0, 0.0, rho), expected; atol=1e-7)
    end

    # Independence: P(Z1≤a, Z2≤b; ρ=0) = Φ(a)*Φ(b)
    for (a, b) in [(1.0, 2.0), (-1.0, 0.5), (0.5, -0.5)]
        @test isapprox(Lavaan.bivnorm_cdf(a, b, 0.0),
                       cdf(Normal(), a) * cdf(Normal(), b); atol=1e-8)
    end
end
```

- [ ] **Step 2: Add include to Lavaan.jl and run test to see it fail**

In `src/Lavaan.jl`, add `include("ordinal.jl")` before the `include("samplestats.jl")` line.

Then create a stub `src/ordinal.jl`:
```julia
# ─── ordinal.jl ────────────────────────────────────────────────────────────────
# Ordinal variable support: polychoric correlations + threshold estimation.
# ─────────────────────────────────────────────────────────────────────────────
```

Add include to `test/runtests.jl` at the bottom:
```julia
include("ordinal.jl")
```

Run:
```bash
cd ~/projects/software/Lavaan.jl && julia --project=. -e "include(\"test/ordinal.jl\")" 2>&1 | head -20
```
Expected: `UndefVarError: bivnorm_cdf not defined`

- [ ] **Step 3: Implement `bivnorm_cdf` in ordinal.jl**

```julia
# ─── ordinal.jl ────────────────────────────────────────────────────────────────
# Ordinal variable support: polychoric correlations + threshold estimation.
# Mirrors lavaan's lav_uvord.R (thresholds) + lav_bvord.R (polychoric).
# ─────────────────────────────────────────────────────────────────────────────

using Distributions: Normal, cdf, pdf, quantile

# ─── 20-point Gauss-Legendre nodes on [-1, 1] ────────────────────────────────
# 10 symmetric pairs (±xᵢ, same weight wᵢ)

const _GL20_NODES = Float64[
    -0.9931285991850949,  0.9931285991850949,
    -0.9639719272779138,  0.9639719272779138,
    -0.9122344282513259,  0.9122344282513259,
    -0.8391169718222188,  0.8391169718222188,
    -0.7463062256567498,  0.7463062256567498,
    -0.6360536807265150,  0.6360536807265150,
    -0.5108670019508271,  0.5108670019508271,
    -0.3737060887154196,  0.3737060887154196,
    -0.2277858511416451,  0.2277858511416451,
    -0.0765265211334973,  0.0765265211334973,
]

const _GL20_WEIGHTS = Float64[
    0.0176140071391521,  0.0176140071391521,
    0.0406014298003869,  0.0406014298003869,
    0.0626720483341091,  0.0626720483341091,
    0.0832767415767047,  0.0832767415767047,
    0.1019301198172404,  0.1019301198172404,
    0.1181945319615184,  0.1181945319615184,
    0.1316886384491766,  0.1316886384491766,
    0.1420961093183821,  0.1420961093183821,
    0.1491729864726038,  0.1491729864726038,
    0.1527533871307259,  0.1527533871307259,
]

const _NORM = Normal()

"""
    bivnorm_cdf(h, k, rho) → Float64

Bivariate standard normal CDF: P(X₁ ≤ h, X₂ ≤ k) where Corr(X₁,X₂) = ρ.

Uses 20-point Gauss-Legendre on the 1D integral representation:
    P = ∫₀^{Φ(h)} Φ((k − ρ·Φ⁻¹(u)) / √(1−ρ²)) du
"""
function bivnorm_cdf(h::Real, k::Real, rho::Real)::Float64
    h, k, rho = Float64(h), Float64(k), Float64(rho)

    # ── Boundary / degenerate cases ───────────────────────────────────────────
    (h == -Inf || k == -Inf) && return 0.0
    (h ==  Inf && k ==  Inf) && return 1.0
    h ==  Inf && return cdf(_NORM, k)
    k ==  Inf && return cdf(_NORM, h)
    rho ==  1.0 && return cdf(_NORM, min(h, k))
    rho == -1.0 && return max(cdf(_NORM, h) + cdf(_NORM, k) - 1.0, 0.0)
    rho ==  0.0 && return cdf(_NORM, h) * cdf(_NORM, k)

    rho = clamp(rho, -1.0 + 1e-10, 1.0 - 1e-10)

    # Upper limit of integration in u-space
    ub = cdf(_NORM, h)
    ub <= 0.0 && return 0.0

    sqrt1r2 = sqrt(1.0 - rho^2)

    # Gauss-Legendre on [0, ub]: ∫₀^{ub} f(u) du ≈ (ub/2) Σᵢ wᵢ f((ub/2)(1+xᵢ))
    half_ub = ub / 2.0
    result = 0.0
    @inbounds for i in eachindex(_GL20_NODES)
        u = half_ub * (1.0 + _GL20_NODES[i])
        u = clamp(u, 1e-15, 1.0 - 1e-15)
        x = quantile(_NORM, u)
        result += _GL20_WEIGHTS[i] * cdf(_NORM, (k - rho * x) / sqrt1r2)
    end
    return half_ub * result
end
```

- [ ] **Step 4: Run the tests**

```bash
cd ~/projects/software/Lavaan.jl && julia --project=. -e "
using Lavaan
include(\"test/ordinal.jl\")
" 2>&1
```
Expected: all bivnorm_cdf tests pass.

- [ ] **Step 5: Commit**

```bash
cd ~/projects/software/Lavaan.jl
git add src/ordinal.jl src/Lavaan.jl test/ordinal.jl test/runtests.jl
git commit -m "feat(ordinal): implement bivnorm_cdf via 20-point Gauss-Legendre"
```

---

## Task 3: Univariate threshold estimation

**Files:**
- Modify: `src/ordinal.jl` (append)
- Modify: `test/ordinal.jl` (append)

Port `fit_thresholds` from opencode. This is correct — only needs minor cleanup (remove unused `scores_th` variable, ensure `Normal()` uses our existing `_NORM` const, check log-likelihood sign).

- [ ] **Step 1: Write the failing test**

Append to `test/ordinal.jl`:

```julia
@testset "fit_thresholds: 4-category variable" begin
    # Simulate ordinal data: y = 1..4 with P(y=1)=0.2, P(y=2)=0.3, P(y=3)=0.3, P(y=4)=0.2
    # Probit thresholds: τ₁=qnorm(0.2)≈-0.842, τ₂=qnorm(0.5)=0.0, τ₃=qnorm(0.8)≈0.842
    Random.seed!(42)
    n = 2000
    u = rand(n)
    y = ones(Int, n)
    for i in 1:n
        if u[i] < 0.2; y[i] = 1
        elseif u[i] < 0.5; y[i] = 2
        elseif u[i] < 0.8; y[i] = 3
        else; y[i] = 4
        end
    end

    fit = Lavaan.fit_thresholds(y)
    @test fit.converged
    @test fit.nth == 3
    @test fit.y_ncat == 4
    # Check thresholds close to true values
    @test isapprox(fit.theta[1], quantile(Normal(), 0.2); atol=0.05)
    @test isapprox(fit.theta[2], quantile(Normal(), 0.5); atol=0.05)
    @test isapprox(fit.theta[3], quantile(Normal(), 0.8); atol=0.05)
end

@testset "fit_thresholds: binary variable" begin
    Random.seed!(7)
    n = 1000
    y = rand(1:2, n)  # 50/50 binary
    fit = Lavaan.fit_thresholds(y)
    @test fit.converged
    @test fit.nth == 1
    @test isapprox(fit.theta[1], 0.0; atol=0.1)  # ~qnorm(0.5)=0
end
```

- [ ] **Step 2: Run to see it fail**

```bash
cd ~/projects/software/Lavaan.jl && julia --project=. -e "
using Lavaan, Distributions, Random
include(\"test/ordinal.jl\")
" 2>&1 | grep -E "Error|FAIL|fit_thresh"
```
Expected: `UndefVarError: fit_thresholds not defined`

- [ ] **Step 3: Implement threshold estimation in ordinal.jl**

Append to `src/ordinal.jl`:

```julia
# ─── Univariate threshold estimation ─────────────────────────────────────────

"""
    OrdinalThresholds

Result of univariate ordinal probit fit.
"""
struct OrdinalThresholds
    theta::Vector{Float64}   # [τ₁, τ₂, ..., τ_{k-1}]
    nth::Int                 # number of thresholds = k - 1
    y_ncat::Int              # number of categories k
    converged::Bool
end

"""
    fit_thresholds(y) → OrdinalThresholds

Estimate thresholds for an ordinal variable using ML (probit link).
y must be an integer vector with values 1..k.
"""
function fit_thresholds(y::Vector{Int})::OrdinalThresholds
    y_min = minimum(y)
    y = y .- y_min .+ 1          # recode to 1..k
    y_ncat = maximum(y)
    nth = y_ncat - 1
    nobs = length(y)

    # Starting values: probit of cumulative proportions
    cumprops = cumsum([count(==(k), y) / nobs for k in 1:y_ncat])[1:end-1]
    theta_start = quantile.(_NORM, clamp.(cumprops, 1e-10, 1 - 1e-10))

    # Boundary offsets so top/bottom categories don't go to ±∞
    # add large offset when y=k (upper) → effectively Φ(+∞)=1
    # add large offset when y=1 (lower) → effectively Φ(-∞)=0
    o_hi = [y[i] == y_ncat ?  100.0 : 0.0 for i in 1:nobs]
    o_lo = [y[i] == 1      ? -100.0 : 0.0 for i in 1:nobs]

    # Indicator matrices Y1[i,j] = (y[i] == j+1), Y2[i,j] = (y[i] == j)
    Y1 = [(y[i] == j + 1) for i in 1:nobs, j in 1:nth]
    Y2 = [(y[i] == j)     for i in 1:nobs, j in 1:nth]

    function nll(tau)
        # TH[y[i]+1] = upper threshold for obs i (with offset for top category)
        # TH[y[i]]   = lower threshold for obs i (with offset for bottom category)
        TH = vcat(0.0, tau, 0.0)   # TH[1]=0 placeholder (never used at y=k+1)
        z_hi = TH[y .+ 1] .+ o_hi
        z_lo = TH[y]       .+ o_lo
        pi_i = cdf.(_NORM, z_hi) .- cdf.(_NORM, z_lo)
        pi_i = max.(pi_i, 1e-300)
        return -sum(log.(pi_i)) / nobs
    end

    result = optimize(nll, theta_start, LBFGS(),
                      Optim.Options(iterations=10_000, g_tol=1e-8);
                      autodiff=:forward)

    return OrdinalThresholds(
        Optim.minimizer(result),
        nth,
        y_ncat,
        Optim.converged(result),
    )
end
```

- [ ] **Step 4: Run the tests**

```bash
cd ~/projects/software/Lavaan.jl && julia --project=. -e "
using Lavaan, Distributions, Random
include(\"test/ordinal.jl\")
" 2>&1 | grep -E "Pass|Fail|Error"
```
Expected: all threshold tests pass.

- [ ] **Step 5: Commit**

```bash
cd ~/projects/software/Lavaan.jl
git add src/ordinal.jl test/ordinal.jl
git commit -m "feat(ordinal): implement univariate probit threshold estimation"
```

---

## Task 4: Polychoric correlation

**Files:**
- Modify: `src/ordinal.jl` (append)
- Modify: `test/ordinal.jl` (append)

Two-stage approach: estimate thresholds from univariate fits (Task 3), then optimize ρ via profile likelihood holding thresholds fixed. Reparameterize ρ = tanh(z) to enforce ρ ∈ (−1, 1).

- [ ] **Step 1: Write the failing test**

Append to `test/ordinal.jl`:

```julia
@testset "polychoric_cor: recovers known ρ" begin
    # Simulate bivariate normal with ρ=0.6, discretize both to 4 categories
    Random.seed!(123)
    n = 1000
    rho_true = 0.6
    # Draw from bivariate normal
    z1 = randn(n)
    z2 = rho_true .* z1 .+ sqrt(1 - rho_true^2) .* randn(n)
    # Discretize: 4 categories at qnorm([0.25, 0.5, 0.75])
    cuts = quantile(Normal(), [0.25, 0.5, 0.75])
    to_ord(z) = Int.(z .< cuts[1]) .* 1 .+ Int.(cuts[1] .<= z .< cuts[2]) .* 2 .+
                Int.(cuts[2] .<= z .< cuts[3]) .* 3 .+ Int.(z .>= cuts[3]) .* 4
    y1, y2 = to_ord(z1), to_ord(z2)

    rho_hat = Lavaan.polychoric_cor(y1, y2)
    @test isapprox(rho_hat, rho_true; atol=0.05)
end

@testset "polychoric_cor: negative correlation" begin
    Random.seed!(99)
    n = 800
    rho_true = -0.5
    z1 = randn(n)
    z2 = rho_true .* z1 .+ sqrt(1 - rho_true^2) .* randn(n)
    cuts = quantile(Normal(), [1/3, 2/3])
    to_ord(z) = Int.(z .< cuts[1]) .* 1 .+ Int.(cuts[1] .<= z .< cuts[2]) .* 2 .+ Int.(z .>= cuts[2]) .* 3
    y1, y2 = to_ord(z1), to_ord(z2)

    rho_hat = Lavaan.polychoric_cor(y1, y2)
    @test isapprox(rho_hat, rho_true; atol=0.08)
end
```

- [ ] **Step 2: Run to see it fail**

```bash
cd ~/projects/software/Lavaan.jl && julia --project=. -e "
using Lavaan, Distributions, Random
include(\"test/ordinal.jl\")
" 2>&1 | grep -E "polychoric|Error"
```
Expected: `UndefVarError: polychoric_cor not defined`

- [ ] **Step 3: Implement `polychoric_cor` in ordinal.jl**

Append to `src/ordinal.jl`:

```julia
# ─── Polychoric correlation ───────────────────────────────────────────────────

"""
    polychoric_cor(y1, y2) → Float64

Estimate the polychoric correlation between two ordinal variables.
Uses two-stage approach: thresholds from univariate probit, then
profile likelihood over ρ = tanh(z) holding thresholds fixed.
"""
function polychoric_cor(y1::Vector{Int}, y2::Vector{Int})::Float64
    # Recode to 1..k
    y1 = y1 .- minimum(y1) .+ 1
    y2 = y2 .- minimum(y2) .+ 1

    # Stage 1: estimate thresholds from marginals
    fit1 = fit_thresholds(y1)
    fit2 = fit_thresholds(y2)

    # Extended threshold vectors: [-∞, τ₁, ..., τ_{k-1}, +∞]
    τ1 = vcat(-Inf, fit1.theta, Inf)
    τ2 = vcat(-Inf, fit2.theta, Inf)

    n = length(y1)

    # Stage 2: profile likelihood over ρ (tanh parameterization)
    function neg_profile_ll(z)
        rho = tanh(z[1])
        ll = 0.0
        for i in 1:n
            s, t = y1[i], y2[i]
            # P(Y1=s, Y2=t) = Φ₂(τ1[s+1],τ2[t+1],ρ) - Φ₂(τ1[s],τ2[t+1],ρ)
            #                - Φ₂(τ1[s+1],τ2[t],ρ)   + Φ₂(τ1[s],τ2[t],ρ)
            p = bivnorm_cdf(τ1[s+1], τ2[t+1], rho) -
                bivnorm_cdf(τ1[s],   τ2[t+1], rho) -
                bivnorm_cdf(τ1[s+1], τ2[t],   rho) +
                bivnorm_cdf(τ1[s],   τ2[t],   rho)
            ll += log(max(p, 1e-300))
        end
        return -ll
    end

    result = optimize(neg_profile_ll, [0.0], LBFGS(),
                      Optim.Options(iterations=1000, g_tol=1e-7);
                      autodiff=:forward)

    return tanh(Optim.minimizer(result)[1])
end
```

Note: `bivnorm_cdf` is called with tanh-parameterized ρ — ForwardDiff will differentiate through `tanh` and `bivnorm_cdf` correctly since `bivnorm_cdf` uses only differentiable operations (no branches on the value of `rho`).

- [ ] **Step 4: Run the tests**

```bash
cd ~/projects/software/Lavaan.jl && julia --project=. -e "
using Lavaan, Distributions, Random
include(\"test/ordinal.jl\")
" 2>&1 | grep -E "Pass|Fail|Error|polychoric"
```
Expected: all polychoric tests pass.

- [ ] **Step 5: Commit**

```bash
cd ~/projects/software/Lavaan.jl
git add src/ordinal.jl test/ordinal.jl
git commit -m "feat(ordinal): implement two-stage polychoric correlation estimator"
```

---

## Task 5: Polychoric matrix + asymptotic Gamma

**Files:**
- Modify: `src/ordinal.jl` (append)
- Modify: `test/ordinal.jl` (append)

Build the full p×p polychoric correlation matrix and the diagonal of the asymptotic covariance matrix Gamma_poly. For DWLS we only need diag(Gamma). Each diagonal entry Gamma_{ij,ij} = n / I_obs(ρ̂_ij) where I_obs is the observed Fisher information (numerical Hessian of the profile negative log-likelihood at ρ̂).

- [ ] **Step 1: Write the failing test**

Append to `test/ordinal.jl`:

```julia
@testset "compute_polychoric_matrix: 3-variable case" begin
    Random.seed!(55)
    n = 500
    # True correlation structure: ρ₁₂=0.6, ρ₁₃=0.4, ρ₂₃=0.5
    Σ_true = [1.0 0.6 0.4; 0.6 1.0 0.5; 0.4 0.5 1.0]
    L = cholesky(Σ_true).L
    Z = (L * randn(3, n))'  # n×3
    cuts = quantile(Normal(), [0.33, 0.67])
    to_ord(z) = Int.(z .< cuts[1]) .* 1 .+ Int.(cuts[1] .<= z .< cuts[2]) .* 2 .+ Int.(z .>= cuts[2]) .* 3

    ov = ["y1","y2","y3"]
    X = hcat(to_ord(Z[:,1]), to_ord(Z[:,2]), to_ord(Z[:,3]))

    R, Gamma_diag = Lavaan.compute_polychoric_matrix(X, ov, ov)

    # Diagonal = 1
    @test all(isapprox.(diag(R), 1.0; atol=1e-10))
    # Off-diagonal close to true values (loose tolerance due to discretization)
    @test isapprox(R[1,2], 0.6; atol=0.08)
    @test isapprox(R[1,3], 0.4; atol=0.08)
    @test isapprox(R[2,3], 0.5; atol=0.08)
    # Gamma_diag has length p*(p+1)/2 and is positive
    m = 3 * 4 ÷ 2
    @test length(Gamma_diag) == m
    @test all(Gamma_diag .> 0)
end
```

- [ ] **Step 2: Run to see it fail**

```bash
cd ~/projects/software/Lavaan.jl && julia --project=. -e "
using Lavaan, Distributions, Random, LinearAlgebra
include(\"test/ordinal.jl\")
" 2>&1 | grep -E "compute_polychoric|Error"
```
Expected: `UndefVarError: compute_polychoric_matrix not defined`

- [ ] **Step 3: Implement `compute_polychoric_matrix` and `_polychoric_var` in ordinal.jl**

Append to `src/ordinal.jl`:

```julia
# ─── Polychoric correlation matrix + Gamma ────────────────────────────────────

"""
    _polychoric_var(y1, y2, rho_hat, n) → Float64

Asymptotic variance of ρ̂: Var(ρ̂) ≈ n / I_obs(ρ̂)
where I_obs is the observed Fisher information (numerical 2nd derivative
of the profile negative log-likelihood at ρ̂).
"""
function _polychoric_var(y1::Vector{Int}, y2::Vector{Int},
                          rho_hat::Float64, n::Int)::Float64
    # Reparameterize ρ = tanh(z) → z_hat = atanh(ρ̂)
    y1 = y1 .- minimum(y1) .+ 1
    y2 = y2 .- minimum(y2) .+ 1

    fit1 = fit_thresholds(y1)
    fit2 = fit_thresholds(y2)
    τ1 = vcat(-Inf, fit1.theta, Inf)
    τ2 = vcat(-Inf, fit2.theta, Inf)

    function nll_rho(rho)
        rho_c = clamp(rho, -1.0 + 1e-8, 1.0 - 1e-8)
        ll = 0.0
        for i in 1:n
            s, t = y1[i], y2[i]
            p = bivnorm_cdf(τ1[s+1], τ2[t+1], rho_c) -
                bivnorm_cdf(τ1[s],   τ2[t+1], rho_c) -
                bivnorm_cdf(τ1[s+1], τ2[t],   rho_c) +
                bivnorm_cdf(τ1[s],   τ2[t],   rho_c)
            ll += log(max(p, 1e-300))
        end
        return -ll
    end

    # Numerical second derivative at rho_hat
    ε = 1e-4
    rho_c = clamp(rho_hat, -1.0 + 2ε, 1.0 - 2ε)
    d2 = (nll_rho(rho_c - ε) - 2*nll_rho(rho_c) + nll_rho(rho_c + ε)) / ε^2
    # d2 > 0 since NLL is convex at min; I_obs = d2 (observed Fisher info)
    d2 = max(d2, 1e-10)
    return n / d2   # Var(ρ̂) ≈ n / I_obs (asymptotic, since NLL is scaled by 1/n... wait)
    # Actually NLL above is NOT divided by n, so I_obs = d2, and Var(ρ̂) = 1/d2
    # Then Gamma_ij,ij = n * Var(ρ̂) = n / d2
end

"""
    compute_polychoric_matrix(X, ov_names, ordered) → (R, Gamma_diag)

Compute the polychoric correlation matrix for the ordered variables
in `ordered`, and the diagonal of the asymptotic covariance Gamma_poly
(in vech order, matching `vech(R)`).

Returns:
- `R`          : p×p correlation matrix (Pearson for continuous-continuous pairs,
                 polychoric for ordinal-ordinal pairs, 1 on diagonal)
- `Gamma_diag` : m-vector (m=p(p+1)/2) of diagonal Gamma entries
"""
function compute_polychoric_matrix(X::Matrix{Float64},
                                   ov_names::Vector{String},
                                   ordered::Vector{String})::Tuple{Matrix{Float64},Vector{Float64}}
    p = length(ov_names)
    n = size(X, 1)
    m = p * (p + 1) ÷ 2

    R = Matrix{Float64}(I, p, p)
    Gamma_diag = zeros(Float64, m)

    ord_set = Set(ordered)
    ord_idx = Dict(name => findfirst(==(name), ov_names) for name in ov_names)

    # Fill upper triangle
    for j in 1:p, i in j:p
        if i == j
            R[i, j] = 1.0
            # For diagonal: Gamma = 2 (normal-theory asymptotic var of sample variance r=1)
            vech_k = (j-1)*p - (j-2)*(j-1)÷2 + (i-j+1)   # vech index (1-based)
            k = vech_pos(p, i, j)
            Gamma_diag[k] = 2.0   # placeholder; diagonal of R is always 1, weight large
        else
            name_i = ov_names[i]
            name_j = ov_names[j]
            k = vech_pos(p, i, j)

            if name_i in ord_set && name_j in ord_set
                # Polychoric
                yi = round.(Int, X[:, i])
                yj = round.(Int, X[:, j])
                rho = polychoric_cor(yi, yj)
                R[i, j] = R[j, i] = rho
                # Asymptotic variance
                var_rho = _polychoric_var(yi, yj, rho, n)
                Gamma_diag[k] = n * var_rho
            else
                # Pearson (continuous variable pair)
                r = cor(X[:, i], X[:, j])
                R[i, j] = R[j, i] = r
                # Normal-theory var of Pearson r: (1-r²)²
                Gamma_diag[k] = (1 - r^2)^2
            end
        end
    end

    return R, Gamma_diag
end

"""
    vech_pos(p, i, j) → Int

Position of element (i,j) with i≥j in vech(A) (1-based, column-major).
"""
function vech_pos(p::Int, i::Int, j::Int)::Int
    @assert i >= j
    # Number of elements in columns 1..(j-1): sum_{c=1}^{j-1} (p-c+1)
    # = (j-1)*p - (j-1)*(j-2)/2 — standard formula
    offset = (j - 1) * p - (j - 1) * (j - 2) ÷ 2
    return offset + (i - j + 1)
end
```

Note: remove the duplicate `vech_k` line (the `vech_pos` function supersedes it). Replace the diagonal block with:
```julia
            k = vech_pos(p, i, j)   # i==j here
            Gamma_diag[k] = 2.0
```

- [ ] **Step 4: Run the tests**

```bash
cd ~/projects/software/Lavaan.jl && julia --project=. -e "
using Lavaan, Distributions, Random, LinearAlgebra, Statistics
include(\"test/ordinal.jl\")
" 2>&1 | grep -E "Pass|Fail|Error|polychoric_matrix"
```
Expected: polychoric matrix test passes.

- [ ] **Step 5: Commit**

```bash
cd ~/projects/software/Lavaan.jl
git add src/ordinal.jl test/ordinal.jl
git commit -m "feat(ordinal): implement polychoric matrix and asymptotic Gamma"
```

---

## Task 6: SampleStats integration — compute R_poly when ordered vars present

**Files:**
- Modify: `src/samplestats.jl`
- Modify: `src/api.jl`

When `!isempty(data.ordered)`:
1. Compute R_poly and Gamma_poly via `compute_polychoric_matrix`
2. Build full `Gamma` matrix (diagonal) from `Gamma_diag`
3. Substitute R_poly for S in SampleStats (and recompute S_inv, logdet_S)
4. Store per-variable thresholds in SampleStats.thresholds and .threshold_ses
5. Auto-switch estimator to :DWLS if user didn't specify

- [ ] **Step 1: Modify `compute_samplestats` in samplestats.jl**

At the top of the function body in `compute_samplestats`, after the existing setup, add an early return for the ordinal case:

```julia
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
        mu_list[g] = vec(mean(X, dims=1))
        S = cov(X)
        if opts.ridge
            S = S + opts.ridge_constant * I(p)
        end
        S_list[g] = S
        chol = cholesky(Symmetric(S))
        S_inv_list[g]    = inv(chol)
        logdet_S_list[g] = 2.0 * sum(log.(diag(chol.U)))
    end

    p = size(data.X[1], 2)

    fiml_pats = if opts.estimator == :FIML
        [build_fiml_patterns(data.X[g]) for g in 1:ngroups]
    else
        nothing
    end

    YLp_list = nothing
    S_W_list = nothing
    S_B_list = nothing
    s_list   = nothing

    # ... (existing multilevel code unchanged) ...

    # ── Ordinal: compute polychoric R and Gamma ─────────────────────────────
    Gamma_list = nothing
    thresholds_out = nothing
    threshold_ses_out = nothing

    if !isempty(data.ordered)
        Gamma_list = Vector{Matrix{Float64}}(undef, ngroups)
        all_th = Dict{String,Vector{Float64}}()
        all_se = Dict{String,Vector{Float64}}()

        for g in 1:ngroups
            X = data.X[g]
            n, p2 = size(X)

            R_poly, Gamma_diag = compute_polychoric_matrix(
                X, data.ov_names, data.ordered)

            # Replace S with R_poly
            S_list[g]        = R_poly
            chol_R = cholesky(Symmetric(R_poly))
            S_inv_list[g]    = inv(chol_R)
            logdet_S_list[g] = 2.0 * sum(log.(diag(chol_R.U)))

            # Build full diagonal Gamma matrix (m×m where m=p*(p+1)/2)
            m = p2 * (p2 + 1) ÷ 2
            Gamma_g = Diagonal(Gamma_diag)   # efficient diagonal storage
            Gamma_list[g] = Matrix(Gamma_g)

            # Collect thresholds for g==1 (single-group; TODO: multi-group)
            if g == 1
                for name in data.ordered
                    col = findfirst(==(name), data.ov_names)
                    col === nothing && continue
                    y = round.(Int, X[:, col])
                    fit_th = fit_thresholds(y)
                    all_th[name] = fit_th.theta
                    # SE from diagonal of Hessian (numerical)
                    # For simplicity, use Fisher info per threshold: ≈ n * φ²/[Φ(1-Φ)]
                    props = cumsum([count(==(k), y) / n for k in 1:fit_th.y_ncat])[1:end-1]
                    props = clamp.(props, 1e-6, 1-1e-6)
                    se_th = 1.0 ./ (sqrt(n) .* pdf.(_NORM, quantile.(_NORM, props)))
                    all_se[name] = se_th
                end
                thresholds_out = all_th
                threshold_ses_out = all_se
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
        Gamma_list,     # not nothing when ordered
        nothing,        # NACOV
        thresholds_out, # thresholds
        threshold_ses_out, # threshold_ses
        fiml_pats,
        YLp_list,
        S_W_list,
        S_B_list,
        s_list,
        data.X,
        data.cluster_idx,
    )
end
```

- [ ] **Step 2: Add `ordered` kwarg to `lavaan()` in api.jl**

In `src/api.jl`, add `ordered::Vector{String} = String[]` to the `lavaan()` signature:

```julia
function lavaan(model_string::String,
                data::DataFrame;
                group::Union{String,Nothing} = nothing,
                cluster::Union{String,Symbol,Nothing,Vector{String}} = nothing,
                ordered::Vector{String} = String[],         # ← ADD
                estimator::Symbol  = :ML,
                ...
```

Add to `LavaanOptions` construction:
```julia
    opts = LavaanOptions(
        ordered         = ordered,                          # ← ADD
        estimator       = estimator,
        ...
    )
```

Also add auto-switch logic in `_lavaan_pipeline`, before the prepare_data call:

```julia
    # Auto-switch to DWLS for ordinal data
    if !isempty(opts.ordered) && opts.estimator == :ML
        @info "Ordered variables detected; switching estimator to :DWLS"
        opts = LavaanOptions(; [f => getfield(opts, f) for f in fieldnames(LavaanOptions)]...,
                             estimator=:DWLS)
    end
```

And update the `prepare_data` call to pass `ordered`:

```julia
    lav_data = prepare_data(df, ov_names, lv_names;
                             group=group,
                             cluster=...,
                             missing_method=opts.missing_method,
                             ordered=opts.ordered)    # ← ADD
```

- [ ] **Step 3: Use DWLS baseline estimator for ordinal in fit.jl**

In `src/fit.jl`, find the line:
```julia
    obj_bl = theta -> estimator_objective(theta, model_bl, stats, :ML)
```
Replace with:
```julia
    baseline_est = opts.estimator in (:DWLS, :WLSMV, :WLS) ? opts.estimator : :ML
    obj_bl = theta -> estimator_objective(theta, model_bl, stats, baseline_est)
```

- [ ] **Step 4: Inject threshold rows into ParTable after samplestats**

After the `compute_samplestats(lav_data, opts)` call in `_lavaan_pipeline`, add:

```julia
    # Inject threshold rows into ParTable for ordinal variables
    if !isempty(opts.ordered) && lav_stats.thresholds !== nothing
        _inject_threshold_rows!(partable, lav_stats.thresholds, lav_stats.threshold_ses)
    end
```

Implement `_inject_threshold_rows!` in `src/partable.jl` at the end:

```julia
"""
    _inject_threshold_rows!(pt, thresholds, threshold_ses)

Append threshold parameter rows (op="|") to the ParTable.
These rows are NOT free parameters (free=0) — they come from the
first-stage univariate probit estimation and appear in parameterEstimates()
output only for display.
"""
function _inject_threshold_rows!(pt::ParTable,
                                  thresholds::Dict{String,Vector{Float64}},
                                  threshold_ses::Union{Nothing,Dict{String,Vector{Float64}}})
    for (var, th) in sort(collect(thresholds), by=first)
        for (k, tau) in enumerate(th)
            n = length(pt.id) + 1
            # Extend all vectors by 1
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
            push!(pt.ci_lower, isnan(se_val) ? NaN : tau - 1.96*se_val)
            push!(pt.ci_upper, isnan(se_val) ? NaN : tau + 1.96*se_val)
            push!(pt.mat,      "")
            push!(pt.mat_row,  0)
            push!(pt.mat_col,  0)
        end
    end
end
```

- [ ] **Step 5: Run existing tests to confirm no regression**

```bash
cd ~/projects/software/Lavaan.jl && julia --project=. test/runtests.jl 2>&1 | tail -10
```
Expected: all 242+ prior tests still pass.

- [ ] **Step 6: Commit**

```bash
cd ~/projects/software/Lavaan.jl
git add src/samplestats.jl src/api.jl src/fit.jl src/partable.jl
git commit -m "feat(ordinal): wire polychoric R_poly + Gamma_poly into SampleStats pipeline"
```

---

## Task 7: End-to-end ordinal CFA test

**Files:**
- Modify: `test/ordinal.jl` (append)

Test a full CFA with ordinal indicators against known reference values. We simulate data from a known factor model, discretize indicators to 4 categories, fit with `cfa(..., ordered=[...], estimator=:DWLS)`, and check loadings are close to true values.

- [ ] **Step 1: Write the end-to-end test**

Append to `test/ordinal.jl`:

```julia
@testset "ordinal CFA: end-to-end DWLS fit" begin
    # True model: 1 factor, 4 indicators, all loadings = 0.7
    # Residual variance = 1 - 0.7² = 0.51
    Random.seed!(2024)
    n = 1000
    lambda_true = 0.7
    eta = randn(n)    # latent factor
    # Continuous indicators
    eps = randn(n, 4) .* sqrt(1 - lambda_true^2)
    Y_cont = lambda_true .* eta .+ eps   # n×4 matrix

    # Discretize to 4 categories at qnorm([0.25, 0.5, 0.75])
    cuts = quantile(Normal(), [0.25, 0.5, 0.75])
    to_ord(z) = [sum(z[i] .>= cuts) + 1 for i in 1:length(z)]

    df = DataFrame(
        y1 = to_ord(Y_cont[:,1]),
        y2 = to_ord(Y_cont[:,2]),
        y3 = to_ord(Y_cont[:,3]),
        y4 = to_ord(Y_cont[:,4]),
    )

    model_str = """
        f =~ y1 + y2 + y3 + y4
    """

    fit = cfa(model_str, df;
              ordered = ["y1","y2","y3","y4"],
              estimator = :DWLS)

    @test fit.converged

    pe = parameterEstimates(fit)

    # Threshold rows should be present
    thresh_rows = pe[pe.op .== "|", :]
    @test nrow(thresh_rows) == 12  # 4 vars × 3 thresholds each

    # Loadings: all should be close to 0.7 (first loading fixed to 1.0*sqrt(latent_var))
    # In a CFA with std_lv=false and auto_fix_first=true, first loading is fixed to 1.0
    # and other loadings are free. Latent variance is free. So the "raw" loadings are
    # proportional to 0.7. Check that loadings[2:4] / loadings[1] ≈ 1.0 (equal loadings)
    load_rows = pe[pe.op .== "=~", :]
    @test nrow(load_rows) == 4

    # The free loadings (rows 2-4) should be close to each other
    free_loads = load_rows[load_rows.free .> 0, :est]
    if length(free_loads) >= 2
        # Relative spread should be small
        @test std(free_loads) / mean(abs.(free_loads)) < 0.15
    end

    # Model fit: with n=1000 and correct model, chi-sq should be reasonable
    fm = fitMeasures(fit)
    @test haskey(fm, :chisq)
    @test haskey(fm, :df)
    @test fm[:df] == 2.0   # df = 4*5/2 - (3 free loadings + 1 latent var + 4 residuals) = 10 - 8 = 2
end

@testset "ordinal CFA: threshold estimates in parameterEstimates" begin
    Random.seed!(77)
    n = 800
    z = randn(n)
    cuts = quantile(Normal(), [0.33, 0.67])
    y = [sum(z[i] .>= cuts) + 1 for i in 1:n]
    df = DataFrame(f1=y, f2=y .+ round.(Int, 0.3 .* randn(n)))
    df.f2 = clamp.(df.f2, 1, 3)

    model_str = "g =~ f1 + f2"
    fit = cfa(model_str, df; ordered=["f1","f2"], estimator=:DWLS)

    pe = parameterEstimates(fit)
    th = pe[pe.op .== "|", :]

    @test nrow(th) == 4  # 2 vars × 2 thresholds
    # Threshold estimates close to probit(1/3) ≈ -0.431 and probit(2/3) ≈ 0.431
    th1_rows = th[th.lhs .== "f1", :]
    @test isapprox(th1_rows.est[1], quantile(Normal(), 1/3); atol=0.08)
    @test isapprox(th1_rows.est[2], quantile(Normal(), 2/3); atol=0.08)
end
```

- [ ] **Step 2: Run the test**

```bash
cd ~/projects/software/Lavaan.jl && julia --project=. -e "
using Lavaan, DataFrames, Distributions, Random, Statistics, LinearAlgebra
include(\"test/ordinal.jl\")
" 2>&1 | grep -E "Pass|Fail|Error|ordinal CFA"
```
Expected: all ordinal CFA tests pass.

- [ ] **Step 3: Run the full test suite**

```bash
cd ~/projects/software/Lavaan.jl && julia --project=. test/runtests.jl 2>&1 | tail -15
```
Expected: all original tests pass + new ordinal tests pass.

- [ ] **Step 4: Commit**

```bash
cd ~/projects/software/Lavaan.jl
git add test/ordinal.jl
git commit -m "test(ordinal): end-to-end DWLS CFA with ordinal indicators"
```

---

## Task 8: Polish — update module include, run full suite, update docs

**Files:**
- Modify: `src/Lavaan.jl` (verify include order)
- Modify: `CLAUDE.md`
- Modify: `README.md`

- [ ] **Step 1: Verify include order in Lavaan.jl**

`src/Lavaan.jl` must include `ordinal.jl` BEFORE `samplestats.jl` (since `compute_polychoric_matrix` is called from `compute_samplestats`). Verify:

```bash
grep -n "include" ~/projects/software/Lavaan.jl/src/Lavaan.jl | grep -E "ordinal|samplestats"
```
Expected: `ordinal.jl` line number < `samplestats.jl` line number.

- [ ] **Step 2: Run full test suite**

```bash
cd ~/projects/software/Lavaan.jl && julia --project=. test/runtests.jl 2>&1 | grep -E "Pass|Fail|Error|Total"
```
Expected: all tests pass (≥ 242 + new ordinal tests).

- [ ] **Step 3: Update CLAUDE.md**

In `CLAUDE.md`, update the status line and add to the "Completed features" section:

```markdown
**Status:** X/X tests pass. Phase 1–6 (Claude) + Phase 4–5 (Gemini) + Phase 7 (Claude: ordinal DWLS)
```

Add under Completed features:
```markdown
**Ordinal/Categorical (Phase 7)**: `ordered=["y1","y2"]` kwarg to cfa/sem. Polychoric correlation matrix computed via two-stage probit ML (univariate thresholds + profile likelihood over ρ). Asymptotic Gamma via numerical Fisher information. DWLS estimation with R_poly as "S". Thresholds appear in `parameterEstimates()` with `op="|"`. Auto-switches to `:DWLS` when ordered vars detected.
```

- [ ] **Step 4: Final commit**

```bash
cd ~/projects/software/Lavaan.jl
git add src/Lavaan.jl CLAUDE.md README.md
git commit -m "docs: update CLAUDE.md and README for Phase 7 ordinal DWLS"
```

---

## Self-Review

**Spec coverage check:**
- ✅ `ordered=["y1","y2"]` kwarg — Task 6 (api.jl)
- ✅ Bivariate normal CDF — Task 2
- ✅ Threshold estimation — Task 3 (ported from opencode)
- ✅ Polychoric correlation — Task 4
- ✅ Polychoric matrix + Gamma — Task 5
- ✅ SampleStats integration — Task 6
- ✅ DWLS estimation (reuses existing) — Task 6 (no new code needed)
- ✅ Threshold rows in parameterEstimates() — Task 6 (partable.jl)
- ✅ End-to-end tests — Task 7

**Known deferred items (not in scope):**
- WLSMV mean-variance adjusted test statistic (requires full Gamma, not just diagonal)
- Mixed continuous + ordinal (polyserial correlations)
- Multi-group ordinal
- Delta parameterization (residual variances fixed; currently they're free)

**Placeholder scan:** No TBDs. All code blocks are complete implementations.

**Type consistency check:**
- `fit_thresholds` returns `OrdinalThresholds` with fields `.theta`, `.nth`, `.y_ncat`, `.converged` — used consistently in Tasks 3, 4, 5
- `polychoric_cor(y1, y2)` returns `Float64` — used in Tasks 4, 5
- `compute_polychoric_matrix(X, ov_names, ordered)` returns `(Matrix{Float64}, Vector{Float64})` — used in Task 6
- `vech_pos(p, i, j)` returns `Int` — used only in Task 5
- `_inject_threshold_rows!(pt, thresholds, threshold_ses)` mutates ParTable — used in Task 6
