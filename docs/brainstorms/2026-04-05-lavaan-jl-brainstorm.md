---
date: 2026-04-05
topic: lavaan-jl
---

# Lavaan.jl — Full-Featured SEM in Julia

## What We're Building

A complete Julia port of the R `lavaan` package for Structural Equation Modeling. Targets R users migrating to Julia and Julia researchers needing SEM, CFA, EFA, multilevel SEM, and SAM. Uses identical string-based model syntax (`"visual =~ x1 + x2 + x3"`) for drop-in familiarity, but replaces lavaan's internals with Julia-native acceleration: `ForwardDiff.jl` autodiff for gradients, `Base.Threads` for bootstrap and multiple-group parallelism, and optional CUDA for large-model matrix operations.

## Why This Approach

### Syntax: string DSL (not macros)
Users can paste lavaan model strings unchanged. A macro frontend would be Julian but creates a migration barrier. String parsing is already solved in lavaan's `lav_syntax_parser.R` (1090 lines) — translating it is mechanical.

### Estimators: all, developed in parallel
ML first internally (it unblocks CFA/SEM immediately), but DWLS, FIML, GLS, PML, ULS scaffolded from day one so the API surface is stable.

### Performance: Julia internals, faithful API
ForwardDiff replaces all hand-coded analytic gradients. Multithreading handles bootstrap replications and multiple groups. GPU (CUDA.jl) is an optional opt-in for large covariance matrix ops. The user-facing API matches lavaan exactly.

## Key Decisions

- **String syntax parser**: port `lav_syntax_parser.R` verbatim → `syntax.jl`. Operators: `=~` (measurement), `~` (regression), `~~` (covariance), `~1` (intercept), `%~%` (unit variance).
- **Internal representation**: LISREL matrix system (Lambda, Psi, Theta, Beta, Phi, Alpha) — same as lavaan, enables direct numerical validation against R output.
- **Optimization**: `Optim.jl` (L-BFGS-B) + ForwardDiff gradients; fall back to finite differences for PML/WLS where needed.
- **vcov**: sandwich estimator for MLR; polychoric/polyserial correlations via `lav_bvord.R` logic for ordinal.
- **Parallelism**: bootstrap replications via `Threads.@threads`; multiple groups estimated in parallel; GPU opt-in via `CUDA.jl` extension.
- **Fit indices**: full set — CFI, TLI, RMSEA (+CI), SRMR, GFI, AGFI, AIC, BIC, SABIC.
- **Tests**: chi-square, Satorra-Bentler (SB), Yuan-Bentler (YB), LRT via `lavTestLRT`.

## Package Structure

```
Lavaan.jl/
├── src/
│   ├── Lavaan.jl           # module + re-exports
│   ├── syntax.jl           # model string parser → parTable
│   ├── partable.jl         # parameter table struct + operations
│   ├── model.jl            # LISREL/RAM matrix construction
│   ├── data.jl             # data prep, missing patterns
│   ├── samplestats.jl      # Σ_hat, μ_hat, polychoric/polyserial
│   ├── estimators/
│   │   ├── ml.jl           # ML / MLR / MLM
│   │   ├── fiml.jl         # full-information ML
│   │   ├── wls.jl          # WLS / DWLS / WLSMV
│   │   ├── gls.jl          # GLS
│   │   ├── pml.jl          # pairwise ML
│   │   └── uls.jl          # ULS
│   ├── optim.jl            # optimization engine (Optim.jl + ForwardDiff)
│   ├── vcov.jl             # SE estimation, sandwich, bootstrap
│   ├── fit.jl              # fit indices
│   ├── test.jl             # chi-square, SB, YB, LRT, Score, Wald
│   ├── bootstrap.jl        # multithreaded bootstrap
│   ├── standardize.jl      # std. solution (std.lv, std.all, std.nox)
│   ├── modindices.jl       # modification indices
│   ├── efa.jl              # exploratory factor analysis
│   ├── sam.jl              # structural after measurement
│   ├── multilevel.jl       # multilevel SEM (between/within)
│   ├── groups.jl           # multiple groups (parallel threads)
│   ├── simulate.jl         # simulate data from model
│   ├── predict.jl          # factor scores, predicted values
│   ├── print.jl            # show/summary methods
│   └── api.jl              # sem(), cfa(), growth(), efa(), sam()
├── test/
│   └── runtests.jl
├── vignettes/
│   ├── introduction.qmd
│   └── paper-results.qmd
├── data/
│   └── HolzingerSwineford1939.csv
├── Project.toml
└── README.md
```

## Implementation Phases

### Phase 1 — Core pipeline (unblocks all other work)
1. `syntax.jl` — parse model string to parTable
2. `partable.jl` — parTable struct
3. `model.jl` — parTable → LISREL matrices
4. `samplestats.jl` — compute Σ_obs, μ_obs from data
5. `estimators/ml.jl` — ML objective, ForwardDiff gradient
6. `optim.jl` — optimize, return fitted params
7. `vcov.jl` — standard ML SEs
8. `fit.jl` — chi-square, CFI, RMSEA, SRMR
9. `api.jl` — `cfa()`, `sem()`
10. `print.jl` — `summary()`, `parameterEstimates()`

### Phase 2 — Estimators
- FIML, DWLS/WLS/WLSMV, GLS, PML, ULS
- MLR/MLM robust SEs (sandwich)

### Phase 3 — Inference
- `bootstrap.jl` (multithreaded)
- `test.jl` — SB, YB, LRT, Score, Wald
- `modindices.jl`

### Phase 4 — Advanced models
- `groups.jl` — multiple groups
- `multilevel.jl` — two-level SEM
- `efa.jl`
- `sam.jl`

### Phase 5 — Polish
- `standardize.jl`
- `simulate.jl`
- `predict.jl`
- GPU extension (CUDA.jl)
- Full vignette suite

## Validation Strategy
Every estimator validated numerically against R lavaan output on:
- `HolzingerSwineford1939` (CFA benchmark)
- `PoliticalDemocracy` (SEM benchmark)
- Simulated ordinal data (DWLS benchmark)

Tolerance: parameter estimates within 1e-5, SEs within 1e-4.

## Open Questions
- EFA rotation methods: all 20+ lavaan rotations or a core subset first?
- Multilevel: two-level only, or arbitrary depth?
- SAM: local vs. global SAM both in Phase 4?

## Next Steps
→ `/workflows:plan` for Phase 1 implementation details
