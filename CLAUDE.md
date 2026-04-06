# Lavaan.jl — Project Notes

## Overview

Julia port of R's lavaan SEM package. Phase 7 complete: ordinal/categorical WLSMV added.

## Architecture

**17-step pipeline** (mirrors lavaan's R internals):
1. `parse_model_string()` — syntax.jl (supports `level: 1` and `level: 2`)
2. `build_partable()` — partable.jl (handles dual-block and n-level structures)
3. `prepare_data()` + `compute_samplestats()` — data.jl, samplestats.jl (handles clustered and crossed data)
4. `validate_model_data()` — validate.jl
5. `build_model()` — model.jl (LISREL matrices)
6. `set_starting_values!()` — start.jl
7. `optimize_model()` — optim.jl (MUML for multilevel, Sparse FIML for crossed)
8. `compute_implied()` — implied.jl
9. `compute_vcov()` — vcov.jl
10. `compute_fit_measures!()` — fit.jl

**ForwardDiff compatibility**: All hot-path functions (fill_block, compute_implied_typed, ml_objective) use `T <: Real` generics so `Dual` numbers from ForwardDiff flow through. NEVER use `real()` on outputs from these functions — it silently zeros all derivatives.
- ForwardDiff closure trap: inner function using same variable name as outer scope causes Julia to box the captured var — use distinct names inside nested closures (fixed in `_polychoric_var`)
- `bivnorm_cdf` casts ρ to Float64 (for performance); use `_bivnorm_cdf_rho` when ForwardDiff must differentiate through ρ

**Key invariants**:
- NEVER throw on non-convergence (Gap G2) — return NaN-filled results
- LavaanFit is defined FIRST in types.jl (Gap G1)
- Baseline model is fitted inside fit.jl and cached in LavaanFit.baseline (Gap G6)
- `_vech_indices` defined in `estimators/robust.jl` — MUST be included before `vcov.jl`

## Running tests

```julia
cd Lavaan.jl && julia --project=. test/runtests.jl
```

Or from the Julia REPL:
```julia
using Pkg; Pkg.test()
```

## Reference values

R lavaan 0.6-21 on Holzinger-Swineford 3-factor CFA:
- chisq=85.306, df=24, CFI=0.931, TLI=0.896, RMSEA=0.092, SRMR=0.065
- RMSEA 90% CI: [0.071, 0.114]
- x2 loading=0.554, x5 loading=1.113, visual variance=0.812

R lavaan 0.6-21 on Political Democracy SEM:
- chisq=72.462, df=41, CFI=0.953, RMSEA=0.101
- dem60~ind60=1.474, dem65~dem60=0.864

## Completed features (Phase 1–5)

**Estimators**: ML, MLR (sandwich SEs), GLS, ULS, WLS, DWLS, FIML, MUML (multilevel), Sparse FIML (crossed)
**SE types**: standard, robust/MLR (Huber-White sandwich), bootstrap (nonparametric, parallel)
**Scaled tests**: Satorra-Bentler MLM (estimator=:MLM, scaling_factor in fitMeasures)
**Diagnostics**: lavTestLRT (nested model comparison), modindices (score test MI + EPC)
**Results**: parameterEstimates, fitMeasures, standardizedSolution, coef, vcov, residuals, lavInspect
**Multilevel**: Supports two-level SEM with `cluster` argument and `level:` modifiers.
**Crossed Effects**: Supports arbitrary non-nested clustering factors via sparse matrix estimation.
**R-squared**: `rsquare(fit)` — R² for observed indicators and endogenous latents.
**simulateData**: `simulateData(fit)` or `simulateData(model_string; n)` — draw from model-implied MVN.
**lavPredict**: `lavPredict(fit)` — regression factor scores (W = Var_η Λᵀ Σ⁻¹).
**EFA**: `efa(data, k; rotation)` — ML extraction + rotation (:geomin, :varimax, :quartimax, :oblimin, :none).
**Ordinal/Categorical (Phase 7)**: `ordered=["y1","y2"]` kwarg to `cfa()`/`sem()`. Auto-switches to `:DWLS`. Polychoric correlation matrix via two-stage probit ML (univariate thresholds → profile likelihood over ρ=tanh(z)). Asymptotic Gamma_poly via numerical Fisher information. DWLS estimation with R_poly substituted for S. Thresholds appear in `parameterEstimates()` with `op="|"`. `bivnorm_cdf` via 20-point Gauss-Legendre on 1D integral.

## Modindices note

Uses diagonal MI approximation (no Schur complement). Test threshold > 3.0 is safe (MI ≈ 4.03 for x2 loading fixed at 1.0, true ≈ 0.554).

## Known limitations

- Ordinal support uses DWLS (not full WLSMV mean/variance adjustment); suitable for most applied work
- `standardizedSolution()` is approximate (uses proxy for latent variances)
- No SAM (Structural After Measurement) — planned
- EFA: ML extraction + 5 rotations implemented; no confirmatory/bifactor EFA
- Bootstrap SE test uses loose ratio bounds (0.3–3.0) due to sampling variability
