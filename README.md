# Lavaan.jl

Julia port of the R [lavaan](https://lavaan.ugent.be) package for Structural Equation Modeling (SEM).

Uses the same model syntax as R lavaan for easy migration.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/xiangao/Lavaan.jl")
```

## Usage

```julia
using Lavaan, DataFrames

# Load built-in dataset
HS = holzinger_swineford()

# Define model with lavaan syntax
model = """
  visual  =~ x1 + x2 + x3
  textual =~ x4 + x5 + x6
  speed   =~ x7 + x8 + x9
"""

# Fit CFA
fit = cfa(model, HS)

# Results
summary(fit, fit_measures=true)
parameterEstimates(fit)
fitMeasures(fit, [:cfi, :tli, :rmsea, :srmr])
```

## Supported models

- Confirmatory Factor Analysis (CFA): `cfa()`
- Structural Equation Models (SEM): `sem()`
- Latent Growth Curve Models: `growth()`
- General lavaan interface: `lavaan()`

## Model syntax

Identical to R lavaan:

| Operator | Meaning                     | Example              |
|----------|-----------------------------|----------------------|
| `=~`     | factor indicator            | `f =~ x1 + x2 + x3` |
| `~~`     | (co)variance                | `x1 ~~ x1`           |
| `~`      | regression                  | `y ~ x1 + x2`        |
| `~1`     | intercept                   | `y ~1`               |
| `:=`     | defined parameter           | `indirect := a*b`    |

## Estimators

| Estimator | Description |
|-----------|-------------|
| `:ML`     | Maximum Likelihood (default) |
| `:MLR`    | ML with Huber-White sandwich SEs |
| `:MLM`    | ML with Satorra-Bentler scaled test statistic |
| `:GLS`    | Generalized Least Squares |
| `:ULS`    | Unweighted Least Squares |
| `:WLS`    | Weighted Least Squares |
| `:DWLS`   | Diagonally Weighted Least Squares (auto-selected for ordinal data) |
| `:FIML`   | Full Information Maximum Likelihood (missing data) |

## Ordinal / categorical data

```julia
# Specify ordinal variables — estimator auto-switches to :DWLS
fit = cfa(model, data; ordered=["y1", "y2", "y3"])

# Thresholds appear in parameterEstimates() with op="|"
parameterEstimates(fit)
```

Polychoric correlations are estimated via two-stage probit ML. Asymptotic standard errors use the numerical Fisher information of the bivariate normal likelihood.

## Standard errors

```julia
# Standard (default)
fit = cfa(model, data)

# Robust sandwich (MLR)
fit = cfa(model, data; se=:robust)

# Satorra-Bentler scaled test (non-normal data)
fit = cfa(model, data; estimator=:MLM)

# Bootstrap (nonparametric, parallel)
fit = cfa(model, data; se=:bootstrap, nboot=1000)
```

## Fit indices

`chisq`, `df`, `pvalue`, `cfi`, `tli`, `rmsea` (+ 90% CI), `srmr`, `aic`, `bic`, `sabic`, `logl`

For MLM estimator: also `chisq_scaled`, `pvalue_scaled`, `scaling_factor`

## Diagnostics

```julia
# Nested model comparison (likelihood ratio test)
lavTestLRT(fit_restricted, fit_full)

# Modification indices
modindices(fit)           # all
modindices(fit; sort=true, minimum_value=3.0)
```

## Performance

Uses [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) for automatic differentiation and [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) L-BFGS for optimization. Bootstrap parallelized via `Threads.@threads`.

## Test suite

152 tests covering: CFA, SEM, syntax parsing, non-convergence safety, fit measures, parameter estimates, FIML, lavTestLRT, modindices, bootstrap SEs, and Satorra-Bentler scaled test.

## References

Rosseel, Y. (2012). lavaan: An R Package for Structural Equation Modeling. *Journal of Statistical Software*, 48(2), 1–36.

Satorra, A. & Bentler, P.M. (1994). Corrections to test statistics and standard errors in covariance structure analysis. In A. von Eye & C.C. Clogg (Eds.), *Latent variables analysis*, pp. 399–419.
