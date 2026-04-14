# Lavaan.jl

`Lavaan.jl` is a Julia port of the R [lavaan](https://lavaan.ugent.be) package for Structural Equation Modeling (SEM). It uses the same model syntax as R lavaan for easy migration.

## Features

- **R-Compatible Syntax**: Same model strings as R's `lavaan` — no relearning required.
- **Full Estimator Suite**: ML, FIML, GLS, WLS, DWLS, MLR, MLM, GSEM.
- **Ordinal/Categorical Data**: Polychoric correlations + DWLS, auto-selected.
- **Generalized SEM**: Poisson/mixed indicators via Gauss-Hermite quadrature.
- **SAM**: Structural After Measurement (Rosseel & Loh 2022).
- **Multilevel & Crossed**: Two-level SEM and arbitrary non-nested clustering.
- **Automatic Differentiation**: ForwardDiff.jl for robust convergence.
- **Bootstrap SEs**: Parallel nonparametric bootstrap via `Threads.@threads`.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/xiangao/Lavaan.jl")
```

## Quick Start

```julia
using Lavaan, DataFrames

HS = holzinger_swineford()

model = """
  visual  =~ x1 + x2 + x3
  textual =~ x4 + x5 + x6
  speed   =~ x7 + x8 + x9
"""

fit = cfa(model, HS)
summary(fit, fit_measures=true)
parameterEstimates(fit)
fitMeasures(fit, [:cfi, :tli, :rmsea, :srmr])
```

## Model syntax

| Operator | Meaning            | Example                  |
|----------|--------------------|--------------------------|
| `=~`     | factor indicator   | `f =~ x1 + x2 + x3`     |
| `~~`     | (co)variance       | `x1 ~~ x1`               |
| `~`      | regression         | `y ~ x1 + x2`            |
| `~1`     | intercept          | `y ~1`                   |
| `:=`     | defined parameter  | `indirect := a*b`        |
| `a * x`  | labeled path       | `textual ~ a * visual`   |

## Vignettes

| Vignette | Description |
|----------|-------------|
| [Introduction](vignettes/Introduction.md) | CFA basics, Holzinger-Swineford example |
| [Model Syntax](vignettes/Model_Syntax.md) | All operators, growth curves, multi-group |
| [Mediation Analysis](vignettes/Mediation_Analysis.md) | Labeled paths, `:=` defined parameters, latent mediation |
| [Ordinal Data](vignettes/Ordinal_Data.md) | Polychoric correlations, DWLS, threshold estimates |
| [Multilevel & Crossed](vignettes/Multilevel_Crossed.md) | Two-level SEM, crossed random effects |
| [GSEM](vignettes/GSEM.md) | Poisson indicators, Gauss-Hermite quadrature |
| [SAM](vignettes/SAM.md) | Structural After Measurement vs standard SEM |
