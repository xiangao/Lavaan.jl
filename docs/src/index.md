# Lavaan.jl

`Lavaan.jl` is a Julia port of the R
[lavaan](https://lavaan.ugent.be) model syntax. The goal is to write the same
SEM model strings from Julia and estimate them with Julia tools.

## What is included

- CFA, SEM, growth models, and the general `lavaan()` interface.
- ML, FIML, GLS, WLS, DWLS, MLR, MLM, and GSEM.
- Ordinal indicators through polychoric correlations and DWLS.
- Structural After Measurement (Rosseel and Loh 2022).
- Multilevel and crossed-cluster examples.
- Bootstrap standard errors.

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
