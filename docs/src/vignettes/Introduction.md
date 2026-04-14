# Introduction to Lavaan.jl

```@meta
CurrentModule = Lavaan
```

`Lavaan.jl` is a Julia port of the R package `lavaan` (Latent Variable Analysis). It allows for the estimation of a wide range of structural equation models, including confirmatory factor analysis (CFA), path analysis, and full structural equation modeling (SEM).

## Why Lavaan.jl?

- **R-Compatible Syntax**: Use the same model strings you are familiar with from R's `lavaan`.
- **Julia Performance**: Benefit from Julia's speed and advanced optimization libraries like `Optim.jl`.
- **Automatic Differentiation**: High-precision gradients via `ForwardDiff.jl` ensure robust and fast convergence.
- **Advanced Features**: Native support for multilevel and crossed random effects.

## A First Example: Confirmatory Factor Analysis (CFA)

The classic "Holzinger and Swineford" dataset is included in `Lavaan.jl`. This dataset consists of mental ability test scores of seventh- and eighth-grade children from two different schools.

```@example lavaan_intro
using Lavaan, DataFrames

# Load the classic Holzinger and Swineford (1939) dataset
HS = holzinger_swineford()

# Define the model syntax
# =~ is used for factor indicators
# ~~ is used for (co)variances
model = """
  visual  =~ x1 + x2 + x3
  textual =~ x4 + x5 + x6
  speed   =~ x7 + x8 + x9
"""

# Fit the CFA model
fit = cfa(model, HS)

# Summarize the results
summary(fit, fit_measures=true)
```

## Structural Equation Modeling (SEM)

For full SEM models involving regressions between latent variables, use the `sem()` function.

```@example lavaan_intro
# Political Democracy model from Bollen (1989)
model = """
  # latent variable definitions
     ind60 =~ x1 + x2 + x3
     dem60 =~ y1 + y2 + y3 + y4
     dem65 =~ y5 + y6 + y7 + y8

  # regressions
    dem60 ~ ind60
    dem65 ~ ind60 + dem60

  # residual correlations
    y1 ~~ y5
    y2 ~~ y4 + y6
    y3 ~~ y7
    y4 ~~ y8
    y6 ~~ y8
"""

fit = sem(model, political_democracy())
summary(fit)
```

## Model Components

- **Factor Loadings (`=~`)**: Defines which observed variables indicate which latent factors.
- **Regressions (`~`)**: Defines directional relationships between variables.
- **(Co)variances (`~~`)**: Defines variances of variables and covariances between them.
- **Intercepts (`~ 1`)**: Explicitly models the mean structure.
- **Defined Parameters (`:=`)**: Create new parameters as functions of existing ones (useful for indirect effects in mediation).

## Next Steps

Explore the more advanced vignettes to learn about:
- [Model Syntax and Operators](Model_Syntax.md)
- [Multilevel and Crossed Models](Multilevel_Crossed.md)
- [Categorical and Ordinal Data](Ordinal_Data.md)
