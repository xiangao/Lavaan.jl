# Mediation Analysis in Lavaan.jl

```@meta
CurrentModule = Lavaan
```

Mediation analysis is a common application of SEM, where the researcher explores the mechanism by which one variable affects another through a third variable (the mediator).

```@setup lavaan_mediation_walkthrough
using Lavaan, DataFrames, Random

Random.seed!(123)
n = 250

x = randn(n)
m = 0.7 .* x .+ randn(n)
y = 0.6 .* m .+ 0.3 .* x .+ randn(n)
data = DataFrame(x = x, m = m, y = y)

m1 = 0.6 .* x .+ randn(n)
m2 = 0.4 .* x .+ randn(n)
y_multi = 0.5 .* m1 .+ 0.4 .* m2 .+ 0.2 .* x .+ randn(n)
data_multi = DataFrame(x = x, m1 = m1, m2 = m2, y = y_multi)

latent_x = randn(n)
latent_m = 0.7 .* latent_x .+ 0.5 .* randn(n)
latent_y = 0.6 .* latent_m .+ 0.3 .* latent_x .+ 0.5 .* randn(n)
data_latent = DataFrame(
    x1 = latent_x .+ 0.3 .* randn(n),
    x2 = 0.8 .* latent_x .+ 0.3 .* randn(n),
    x3 = 0.6 .* latent_x .+ 0.3 .* randn(n),
    m1 = latent_m .+ 0.3 .* randn(n),
    m2 = 0.8 .* latent_m .+ 0.3 .* randn(n),
    m3 = 0.6 .* latent_m .+ 0.3 .* randn(n),
    y1 = latent_y .+ 0.3 .* randn(n),
    y2 = 0.8 .* latent_y .+ 0.3 .* randn(n),
    y3 = 0.6 .* latent_y .+ 0.3 .* randn(n),
)
```

## The Mediation Model

In a simple mediation model, we have:
- **Independent variable ($X$)**
- **Mediator variable ($M$)**
- **Dependent variable ($Y$)**

We want to test the **indirect effect** ($a \cdot b$) and the **direct effect** ($c$).

### Specification in Lavaan.jl

Use labels to name the paths and the `:=` operator to define the indirect and total effects.

```@example lavaan_mediation_walkthrough
model = """
  # Regressions
    m ~ a * x
    y ~ b * m + c * x

  # Defined parameters
    indirect := a * b
    direct   := c
    total    := direct + indirect
"""

# Fit the model
fit = sem(model, data)

# See the results for 'indirect', 'direct', and 'total'
parameterEstimates(fit)
```

## Standard Errors for Indirect Effects

By default, `Lavaan.jl` calculates standard errors for defined parameters using the **Delta Method**.

### Bootstrap Standard Errors

For mediation analysis, many researchers prefer **nonparametric bootstrap standard errors**, as the product of two coefficients ($a \cdot b$) is often not normally distributed.

```@example lavaan_mediation_walkthrough
# Use a small number of bootstrap iterations to keep docs builds fast
fit = sem(model, data; se=:bootstrap, nboot=20)

# The summary will now report bootstrap SEs and p-values
parameterEstimates(fit)[parameterEstimates(fit).op .== ":=", [:lhs, :op, :rhs, :est, :se, :pvalue]]
```

`Lavaan.jl` parallelizes bootstrap calculations using Julia's `Threads.@threads`, making it significantly faster on multi-core systems.

## Multiple Mediators

You can easily extend this to models with multiple mediators, whether they are in parallel or in a serial sequence.

```@example lavaan_mediation_walkthrough
# Parallel mediators
model = """
  m1 ~ a1 * x
  m2 ~ a2 * x
  y  ~ b1 * m1 + b2 * m2 + c * x

  ind_m1 := a1 * b1
  ind_m2 := a2 * b2
  total_indirect := ind_m1 + ind_m2
"""

fit = sem(model, data_multi)
parameterEstimates(fit)[parameterEstimates(fit).op .== ":=", [:lhs, :op, :rhs, :est, :se]]
```

## Mediation with Latent Variables

The real power of `Lavaan.jl` comes when you use latent variables for your mediators or predictors.

```@example lavaan_mediation_walkthrough
model = """
  # Measurement model
    X =~ x1 + x2 + x3
    M =~ m1 + m2 + m3
    Y =~ y1 + y2 + y3

  # Structural model
    M ~ a * X
    Y ~ b * M + c * X

  # Indirect effect
    indirect := a * b
"""

fit = sem(model, data_latent)
parameterEstimates(fit)[parameterEstimates(fit).op .== ":=", [:lhs, :op, :rhs, :est, :se]]
```

By using latent variables, you can account for measurement error, which often biases estimates of indirect effects in standard regression-based mediation.

### Example: Latent Mediation with Holzinger-Swineford

```@example lavaan_mediation
using Lavaan, DataFrames

df = holzinger_swineford()
model = """
  # Measurement model
    visual  =~ x1 + x2 + x3
    textual =~ x4 + x5 + x6
    speed   =~ x7 + x8 + x9

  # Regressions
    textual ~ a * visual
    speed   ~ b * textual + c * visual

  # Indirect and total effect
    indirect := a * b
    total    := indirect + c
"""

fit = sem(model, df)
pe = parameterEstimates(fit)

# Show only the defined parameters
pe[pe.op .== ":=", [:lhs, :op, :rhs, :est, :se, :z, :pvalue]]
```
