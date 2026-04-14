# Mediation Analysis in Lavaan.jl

Mediation analysis is a common application of SEM, where the researcher explores the mechanism by which one variable affects another through a third variable (the mediator).

## The Mediation Model

In a simple mediation model, we have:
- **Independent variable ($X$)**
- **Mediator variable ($M$)**
- **Dependent variable ($Y$)**

We want to test the **indirect effect** ($a \cdot b$) and the **direct effect** ($c$).

### Specification in Lavaan.jl

Use labels to name the paths and the `:=` operator to define the indirect and total effects.

```julia
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

```julia
# Use 1000 bootstrap iterations
fit = sem(model, data; se=:bootstrap, nboot=1000)

# The summary will now report bootstrap SEs and p-values
summary(fit)
```

`Lavaan.jl` parallelizes bootstrap calculations using Julia's `Threads.@threads`, making it significantly faster on multi-core systems.

## Multiple Mediators

You can easily extend this to models with multiple mediators, whether they are in parallel or in a serial sequence.

```julia
# Parallel mediators
model = """
  m1 ~ a1 * x
  m2 ~ a2 * x
  y  ~ b1 * m1 + b2 * m2 + c * x

  ind_m1 := a1 * b1
  ind_m2 := a2 * b2
  total_indirect := ind_m1 + ind_m2
"""

fit = sem(model, data)
```

## Mediation with Latent Variables

The real power of `Lavaan.jl` comes when you use latent variables for your mediators or predictors.

```julia
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

fit = sem(model, data)
```

By using latent variables, you can account for measurement error, which often biases estimates of indirect effects in standard regression-based mediation.

### Example: Latent Mediation with Holzinger-Swineford

```julia
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

**Output:**

```text
2×7 DataFrame
 Row │ lhs       op      rhs           est        se         z        pvalue    
     │ String    String  String        Float64    Float64    Float64  Float64   
─────┼──────────────────────────────────────────────────────────────────────────
   1 │ indirect  :=      a * b         0.0268808  0.0265923  1.01085  0.31209
   2 │ total     :=      indirect + c  0.324008   0.0697292  4.64666  3.3735e-6
```
