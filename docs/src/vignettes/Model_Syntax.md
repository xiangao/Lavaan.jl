# Model Syntax and Operators

```@meta
CurrentModule = Lavaan
```

```@setup lavaan_syntax
using Lavaan
using DataFrames

HS = holzinger_swineford()
```

`Lavaan.jl` uses the same basic model syntax as R's `lavaan`. The main
operators are the ones below.

## The Core Operators

| Operator | Meaning | Example |
| :--- | :--- | :--- |
| `=~` | Measured by (latent variable definition) | `f =~ x1 + x2 + x3` |
| `~` | Regressed on (directional path) | `y ~ x1 + x2` |
| `~~` | Correlated with (variance/covariance) | `x1 ~~ x2` or `x1 ~~ x1` |
| `~ 1` | Intercept (mean) | `y ~ 1` |
| `:=` | Parameter definition (defined functions) | `indirect := a * b` |
| `==` | Equality constraint | `a == b` |
| `<` | Inequality constraint | `a > 0` |
| `>` | Inequality constraint | `b < 1` |

## Labeling and Constraints

You can label parameters by prefixing them with a label and `*`. This is useful for defining indirect effects or applying constraints.

### 1. Equality Constraints

Label two parameters with the same name to constrain them to be equal:

```@example lavaan_syntax
model = """
  # Constrain factor loadings of x2 and x3 to be the same
  f =~ x1 + L1*x2 + L1*x3
"""

Lavaan.parse_model_string(model)
```

### 2. Parameter Definitions (`:=`)

Use labeling to compute new parameters, such as indirect effects in a mediation model:

```@example lavaan_syntax
# Simple mediation: x -> m -> y
model = """
  y ~ b*m + c*x
  m ~ a*x

  # Indirect effect (a * b)
  indirect := a * b

  # Total effect (indirect + direct)
  total := indirect + c
"""

Lavaan.parse_model_string(model)
```

## Residual Variances and Covariances

By default, `Lavaan.jl` automatically adds residual variances for all endogenous variables. You can explicitly specify them or their covariances using the `~~` operator.

```@example lavaan_syntax
model = """
  visual =~ x1 + x2 + x3
  
  # Explicitly allow x1 and x2 errors to correlate
  x1 ~~ x2
  
  # Fix a variance to a specific value
  x3 ~~ 0.5*x3
"""

Lavaan.parse_model_string(model)
```

## Latent Growth Curves (`growth()`)

Growth curve models are a special case of SEM where factor loadings are fixed to represent time.

```@example lavaan_syntax
# Linear growth model
model = """
  # Intercept (i) and slope (s) factors
  i =~ 1*t1 + 1*t2 + 1*t3 + 1*t4
  s =~ 0*t1 + 1*t2 + 2*t3 + 3*t4
"""

model
```

## Multiple Groups

You can specify a grouping variable to perform multi-group SEM:

```@example lavaan_syntax
# Fit the model across different schools
model = """
  visual  =~ x1 + x2 + x3
  textual =~ x4 + x5 + x6
  speed   =~ x7 + x8 + x9
"""

fit = cfa(model, HS; group="school")
parameterEstimates(fit)[1:6, [:lhs, :op, :rhs, :est, :se]]
```

This will automatically allow parameters to vary across groups unless otherwise constrained.
