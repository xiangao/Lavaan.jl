# Multilevel and Crossed Models in Lavaan.jl

This page shows how `Lavaan.jl` handles nested and crossed dependence.

## Nested Multilevel SEM

When observations are nested within groups (e.g., students within schools), standard SEM can lead to biased standard errors. `Lavaan.jl` uses the **MUML (Muthen's Maximum Likelihood)** estimator to correctly handle these nested structures.

### Syntax for Multilevel Models

Use the `level: 1` and `level: 2` blocks to define your within- and between-group models separately.

```julia
model = """
  # Within-group model (Level 1)
  level: 1
    f_w =~ x1 + x2 + x3
    f_w ~ y_w

  # Between-group model (Level 2)
  level: 2
    f_b =~ x1 + x2 + x3
    f_b ~ y_b
"""

# Fit with the 'cluster' argument
fit = sem(model, data; cluster="school_id")

# View results
summary(fit)
```

**Output:**

```text
────────────────────────────────────────────────────────────────────────
lavaan 0.1.0-dev -- SEM model

  ✓  Model converged normally
     Number of observations:             301
     Estimator:                           ML
     Number of model parameters:          12

Model Test User Model:
  Test statistic:               1042.906
  Degrees of freedom:           0
  P-value (Chi-square):            NA

Parameter Estimates:
  Standard errors:              standard
  Information:                  expected

Latent Variable Definitions:

  Observed          Variable     Estimate  Std.Err z-value  P(>|z|)
  ────────────────────────────────────────────────────────────
  visual       =~   x1              1.000   (fixed)
               =~   x2              0.825    0.140  5.891    0.000
               =~   x3              1.170    0.217  5.385    0.000
               =~   x1              1.000   (fixed)
               =~   x2              0.299 3118.081  0.000    1.000
               =~   x3              0.428 3841.539  0.000    1.000

Variances and Covariances:

  Observed          Variable     Estimate  Std.Err z-value  P(>|z|)
  ────────────────────────────────────────────────────────────
  x1           ~~   x1              0.867    0.112  7.746    0.000
  x2           ~~   x2              1.037    0.103 10.053    0.000
  x3           ~~   x3              0.540    0.127  4.258    0.000
  x1           ~~   x1              0.000    0.000              NA
  x2           ~~   x2              0.000    0.000              NA
  x3           ~~   x3              0.000    0.000              NA
  visual       ~~   visual          0.496    0.122  4.062    0.000
               ~~   visual          0.000    0.000              NA
```

`Lavaan.jl` will automatically compute the pooled within-group and between-group sample moments to estimate the model.

## Crossed Random Effects

For more complex data where observations are nested within multiple, non-nested factors (e.g., students nested in both schools and neighborhoods), `Lavaan.jl` implements a **Global Sparse FIML** estimator.

Unlike standard R `lavaan`, `Lavaan.jl` can natively handle these non-nested dependencies using sparse matrix Cholesky factorization for high performance.

### Syntax for Crossed Models

Specify multiple clustering variables in the `cluster` argument as an array.

```julia
model = """
  # Define your model as usual
  f1 =~ x1 + x2 + x3
  f2 =~ y1 + y2 + y3
  f2 ~ f1
"""

# Fit with multiple crossed clusters
fit = sem(model, data; cluster=["school", "neighborhood"])
```

### Performance Considerations

The crossed random effects code uses `SparseArrays.jl` to represent the global
$Np \times Np$ covariance matrix without storing all entries densely.

## When to use which?

| Structure | Example | Method | Argument |
| :--- | :--- | :--- | :--- |
| **Nested** | Students in Schools | MUML | `cluster="school"` |
| **Crossed** | Students in Schools & Neighborhoods | Sparse FIML | `cluster=["school", "nh"]` |
| **Multiple levels** | Students in Classes in Schools | MUML (extended) | `cluster=["class", "school"]` |

The same `ForwardDiff.jl` machinery is used for these likelihoods.
