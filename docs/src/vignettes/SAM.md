# Structural After Measurement (SAM) in Lavaan.jl

```@meta
CurrentModule = Lavaan
```

**Structural After Measurement (SAM)** is a robust estimation approach for structural equation models (Rosseel & Loh, 2022). Unlike standard SEM, which estimates measurement and structural parameters simultaneously, SAM separates these steps. This prevents measurement model misspecification from biasing the structural path estimates.

## Why use SAM?

- **Robustness**: If one part of your measurement model is wrong (e.g., a missing cross-loading), it won't contaminate the regressions between other latent variables.
- **Convergence**: SAM often converges more easily than full SEM for complex models.
- **Interpretability**: By fitting measurement models separately, you can ensure they are well-defined before looking at structural relationships.

## The Two-Step Approach

`Lavaan.jl` implements **Local SAM**, which follows these steps:
1. **Measurement Step**: For each latent variable, a small CFA model is fit to its indicators to estimate loadings and residual variances.
2. **Structural Step**: A mapping matrix (Bartlett weights) is used to derive a "latent" covariance matrix from the observed data. The structural regressions are then fit to this latent covariance matrix.

## Example: Political Democracy Model

The following example compares standard SEM with the SAM approach using the Bollen (1989) Industrialization and Democracy dataset.

```@example lavaan_sam
using Lavaan, DataFrames

# Load the dataset
df = political_democracy()

# Define the full model
model = """
  # measurement model
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

# 1. Fit using standard SEM
fit_sem = sem(model, df)

# 2. Fit using SAM (Local)
fit_sam = sam(model, df; sam_method=:local)

# Compare the structural path estimates
function get_path(fit, lhs, op, rhs)
    pe = parameterEstimates(fit)
    row = pe[(pe.lhs .== lhs) .& (pe.op .== op) .& (pe.rhs .== rhs), :]
    return isempty(row) ? NaN : row.est[1]
end

println("SEM Beta (dem60 ~ ind60): ", round(get_path(fit_sem, "dem60", "~", "ind60"), digits=4))
println("SAM Beta (dem60 ~ ind60): ", round(get_path(fit_sam, "f_dem60", "~", "f_ind60"), digits=4))
```

## Fitting from Summary Statistics

Because the structural step of SAM relies only on latent covariances, `Lavaan.jl` also supports fitting models directly from a covariance matrix and sample size. This is used internally by SAM but can be called manually via `lavaan()`.

```julia
# Fit a model using only a covariance matrix
fit = lavaan(model_syntax, nothing; 
             sample_cov = my_cov_matrix, 
             sample_nobs = 500)
```

## Implementation Notes

- **Variable Naming**: In the current implementation of SAM Step 2, latent variables are automatically prefixed with `f_` (e.g., `ind60` becomes `f_ind60`).
- **Bias Correction**: SAM includes an internal correction to account for measurement error, ensuring that latent path coefficients are not attenuated.
- **Local vs Global**: Currently, `Lavaan.jl` prioritizes the **Local SAM** method, where each latent factor is fit individually.

## References

- Rosseel, Y., & Loh, W. W. (2022). A structural after measurement (SAM) approach to structural equation modeling. *Psychological Methods*.
