# Generalized SEM (GSEM) in Lavaan.jl

Standard SEM assumes that all observed indicators follow a multivariate normal distribution. However, in many research settings, you may have count data, binary outcomes, or other non-Gaussian distributions. 

`Lavaan.jl` supports **Generalized SEM (GSEM)**, allowing you to specify different distributions (families) for different outcome variables within the same model.

## Supported Families

Currently, `Lavaan.jl` supports the following families:
- `:gaussian` (default): Linear regression with normal errors.
- `:poisson`: Poisson regression for count data (using a log link).

## How it Works

GSEM uses **Gauss-Hermite quadrature** to integrate the conditional likelihood over the latent variable distribution. By default, `Lavaan.jl` uses 15 quadrature points per latent dimension to ensure high accuracy.

## Example: Poisson Indicators in CFA

In this example, we simulate and fit a CFA model where the indicators are count variables following a Poisson distribution.

```julia
using Lavaan, DataFrames, Random

# Simulate some Poisson data
Random.seed!(123)
n = 500
f = randn(n)  # Latent factor
y1 = [rand(Poisson(exp(0.0 + 1.0 * f_i))) for f_i in f]
y2 = [rand(Poisson(exp(0.0 + 0.8 * f_i))) for f_i in f]
y3 = [rand(Poisson(exp(0.0 + 0.6 * f_i))) for f_i in f]

df = DataFrame(y1=y1, y2=y2, y3=y3)

# Define the model syntax
model = """
  F =~ y1 + y2 + y3
"""

# Fit using the family keyword argument
# Mapping variable names to their respective distributions
fit = cfa(model, df; family=Dict("y1" => :poisson, "y2" => :poisson, "y3" => :poisson))

# View results
summary(fit)
```

**Output:**

```text
────────────────────────────────────────────────────────────────────────
lavaan 0.1.0-dev -- CFA model

  ✓  Model converged normally
     Number of observations:             500
     Estimator:                         GSEM
     Number of model parameters:           6

Model Test User Model:
  Test statistic:               4462.736
  Degrees of freedom:           3
  P-value (Chi-square):         0.000

Parameter Estimates:
  Standard errors:              standard
  Information:                  expected

Latent Variable Definitions:

  Observed          Variable     Estimate  Std.Err z-value  P(>|z|)
  ────────────────────────────────────────────────────────────
  F            =~   y1              1.000   (fixed)
               =~   y2              0.830    0.060 13.770    0.000
               =~   y3              0.577    0.058  9.987    0.000

Variances and Covariances:

  Observed          Variable     Estimate  Std.Err z-value  P(>|z|)
  ────────────────────────────────────────────────────────────
  y1           ~~   y1              0.000   (fixed)
  y2           ~~   y2              0.000   (fixed)
  y3           ~~   y3              0.000   (fixed)
  F            ~~   F               1.059    0.115  9.188    0.000

Intercepts:

  Observed          Variable     Estimate  Std.Err z-value  P(>|z|)
  ────────────────────────────────────────────────────────────
  y1           ~1   1              -0.028    0.072 -0.393    0.694
  y2           ~1   1              -0.036    0.066 -0.540    0.589
  y3           ~1   1              -0.058    0.057 -1.013    0.311


Warnings:
  ⚠ Baseline model did not converge. CFI/TLI are NaN.
────────────────────────────────────────────────────────────────────────
```

## Mixed Outcomes

You can mix Gaussian and Poisson indicators in the same model. Any variable not specified in the `family` dictionary defaults to `:gaussian`.

```julia
# Model where x1, x2 are normal and y1 is Poisson
model = """
  F =~ x1 + x2 + y1
"""

fit = cfa(model, df; family=Dict("y1" => :poisson))
```

## Implementation Details

- **Intercepts**: When using non-Gaussian families, intercepts are automatically enabled (`meanstructure=true`) as they are essential for the link function.
- **Residual Variances**: For Poisson indicators, the residual variance is fixed to zero in the ParTable, as the variance is determined by the mean (λ).
- **Optimization**: GSEM models are minimized using the `:GSEM` estimator, which optimizes the marginal log-likelihood directly.

## Performance Note

Gauss-Hermite quadrature scale exponentially with the number of latent factors ($Q^q$). For models with 3 or more latent factors, consider reducing `n_quad_points` (e.g., to 7 or 9) if optimization is slow.

```julia
fit = sem(model, data; n_quad_points=7, family=my_families)
```
