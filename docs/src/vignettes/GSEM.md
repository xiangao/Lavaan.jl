# Generalized SEM (GSEM) in Lavaan.jl

```@meta
CurrentModule = Lavaan
```

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

```@example lavaan_gsem
using Lavaan, DataFrames, Random
using Distributions

# Simulate some Poisson data
Random.seed!(123)
n = 500
f = randn(n)  # Latent factor
y1 = [rand(Poisson(exp(0.0 + 1.0 * f_i))) for f_i in f]
y2 = [rand(Poisson(exp(0.0 + 0.8 * f_i))) for f_i in f]
y3 = [rand(Poisson(exp(0.0 + 0.6 * f_i))) for f_i in f]
x1 = 0.8 .* f .+ randn(n)
x2 = 0.6 .* f .+ randn(n)

df = DataFrame(y1=y1, y2=y2, y3=y3, x1=x1, x2=x2)

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

## Mixed Outcomes

You can mix Gaussian and Poisson indicators in the same model. Any variable not specified in the `family` dictionary defaults to `:gaussian`.

```@example lavaan_gsem
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
