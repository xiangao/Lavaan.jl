# Categorical and Ordinal Data

```@meta
CurrentModule = Lavaan
```

In `Lavaan.jl`, variables can be declared as ordinal, enabling estimation using **Diagonally Weighted Least Squares (DWLS)**, which is appropriate for Likert-scale or binary outcomes.

## Declaring Ordinal Variables

To specify that certain variables are ordinal, use the `ordered` argument in `cfa()` or `sem()`.

```@example lavaan_ordinal
using Lavaan, DataFrames, Random, Distributions

Random.seed!(123)
n = 400
f = randn(n)
z1 = 0.9 .* f .+ randn(n)
z2 = 0.7 .* f .+ randn(n)
z3 = 0.8 .* f .+ randn(n)
z4 = 0.6 .* f .+ randn(n)

cuts = [-0.5, 0.4]
ordinalize(z) = map(x -> x < cuts[1] ? 1 : x < cuts[2] ? 2 : 3, z)
data = DataFrame(
    y1 = ordinalize(z1),
    y2 = ordinalize(z2),
    y3 = ordinalize(z3),
    y4 = ordinalize(z4),
)

# Model with ordinal indicators
model = """
  f1 =~ y1 + y2 + y3 + y4
"""

# Fit with 'ordered' as an array of column names
fit = cfa(model, data; ordered=["y1", "y2", "y3", "y4"])

# View results
parameterEstimates(fit)
```

When `ordered` is specified, `Lavaan.jl` automatically switches the estimator from Maximum Likelihood (`:ML`) to `:DWLS`.

## Under the Hood: Polychoric Correlations

When dealing with ordinal data, `Lavaan.jl` performs a two-stage estimation process:
1. **Threshold Estimation**: For each ordinal variable, thresholds are estimated assuming a latent underlying normal distribution.
2. **Correlation Estimation**: Polychoric correlations (for ordinal-ordinal pairs) or polyserial correlations (for ordinal-continuous pairs) are calculated.

The resulting correlation matrix is then used in the DWLS estimation.

## Interpreting Results

In the results from `parameterEstimates(fit)`, you will see a new operator `|`, which represents the **thresholds** for the ordinal variables.

| lhs | op | rhs | label | est | se | z | pvalue |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| y1 | &#124; | t1 | | -0.523 | 0.045 | -11.622 | 0.000 |
| y1 | &#124; | t2 | | 0.841 | 0.051 | 16.490 | 0.000 |

A variable with $k$ categories will have $k-1$ thresholds.

## Standard Errors and Scaled Tests

For ordinal models, `Lavaan.jl` provides robust standard errors and scaled test statistics (similar to `lavaan`'s `WLSMV` but focused on `DWLS`).

```julia
# Using the Satorra-Bentler scaled test
fit = cfa(model, data; ordered=["y1", "y2"], estimator=:MLM)
```

## Summary of Estimators

| Estimator | Typical Use Case |
| :--- | :--- |
| `:ML` | Continuous, normally distributed data. |
| `:DWLS` | Ordinal/Categorical data (default when `ordered` is used). |
| `:ULS` | Unweighted Least Squares (useful for some exploratory work). |
| `:FIML` | Full Information Maximum Likelihood (for missing data). |

By providing native support for polychoric correlations, `Lavaan.jl` ensures that your SEM models remain valid even when the underlying data is categorical.
