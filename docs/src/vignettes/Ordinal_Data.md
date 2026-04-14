# Categorical and Ordinal Data

In `Lavaan.jl`, variables can be declared as ordinal, enabling estimation using **Diagonally Weighted Least Squares (DWLS)**, which is appropriate for Likert-scale or binary outcomes.

## Declaring Ordinal Variables

To specify that certain variables are ordinal, use the `ordered` argument in `cfa()` or `sem()`.

```julia
# Model with ordinal indicators
model = """
  f1 =~ y1 + y2 + y3 + y4
"""

# Fit with 'ordered' as an array of column names
fit = cfa(model, data; ordered=["y1", "y2", "y3", "y4"])

# View results
parameterEstimates(fit)
```

**Output:**

```text
8×6 DataFrame
 Row │ lhs     op      rhs     est          se          z            
     │ String  String  String  Float64      Float64     Float64      
─────┼───────────────────────────────────────────────────────────────
   1 │ F       =~      y1       1.0         NaN         NaN
   2 │ F       =~      y2       0.627364    108.025       0.00580758
   3 │ y1      ~~      y1       0.437676     96.8258      0.00452024
   4 │ y2      ~~      y2       0.778677     38.1094      0.0204327
   5 │ F       ~~      F        0.562324     96.8258      0.00580758
   6 │ y1      |       t1      -0.310738      0.117645   -2.64132
   7 │ y1      |       t2       0.36381       0.119769    3.03759
   8 │ y2      |       t1      -0.00501328    0.112101   -0.044721
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
