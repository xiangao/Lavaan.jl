# ─── simulate.jl ─────────────────────────────────────────────────────────────
# Generate data from a fitted model or a fully-specified model string.
# Mirrors lavaan's simulateData() function.
#
# Two call signatures:
#
#   simulateData(fit; n, seed)
#     → simulate from the estimated Σ(θ̂) and μ(θ̂) of a fitted model.
#
#   simulateData(model_string; n, seed, ...)
#     → all parameters must be fixed in the syntax (e.g. "f =~ 1*x1 + 0.8*x2").
#       Builds a zero-df model internally, no optimization needed.
# ─────────────────────────────────────────────────────────────────────────────

"""
    simulateData(fit; n=nothing, seed=nothing) → DataFrame

Generate multivariate-normal data from the model-implied covariance Σ(θ̂)
and mean μ(θ̂) of a previously-fitted LavaanFit.

# Arguments
- `fit`  : a fitted LavaanFit object
- `n`    : sample size (default: same as original data per group)
- `seed` : integer seed for reproducibility

# Returns
A DataFrame with the same observed variable names as the original data.

# Example
```julia
fit = cfa(model, data)
sim = simulateData(fit; n=500, seed=42)
```
"""
function simulateData(fit::LavaanFit;
                      n::Union{Nothing,Int} = nothing,
                      seed::Union{Nothing,Int} = nothing)::DataFrame

    rng      = seed !== nothing ? Random.MersenneTwister(seed) : Random.default_rng()
    ov_names = fit.data.ov_names
    dfs      = DataFrame[]

    nblocks = fit.model.nblocks
    for g in 1:nblocks
        Sigma_g = Matrix(Symmetric(fit.implied.Sigma[g]))
        Mu_g    = fit.implied.Mu[g]
        n_g     = n !== nothing ? (g == nblocks ? n - (nblocks-1)*(n÷nblocks) : n÷nblocks) :
                                  fit.data.nobs[g]

        # Ensure positive definite (add small ridge if needed)
        min_eig = minimum(eigvals(Symmetric(Sigma_g)))
        if min_eig < 1e-10
            Sigma_g += (abs(min_eig) + 1e-10) .* I
        end

        dist  = MvNormal(Mu_g, Symmetric(Sigma_g))
        X_sim = rand(rng, dist, n_g)'   # n_g × p

        push!(dfs, DataFrame(X_sim, ov_names))
    end

    return vcat(dfs...)
end

"""
    simulateData(model_string; n=500, seed=nothing, kwargs...) → DataFrame

Generate data from a fully-specified lavaan model string.

All parameters must be fixed via the `*` modifier in the model syntax,
e.g.:
```julia
model = \"\"\"
  f =~ 1*x1 + 0.8*x2 + 0.6*x3
  f ~~ 1*f
  x1 ~~ 0.36*x1
  x2 ~~ 0.36*x2
  x3 ~~ 0.64*x3
\"\"\"
df = simulateData(model; n=300, seed=1)
```

Any `kwargs` are passed to `lavaan()` (e.g. `meanstructure=true`).

!!! note
    All parameters must be fixed. Free parameters will be initialized at
    starting values, which may not match your intended population model.
"""
function simulateData(model_string::String;
                      n::Int = 500,
                      seed::Union{Nothing,Int} = nothing,
                      meanstructure::Bool = false,
                      kwargs...)::DataFrame

    # Extract variable names from model string (no real data needed)
    parsed   = parse_model_string(model_string)
    lv_names = lv_names_from_rows(parsed)
    ov_names = ov_names_from_rows(parsed, lv_names)

    # Build a minimal dummy DataFrame (just enough for samplestats)
    n_dummy = max(length(ov_names) * 3, 30)
    Random.seed!(seed !== nothing ? seed : 0)
    df_dummy = DataFrame(Dict(v => randn(n_dummy) for v in ov_names))

    # Fit with do_fit=false — no optimization, fixed params are taken as-is
    fit = lavaan(model_string, df_dummy;
                 do_fit       = false,
                 meanstructure = meanstructure,
                 kwargs...)

    return simulateData(fit; n=n, seed=seed)
end
