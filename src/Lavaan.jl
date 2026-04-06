module Lavaan

using LinearAlgebra
using Statistics
using Random
using Printf
using DataFrames
using CSV
using Distributions
using ForwardDiff
using Optim
using RecipesBase
using StatsAPI
using SparseArrays

import StatsAPI: coef, vcov, fitted, residuals

# ─── Type definitions (first) ─────────────────────────────────────────────────
include("types.jl")

# ─── Pipeline stages ──────────────────────────────────────────────────────────
include("options.jl")
include("syntax.jl")
include("partable.jl")
include("data.jl")
include("validate.jl")
include("ordinal.jl")
include("samplestats.jl")
include("model.jl")
include("start.jl")
include("implied.jl")
include("estimators/ml.jl")
include("estimators/wls.jl")
include("estimators/fiml.jl")
include("estimators/robust.jl")
include("optim.jl")
include("vcov.jl")
include("fit.jl")
include("results.jl")
include("simulate.jl")
include("efa.jl")
include("print.jl")
include("api.jl")

# ─── Public API ───────────────────────────────────────────────────────────────
export lavaan, cfa, sem, growth

# Results extraction
export parameterEstimates, fitMeasures, standardizedSolution
export coef, vcov, fitted, residuals
export lavInspect, lavTestLRT, modindices
export rsquare, lavPredict, simulateData

# EFA
export efa, EFAResult

# Data utilities
export holzinger_swineford, political_democracy

# Types (for dispatch)
export LavaanFit, LavaanOptions, ParTable

end # module Lavaan
