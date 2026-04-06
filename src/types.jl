# ─── types.jl ────────────────────────────────────────────────────────────────
# Central type definitions for Lavaan.jl.
# Written FIRST — every other file depends on these structs.
# ─────────────────────────────────────────────────────────────────────────────

# ─── LavaanOptions ────────────────────────────────────────────────────────────

"""
    LavaanOptions

All user-controllable options for a lavaan model fit.
Immutable: constructed once at the `sem()`/`cfa()` call site.
"""
Base.@kwdef struct LavaanOptions
    # Estimator
    estimator::Symbol        = :ML      # :ML :MLR :MLM :FIML :GLS :WLS :DWLS :WLSMV :PML :ULS
    # Standard errors
    se::Symbol               = :standard # :standard :robust_sem :robust_huber_white :bootstrap :none
    # Missing data
    clusters::Vector{String} = String[] # clustering variable names (vector for crossed)
    cluster::Union{Symbol,String,Nothing} = nothing # deprecated: use clusters
    # Chi-square test(s)
    test::Vector{Symbol}     = [:standard]
    # Missing data
    missing_method::Symbol   = :listwise # :listwise :fiml :pairwise
    # Mean structure
    meanstructure::Bool      = false
    # Variable roles
    fixed_x::Bool            = true     # fix exogenous x covariances to sample values
    conditional_x::Bool      = false
    # Latent variable scaling
    std_lv::Bool             = false    # standardize latent variables (std.lv)
    orthogonal::Bool         = false
    orthogonal_y::Bool       = false
    orthogonal_x::Bool       = false
    # Internal representation
    representation::Symbol   = :LISREL  # :LISREL or :RAM
    parameterization::Symbol = :delta   # :delta or :theta (for categorical)
    # Ordered/categorical variables
    ordered::Vector{String} = String[]  # variable names declared ordinal
    # Auto-parameter defaults
    int_ov_free::Bool        = false    # free observed intercepts
    int_lv_free::Bool        = false    # free latent intercepts
    auto_fix_first::Bool     = true     # fix first loading per factor to 1
    auto_fix_single::Bool    = true     # fix single-indicator loadings to 1
    auto_var::Bool           = true     # add residual variances automatically
    auto_cov_lv_x::Bool      = true     # add covariances among exo latents
    auto_cov_y::Bool         = false    # add covariances among endo observed
    auto_th::Bool            = true     # add thresholds for ordered variables
    auto_delta::Bool         = true     # add scaling parameters for ordered
    # Optimization
    do_fit::Bool             = true
    bounds::Symbol           = :none    # :none :default :wide :wide_zerovar
    ridge::Bool              = false
    ridge_constant::Float64  = 1e-5
    optim_method::Symbol     = :lbfgs   # :lbfgs :nlminb
    optim_tol::Float64       = 1e-7
    optim_iter::Int          = 10_000
    # Information matrix type
    information::Symbol      = :expected # :expected or :observed
    # Bootstrap
    nboot::Int               = 1000
    boot_type::Symbol        = :ordinary # :ordinary :bollen_stine :parametric
    # EFA
    rotation::Symbol         = :geomin
    # Model type (used internally by cfa/sem/growth)
    model_type::Symbol       = :sem     # :sem :cfa :growth
    # Verbosity
    verbose::Bool            = false
    debug::Bool              = false
end

# ─── ParTable ─────────────────────────────────────────────────────────────────

"""
    ParTable

The parameter table: a columnar representation of all model parameters.
Mirrors lavaan's internal `lavParTable` (a named list of equal-length vectors).

Columns present before fitting:
  id, lhs, op, rhs, user, block, group, level, free, ustart, exo,
  label, plabel, eq_id, lower, upper, start

Columns added after fitting:
  est, se, z, pvalue, ci_lower, ci_upper

Columns added after model.jl:
  mat, mat_row, mat_col
"""
mutable struct ParTable
    # Core identification
    id::Vector{Int}
    lhs::Vector{String}
    op::Vector{String}
    rhs::Vector{String}
    # Source
    user::Vector{Int}           # 1=user-specified, 0=auto-added
    block::Vector{Int}          # block index (group × level)
    group::Vector{String}       # group label
    level::Vector{String}       # level label (multilevel)
    # Parameter identification
    free::Vector{Int}           # 0=fixed; >0=index in θ vector
    ustart::Vector{Union{Float64,Missing}}  # user-provided starting value
    exo::Vector{Bool}           # exogenous variable?
    # Labels
    label::Vector{String}       # user-specified label (e.g. "b1")
    plabel::Vector{String}      # auto-generated label (.p1., .p2., ...)
    eq_id::Vector{Int}          # equality constraint group (0=none)
    # Bounds
    lower::Vector{Float64}
    upper::Vector{Float64}
    # After partable.jl: starting values
    start::Vector{Float64}
    # After optim.jl: estimates
    est::Vector{Float64}
    se::Vector{Float64}
    z::Vector{Float64}
    pvalue::Vector{Float64}
    ci_lower::Vector{Float64}
    ci_upper::Vector{Float64}
    # After model.jl: LISREL matrix location
    mat::Vector{String}         # "lambda","theta","psi","beta","nu","alpha",""
    mat_row::Vector{Int}
    mat_col::Vector{Int}
end

"""
    ParTable(n::Int)

Construct an empty ParTable for `n` rows.
"""
function ParTable(n::Int)
    ParTable(
        zeros(Int, n),             # id
        fill("", n),               # lhs
        fill("", n),               # op
        fill("", n),               # rhs
        zeros(Int, n),             # user
        ones(Int, n),              # block (default: block 1)
        fill("", n),               # group
        fill("", n),               # level
        zeros(Int, n),             # free
        fill(missing, n),          # ustart
        fill(false, n),            # exo
        fill("", n),               # label
        fill("", n),               # plabel
        zeros(Int, n),             # eq_id
        fill(-Inf, n),             # lower
        fill(+Inf, n),             # upper
        zeros(Float64, n),         # start
        fill(NaN, n),              # est
        fill(NaN, n),              # se
        fill(NaN, n),              # z
        fill(NaN, n),              # pvalue
        fill(NaN, n),              # ci_lower
        fill(NaN, n),              # ci_upper
        fill("", n),               # mat
        zeros(Int, n),             # mat_row
        zeros(Int, n),             # mat_col
    )
end

Base.length(pt::ParTable) = length(pt.id)
Base.size(pt::ParTable) = (length(pt.id),)

"""Number of free parameters."""
nfree(pt::ParTable) = maximum(pt.free; init=0)

"""Indices of free parameter rows."""
free_idx(pt::ParTable) = findall(pt.free .> 0)

"""Indices of user-specified rows only."""
user_idx(pt::ParTable) = findall(pt.user .== 1)

"""Get starting value vector θ₀."""
function get_start(pt::ParTable)
    nθ = nfree(pt)
    θ = zeros(nθ)
    for i in eachindex(pt.free)
        if pt.free[i] > 0
            θ[pt.free[i]] = pt.start[i]
        end
    end
    return θ
end

"""Get estimate vector θ̂."""
function get_est(pt::ParTable)
    nθ = nfree(pt)
    θ = zeros(nθ)
    for i in eachindex(pt.free)
        if pt.free[i] > 0
            θ[pt.free[i]] = pt.est[i]
        end
    end
    return θ
end

"""Write estimate vector θ̂ back into parTable."""
function set_est!(pt::ParTable, θ::Vector{Float64})
    for i in eachindex(pt.free)
        if pt.free[i] > 0
            pt.est[i] = θ[pt.free[i]]
        elseif !ismissing(pt.ustart[i])
            pt.est[i] = pt.ustart[i]
        else
            pt.est[i] = pt.start[i]
        end
    end
end

# ─── LavaanData ───────────────────────────────────────────────────────────────

"""
    LavaanData

Prepared data for model fitting.  One matrix per group.
"""
struct LavaanData
    X::Vector{Matrix{Float64}}           # [ngroups] rows=obs, cols=ov_names
    ov_names::Vector{String}             # observed variable names
    lv_names::Vector{String}             # latent variable names (from parTable)
    ngroups::Int
    nobs::Vector{Int}                    # number of observations per group
    group_labels::Vector{String}         # group label strings
    weights::Vector{Union{Vector{Float64},Nothing}}
    # original DataFrame reference (for bootstrap resampling)
    original_df::Any                     # DataFrame or nothing
    group_col::Union{String,Nothing}
    cluster_col::Union{String,Nothing} # deprecated: use cluster_cols
    cluster_cols::Vector{String}
    cluster_idx::Vector{Vector{Vector{Int}}}     # per group: list of indices for each cluster factor
    cluster_sizes::Vector{Vector{Vector{Int}}}   # per group: cluster sizes for each factor
    ordered::Vector{String}                              # ordinal variable names
    num_thresholds::Dict{String,Int}                     # k-1 thresholds per var (empty until samplestats)
end

# ─── SampleStats ──────────────────────────────────────────────────────────────

"""
    FIMLPattern

One distinct missing-data pattern for FIML estimation.
Observations are grouped by pattern to share Cholesky decompositions.
"""
struct FIMLPattern
    obs_idx::Vector{Int}       # which columns are observed
    rows::Vector{Int}          # which row indices have this pattern
    X_obs::Matrix{Float64}     # n_pattern × p_obs submatrix of observed data
end

"""
    SampleStats

Observed summary statistics computed from `LavaanData`.
"""
struct SampleStats
    S::Vector{Matrix{Float64}}           # sample covariance per group
    S_inv::Vector{Matrix{Float64}}       # S⁻¹
    logdet_S::Vector{Float64}            # log|S|
    mu::Vector{Vector{Float64}}          # sample means per group
    nobs::Vector{Int}                    # effective N per group
    ntotal::Int                          # sum of nobs
    p::Int                               # number of observed variables
    # For WLS/GLS (computed on demand)
    Gamma::Union{Nothing,Vector{Matrix{Float64}}}     # asymp. cov of vech(S)
    NACOV::Union{Nothing,Vector{Matrix{Float64}}}     # user-supplied NACOV
    # For ordinal DWLS (polychoric)
    thresholds::Union{Nothing,Dict{String,Vector{Float64}}}      # per-var threshold estimates
    threshold_ses::Union{Nothing,Dict{String,Vector{Float64}}}   # per-var threshold SEs
    # For FIML (raw observation patterns)
    fiml_patterns::Union{Nothing,Vector{Vector{FIMLPattern}}}
    # For Multilevel (MUML or Crossed)
    YLp::Union{Nothing,Vector{Vector{Matrix{Float64}}}}       # Per group: list of cluster means for each factor
    S_W::Union{Nothing,Vector{Matrix{Float64}}}               # Pooled within-cluster cov (only for nested)
    S_B::Union{Nothing,Vector{Vector{Matrix{Float64}}}}       # Per group: list of between-cluster cov for each factor
    s::Union{Nothing,Vector{Vector{Float64}}}                 # Per group: list of scaling factors for each factor
    X::Union{Nothing,Vector{Matrix{Float64}}}                 # Raw data (optional, for crossed)
    cluster_idx::Union{Nothing,Vector{Vector{Vector{Int}}}}   # Raw indices (optional, for crossed)
end

# ─── LISREL Block ─────────────────────────────────────────────────────────────

"""
    LISRELBlock

LISREL matrix system for one block (group × level).

The model-implied covariance is:
    Σ(θ) = Λ (I-B)⁻¹ Ψ (I-B)⁻ᵀ Λᵀ + Θ

The model-implied mean vector is:
    μ(θ) = ν + Λ (I-B)⁻¹ α

Matrix dimensions:
    p = number of observed variables
    q = number of latent variables
    Lambda : p × q
    Theta  : p × p (symmetric)
    Psi    : q × q (symmetric)
    Beta   : q × q (lower triangular; structural paths among latents)
    Nu     : p      (observed intercepts)
    Alpha  : q      (latent intercepts)
"""
mutable struct LISRELBlock
    # Template matrices (filled from θ via free_map)
    Lambda::Matrix{Float64}    # p × q  factor loadings
    Theta::Matrix{Float64}     # p × p  residual covariances of observed
    Psi::Matrix{Float64}       # q × q  latent covariances
    Beta::Matrix{Float64}      # q × q  structural regression coefficients
    Nu::Vector{Float64}        # p      observed intercepts
    Alpha::Vector{Float64}     # q      latent intercepts

    # Fixed-value mask: true if element is fixed (not in θ)
    Lambda_free::Matrix{Bool}
    Theta_free::Matrix{Bool}
    Psi_free::Matrix{Bool}
    Beta_free::Matrix{Bool}
    Nu_free::Vector{Bool}
    Alpha_free::Vector{Bool}

    # Maps each free parameter index → (matrix_symbol, row, col)
    # Length = total nfree for this block
    free_map::Vector{Tuple{Symbol,Int,Int}}   # (:lambda/:theta/etc, row, col)
    free_idx::Vector{Int}                     # which θ index each map entry uses

    # Dimension info
    p::Int   # observed vars
    q::Int   # latent vars
end

# ─── LavaanModel ──────────────────────────────────────────────────────────────

"""
    LavaanModel

The LISREL-parameterized model: one `LISRELBlock` per group/level block.
"""
struct LavaanModel
    blocks::Vector{LISRELBlock}
    nblocks::Int
    nfree::Int                   # total free parameters across all blocks
    ov_names::Vector{String}
    lv_names::Vector{String}
    en_names::Vector{String}     # endogenous variable names
    ex_names::Vector{String}     # exogenous latent variable names
    partable::ParTable
    options::LavaanOptions
end

# ─── ImpliedMoments ───────────────────────────────────────────────────────────

"""
    ImpliedMoments

Model-implied covariance matrices and mean vectors at a given θ.
"""
struct ImpliedMoments
    Sigma::Vector{Matrix{Float64}}    # model-implied cov per block
    Mu::Vector{Vector{Float64}}       # model-implied mean per block
end

# ─── LavaanFit ────────────────────────────────────────────────────────────────

"""
    LavaanFit

The main result object returned by `cfa()`, `sem()`, `lavaan()`, etc.

Fields are populated in pipeline order; see `src/api.jl`.
"""
mutable struct LavaanFit
    # ── inputs ─────────────────────────────────────────────────────────────
    model_string::String
    options::LavaanOptions

    # ── pipeline stages ────────────────────────────────────────────────────
    partable::ParTable
    data::LavaanData
    samplestats::SampleStats
    model::LavaanModel
    implied::ImpliedMoments
    vcov_theta::Matrix{Float64}    # parameter covariance matrix (nfree × nfree)

    # ── results ────────────────────────────────────────────────────────────
    fit_measures::Dict{Symbol,Float64}
    baseline::Union{LavaanFit,Nothing}   # independence model (for CFI/TLI)
    optim_result::Any                    # raw Optim.jl result

    # ── status ─────────────────────────────────────────────────────────────
    converged::Bool
    iterations::Int
    warnings::Vector{String}
    timing::Dict{Symbol,Float64}
end
