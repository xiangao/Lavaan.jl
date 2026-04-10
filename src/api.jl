# ─── api.jl ───────────────────────────────────────────────────────────────────
# Public entry points: lavaan(), cfa(), sem(), growth()
# Mirrors lavaan's lav_lavaan.R — the 17-step pipeline orchestrator.
# ─────────────────────────────────────────────────────────────────────────────

"""
    lavaan(model_string, data; kwargs...) → LavaanFit

Fit a general SEM model. All options are passed as keyword arguments.

# Arguments
- `model_string` : lavaan syntax string (e.g. `"y =~ x1 + x2 + x3"`)
- `data`         : DataFrame with observed variable columns
- `group`        : column name for multi-group analysis (optional)
- `estimator`    : `:ML`, `:MLR`, `:GLS`, `:ULS`, etc.  (default `:ML`)
- `se`           : `:standard`, `:robust_sem`, `:none`, etc. (default `:standard`)
- `missing`      : `:listwise` or `:fiml` (default `:listwise`)
- `meanstructure`: `true`/`false` (default `false`)
- `std_lv`       : standardize latent variables (default `false`)

# Returns
A `LavaanFit` object. Use `summary(fit)`, `parameterEstimates(fit)`,
`fitMeasures(fit)` to inspect results.

# Example
```julia
model = \"\"\"
  visual  =~ x1 + x2 + x3
  textual =~ x4 + x5 + x6
  speed   =~ x7 + x8 + x9
\"\"\"
HS = holzinger_swineford()
fit = cfa(model, HS)
summary(fit, fit_measures=true)
```
"""
function lavaan(model_string::String,
                data::DataFrame;
                group::Union{String,Nothing} = nothing,
                cluster::Union{String,Symbol,Nothing,Vector{String}} = nothing,
                ordered::Vector{String} = String[],
                estimator::Symbol  = :ML,
                se::Symbol         = :standard,
                test::Vector{Symbol} = [:standard],
                missing_method::Symbol = :listwise,
                meanstructure::Bool = false,
                fixed_x::Bool = true,
                std_lv::Bool = false,
                int_ov_free::Bool = false,
                int_lv_free::Bool = false,
                auto_fix_first::Bool = true,
                auto_fix_single::Bool = true,
                auto_var::Bool = true,
                auto_cov_lv_x::Bool = true,
                do_fit::Bool = true,
                optim_tol::Float64 = 1e-7,
                optim_iter::Int = 10_000,
                information::Symbol = :expected,
                verbose::Bool = false,
                debug::Bool = false,
                model_type::Symbol = :sem,
                nboot::Int = 1000,
                family::Dict{String,Symbol} = Dict{String,Symbol}(),
                n_quad_points::Int = 15,
                sam::Bool = false,
                sam_method::Symbol = :local,
                )::LavaanFit

    cluster_list = if cluster === nothing
        String[]
    elseif cluster isa Vector
        string.(cluster)
    else
        [string(cluster)]
    end

    opts = LavaanOptions(
        estimator       = estimator,
        se              = se,
        ordered         = ordered,
        clusters        = cluster_list,
        cluster         = isempty(cluster_list) ? nothing : cluster_list[1],
        test            = test,
        missing_method  = missing_method,
        meanstructure   = meanstructure,
        fixed_x         = fixed_x,
        std_lv          = std_lv,
        int_ov_free     = int_ov_free,
        int_lv_free     = int_lv_free,
        auto_fix_first  = auto_fix_first,
        auto_fix_single = auto_fix_single,
        auto_var        = auto_var,
        auto_cov_lv_x   = auto_cov_lv_x,
        do_fit          = do_fit,
        optim_tol       = optim_tol,
        optim_iter      = optim_iter,
        information     = information,
        verbose         = verbose,
        debug           = debug,
        model_type      = model_type,
        nboot           = nboot,
        family          = family,
        n_quad_points   = n_quad_points,
        sam             = sam,
        sam_method      = sam_method,
    )

    if opts.sam
        return run_sam_pipeline(model_string, data, opts; group=group)
    end

    return _lavaan_pipeline(model_string, data, opts; group=group)
end

"""
    sam(model_string, data; sam_method=:local, kwargs...) → LavaanFit

Structural After Measurement (SAM) approach for robust SEM estimation.
Rosseel & Loh (2022).
"""
function sam(model_string::String, data::DataFrame; sam_method::Symbol=:local, kwargs...)::LavaanFit
    return lavaan(model_string, data; sam=true, sam_method=sam_method, kwargs...)
end

"""
    cfa(model_string, data; kwargs...) → LavaanFit

Confirmatory Factor Analysis. Thin wrapper around `lavaan()` with
`model_type=:cfa` defaults (auto_var=true, auto_cov_lv_x=true).
"""
function cfa(model_string::String, data::DataFrame; kwargs...)::LavaanFit
    return lavaan(model_string, data; model_type=:cfa, kwargs...)
end

"""
    sem(model_string, data; kwargs...) → LavaanFit

Full Structural Equation Model. Thin wrapper around `lavaan()`.
"""
function sem(model_string::String, data::DataFrame; kwargs...)::LavaanFit
    return lavaan(model_string, data; model_type=:sem, kwargs...)
end

"""
    growth(model_string, data; kwargs...) → LavaanFit

Latent Growth Curve Model. Thin wrapper around `lavaan()` with
`meanstructure=true` (required for growth models).
"""
function growth(model_string::String, data::DataFrame; kwargs...)::LavaanFit
    return lavaan(model_string, data;
                  model_type=:growth,
                  meanstructure=true,
                  int_ov_free=false,
                  int_lv_free=true,
                  kwargs...)
end

# ─── Pipeline orchestrator ────────────────────────────────────────────────────

function _lavaan_pipeline(model_string::String,
                           data::DataFrame,
                           opts::LavaanOptions;
                           group::Union{String,Nothing} = nothing)::LavaanFit

    warnings = String[]
    timing   = Dict{Symbol,Float64}()
    t0 = time()

    # ── step 01: Parse model string → ParsedRow[] ────────────────────────────
    t1 = time()
    parsed_rows = parse_model_string(model_string)
    timing[:parse] = time() - t1

    # ── step 02: Derive variable names ───────────────────────────────────────
    lv_names = lv_names_from_rows(parsed_rows)
    ov_names = ov_names_from_rows(parsed_rows, lv_names)

    # ── step 03: Validate data ────────────────────────────────────────────────
    t1 = time()
    check_ov_names_in_data(ov_names, data)
    timing[:validate] = time() - t1

    # ── step 04: Build ParTable ───────────────────────────────────────────────
    t1 = time()
    ngroups = group === nothing ? 1 : length(unique(skipmissing(data[!, group])))
    pt = build_partable(parsed_rows, ov_names, lv_names, opts; ngroups=ngroups)
    _fix_poisson_residuals!(pt, opts.family)
    timing[:partable] = time() - t1

    # ── step 05: Prepare data + sample statistics ─────────────────────────────
    # Auto-switch estimator: ordinal → DWLS; Poisson family → GSEM
    if !isempty(opts.ordered) && opts.estimator == :ML
        opts = LavaanOptions(; [f => getfield(opts, f) for f in fieldnames(LavaanOptions)]...,
                             estimator=:DWLS)
    end
    has_poisson = any(v == :poisson for v in values(opts.family))
    if has_poisson && opts.estimator ∉ (:GSEM,)
        opts = LavaanOptions(; [f => getfield(opts, f) for f in fieldnames(LavaanOptions)]...,
                             estimator=:GSEM,
                             meanstructure=true,
                             int_ov_free=true)
    end

    t1 = time()
    lavdata = prepare_data(data, ov_names, lv_names;
                           group=group, cluster=opts.clusters, missing_method=opts.missing_method,
                           ordered=opts.ordered)
    stats = compute_samplestats(lavdata, opts)
    timing[:samplestats] = time() - t1

    # Inject threshold rows into ParTable for ordinal output
    if !isempty(opts.ordered) && stats.thresholds !== nothing
        _inject_threshold_rows!(pt, stats.thresholds, stats.threshold_ses)
    end

    # ── step 06: Input validation ─────────────────────────────────────────────
    validation_warnings = validate_model_data(pt, lavdata, opts)
    append!(warnings, validation_warnings)

    # ── step 07: Build LISREL model ───────────────────────────────────────────
    t1 = time()
    model = build_model(pt, ov_names, lv_names, opts)
    timing[:model] = time() - t1

    # ── step 08: Starting values ──────────────────────────────────────────────
    t1 = time()
    set_starting_values!(pt, stats, ov_names, lv_names, opts)
    timing[:start] = time() - t1

    # ── step 09: Optimization ─────────────────────────────────────────────────
    t1 = time()
    converged = false
    iters = 0
    optim_result = nothing
    theta_hat = get_start(pt)

    if opts.do_fit && model.nfree > 0
        theta_hat, converged, iters, optim_result =
            optimize_model(model, stats, opts)
    else
        converged = true    # nothing to optimize
    end
    timing[:optim] = time() - t1

    # Write θ̂ back to parTable
    set_est!(pt, theta_hat)

    # ── step 10: Model-implied moments ────────────────────────────────────────
    t1 = time()
    implied = compute_implied(model, theta_hat)
    timing[:implied] = time() - t1

    # ── step 11: vcov + SEs ───────────────────────────────────────────────────
    t1 = time()
    vcov_theta = compute_vcov(model, stats, opts, lavdata, theta_hat, converged, warnings)
    fill_ses!(pt, theta_hat, vcov_theta)
    compute_defined_params!(pt, vcov_theta)
    timing[:vcov] = time() - t1

    # ── Assemble LavaanFit (before fit measures, to allow fit.baseline) ───────
    fit = LavaanFit(
        model_string,
        opts,
        pt,
        lavdata,
        stats,
        model,
        implied,
        vcov_theta,
        Dict{Symbol,Float64}(),   # fit_measures — filled next
        nothing,                  # baseline — filled inside compute_fit_measures!
        optim_result,
        converged,
        iters,
        warnings,
        timing,
    )

    # ── step 12: Fit measures ──────────────────────────────────────────────────
    t1 = time()
    compute_fit_measures!(fit)
    timing[:fit] = time() - t1

    timing[:total] = time() - t0

    if opts.verbose
        @info "lavaan fit complete in $(round(timing[:total]*1000, digits=1)) ms" *
              " (optim: $(round(timing[:optim]*1000, digits=1)) ms)"
    end

    return fit
end
