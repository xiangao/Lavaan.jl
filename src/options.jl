# ─── options.jl ───────────────────────────────────────────────────────────────
# LavaanOptions is defined in types.jl.
# This file provides helper constructors for common model types.
# ─────────────────────────────────────────────────────────────────────────────

"""
    cfa_options(; kwargs...)

Default options for CFA models.
"""
function cfa_options(; kwargs...)
    LavaanOptions(;
        model_type      = :cfa,
        auto_fix_first  = true,
        auto_var        = true,
        auto_cov_lv_x   = true,
        auto_cov_y      = false,
        kwargs...
    )
end

"""
    sem_options(; kwargs...)

Default options for SEM models.
"""
function sem_options(; kwargs...)
    LavaanOptions(;
        model_type      = :sem,
        auto_fix_first  = true,
        auto_var        = true,
        auto_cov_lv_x   = true,
        auto_cov_y      = true,
        kwargs...
    )
end

"""
    growth_options(; kwargs...)

Default options for latent growth curve models.
"""
function growth_options(; kwargs...)
    LavaanOptions(;
        model_type      = :growth,
        meanstructure   = true,
        auto_fix_first  = false,   # growth models fix loadings explicitly
        int_ov_free     = false,
        int_lv_free     = true,
        auto_var        = true,
        auto_cov_lv_x   = true,
        kwargs...
    )
end
