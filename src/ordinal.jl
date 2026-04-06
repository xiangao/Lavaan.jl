# ─── ordinal.jl ────────────────────────────────────────────────────────────────
# Ordinal variable support: polychoric correlations + threshold estimation.
# Mirrors lavaan's lav_uvord.R (thresholds) + lav_bvord.R (polychoric).
# ─────────────────────────────────────────────────────────────────────────────

using Distributions: Normal, cdf, pdf, quantile

# ─── 20-point Gauss-Legendre nodes on [-1, 1] ────────────────────────────────
# 10 symmetric pairs (±xᵢ, same weight wᵢ)

const _GL20_NODES = Float64[
    -0.9931285991850949,  0.9931285991850949,
    -0.9639719272779138,  0.9639719272779138,
    -0.9122344282513259,  0.9122344282513259,
    -0.8391169718222188,  0.8391169718222188,
    -0.7463062256567498,  0.7463062256567498,
    -0.6360536807265150,  0.6360536807265150,
    -0.5108670019508271,  0.5108670019508271,
    -0.3737060887154196,  0.3737060887154196,
    -0.2277858511416451,  0.2277858511416451,
    -0.0765265211334973,  0.0765265211334973,
]

const _GL20_WEIGHTS = Float64[
    0.0176140071391521,  0.0176140071391521,
    0.0406014298003869,  0.0406014298003869,
    0.0626720483341091,  0.0626720483341091,
    0.0832767415767047,  0.0832767415767047,
    0.1019301198172404,  0.1019301198172404,
    0.1181945319615184,  0.1181945319615184,
    0.1316886384491766,  0.1316886384491766,
    0.1420961093183821,  0.1420961093183821,
    0.1491729864726038,  0.1491729864726038,
    0.1527533871307259,  0.1527533871307259,
]

const _NORM = Normal()

"""
    bivnorm_cdf(h, k, rho) → Float64

Bivariate standard normal CDF: P(X₁ ≤ h, X₂ ≤ k) where Corr(X₁,X₂) = ρ.

Uses 20-point Gauss-Legendre on the 1D integral representation:
    P = ∫₀^{Φ(h)} Φ((k − ρ·Φ⁻¹(u)) / √(1−ρ²)) du
"""
function bivnorm_cdf(h::Real, k::Real, rho::Real)::Float64
    h, k, rho = Float64(h), Float64(k), Float64(rho)

    # ── Boundary / degenerate cases ───────────────────────────────────────────
    (h == -Inf || k == -Inf) && return 0.0
    (h ==  Inf && k ==  Inf) && return 1.0
    h ==  Inf && return cdf(_NORM, k)
    k ==  Inf && return cdf(_NORM, h)
    rho ==  1.0 && return cdf(_NORM, min(h, k))
    rho == -1.0 && return max(cdf(_NORM, h) + cdf(_NORM, k) - 1.0, 0.0)
    rho ==  0.0 && return cdf(_NORM, h) * cdf(_NORM, k)

    rho = clamp(rho, -1.0 + 1e-10, 1.0 - 1e-10)

    # Upper limit of integration in u-space
    ub = cdf(_NORM, h)
    ub <= 0.0 && return 0.0

    sqrt1r2 = sqrt(1.0 - rho^2)

    # Gauss-Legendre on [0, ub]: ∫₀^{ub} f(u) du ≈ (ub/2) Σᵢ wᵢ f((ub/2)(1+xᵢ))
    half_ub = ub / 2.0
    result = 0.0
    @inbounds for i in eachindex(_GL20_NODES)
        u = half_ub * (1.0 + _GL20_NODES[i])
        u = clamp(u, 1e-15, 1.0 - 1e-15)
        x = quantile(_NORM, u)
        result += _GL20_WEIGHTS[i] * cdf(_NORM, (k - rho * x) / sqrt1r2)
    end
    return half_ub * result
end
