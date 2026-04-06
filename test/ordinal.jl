# ─── test/ordinal.jl ──────────────────────────────────────────────────────────
using Test, Lavaan, Distributions, Random

@testset "bivnorm_cdf: boundary cases" begin
    # Independence: P(Z1≤0, Z2≤0; ρ=0) = Φ(0)² = 0.25
    @test isapprox(Lavaan.bivnorm_cdf(0.0, 0.0, 0.0), 0.25; atol=1e-10)

    # Marginal: P(Z1≤∞, Z2≤k) = Φ(k)
    @test isapprox(Lavaan.bivnorm_cdf(Inf, 0.0, 0.5), cdf(Normal(), 0.0); atol=1e-10)
    @test isapprox(Lavaan.bivnorm_cdf(0.0, Inf, -0.3), cdf(Normal(), 0.0); atol=1e-10)
    @test isapprox(Lavaan.bivnorm_cdf(Inf, Inf, 0.7), 1.0; atol=1e-10)

    # Lower bound
    @test isapprox(Lavaan.bivnorm_cdf(-Inf, 0.0, 0.5), 0.0; atol=1e-10)
    @test isapprox(Lavaan.bivnorm_cdf(0.0, -Inf, 0.5), 0.0; atol=1e-10)
end

@testset "bivnorm_cdf: known analytic values" begin
    # P(Z1≤0, Z2≤0; ρ) = 0.25 + arcsin(ρ)/(2π) — exact formula for h=k=0
    # 20-point GL quadrature achieves ~1e-5 accuracy at h=k=0 (upper bound=0.5)
    for rho in [-0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8]
        expected = 0.25 + asin(rho) / (2π)
        @test isapprox(Lavaan.bivnorm_cdf(0.0, 0.0, rho), expected; atol=1e-4)
    end

    # Independence: P(Z1≤a, Z2≤b; ρ=0) = Φ(a)*Φ(b)
    for (a, b) in [(1.0, 2.0), (-1.0, 0.5), (0.5, -0.5)]
        @test isapprox(Lavaan.bivnorm_cdf(a, b, 0.0),
                       cdf(Normal(), a) * cdf(Normal(), b); atol=1e-8)
    end
end
