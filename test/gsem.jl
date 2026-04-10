# ─── Test Poisson GSEM ────────────────────────────────────────────────────────
using Test
using Lavaan
using DataFrames
using Random
using Distributions
using LinearAlgebra

@testset "GSEM types: family and n_quad_points" begin
    opts = LavaanOptions(family=Dict("y1"=>:poisson, "y2"=>:poisson))
    @test opts.family["y1"] == :poisson
    @test opts.family["y2"] == :poisson
    @test opts.n_quad_points == 15   # default

    opts2 = LavaanOptions(n_quad_points=7)
    @test opts2.n_quad_points == 7
end

@testset "GH quadrature: nodes and weights" begin
    # Access internal function via Lavaan
    nodes, weights = Lavaan._gh_nodes_weights(15)
    @test length(nodes) == 15
    @test length(weights) == 15
    # Weights sum to √π (normalisation for exp(-x²) weight)
    @test isapprox(sum(weights), sqrt(π); atol=1e-10)
    # Nodes are symmetric
    @test isapprox(nodes[1], -nodes[end]; atol=1e-12)
    # Centre node is 0
    @test isapprox(nodes[8], 0.0; atol=1e-12)

    nodes7, weights7 = Lavaan._gh_nodes_weights(7)
    @test length(nodes7) == 7
    @test isapprox(sum(weights7), sqrt(π); atol=1e-10)
end

@testset "GSEM: objective callable with :GSEM estimator" begin
    # Pure Poisson CFA: 3 indicators of 1 latent factor
    Random.seed!(42)
    n = 200
    eta = randn(n)            # latent factor
    y1 = [rand(Poisson(exp(0.5 + 0.8*eta[i]))) for i in 1:n]
    y2 = [rand(Poisson(exp(0.3 + 1.0*eta[i]))) for i in 1:n]
    y3 = [rand(Poisson(exp(0.1 + 0.6*eta[i]))) for i in 1:n]
    df = DataFrame(y1=Float64.(y1), y2=Float64.(y2), y3=Float64.(y3))

    model_str = """
        F =~ y1 + y2 + y3
    """
    # fit with :GSEM family — should not throw
    fit = cfa(model_str, df;
              family=Dict("y1"=>:poisson,"y2"=>:poisson,"y3"=>:poisson),
              estimator=:GSEM)
    @test fit !== nothing
    @test !all(isnan.(coef(fit)))
end

@testset "GSEM: pure Poisson CFA — parameter recovery" begin
    # Simulate from: F =~ y1 + y2 + y3  (Poisson indicators)
    # True params: nu = [0.5, 0.3, 0.1], lambda = [1.0, 0.8, 0.6], psi = 1.0
    Random.seed!(123)
    n = 500
    eta = randn(n)
    y1 = [rand(Poisson(exp(0.5 + 1.0*eta[i]))) for i in 1:n]
    y2 = [rand(Poisson(exp(0.3 + 0.8*eta[i]))) for i in 1:n]
    y3 = [rand(Poisson(exp(0.1 + 0.6*eta[i]))) for i in 1:n]
    df = DataFrame(y1=Float64.(y1), y2=Float64.(y2), y3=Float64.(y3))

    model_str = """
        F =~ y1 + y2 + y3
    """
    fit = cfa(model_str, df;
              family=Dict("y1"=>:poisson, "y2"=>:poisson, "y3"=>:poisson))

    pe = parameterEstimates(fit)
    lambdas = pe[pe.op .== "=~", :]

    # y1 loading fixed to 1.0 (identification)
    y1_load = lambdas[lambdas.rhs .== "y1", :est]
    @test !isempty(y1_load)
    @test isapprox(y1_load[1], 1.0; atol=1e-6)

    # y2 loading ≈ 0.8 (true value), allow ±0.20 tolerance (finite sample)
    y2_load = lambdas[lambdas.rhs .== "y2", :est]
    @test !isempty(y2_load)
    @test isapprox(y2_load[1], 0.8; atol=0.20)

    # y3 loading ≈ 0.6
    y3_load = lambdas[lambdas.rhs .== "y3", :est]
    @test !isempty(y3_load)
    @test isapprox(y3_load[1], 0.6; atol=0.20)

    # No NaN in coefficients
    @test !any(isnan.(coef(fit)))
end

@testset "GSEM: mixed Gaussian + Poisson indicators" begin
    # Simulate: F =~ x1 + x2 (Gaussian) + y1 + y2 (Poisson)
    Random.seed!(456)
    n = 400
    eta = randn(n)
    x1 = 0.2 .+ 0.9*eta .+ 0.3*randn(n)
    x2 = 0.1 .+ 0.7*eta .+ 0.3*randn(n)
    y1 = [rand(Poisson(exp(0.4 + 0.8*eta[i]))) for i in 1:n]
    y2 = [rand(Poisson(exp(0.2 + 0.6*eta[i]))) for i in 1:n]
    df = DataFrame(x1=x1, x2=x2, y1=Float64.(y1), y2=Float64.(y2))

    model_str = """
        F =~ x1 + x2 + y1 + y2
    """
    fit = cfa(model_str, df;
              family=Dict("y1"=>:poisson, "y2"=>:poisson))

    @test !any(isnan.(coef(fit)))

    pe = parameterEstimates(fit)
    @test nrow(pe) > 0

    # Gaussian residual variances should be estimated (non-zero)
    resid_x1 = pe[(pe.lhs .== "x1") .& (pe.op .== "~~") .& (pe.rhs .== "x1"), :est]
    @test !isempty(resid_x1)
    @test resid_x1[1] > 0

    # Poisson residual variances should be fixed to 0
    resid_y1 = pe[(pe.lhs .== "y1") .& (pe.op .== "~~") .& (pe.rhs .== "y1"), :est]
    # Poisson vars have no residual variance row (fixed and removed)
    # Either not in output or fixed to 0
    if !isempty(resid_y1)
        @test isapprox(resid_y1[1], 0.0; atol=1e-10)
    end
end

@testset "GSEM: cfa() and sem() pass family through" begin
    Random.seed!(789)
    n = 100
    eta = randn(n)
    y1 = [rand(Poisson(exp(0.3*eta[i]))) for i in 1:n]
    y2 = [rand(Poisson(exp(0.5*eta[i]))) for i in 1:n]
    df = DataFrame(y1=Float64.(y1), y2=Float64.(y2))

    model_str = "F =~ y1 + y2"
    fit_cfa = cfa(model_str, df; family=Dict("y1"=>:poisson,"y2"=>:poisson))
    fit_sem = sem(model_str, df; family=Dict("y1"=>:poisson,"y2"=>:poisson))

    @test !any(isnan.(coef(fit_cfa)))
    @test !any(isnan.(coef(fit_sem)))
end
