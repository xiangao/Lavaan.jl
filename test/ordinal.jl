# ─── test/ordinal.jl ──────────────────────────────────────────────────────────
using Test, Lavaan, Distributions, Random, DataFrames

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

@testset "fit_thresholds: 4-category variable" begin
    # Simulate ordinal data: y=1..4 with P(y=k) = 0.25 each
    # True thresholds: qnorm([0.25, 0.50, 0.75]) ≈ [-0.674, 0.0, 0.674]
    Random.seed!(42)
    n = 2000
    u = rand(n)
    y = ones(Int, n)
    for i in 1:n
        if u[i] < 0.25; y[i] = 1
        elseif u[i] < 0.50; y[i] = 2
        elseif u[i] < 0.75; y[i] = 3
        else; y[i] = 4
        end
    end

    fit = Lavaan.fit_thresholds(y)
    @test fit.converged
    @test fit.nth == 3
    @test fit.y_ncat == 4
    @test isapprox(fit.theta[1], quantile(Normal(), 0.25); atol=0.05)
    @test isapprox(fit.theta[2], quantile(Normal(), 0.50); atol=0.05)
    @test isapprox(fit.theta[3], quantile(Normal(), 0.75); atol=0.05)
end

@testset "fit_thresholds: binary variable" begin
    Random.seed!(7)
    n = 1000
    y = rand(1:2, n)  # 50/50 binary
    fit = Lavaan.fit_thresholds(y)
    @test fit.converged
    @test fit.nth == 1
    @test isapprox(fit.theta[1], 0.0; atol=0.1)
end

@testset "polychoric_cor: recovers known ρ=0.6" begin
    Random.seed!(123)
    n = 1000
    rho_true = 0.6
    z1 = randn(n)
    z2 = rho_true .* z1 .+ sqrt(1 - rho_true^2) .* randn(n)
    cuts = quantile(Normal(), [0.25, 0.5, 0.75])
    to_ord(z) = [sum(z[i] .>= cuts) + 1 for i in 1:length(z)]
    y1, y2 = to_ord(z1), to_ord(z2)

    rho_hat = Lavaan.polychoric_cor(y1, y2)
    @test isapprox(rho_hat, rho_true; atol=0.06)
end

@testset "polychoric_cor: negative correlation ρ=-0.5" begin
    Random.seed!(99)
    n = 800
    rho_true = -0.5
    z1 = randn(n)
    z2 = rho_true .* z1 .+ sqrt(1 - rho_true^2) .* randn(n)
    cuts = quantile(Normal(), [1/3, 2/3])
    to_ord(z) = [sum(z[i] .>= cuts) + 1 for i in 1:length(z)]
    y1, y2 = to_ord(z1), to_ord(z2)

    rho_hat = Lavaan.polychoric_cor(y1, y2)
    @test isapprox(rho_hat, rho_true; atol=0.08)
end

@testset "vech_pos: position mapping" begin
    # p=3: vech order is (1,1),(2,1),(3,1),(2,2),(3,2),(3,3)
    @test Lavaan.vech_pos(3, 1, 1) == 1
    @test Lavaan.vech_pos(3, 2, 1) == 2
    @test Lavaan.vech_pos(3, 3, 1) == 3
    @test Lavaan.vech_pos(3, 2, 2) == 4
    @test Lavaan.vech_pos(3, 3, 2) == 5
    @test Lavaan.vech_pos(3, 3, 3) == 6
end

@testset "compute_polychoric_matrix: 3-variable case" begin
    Random.seed!(55)
    n = 500
    # True correlations: ρ₁₂=0.6, ρ₁₃=0.4, ρ₂₃=0.5
    Σ_true = [1.0 0.6 0.4; 0.6 1.0 0.5; 0.4 0.5 1.0]
    L = cholesky(Σ_true).L
    Z = (L * randn(3, n))'  # n×3
    cuts = quantile(Normal(), [0.33, 0.67])
    to_ord(z) = [sum(z[i] .>= cuts) + 1 for i in 1:length(z)]

    ov = ["y1","y2","y3"]
    X = Float64.(hcat(to_ord(Z[:,1]), to_ord(Z[:,2]), to_ord(Z[:,3])))

    R, Gamma_d = Lavaan.compute_polychoric_matrix(X, ov, ov)

    # Diagonal = 1
    @test all(isapprox.(diag(R), 1.0; atol=1e-10))
    # Off-diagonal close to true values
    @test isapprox(R[2,1], 0.6; atol=0.10)
    @test isapprox(R[3,1], 0.4; atol=0.10)
    @test isapprox(R[3,2], 0.5; atol=0.10)
    # Gamma_diag: length = p*(p+1)/2, all positive
    @test length(Gamma_d) == 6
    @test all(Gamma_d .> 0)
end

@testset "ordinal CFA: end-to-end DWLS fit" begin
    # True model: 1 factor, 4 indicators, all loadings = 0.7
    # Residual variance = 1 - 0.7² = 0.51
    Random.seed!(2024)
    n = 800
    lambda_true = 0.7
    eta = randn(n)
    eps = randn(n, 4) .* sqrt(1 - lambda_true^2)
    Y_cont = lambda_true .* eta .+ eps   # n×4

    # Discretize to 4 categories at qnorm([0.25, 0.5, 0.75])
    cuts = quantile(Normal(), [0.25, 0.5, 0.75])
    to_ord(z) = [sum(z[i] .>= cuts) + 1 for i in 1:length(z)]

    df = DataFrame(
        y1 = to_ord(Y_cont[:,1]),
        y2 = to_ord(Y_cont[:,2]),
        y3 = to_ord(Y_cont[:,3]),
        y4 = to_ord(Y_cont[:,4]),
    )

    model_str = "f =~ y1 + y2 + y3 + y4"

    fit = cfa(model_str, df; ordered=["y1","y2","y3","y4"])

    @test fit.converged

    pe = parameterEstimates(fit)

    # Threshold rows should be present (4 vars × 3 thresholds each = 12)
    thresh_rows = pe[pe.op .== "|", :]
    @test nrow(thresh_rows) == 12

    # Loading rows
    load_rows = pe[pe.op .== "=~", :]
    @test nrow(load_rows) == 4

    # The free loadings (y2,y3,y4 — y1 fixed to 1.0) should be close to each other
    # All loadings proportional to 0.7; since auto_fix_first=true, y1 loading = 1.0 (fixed)
    # Free loadings for y2,y3,y4 should all be ≈ 1.0 (since all true lambdas equal)
    free_loads = load_rows[load_rows.free .> 0, :est]
    @test length(free_loads) == 3
    @test all(isapprox.(free_loads, 1.0; atol=0.25))

    # Fit measures exist
    fm = fitMeasures(fit)
    @test haskey(fm, :chisq)
    @test haskey(fm, :df)
    @test fm[:df] >= 0.0
end

@testset "ordinal CFA: threshold output in parameterEstimates" begin
    Random.seed!(77)
    n = 600
    # Simple binary factor model
    eta = randn(n)
    y1_cont = 0.6 .* eta .+ sqrt(1 - 0.36) .* randn(n)
    y2_cont = 0.7 .* eta .+ sqrt(1 - 0.49) .* randn(n)

    # Discretize to 3 categories at qnorm([1/3, 2/3])
    cuts3 = quantile(Normal(), [1/3, 2/3])
    to_ord3(z) = [sum(z[i] .>= cuts3) + 1 for i in 1:length(z)]

    df2 = DataFrame(f1 = to_ord3(y1_cont), f2 = to_ord3(y2_cont))

    fit2 = cfa("g =~ f1 + f2", df2; ordered=["f1","f2"])

    pe2 = parameterEstimates(fit2)
    th = pe2[pe2.op .== "|", :]

    # 2 vars × 2 thresholds = 4 rows
    @test nrow(th) == 4

    # Threshold estimates should be close to probit(1/3) ≈ -0.431 and probit(2/3) ≈ 0.431
    th_f1 = th[th.lhs .== "f1", :]
    @test isapprox(th_f1.est[1], quantile(Normal(), 1/3); atol=0.10)
    @test isapprox(th_f1.est[2], quantile(Normal(), 2/3); atol=0.10)

    # SEs should be positive and finite
    @test all(th.se .> 0)
    @test all(isfinite.(th.se))
end
