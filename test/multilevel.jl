using Test, Lavaan, DataFrames, Random

@testset "Multilevel SEM Estimation" begin
    # 1. Create mock clustered data with distinct within/between covariance structures
    Random.seed!(123)
    n_clusters = 50
    n_per_cluster = 10
    N = n_clusters * n_per_cluster
    
    # Between-level data (clusters)
    u1 = randn(n_clusters)
    u2 = 0.5 * u1 .+ sqrt(0.75) * randn(n_clusters)
    
    df = DataFrame(
        id = repeat(1:n_clusters, inner=n_per_cluster),
        y1 = zeros(N),
        y2 = zeros(N)
    )
    
    # Within-level data (individuals) + between-level
    for i in 1:N
        c = df.id[i]
        # within-level correlation is different from between
        w1 = randn()
        w2 = -0.3 * w1 + sqrt(0.91) * randn()
        
        df.y1[i] = u1[c] + w1
        df.y2[i] = u2[c] + w2
    end
    
    # 2. Fit a basic multilevel model
    model = """
      level: 1
        y1 ~~ y1
        y2 ~~ y2
        y1 ~~ y2
      level: 2
        y1 ~~ y1
        y2 ~~ y2
        y1 ~~ y2
    """
    
    fit = sem(model, df; cluster="id", information=:expected, fixed_x=false)
    
    @test fit.converged
    @test fit.options.cluster == "id"
    
    # 3. Check parameters
    pt = fit.partable
    
    # We should have parameters for both blocks.
    # We expect y1~~y2 for block 1 (within) to be negative approx -0.3
    # and for block 2 (between) to be positive approx 0.5
    w_idx = findfirst((pt.lhs .== "y1") .& (pt.op .== "~~") .& (pt.rhs .== "y2") .& (pt.block .== 1))
    b_idx = findfirst((pt.lhs .== "y1") .& (pt.op .== "~~") .& (pt.rhs .== "y2") .& (pt.block .== 2))
    
    w_cov = pt.est[w_idx]
    b_cov = pt.est[b_idx]
    
    @test w_cov < 0.0
    @test b_cov > 0.0
end
