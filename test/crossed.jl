using Test, Lavaan, DataFrames, Random, LinearAlgebra

@testset "Crossed Random Effects SEM" begin
    # 1. Simulate crossed data (Students in Schools and Neighborhoods)
    Random.seed!(123)
    n_schools = 10
    n_neighborhoods = 10
    n_students = 100
    
    school_ids = rand(1:n_schools, n_students)
    hood_ids = rand(1:n_neighborhoods, n_students)
    
    # Between effects
    u_school = randn(n_schools)
    u_hood = randn(n_neighborhoods)
    
    df = DataFrame(
        student = 1:n_students,
        school = school_ids,
        hood = hood_ids,
        y1 = zeros(n_students)
    )
    
    for i in 1:n_students
        df.y1[i] = u_school[df.school[i]] + u_hood[df.hood[i]] + randn()
    end
    
    # 2. Fit a model with two crossed levels
    # level: 1 (within student)
    # level: 2 (school)
    # level: 3 (hood)
    model = """
      level: 1
        y1 ~~ y1
      level: 2
        y1 ~~ y1
      level: 3
        y1 ~~ y1
    """
    
    # Passing multiple clusters
    # Note: Currently Lavaan.jl syntax parser maps 'level: name' to blocks sequentially.
    # We need to make sure the parser handles more than 2 levels.
    
    fit = sem(model, df; cluster=["school", "hood"], fixed_x=false)
    
    @test fit.converged
    @test length(fit.options.clusters) == 2
    
    pt = fit.partable
    # We expect 3 variance parameters
    @test count(pt.op .== "~~") >= 3
    
    # Check that estimates are positive
    @test all(pt.est[pt.op .== "~~"] .>= 0.0)
end
