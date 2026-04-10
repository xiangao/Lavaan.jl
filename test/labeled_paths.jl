# ─── test/labeled_paths.jl ────────────────────────────────────────────────────
# Tests for labeled path syntax (a * x) and := defined parameters.
# ─────────────────────────────────────────────────────────────────────────────
using Test, Lavaan, DataFrames

@testset "Labeled path syntax: a * x parsed as label + variable" begin
    df = holzinger_swineford()
    model = """
      visual  =~ x1 + x2 + x3
      textual =~ x4 + x5 + x6
      speed   =~ x7 + x8 + x9
      textual ~ a * visual
      speed   ~ b * textual + c * visual
    """
    fit = sem(model, df)
    @test fit.converged

    pe = parameterEstimates(fit)
    reg = pe[pe.op .== "~", :]

    # Labels correctly assigned
    @test "a" in reg.label
    @test "b" in reg.label
    @test "c" in reg.label

    # Estimates not zero (model actually ran)
    a_est = reg[reg.label .== "a", :est][1]
    b_est = reg[reg.label .== "b", :est][1]
    c_est = reg[reg.label .== "c", :est][1]
    @test abs(a_est) > 0.01
    @test abs(c_est) > 0.01
end

@testset "Defined parameters (:=) with delta-method SEs" begin
    df = holzinger_swineford()
    model = """
      visual  =~ x1 + x2 + x3
      textual =~ x4 + x5 + x6
      speed   =~ x7 + x8 + x9
      textual ~ a * visual
      speed   ~ b * textual + c * visual
      indirect := a * b
      total    := indirect + c
    """
    fit = sem(model, df)
    @test fit.converged

    pe = parameterEstimates(fit)
    def = pe[pe.op .== ":=", :]

    # Both defined params present
    @test "indirect" in def.lhs
    @test "total" in def.lhs

    indirect = def[def.lhs .== "indirect", :][1, :]
    total    = def[def.lhs .== "total",    :][1, :]

    # Estimates match R lavaan reference values
    @test isapprox(indirect.est, 0.0269; atol=0.005)
    @test isapprox(total.est,    0.3240; atol=0.005)

    # Delta-method SEs are non-NaN and positive
    @test !isnan(indirect.se) && indirect.se > 0
    @test !isnan(total.se)    && total.se > 0

    # Chained definition: total = indirect + c → test numerically
    c_est = pe[(pe.op .== "~") .& (pe.label .== "c"), :est][1]
    @test isapprox(total.est, indirect.est + c_est; atol=1e-8)
end
