using Test
using Lavaan
using DataFrames

@testset "SAM (Structural After Measurement)" begin
    # ── Simple SEM model ──────────────────────────────────────────────────────
    # Holzinger & Swineford (1939) with a structural path
    df = holzinger_swineford()
    
    model = """
      # measurement model
        visual  =~ x1 + x2 + x3
        textual =~ x4 + x5 + x6
        speed   =~ x7 + x8 + x9

      # regression
        speed ~ visual + textual
    """

    # Fit using standard SEM
    fit_sem = sem(model, df)
    
    # Fit using SAM (Local)
    fit_sam = sam(model, df)

    # ── Basic checks ─────────────────────────────────────────────────────────
    @test fit_sam.converged
    
    pe_sem = parameterEstimates(fit_sem)
    pe_sam = parameterEstimates(fit_sam)
    
    # Get structural paths f_speed ~ f_visual
    # (SAM uses f_* prefix in current Step 2 implementation)
    beta_sem = pe_sem[(pe_sem.lhs .== "speed") .& (pe_sem.op .== "~") .& (pe_sem.rhs .== "visual"), :est][1]
    beta_sam = pe_sam[(pe_sam.lhs .== "f_speed") .& (pe_sam.op .== "~") .& (pe_sam.rhs .== "f_visual"), :est][1]
    
    # SAM uses synthetic data for Step 2 — allow wider tolerance (0.3)
    @test isapprox(beta_sam, beta_sem, atol=0.3)

    # Check f_speed ~ f_textual
    gamma_sem = pe_sem[(pe_sem.lhs .== "speed") .& (pe_sem.op .== "~") .& (pe_sem.rhs .== "textual"), :est][1]
    gamma_sam = pe_sam[(pe_sam.lhs .== "f_speed") .& (pe_sam.op .== "~") .& (pe_sam.rhs .== "f_textual"), :est][1]
    @test isapprox(gamma_sam, gamma_sem, atol=0.3)
end
