# ─── implied.jl ───────────────────────────────────────────────────────────────
# Compute model-implied moments Σ(θ) and μ(θ) from LISREL matrices.
# Mirrors lavaan's lav_lavaan_step12_implied.R + lav_model_implied.R
#
# Full LISREL formula:
#   IB_inv = (I - B)⁻¹
#   Σ(θ)   = Λ IB_inv Ψ IB_invᵀ Λᵀ + Θ
#   μ(θ)   = ν + Λ IB_inv α
#
# For pure CFA (B = 0):
#   IB_inv = I  →  Σ = Λ Ψ Λᵀ + Θ
#   μ = ν + Λ α
#
# IMPORTANT: all operations must be generic over T <: Real so that
# ForwardDiff can differentiate through them.
# ─────────────────────────────────────────────────────────────────────────────

"""
    compute_implied(model, theta) → ImpliedMoments

Compute model-implied covariance matrices and mean vectors from θ.
"""
function compute_implied(model::LavaanModel,
                         theta::AbstractVector{T})::ImpliedMoments where T <: Real
    nblocks = model.nblocks
    Sigma_list = Vector{Matrix{T}}(undef, nblocks)
    Mu_list    = Vector{Vector{T}}(undef, nblocks)

    for g in 1:nblocks
        block = model.blocks[g]
        mats  = fill_block(block, theta)
        Sigma_list[g], Mu_list[g] = _compute_block_implied(mats, block.p, block.q)
    end

    # Downcast to Float64 if T == Float64 (avoids unnecessary allocations)
    if T == Float64
        return ImpliedMoments(
            [Matrix{Float64}(S) for S in Sigma_list],
            [Vector{Float64}(m) for m in Mu_list],
        )
    else
        return ImpliedMoments(
            [Matrix{Float64}(real.(S)) for S in Sigma_list],
            [Vector{Float64}(real.(m)) for m in Mu_list],
        )
    end
end

"""
    compute_implied_typed(model, theta) → (Sigma, Mu) with type T

Type-preserving version used inside the objective function.
Returns Sigma and Mu with the same numeric type as theta (Float64 or Dual).
"""
function compute_implied_typed(model::LavaanModel,
                                theta::AbstractVector{T}) where T <: Real
    nblocks = model.nblocks
    Sigma_list = Vector{Matrix{T}}(undef, nblocks)
    Mu_list    = Vector{Vector{T}}(undef, nblocks)

    for g in 1:nblocks
        block = model.blocks[g]
        mats  = fill_block(block, theta)
        Sigma_list[g], Mu_list[g] = _compute_block_implied(mats, block.p, block.q)
    end

    return Sigma_list, Mu_list
end

function _compute_block_implied(mats::NamedTuple, p::Int, q::Int)
    T = eltype(mats.Lambda)

    Lambda = mats.Lambda   # p × q
    Theta  = mats.Theta    # p × p
    Psi    = mats.Psi      # q × q
    Beta   = mats.Beta     # q × q
    Nu     = mats.Nu       # p
    Alpha  = mats.Alpha    # q

    # Full LISREL path: Σ = Λ (I-B)⁻¹ Ψ (I-B)⁻ᵀ Λᵀ + Θ
    # We always use the full path so that ForwardDiff can trace Beta gradients
    # even when Beta starts at zero.
    if q > 0
        IB     = Matrix{T}(I, q, q) - Beta
        IB_inv = inv(IB)
        IBPIB  = IB_inv * Psi * IB_inv'
        Sigma  = Lambda * IBPIB * Lambda' + Theta
        Mu     = Nu + Lambda * (IB_inv * Alpha)
    else
        Sigma = Theta   # no latent variables
        Mu    = Nu
    end

    # Symmetrize Sigma to prevent numerical drift
    Sigma = (Sigma + Sigma') / 2

    return Sigma, Mu
end
