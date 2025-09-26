using LinearAlgebra

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

"""
    shift(A, B, σ)

Return the shifted matrix Aσ = A - σB.
This is the core step in the shift-and-invert strategy.
"""
function shift(A::AbstractMatrix{E}, B::AbstractMatrix{E}, σ) where {E<:AbstractFloat}
    C = similar(A)
    for j in axes(A,2), i in axes(A,1)
        C[i,j] = fma(-σ, B[i,j], A[i,j])
    end
    return C
end

function shift(A::AbstractMatrix{Complex{E}}, B::AbstractMatrix{Complex{E}}, σ) where {E<:AbstractFloat}
    C = similar(A)
    for j in axes(A,2), i in axes(A,1)
        C[i,j] = complex(
            fma(-σ, real(B[i,j]), real(A[i,j])),
            fma(-σ, imag(B[i,j]), imag(A[i,j])),
        )
    end
    return C
end

# Custom error type for bounding ||X||
struct EtaXError{T} <: Exception
    etax :: T
    bound :: T
end

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

# TODO: explore different tolerances for pivoted Cholesky and pseudoinverse

"""
    eig_spectral_trans(A, B, σ; method=:LQD, tol=0, pinv_tol=0, ηx_max=500.0)

Compute generalized eigenpairs of Hermitian matrices (A,B) with B ≽ 0,
using a shift-and-invert spectral transform with shift σ.

The algorithm is:

1. Form Aσ = A - σB.
2. Pivoted Cholesky: B ≈ Cb*Cbᵀ.
   - `tol` handles numerical stability issues with problematic pivots.
3. Factorize Aσ using one of:
   - `:LQD`  → custom LQD factorization
   - `:LDLt` → Bunch–Kaufman (symmetric-indefinite)
   - `:LU`   → LU factorization
   - `:Eig`  → full eigendecomposition
4. Apply Aσ⁻¹ to Cb:
   - For `:Eig`, use the spectral identity
         Aσ⁻¹ = Q Λ⁻¹ Qᵀ
     with cutoff `pinv_tol`:
         (Λ⁺)_{ii} = 1/λᵢ if |λᵢ| > pinv_tol else 0.
     This is the **Moore–Penrose pseudoinverse** of Aσ, which is why `pinv_tol`
     is used suggestively (also for zero/near zero eigenvalues - numerical
     stability).
   - For the others, solve directly with the factorization.
5. Form reduced operator W = Cbᵀ Aσ⁻¹ Cb.
6. Solve WU = UΘ.
7. Recover original eigenvalues: λ = σ + 1/θ.
8. Recover eigenvectors:
   - `:LQD`: V = (F')⁻¹ (D * (XU)), X = Aσ⁻¹Cb.
   - Others: V = Cb' \ U.

Returns:
(Cb, U, θ, λ, α, β, V, Y, η, D).
"""
function eig_spectral_trans(A, B, σ; method=:LQD, tol=0, pinv_tol=0, ηx_max=500.0)
    # 1. Shift
    Aσ = shift(A,B,σ)

    # 2. Pivoted Cholesky of B
    Fb = cholesky!(Hermitian(B,:L), RowMaximum(), tol=tol, check=false)
    r  = Fb.rank
    ip = invperm(Fb.p)
    Cb = Fb.factors[:,1:r]

    # 3. Factorize Aσ (method-dependent)
    F = nothing
    if method == :LQD
        F = lqd(Hermitian(Aσ,:L))
    elseif method == :LDLt
        F = bunchkaufman!(Hermitian(Aσ,:L))
    elseif method == :LU
        F = lu!(Aσ)
    elseif method == :Eig
        F = eigen!(Hermitian(Aσ,:L))
    else
        throw(ArgumentError("Unknown method: $method"))
    end

    # 4. Apply inverse once: Y = Aσ⁻¹ * Cb
    Y = copy(Cb)
    if method == :Eig
        # Spectral decomposition: Aσ = Q Λ Qᵀ
        # Apply pseudoinverse: Aσ⁺ = Q Λ⁺ Qᵀ → efficient way to avoid zero or 
        # near zero eigenvalues (numerical instability), so while not computed
        # directly, this is the Moore–Penrose pseudoinverse underneath, so we
        # indicate as much with the notation.
        Y = F.vectors' * Y        # Qᵀ * Cb
        for j in axes(Y,1)
            λj = F.values[j]
            if abs(λj) > pinv_tol
                Y[j,:] ./= λj     # scale by 1/λj
            else
                Y[j,:] .= 0       # cutoff → pseudoinverse sets to 0
            end
        end
        Y = F.vectors * Y         # Q * (Λ⁺ Qᵀ Cb)
    else
        # Direct solve with factorization
        Y .= F \ Y
    end

    # 5. Reduced operator
    W = Hermitian(Cb' * Y, :L)
    θ, U = eigen(W)

    # 6. Recover eigenvalues
    m = length(θ)
    α = similar(θ); β = copy(θ); λ = similar(θ)
    for j in 1:m
        α[j] = 1 + σ*θ[j]
        λ[j] = α[j]/β[j]    # equivalently λ = σ + 1/θ
    end

    # 7. Recover eigenvectors
    V = nothing
    if method == :LQD
        V = F' \ (F.D * (Y * U))
    else
        V = Cb' \ U
    end

    # 8. Return results
    return Cb, U, θ, λ, α, β, V, Y, 0.0, (method == :LQD ? F.D : nothing)
end