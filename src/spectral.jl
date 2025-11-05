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

###################################
# TODO:
# Is this really necessary? Check & perhaps remove it later.
###################################

# Custom error type for bounding ||X||
struct EtaXError{T} <: Exception
    etax :: T
    bound :: T
end

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

###################################
# TODO: implement code profiling with table visualizations later
###################################

"""
    eig_spectral_trans(A, B, σ; method=:LQD, tol=0, ηx_max=500.0)

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
   - For the others, solve directly with the factorization.
5. Form reduced operator W = Cbᵀ Aσ⁻¹ Cb.
6. Solve WU = UΘ.
7. Recover original eigenvalues: λ = σ + 1/θ.
8. Recover eigenvectors:
   - `:LQD`: V = (F')⁻¹ (D * (XU)), X = Aσ⁻¹Cb.
   - Others: V = (Cb')⁻¹ U.

Returns:
(Cb, U, θ, λ, α, β, V, Y, η, D).
"""
function eig_spectral_trans(A, B, σ; method=:LQD, tol=0, ηx_max=500.0)
    # 1. Shift & compute η
    Aσ = shift(A,B,σ)
    η = sqrt(opnorm(Aσ)/opnorm(B))

    # 2. Pivoted Cholesky of B
    Fb = cholesky(Hermitian(B,:L), RowMaximum(), tol=tol, check=false)
    r  = Fb.rank
    ip = invperm(Fb.p)
    Cb = Matrix(Fb.L)[ip,1:r] 

    # 3. Factorize Aσ (method-dependent)

    # For the ||X|| calculation later, LQD is needed regardless of the method.
    ###################################
    # TODO / Question:
    # Should I use copy(Aσ) here to avoid modifying the input Aσ? Or is my comment
    # below correct, namely, that lqd() calls lqd!() on copy(Aσ) internally, so Aσ
    # is unchanged? I ask because I think some of the printed debugging info upon
    # running "paper_experiments.jl" seems to indicate something went wrong after
    # the below change was implemented.
    ###################################
    F_LQD = lqd(Hermitian(Aσ,:L)) # lqd operates in-place on a COPY of Aσ, so Aσ
                                  # is unchanged.

    ###################################
    # Comment: It might be better to pass in a type.  I ran @code_warntype
    # on this and your factorization F is a huge union, which creates
    # some possibility of inefficiency, although it's certainly not an
    # issue for testing and it's more important to get everything working
    # than to worry about it.

    ## Reply: I've decided, for now, to leave as-is. Though I've read the design
    ## pattern I used before is common for Julia, providing the high-level
    ## Factorization type, with various subtypes for the different factorizations.
    ## We can revert back to that if really necessary.
    ###################################
    if method == :LQD
        # Use the pre-computed LQD factorization. This particular branch is more
        # for readability.
        F = F_LQD
    elseif method == :LDLt
        F = bunchkaufman!(Hermitian(Aσ,:L))
    elseif method == :LU
        F = lu!(Aσ)
    elseif method == :Eig
        F = eigen!(Hermitian(Aσ,:L))
    else
        throw(ArgumentError("Unknown method: $method"))
    end

    # 4. Apply inverse once, Y = Aσ⁻¹ * Cb, and compute X
    if method == :LQD
        # LQD: Aσ = L * Q * D * S * D * Q' * L⁻¹
        Y = F' \ (F.S * (F \ Cb))

        # Compute X = D⁻¹ * Qᵀ * L⁻¹ * Cb
        X = F.D \ (F.Q' * (F.L \ Cb))
    elseif method == :LDLt
        # Direct solve with factorization
        Y = F \ Cb

        ######
        # # Compute X = D⁻¹ * L⁻¹ * P * Cb
        # X = F.D \ (F.L \ (Fb.P * Cb))
        ######

        # Use X from the pre-computed LQD factorization
        X = F_LQD.D \ (F_LQD.Q' * (F_LQD.L \ Cb))
    elseif method == :LU
        # Direct solve with factorization
        Y = F \ Cb

        ######
        # # Compute X = U⁻¹ * L⁻¹ * Pᵀ * Cb
        # X = F.U \ (F.L \ (Fb.P' * Cb))
        ######

        # Use X from the pre-computed LQD factorization
        X = F_LQD.D \ (F_LQD.Q' * (F_LQD.L \ Cb))
    elseif method == :Eig
        # Spectral decomposition: Aσ = Q Λ Qᵀ
        Y = F.vectors' * Cb         # Qᵀ * Cb
        Y = Diagonal(F.values) \ Y
        Y = F.vectors * Y         # Q * (Λ⁺ Qᵀ Cb)

        # Compute X = D⁻¹ * Qᵀ * Cb
        Λ = F.values                    # eigenvalues of Aσ from eigendecomposition
        D = Diagonal(sqrt.(abs.(Λ)))    # D = diag(sqrt(|Λ|))
        X = D \ F.vectors' * Cb         # X = D⁻¹ * Qᵀ * Cb
    end

    # 5. Consider the η||X|| max threshold
    normX = opnorm(X)
    # compute the theoretical (residual) threshold
    ηx = η * normX
    # then check against the user-defined (or default) max
    if ηx > ηx_max
        ###################################
        # Comment: For now, just print a warning. Later, we can make this
        # throw an error if we want to enforce it strictly.
        ###################################
        # throw(EtaXError(η, ηx_max))

        # instead, we'll just warn the user of potential inaccuracy
        @warn "Value of η ||X|| is $(ηx), which exceeds the given bound $(ηx_max)."
    end

    # 6. Reduced operator
    W = Hermitian(Cb' * Y, :L)
    θ, U = eigen(W)

    # 7. Recover eigenvalues
    m = length(θ)
    α = similar(θ); β = copy(θ); λ = similar(θ)
    for j in 1:m
        α[j] = 1 + σ*θ[j]
        λ[j] = α[j]/β[j]    # equivalently λ = σ + 1/θ
    end

    # 8. Recover eigenvectors
    if method == :Eig
        V = F.vectors * (Diagonal(F.values) \ (F.vectors' * (Cb * U)))
    elseif method == :LQD
        V = F' \ (F.S * (F \ (Cb * U)))
    else
        V = F \ (Cb * U)
    end

    ###################################
    # TODO:
    # Still need to adjust the return value, but to do that, all the endpoints
    # calling this function need to be updated, too. Not being necessary now,
    # I'll leave it for later.
    #
    # It would also be helpful to return ηx, exposing it to the user for error
    # analysis purposes.
    ###################################

    # 9. Return results
    return Cb, U, θ, λ, α, β, V, Y, ηx, nothing
end
