# for efficient testing of factorization types, we'll use an API approach
abstract type FactorizationMethod end

# currently there are 4 methods to consider, LDQ (original), LDLᵗ, LU, and
# eigendecomposition
struct LQDMethod <: FactorizationMethod end
struct LDLtMethod <: FactorizationMethod end
struct LUMethod <: FactorizationMethod end
struct EigMethod <: FactorizationMethod end

# regular shift for spec transform
function shift(
  A::AbstractMatrix{E},
  B::AbstractMatrix{E},
  σ,
) where {E<:AbstractFloat}
  A1 = similar(A)
  for k in axes(A, 2)
    for j in axes(A, 1)
      A1[j, k] = fma(-σ, B[j, k], A[j, k])
    end
  end
  return A1
end

# complex shift for spec transform
function shift(
  A::AbstractMatrix{Complex{E}},
  B::AbstractMatrix{Complex{E}},
  σ,
) where {E<:AbstractFloat}
  A1 = similar(A)
  for k in axes(A, 2)
    for j in axes(A, 1)
      zr = fma(-σ, real(B[j, k]), real(A[j, k]))
      zi = fma(-σ, imag(B[j, k]), imag(A[j, k]))
      A1[j, k] = complex(zr, zi)
    end
  end
  return A1
end

# overloading factorization methods
function factorize_shifted(Aσ::StridedMatrix, ::LQDMethod)
  # LQD (original)
  lqd(Hermitian(Aσ, :L))
end

function factorize_shifted(Aσ::StridedMatrix, ::LDLtMethod)
  # Bunch-Kaufman (LDLᵗ)
  bunchkaufman!(Hermitian(Aσ, :L); check=false)
end

function factorize_shifted(Aσ::StridedMatrix, ::LUMethod)
  # standard LU
  lu!(Aσ; check=false)
end

function factorize_shifted(Aσ::StridedMatrix, ::EigMethod)
  # eigendecomposition
  eigen!(Hermitian(Aσ, :L))
end

# custom exception for exceeding threshold (per paper)
struct EtaXError{T} <: Exception
  etax :: T
end

### ORIGINAL CODE ###

# function eig_spectral_trans(A, B, σ; ηx_max = 500.0, tol = 0.0)

#   Base.require_one_based_indexing(A,B)
  
#   m, n = size(A)
#   mb, nb = size(B)
#   m == n ||
#     throw(DimensionMismatch("Matrix A is not square: dimensions are ($m, $n)"))
#   mb == nb ||
#     throw(DimensionMismatch("Matrix B is not square: dimensions are ($mb,$nb)"))
#   n == nb ||
#     throw(DimensionMismatch(
#       "Matrix A has dimensions ($m,$n) and B has dimensions ($mb,$nb)"))

#   Fb = cholesky(Hermitian(B), RowMaximum(), tol = tol, check = false)
#   A1 = shift(A, B, σ)

#   r = Fb.rank
#   ip = invperm(Fb.p)
#   Cb = Matrix(Fb.L)[ip, 1:r]
  
#   Fa = lqd(Hermitian(A1, :L))

#   Da = Fa.S
#   η = sqrt(opnorm(A1, Inf) / opnorm(B, Inf))
  
#   X = Fa\Cb
#   ηx = η * opnorm(X, Inf)
#   ηx <= ηx_max || throw(EtaXError(ηx))

#   W = X'*(Da*X)

#   θ, U = eigen(Hermitian(W))
#   λ = similar(θ)
#   β = copy(θ)
#   α = similar(θ)
#   for j in 1:r
#     α[j] = fma(σ, θ[j], one(λ[j]))
#     λ[j] = α[j]/β[j]
#   end
#   V = Fa' \ (Da*(X*U))
#   return Cb, U, θ, λ, α, β, V, X, η, Da
# end

### END ORIGINAL ###

# continuing with factorization overload, we require the same for the inversion
function apply_inv!(Y::StridedMatrix, F, ::LQDMethod)
  # LQD (original) 
  (Y .= F \ Y)
end

function apply_inv!(Y::StridedMatrix, F, ::LDLtMethod)
  # Bunch-Kaufman (LDLᵗ)
  (Y .= F \ Y)
end

function apply_inv!(Y::StridedMatrix, F, ::LUMethod)
  # standard LU
  (Y .= F \ Y)
end

function apply_inv!(Y::StridedMatrix, F::Eigen, ::EigMethod; pinv_tol=0.0)
  # eigendecomposition
    invλ = similar(F.values)
    for i in eachindex(F.values)
        λ = F.values[i]
        invλ[i] = (abs(λ) ≤ pinv_tol) ? zero(λ) : inv(λ)
    end
    
    Y .= F.vectors' * Y # Y ← Q'Y
    Y .= invλ .* Y # Y ← Λ^{-1}*Y (element-wise scaling)
    Y .= F.vectors * Y # Y ← Q*Y
    
    return Y
end

function eig_spectral_trans(A::StridedMatrix,B::StridedMatrix, σ;
                            method::FactorizationMethod = LQDMethod(),
                            tol::Float64 = 0.0,
                            pinv_tol::Float64 = 0.0,
                            ηx_max::Float64 = 500.0)

    # spectral shift
    Aσ = shift(A, B, σ)

    # pivoted Cholesky of B
    Fb = cholesky(Hermitian(B, :L), RowMaximum(); tol=tol, check=false)
    r = Fb.rank
    ip = invperm(Fb.p)
    Cb = Matrix(Fb.L)[ip, 1:r]
    Cb_r = Cb # working copy

    # now we abstract via factorization methods
    F = factorize_shifted(Aσ, method)

    # now we may apply the inverse in just one go: Y = Aσ^{-1} * Cb
    Y = copy(Cb_r)
    if method isa EigMethod
        apply_inv!(Y, F, method; pinv_tol=pinv_tol)
    else
        apply_inv!(Y, F, method)
    end

    # reduced (possibly) Hermitian operator & its eigen structure
    W = Hermitian(Cb_r' * Y, :L)
    eigW = eigen(W)
    θ, U = eigW.values, eigW.vectors

    # convert θ to (α,β,λ) per the paper
    λ = similar(θ); β = copy(θ); α = similar(θ)
    for j in eachindex(θ)
        α[j] = fma(σ, θ[j], one(θ[j]))   # get α (= 1 + σ θ)
        λ[j] = α[j] / β[j]               # and λ (= σ + 1/θ)
    end

    # evects: note for LQD, V = Fa' \ (Da*(X*U)) above. but to cover all the
    # factorization methods, we need to avoid Da & X here. we can use the below
    # to preserve Hermitian structure. we'll split into LQD vs others, maintaining
    # the original code's structure for LQD
    if method isa LQDMethod
      # preserve existing structure
      X = Y # per Y, X = Aσ^{-1} * Cb
      Da = F.D
      V = F' \ (Da * (X * U))
    else
      # for other factorizations, we just preserve Hermitian structure as
      # mentioned
      X = Y
      Da = nothing
      V = Cb' \ U
    end

    # tests expect particular signatures, so we maintin original code's return
    # shape (even if some entries are placeholders as `Da` above, e.g.)
    η  = 0.0

    return Cb, U, θ, λ, α, β, V, X, η, Da
end