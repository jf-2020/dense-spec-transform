using DenseSpectralTransformation
using LinearAlgebra

showall(io, x; compact = true) =
  show(IOContext(io, :compact => compact), "text/plain", x)
showall(x) = showall(stdout, x)

n = 100
x = exp.(-.2 * (1:n))
A = Float64[
    (i == j ? (-1)^i * x[abs(i - j) + 1] : x[abs(i - j) + 1])
    for i = 1:n, j = 1:n
        ]
B = [1/(i+j-1) for i in 1:n, j in 1:n]
nrma = opnorm(A)
nrmb = opnorm(B)
tol = 1e-13
@show σ = 5.0
@show σ0 = σ*nrmb/nrma
@show cond(A-σ*B)
A0 = copy(A)
B0 = copy(B)
# @code_warntype eig_spectral_trans(A, B, σ, ηx_max=100.0)
Cb, U, θ, λ, α, β, V, X, η, Da =
    eig_spectral_trans(A, B, σ, ηx_max=100.0, tol=1e-16, method=:Eig)
R = A0*V*Diagonal(β) - B0*V*Diagonal(α)
z = zeros(length(α))
@views for k in axes(R,2)
    z[k] =
        norm(R[:, k]) / (abs(β[k]) * nrma + abs(α[k]) * nrmb) / norm(V[:, k])
end
@show length(α)
p = sortperm(λ, by = abs)
showall([z[p] λ[p]])
nothing

