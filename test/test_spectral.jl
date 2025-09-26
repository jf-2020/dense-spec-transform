module TestSpectral
using DenseSpectralTransformation
using LinearAlgebra
using Test

function run_test(A, B, σ, tol)
  n = size(A,1)
  nrma = opnorm(A)
  nrmb = opnorm(B)

  Cb, U, θ, λ, α, β, V, X, η, Da =
    eig_spectral_trans(A, B, σ, ηx_max=100.0)

  @testset "Residual test" begin
    R = A*V*Diagonal(β) - B*V*Diagonal(α)
    z = zeros(length(α))
    @views for k in axes(R,2)
      z[k] =
        norm(R[:, k]) / (abs(β[k]) * nrma + abs(α[k]) * nrmb) / norm(V[:, k]) /
        max(1, abs(1 - λ[k] / σ))
    end

    @test norm(z,Inf) ≤ tol
  end

  @testset "Condition test" begin
    z = zeros(length(α))
    @views for k in eachindex(α)
      z[k] = 1/cond(β[k]*A - α[k]*B, Inf)
    end
    @test norm(z,Inf) ≤ tol
  end

end
end