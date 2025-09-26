module DenseSpectralTransformation

using LinearAlgebra

include("lqds.jl")
export lqd!, lqd, LQD

include("spectral.jl")
export eig_spectral_trans, shift, EtaXError

include("../test/test_lqds.jl")
include("../test/test_spectral.jl")

using PrecompileTools

# @setup_workload begin
#   @compile_workload begin
#     n = 5
#     tol = 1e-13
#     Br = [1 / (i + j - 1) for i = 1:n, j = 1:n]
#     Ar = randn(n, n)
#     Ar = Ar + Ar'

#     TestLQDS.run_test(Hermitian(Ar, :L), tol)
#     TestSpectral.run_test(Ar, Br, 5.0, tol)

#     Bc = Complex{Float64}[1 / (i + j - 1) for i = 1:n, j = 1:n]
#     Ac = randn(Complex{Float64}, n, n)
#     Ac = Ac + Ac'

#     TestLQDS.run_test(Hermitian(Ac, :L), tol)
#     TestSpectral.run_test(Ac, Bc, 5.0, tol)
#   end
# end


end # module SymDefGenEig