using LinearAlgebra
using DenseSpectralTransformation
using MatrixMarket
using Plots

include("./condest.jl")
include("./two_sided_inverse.jl")

unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))

function run_standard(A,B; condition=false, plotkws=(;))
  # standard method
  Fb = cholesky(Hermitian(B))
  Cb = Matrix(Fb.U')
  CAC = two_sided_inv(A, Cb)
  λ, V = eigen(Hermitian(CAC))
  p = sortperm(λ)
  λ = λ[p]
  V = V[:,p]
  V = Cb'\V
  α = similar(λ)
  β = similar(λ)

  for j in eachindex(λ)
    α[j] = λ[j]
    β[j] = 1.0
  end

  R = (A * V * Diagonal(β) - B*V*Diagonal(α))
  @views for k in 1:n
    z = (abs(β[k]) * nrma2 + abs(α[k])*nrmb2)*norm(V[:,k])
    R[:,k] ./= z
  end
  Rnorms = map(k -> (k, λ[k], norm(R[:,k])), 1:n)
  xs = (x -> x[2]).(Rnorms)
  ys = (x -> x[3]).(Rnorms)
  if condition
    num = n
    cs = zeros(n)
    C = zeros(n, n)
    Ta, Tb = schur(A, B)
    for j = 1:num
      @. C = β[j] * Ta - α[j] * Tb
      nrmC = opnorm(C, Inf)
      for k = 1:(n - 1)
        @views if abs(C[k + 1, k]) > 1e-13 * nrmC
          (q,) = qr(C[k:(k + 1), k:(k + 1)])
          C[k:(k + 1), k:n] = q' * C[k:(k + 1), k:n]
        end
      end
      cs[j] = (1/inv_norm_est(UpperTriangular(C)))/(abs(β[j])*nrma2 + abs(α[j])*nrmb2)
    end
    ys = (x -> max(1e-25, x)).(cs[1:num])
    xs = xs[1:num]
  end
  kws = Dict(
    :xaxis => :log10,
    :yaxis => :log10,
    :ms => 2,
    :seriestype => :scatter,
  )
  isempty(plotkws) || push!(kws, pairs(plotkws)...)
  xs_pos, ys_pos = unzip(Iterators.filter(p -> p[1] > 0, zip(xs, ys)))
  xs_neg, ys_neg = unzip(Iterators.filter(p -> p[1] <= 0, zip(xs, ys)))
  pl=plot(
    xs_pos,
    ys_pos;
    label = "\$\\lambda\$",
    kws...
  )
  plot!(
    pl,
    abs.(xs_neg),
    ys_neg;
    shape = :utriangle,
    label = "\$-\\lambda\$",
    kws...
  )
  return pl
end

function run_spectral(
  A,
  B,
  σ;
  method = :LQD,
  ηx_max = 500.0,
  r::Union{Int,Nothing} = size(A, 1),
  tol = 0.0,
  condition = false,
  plotkws = (),
  bound_large = nothing,
  bound_small = nothing, 
)
  Cb, U, θ, λ, α, β, V, Y, η, Da =
    eig_spectral_trans(A, B, σ; method = method, ηx_max = ηx_max, tol = tol)
  
  # --- CHECK: numerical symmetry of W = Cb' * Y ---
  W_raw = Cb' * Y
  err = opnorm(W_raw - W_raw', 2)
  rel = err / max(opnorm(W_raw, 2), eps())
  println("    [info] W symmetry: abs = ", err, ", rel = ", rel)
  # -------------------------------------------------
  
  n, _ = size(A)
  r = length(θ)
  println("Rank: $r")
  p = sortperm(λ)
  λ = λ[p]
  V = V[:, p]
  α = α[p]
  β = β[p]
  θ = θ[p]
  R = (A * V * Diagonal(β) - B * V * Diagonal(α))
  @views for k = 1:r
    z = (abs(β[k]) * nrma2 + abs(α[k]) * nrmb2) * norm(V[:, k])
    R[:, k] ./= z
  end
  Rnorms = map(k -> (k, λ[k], norm(R[:, k]), norm(V[:, k])), 1:r)

  xs = (x -> x[2]).(Rnorms)
  ys = (x -> x[3]).(Rnorms)
  if condition
    num = r
    cs = zeros(r)
    C = zeros(n, n)
    Ta, Tb = schur(A, B)
    for j = 1:num
      @. C = β[j] * Ta - α[j] * Tb
      nrmC = opnorm(C, Inf)
      for k = 1:(n - 1)
        @views if abs(C[k + 1, k]) > 1e-13 * nrmC
          (q,) = qr(C[k:(k + 1), k:(k + 1)])
          C[k:(k + 1), k:n] = q' * C[k:(k + 1), k:n]
        end
      end
      cs[j] = (1/inv_norm_est(UpperTriangular(C)))/(abs(β[j])*nrma2 + abs(α[j])*nrmb2)
    end
    ys = (x -> max(1e-25, x)).(cs[1:num])
    xs = xs[1:num]
  end
  kws = Dict(
    :xaxis => :log10,
    :yaxis => :log10,
    :ms => 2,
    :seriestype => :scatter,
  )
  isempty(plotkws) || push!(kws, pairs(plotkws)...)
  xs_pos, ys_pos = unzip(Iterators.filter(p -> p[1] > 0, zip(xs, ys)))
  xs_neg, ys_neg = unzip(Iterators.filter(p -> p[1] <= 0, zip(xs, ys)))

  pl = plot(xs_pos, ys_pos; label = "λ", kws...)
  plot!(pl, abs.(xs_neg), ys_neg; shape = :utriangle, label = "-λ", kws...)

  if !isnothing(bound_small)
    xs1, ys1 = unzip(Iterators.filter(
      p -> p[2] < 1e-10,
      zip(xs, (x -> bound_small * abs(1 - x / σ)).(xs)),
    ))
    plot!(pl, abs.(xs1), ys1, label = nothing)
  end
  if !isnothing(bound_large)
    xs2, ys2 = unzip(Iterators.filter(
      p -> p[2] < 1e-10,
      zip(xs, (x -> bound_large * abs(1 - x / σ) * abs(1 - σ / x)).(xs)),
    ))
    xs2_pos, ys2_pos = unzip(Iterators.filter(p -> p[1] > 0, zip(xs2, ys2)))
    xs2_neg, ys2_neg = unzip(Iterators.filter(p -> p[1] <= 0, zip(xs2, ys2)))
    plot!(pl, xs2_pos, ys2_pos, label = nothing)
    plot!(pl, abs.(xs2_neg), ys2_neg, linestyle = :dot, label = nothing)
  end
  return pl
end

default(
  fontfamily = "Computer modern",
  titlefontsize = 12,
  legendfontsize = 10,
  guidefontsize = 10,
  tickfontsize = 10,
)

# --- Plot directory setup ---
plotdir::String = "./plots/"
isdir(plotdir) || mkpath(plotdir)  # create directory if it doesn't exist
# ----------------------------

# Fluid flow example
B0::Matrix{Float64} = Matrix(MatrixMarket.mmread("MatrixMarket/bcsstm13.mtx"))
n::Int64 = size(B0,2)
A0::Matrix{Float64} = Matrix(MatrixMarket.mmread("MatrixMarket/bcsstk13.mtx"))
# n = 50
B0 = B0[1:n, 1:n]
A0 = A0[1:n, 1:n]
dB::Vector{Float64} = (x -> exp(-0.02*x)).(collect(n:-1:1)) * opnorm(B0)
B::Matrix{Float64} = B0 + Diagonal(dB)
A::Matrix{Float64} = A0
nrma2::Float64 = opnorm(A)
nrmb2::Float64 = opnorm(B)
nrma::Float64 = opnorm(A,Inf)
nrmb::Float64 = opnorm(B,Inf)

@views function test1_standard()
  pl = let
    yticks = [1e-10, 1e-15, 1e-20, 1e-25, 1e-30]
    plotkws = (;
               xticks = [1e10, 1e15, 1e20, 1e25, 1e30, 1e35, 1e40],
               yticks = [1e-5, 1e-10, 1e-15, 1e-20, 1e-25, 1e-30],
               title = "Residual, Computed \$v_i\$",
               xlabel = "\$\\pm \\lambda\$")

    run_standard(A, B, condition = false, plotkws = plotkws)
  end
  GC.gc()
  return pl
end

@views function test1_standard_cond()
  pl = let
    yticks = [1e-10, 1e-15, 1e-20, 1e-25, 1e-30]
    plotkws = (;
               xticks = [1e10, 1e15, 1e20, 1e25, 1e30, 1e35, 1e40],
               yticks = [1e-5, 1e-10, 1e-15, 1e-20, 1e-25, 1e-30],
               title = "Residual, Best \$v_i\$",
               xlabel = "\$\\pm \\lambda\$")

    run_standard(A, B, condition = true, plotkws = plotkws)
  end
  GC.gc()
  return pl
end

@views function test1_spectral_small(; method=:LQD)
  pl = let
    σ0 = 10.0
    @show σ = σ0 * nrma2/nrmb2
    plotkws = (;
               xticks = [1e5, 1e10, 1e15, 1e20, 1e25, 1e30],
               yticks = [1e-5, 1e-10, 1e-15, 1e-20, 1e-25, 1e-30],
               title = "Residual, Computed \$v_i\$",
               xlabel = "\$\\pm \\lambda\$")

    run_spectral(
      A,
      B,
      σ;
      method = method,
      condition = false,
      tol = 0.0,
      bound_small = 1e-14,
      plotkws = plotkws,
    )
  end
  GC.gc()
  return pl
end

@views function test1_spectral_small_cond(; method=:LQD)
  dB = (x -> exp(-0.02 * x)).(collect(n:-1:1)) * opnorm(B0)
  pl = let
    σ0 = 10.0
    @show σ = σ0 * nrma2/nrmb2
    plotkws =
      (;
       xticks = [1e5, 1e10, 1e15, 1e20, 1e25, 1e30],
       yticks = [1e-5, 1e-10, 1e-15, 1e-20, 1e-25, 1e-30],
       title = "Residual, Best \$v_i\$",
       xlabel = "\$\\pm \\lambda\$")
    run_spectral(
      A,
      B,
      σ;
      method = method,
      condition = true,
      tol = tol=0.0,
      plotkws = plotkws,
    )
  end
  GC.gc()
  return pl
end

@views function test1_spectral_large(; method=:LQD)
  pl = let
    σ0 = 1e7
    @show σ = σ0 * nrma2/nrmb2
    plotkws = (;
               xticks = [1e5, 1e10, 1e15, 1e20, 1e25, 1e30],
               yticks = [1e-5, 1e-10, 1e-15, 1e-20, 1e-25, 1e-30],
               title = "Residual, Computed \$v_i\$",
               xlabel = "\$\\pm \\lambda\$")

    run_spectral(
      A,
      B,
      σ;
      method = method,
      condition = false,
      tol = 0.0,
      bound_large = 1e-15,
      plotkws = plotkws,
    )
  end
  GC.gc()
  return pl
end

@views function test1_spectral_large_cond(; method=:LQD)
  dB = (x -> exp(-0.02*x)).(collect(n:-1:1)) * opnorm(B0)
  pl = let
    σ0 = 1e7
    @show σ = σ0 * nrma2/nrmb2
    plotkws = (;
               xticks = [1e5, 1e10, 1e15, 1e20, 1e25, 1e30],
               yticks = [1e-5, 1e-10, 1e-15, 1e-20, 1e-25, 1e-30],
               title = "Residual, Best \$v_i\$",
               xlabel = "\$\\pm \\lambda\$")

    run_spectral(
      A,
      B,
      σ;
      method = method,
      condition = true,
      tol = 0.0,
      plotkws = plotkws,
    )
  end
  GC.gc()
  return pl
end

function show_matrix_info(σ0; tol=0.0)
  let 
    @show σ = σ0 * nrma2/nrmb2
    @show opnorm(A)
    @show cond(A)
    @show opnorm(B)
    @show cond(B)

    Fb = cholesky(Hermitian(B), RowMaximum(), tol = tol, check = false)
    A1 = shift(A, B, σ)
    r = Fb.rank
    ip = invperm(Fb.p)
    Cb = Matrix(Fb.L)[ip, 1:r]
    Fa = lqd(Hermitian(A1))
    η = sqrt(opnorm(A1)/opnorm(B))
    X = Fa\Cb
    @show η*opnorm(X)
  end
  GC.gc()
end

# First run the baseline (independent of factorization method)
println("\n>>> Running standard baseline tests <<<")

pl1_standard = test1_standard()
println("Saving pl1_standard.png")
savefig(pl1_standard, "$plotdir/pl1_standard")
display(pl1_standard)

pl1_standard_cond = test1_standard_cond()
println("Saving pl1_standard_cond.png")
savefig(pl1_standard_cond, "$plotdir/pl1_standard_cond")
display(pl1_standard_cond)

# Then iterate spectral tests over factorization
methods = [:LQD, :LDLt, :LU, :Eig]

for m in methods
  println("\n==============================")
  println("Running spectral experiments with method = $m")

  println("  -> test1_spectral_small ($m)")
  pl1_spectral_small = test1_spectral_small(; method = m)
  savefig(pl1_spectral_small, "$plotdir/pl1_spectral_small_$(m).png")
  display(pl1_spectral_small)

  println("  -> test1_spectral_small_cond ($m)")
  pl1_spectral_small_cond = test1_spectral_small_cond(; method = m)
  savefig(pl1_spectral_small_cond, "$plotdir/pl1_spectral_small_cond_$(m).png")
  display(pl1_spectral_small_cond)

  println("  -> test1_spectral_large ($m)")
  pl1_spectral_large = test1_spectral_large(; method = m)
  savefig(pl1_spectral_large, "$plotdir/pl1_spectral_large_$(m).png")
  display(pl1_spectral_large)

  println("  -> test1_spectral_large_cond ($m)")
  pl1_spectral_large_cond = test1_spectral_large_cond(; method = m)
  savefig(pl1_spectral_large_cond, "$plotdir/pl1_spectral_large_cond_$(m).png")
  display(pl1_spectral_large_cond)
end