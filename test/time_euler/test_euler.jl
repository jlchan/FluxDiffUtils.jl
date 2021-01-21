using BenchmarkTools
using EntropyStableEuler
using FluxDiffUtils
using ForwardDiff
using LinearAlgebra
using StaticArrays
using SparseArrays
using Test

N1D = 5
N = N1D*N1D

rho = 1 .+ rand(N)
u,v = .1*randn(N), .2*randn(N)
p = rho.^Euler{2}().γ
Q = @SVector [rho,u,v,p]

U = prim_to_cons(Euler{2}(),Q)
Q = cons_to_prim_beta(Euler{2}(),U)
Qlog = SVector{6}(Q...,log.(first(Q)),log.(last(Q)))

skew(A) = droptol!(sparse(A-A'),5e-12)
skewTr(A) = droptol!(sparse(-(A-A')),5e-12)
A = randn(N1D,N1D)
Qx,Qy = kron(I(N1D),A), kron(A,I(N1D)) # sparsity for N=4 quad
QxyTr = skewTr.((Qx,Qy))
QxyTrDense = Matrix.(QxyTr)
Qxy = skew.((Qx,Qy))
QxyDense = Matrix.(Qxy)

f2D(uL,uR) = fS_prim_log(Euler{2}(),uL,uR)

rhs1 = zero.(Q)
rhs2 = zero.(Q)
hadamard_sum_ATr!(rhs1,QxyTr,f2D,Qlog)
hadamard_sum_ATr!(rhs2,QxyTrDense,f2D,Qlog)
@test sum(norm.(rhs1 .- rhs2)) < 1e-10

# @btime hadamard_sum_ATr!($rhs1,$QxyTr,$f2D,$Qlog)
# @btime hadamard_sum_ATr!($rhs2,$QxyTrDense,$f2D,$Qlog)

dF(uL,uR) = ntuple(i->ForwardDiff.jacobian(uR->fS(Euler{2}(),uL,uR)[i],uR),2)
Nfields = 4
Asparse = @SMatrix [zero(first(Qxy)) for i=1:Nfields, j=1:Nfields]
Adense = @SMatrix [zero(first(QxyDense)) for i=1:Nfields, j=1:Nfields]
hadamard_jacobian!(Asparse,Qxy,:skew,dF,U)
hadamard_jacobian!(Adense,QxyDense,:skew,dF,U)

# hadamard_jacobian!(Asparse,Qxy,:skew,dF,U)
# hadamard_jacobian!(Adense,QxyDense,:skew,dF,U)

# # julia> Qlog = (1.0, 0.1, 0.2, 2.5250000000000004, 0.0, 0.9262410627273233)
# # julia> @btime fS_prim_log($Euler{2}(),$Qlog,$Qlog);
# #   2.963 μs (0 allocations: 0 bytes)
#
# # julia> @btime getindex.($A_list,$11,$1);
# #   8.105 ns (0 allocations: 0 bytes)
