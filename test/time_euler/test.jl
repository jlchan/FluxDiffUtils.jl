using LinearAlgebra
using Test
using SparseArrays
using StaticArrays

tol = 5e2*eps()
N = 10

# scalar test
A = sparse(randn(N,N))
ATr_list = (Matrix(transpose(A)),)
function flux1(UL,UR)
    return UL .+ UR
end
U = (randn(N),)
rhs_exact = (A*U[1] + U[1].*sum(A,dims=2),)
rhs1, rhs2 = zero.(U), zero.(U)
hadamard_sum_ATr!(rhs1, ATr_list,flux1,U)
hadamard_sum_ATr!(rhs2, sparse.(ATr_list),flux1,U)
@test sum(norm.(rhs1 .- rhs_exact)) < tol
@test sum(norm.(rhs_exact .- rhs2)) < tol
@test sum(norm.(rhs1 .- rhs2)) < tol

# 2 flux test
A,B = randn(N,N),randn(N,N)
ATr_list = map(A->Matrix(transpose(A)),(A,B))
function flux2(UL,UR)
    uL,vL = UL
    uR,vR = UR
    return SVector{2}(uL+uR,0),SVector{2}(0,2*(vL+vR))
end
u,v = (randn(N),randn(N))
U = (u,v)
# computes sum(A.*F1 + B.*F2,dims=2)
#rhs_exact = (sum(A,dims=2), sum(B,dims=2))
rhs_exact = (A*u + u.*sum(A,dims=2),2*B*v + 2*v.*sum(B,dims=2))
# rhs_exact = (A*u + u.*sum(A,dims=2), B*v + v.*sum(B,dims=2))
rhs1, rhs2 = zero.(U), zero.(U)
hadamard_sum_ATr!(rhs1,ATr_list,flux2,U)
hadamard_sum_ATr!(rhs2,sparse(ATr_list[1]),(uL,uR)->flux2(uL,uR)[1],U)
hadamard_sum_ATr!(rhs2,sparse(ATr_list[2]),(uL,uR)->flux2(uL,uR)[2],U)

@test sum(norm.(rhs1 .- rhs_exact)) < tol
@test sum(norm.(rhs_exact .- rhs2)) < tol
@test sum(norm.(rhs1 .- rhs2)) < tol
