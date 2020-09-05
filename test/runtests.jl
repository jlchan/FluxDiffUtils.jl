using FluxDiffUtils
using Test

using SparseArrays
using StaticArrays
using LinearAlgebra
using ForwardDiff

# make 3-field solution
u = collect(LinRange(-1,1,10))
U = SVector{3}(u,u,u)

# define flux function and its derivative
function flux_function(uL,vL,wL,uR,vR,wR)
    avg(x,y) = @. .5*(x+y)
    Fx = @SVector [avg(uL,uR),avg(vL,vR),avg(wL,wR)]
    Fy = @SVector [avg(uL,uR),avg(vL,vR),avg(wL,wR)]
    return @SVector [Fx,Fy]
end

dfx(UL,UR) = ForwardDiff.jacobian(UR->flux_function(UL...,UR...)[1], UR)
dfy(UL,UR) = ForwardDiff.jacobian(UR->flux_function(UL...,UR...)[2], UR)
df(uL,vL,wL,uR,vR,wR) = dfx(SVector{3}(uL,vL,wL),SVector{3}(uR,vR,wR)),
                        dfy(SVector{3}(uL,vL,wL),SVector{3}(uR,vR,wR))

A = randn(10,10)
A = ntuple(x->A,2)
A = (A->A-A').(A) # make skew for exact formula testing
ATr = (A->Matrix(transpose(A))).(A)

@testset "Flux diff tests" begin

    rhs = hadamard_sum(ATr,flux_function,U)
    @test (A[1]*U[1] + U[1].*sum(A[1],dims=2)) ≈ rhs[1]
    @test (A[1]*U[2] + U[2].*sum(A[1],dims=2)) ≈ rhs[2]
    @test (A[1]*U[3] + U[3].*sum(A[1],dims=2)) ≈ rhs[3]

    rhs_sparse = hadamard_sum(sparse.(ATr),flux_function,U)
    @test all(rhs .≈ rhs_sparse)
end

@testset "Jacobian tests" begin
    tol = 5e2*eps()

    Jdense = hadamard_jacobian(A,df,U)
    J = hadamard_jacobian(sparse.(A),df,U)
    # no factor of .5 since Alist = (A,A) so exact jac is A - diag(sum(A,2))
    @test J[1][1] ≈ first(A) - diagm(vec(sum(first(A),dims=1)))
    @test J[2][2] ≈ first(A) - diagm(vec(sum(first(A),dims=1)))
    @test J[3][3] ≈ first(A) - diagm(vec(sum(first(A),dims=1)))
    accum_norm = 0.0
    for i = 1:3, j = 1:3
        if i!=j
            accum_norm += norm(J[i][j])
        end
    end
    @test accum_norm < tol

    test_dense_eq_sparse = Bool[]
    for i=1:3,j=1:3
        push!(test_dense_eq_sparse, J[i][j] ≈ sparse(Jdense[i][j]))
    end

    # for banded diagonal matrix test
    g(u,v) = SMatrix{2,2}([u 0
                           0 v])
    G = banded_matrix_function(g,U[1:2])
    @test G[1][1] ≈ spdiagm(0=>U[1])
    @test norm(G[1][2]) + norm(G[2][1]) < tol
    @test G[2][2] ≈ spdiagm(0=>U[2])
end

# @testset "Jacobian tests" begin
#
#     # make 2-field solution
#     U = SVector(collect(LinRange(-1,1,10)),collect(LinRange(-1,1,10)))
#
#     # define flux function and its derivative
#     function flux_function(uL,vL,uR,vR)
#         return SVector{2}(.5*(uL+uR),.5*(vL+vR))
#     end
#     df(UL,UR) = ForwardDiff.jacobian(UR->flux_function(UL...,UR...), UR)
#
#     # for banded diagonal matrix test
#     g(U) = SMatrix{2,2}([U[1] 0
#                         0 U[2]])
#
#     A = sprandn(10,10,.25)
#     A = sparse(A-A') # make skew
#     ATr = sparse(A')
#
#     rhs = hadamard_sum(ATr,flux_function,U)
#     @test .5*(A*U[1] + U[1].*sum(A,dims=2)) ≈ rhs[1]
#     @test .5*(A*U[2] + U[2].*sum(A,dims=2)) ≈ rhs[2]
#
#     J = hadamard_jacobian(A,df,U)
#     Jexplicit = blockdiag(A,A)
#     @test J ≈ .5*(Jexplicit + diagm(vec(sum(Jexplicit,dims=2)))) # formula from paper
#
#     G = banded_matrix_function(g,U)
#     @test G ≈ blockdiag(spdiagm(0=>U[1]),spdiagm(0=>U[2]))
# end
