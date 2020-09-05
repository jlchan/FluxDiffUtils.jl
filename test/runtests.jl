using FluxDiffJacobians
using Test
using SparseArrays

using StaticArrays
using LinearAlgebra
using ForwardDiff

@testset "Flux diff tests" begin
    # make 2-field solution
    # U = SVector{2}(collect(LinRange(-1,1,10)),collect(LinRange(-1,1,10)))
    U = collect(LinRange(-1,1,10)),collect(LinRange(-1,1,10))

    # define flux function and its derivative
    function flux_function(uL,vL,uR,vR)
        # return SVector{2}(.5*(uL+uR),.5*(vL+vR)),SVector{2}(.5*(uL+uR),.5*(vL+vR))
        return (@. .5*(uL+uR),@. .5*(vL+vR)),(@. .5*(uL+uR),@. .5*(vL+vR))
    end

    df1(UL,UR) = ForwardDiff.jacobian(UR->flux_function(UL...,UR...)[1], UR)
    df2(UL,UR) = ForwardDiff.jacobian(UR->flux_function(UL...,UR...)[2], UR)
    df(UL,UR) = df1(UL,UR),df1(UL,UR)

    # for banded diagonal matrix test
    g(U) = SMatrix{2,2}([U[1] 0
                        0 U[2]])

    A = randn(10,10)
    A = ntuple(x->A,2)
    A = (A->A-A').(A) # make skew
    ATr = (A->Matrix(A')).(A)

    rhs = hadamard_sum(ATr,flux_function,U)
    @test (A[1]*U[1] + U[1].*sum(A[1],dims=2)) ≈ rhs[1]
    @test (A[1]*U[2] + U[2].*sum(A[1],dims=2)) ≈ rhs[2]

    rhs_sparse = hadamard_sum(sparse.(ATr),flux_function,U)
    @test all(rhs .≈ rhs_sparse)
end

@testset "Sparse Jacobian tests" begin

    # make 2-field solution
    U = SVector(collect(LinRange(-1,1,10)),collect(LinRange(-1,1,10)))

    # define flux function and its derivative
    function flux_function(uL,vL,uR,vR)        
        return SVector{2}(.5*(uL+uR),.5*(vL+vR))
    end
    df(UL,UR) = ForwardDiff.jacobian(UR->flux_function(UL...,UR...), UR)

    # for banded diagonal matrix test
    g(U) = SMatrix{2,2}([U[1] 0
                        0 U[2]])

    A = sprandn(10,10,.25)
    A = sparse(A-A') # make skew
    ATr = sparse(A')

    rhs = hadamard_sum(ATr,flux_function,U)
    @test .5*(A*U[1] + U[1].*sum(A,dims=2)) ≈ rhs[1]
    @test .5*(A*U[2] + U[2].*sum(A,dims=2)) ≈ rhs[2]

    J = hadamard_jacobian(A,df,U)
    Jexplicit = blockdiag(A,A)
    @test J ≈ .5*(Jexplicit + diagm(vec(sum(Jexplicit,dims=2)))) # formula from paper

    G = banded_matrix_function(g,U)
    @test G ≈ blockdiag(spdiagm(0=>U[1]),spdiagm(0=>U[2]))

end
