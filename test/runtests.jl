using FluxDiffUtils
using Test

using SparseArrays
using StaticArrays
using LinearAlgebra
using ForwardDiff

@testset "Flux diff tests" begin
    u = collect(LinRange(-1,1,10))
    U = (u,2*u)
    function flux1D(UL,UR)
        uL,vL = UL
        uR,vR = UR
        avg(x,y) = @. .5*(x+y)
        Fx = SVector{2}(avg(uL,uR),avg(vL,vR))
        return Fx
    end
    A = randn(10,10)
    A[8:10,8:10] .= 0 # for testing skip_index
    skip_index(i,j) = (i>8)&&(j>8)
    A = A-A' # make skew for exact formula testing
    ATr = (A->Matrix(transpose(A)))(A)

    rhs = hadamard_sum((A,),tuple∘flux1D,U,skip_index=skip_index)
    @test (A*U[1] + U[1].*sum(A,dims=2)) ≈ 2*rhs[1]
    @test (A*U[2] + U[2].*sum(A,dims=2)) ≈ 2*rhs[2]
end

# make 3-field solution
u = collect(LinRange(-1,1,10))
U = SVector{3}(u,2*u,3*u)

# define flux function and its derivative
function flux_function(UL,UR)
    uL,vL,wL = UL
    uR,vR,wR = UR
    avg(x,y) = @. .5*(x+y)
    Fx = avg(uL,uR),avg(vL,vR),avg(wL,wR)
    Fy = avg(uL,uR),avg(vL,vR),avg(wL,wR)
    return SVector{3}(Fx),SVector{3}(Fy)
end

dfx(UL,UR) = ForwardDiff.jacobian(UR->flux_function(UL,UR)[1], SVector{3}(UR))
dfy(UL,UR) = ForwardDiff.jacobian(UR->flux_function(UL,UR)[2], SVector{3}(UR))
df(UL,UR) = dfx(UL,UR),dfy(UL,UR)

A = randn(10,10)
A[8:10,8:10] .= 0 # for testing skip_index
skip_index(i,j) = (i>8)&(j>8)
A = ntuple(x->A,2)
A = (A->A-A').(A) # make skew for exact formula testing

@testset "Flux diff tests w/StaticArrays" begin

    rhs = hadamard_sum(A,flux_function,U)
    @test (A[1]*U[1] + U[1].*sum(A[1],dims=2)) ≈ rhs[1]
    @test (A[1]*U[2] + U[2].*sum(A[1],dims=2)) ≈ rhs[2]
    @test (A[1]*U[3] + U[3].*sum(A[1],dims=2)) ≈ rhs[3]

    rhs_sparse = hadamard_sum(sparse.(A),flux_function,U)
    @test all(rhs .≈ rhs_sparse)

    rhs_skipped = hadamard_sum(sparse.(A),flux_function,U;
                               skip_index=skip_index)
    @test all(rhs .≈ rhs_skipped)

    # test tuple args too
    v = collect(LinRange(-1,1,10))
    V = (v,2*v,3*v)
    # define flux function and its derivative
    function flux_function_tuple(UL,UR)
        uL,vL,wL = UL
        uR,vR,wR = UR
        avg(x,y) = @. .5*(x+y)
        Fx = avg(uL,uR),avg(vL,vR),avg(wL,wR)
        Fy = avg(uL,uR),avg(vL,vR),avg(wL,wR)
        return SVector{3}(Fx),SVector{3}(Fy)
    end
    rhs2 = hadamard_sum(A,flux_function_tuple,V)
    @test all(rhs .≈ rhs2)

end

@testset "Jacobian tests" begin
    tol = 5e2*eps()

    Jdense = hadamard_jacobian(A,df,U)
    J = hadamard_jacobian(sparse.(A),df,U)
    # no factor of .5 since Alist = (A,A) and exact jac is ∑_i (A_i - diag(sum(A_i,2)))
    @test J[1,1] ≈ first(A) - diagm(vec(sum(first(A),dims=1)))
    @test J[2,2] ≈ first(A) - diagm(vec(sum(first(A),dims=1)))
    @test J[3,3] ≈ first(A) - diagm(vec(sum(first(A),dims=1)))
    accum_norm = 0.0
    for i = 1:3, j = 1:3
        if i!=j
            accum_norm += norm(J[i,j])
        end
    end
    @test accum_norm < tol

    Jskip = hadamard_jacobian(A,df,U,skip_index=skip_index)
    test_dense_eq_sparse = Bool[]
    test_skip = Bool[]
    for i=1:3,j=1:3
        push!(test_dense_eq_sparse, J[i,j] ≈ sparse(Jdense[i,j]))
        push!(test_skip, J[i,j] ≈ sparse(Jskip[i,j]))
    end
    @test all(test_dense_eq_sparse)
    @test all(test_skip)

    # for banded diagonal matrix test
    g(u) = SMatrix{2,2}([u[1] 0
                           0 u[2]])
    G = banded_function_evals(g,U[1:2])
    @test G[1,1] ≈ spdiagm(0=>U[1])
    @test norm(G[1,2]) + norm(G[2,1]) < tol
    @test G[2,2] ≈ spdiagm(0=>U[2])

    # test blockcat
    B = [[i] for i=1:2,j=1:2]
    @test blockcat(size(B,2),B) ≈ [1 1;2 2]

    # test flattening of matrix
    @test blockcat(size(J,2),J) ≈ kron(I(3),first(A) - diagm(vec(sum(first(A),dims=1))))
end

@testset "Extra sparse/dense tests" begin
    tol = 5e2*eps()
    A,B = ntuple(x->sprandn(25,25,.1),2)
    A_list = (A,B)
    function flux2D(UL,UR)
        uL,vL = UL
        uR,vR = UR
        return (uL+uR),2*(vL+vR)
    end
    U = randn(25),randn(25)
    rhs_exact = (A*U[1] + U[1].*sum(A,dims=2), 2*B*U[2] + 2*U[2].*sum(B,dims=2))
    rhs1 = zero.(U)
    rhs2 = zero.(U)
    hadamard_sum_ATr!(rhs1,A_list,flux2D,U)
    hadamard_sum_ATr!(rhs2,Matrix.(A_list),flux2D,U) # dense version
    @test sum(norm.(rhs1 .- rhs2)) < tol
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
end
