using FluxDiffUtils
using Test

using SparseArrays
using StaticArrays
using LinearAlgebra
using ForwardDiff

@testset "1D flux diff tests" begin
    u = collect(LinRange(-1,1,10))
    U = (u,2*u)
    function flux1D(uL,vL,uR,vR)
        avg(x,y) = @. .5*(x+y)
        Fx = avg(uL,uR),avg(vL,vR)
        return Fx
    end
    A = randn(10,10)
    A[8:10,8:10] .= 0 # for testing skip_index
    skip_index(i,j) = (i>8)&(j>8)
    A = (A->A-A')(A) # make skew for exact formula testing
    ATr = (A->Matrix(transpose(A)))(A)

    rhs = hadamard_sum((ATr,),tuple∘flux1D,U,skip_index=skip_index)
    @test (A*U[1] + U[1].*sum(A,dims=2)) ≈ 2*rhs[1]
    @test (A*U[2] + U[2].*sum(A,dims=2)) ≈ 2*rhs[2]
end

# make 3-field solution
u = collect(LinRange(-1,1,10))
# U = (u,u,u)
U = SVector{3}(u,2*u,3*u)

# define flux function and its derivative
function flux_function(uL,vL,wL,uR,vR,wR)
    avg(x,y) = @. .5*(x+y)
    Fx = SVector{3}(avg(uL,uR),avg(vL,vR),avg(wL,wR))
    Fy = SVector{3}(avg(uL,uR),avg(vL,vR),avg(wL,wR))
    return SVector{2}(Fx,Fy)
end

dfx(UL,UR) = ForwardDiff.jacobian(UR->flux_function(UL...,UR...)[1], UR)
dfy(UL,UR) = ForwardDiff.jacobian(UR->flux_function(UL...,UR...)[2], UR)
df(uL,vL,wL,uR,vR,wR) = dfx(SVector{3}(uL,vL,wL),SVector{3}(uR,vR,wR)),
                        dfy(SVector{3}(uL,vL,wL),SVector{3}(uR,vR,wR))

A = randn(10,10)
A[8:10,8:10] .= 0 # for testing skip_index
skip_index(i,j) = (i>8)&(j>8)
A = ntuple(x->A,2)
A = (A->A-A').(A) # make skew for exact formula testing
ATr = (A->Matrix(transpose(A))).(A)

@testset "Flux diff tests w/StaticArrays" begin

    rhs = hadamard_sum(ATr,flux_function,U)
    @test (A[1]*U[1] + U[1].*sum(A[1],dims=2)) ≈ rhs[1]
    @test (A[1]*U[2] + U[2].*sum(A[1],dims=2)) ≈ rhs[2]
    @test (A[1]*U[3] + U[3].*sum(A[1],dims=2)) ≈ rhs[3]

    rhs_sparse = hadamard_sum(sparse.(ATr),flux_function,U)
    @test all(rhs .≈ rhs_sparse)

    rhs_skipped = hadamard_sum(sparse.(ATr),flux_function,U;
                              skip_index=skip_index)
    @test all(rhs .≈ rhs_skipped)

    # test tuple args too
    v = collect(LinRange(-1,1,10))
    V = (v,2*v,3*v)
    # W = [v,v,v]
    # define flux function and its derivative
    function flux_function_tuple(uL,vL,wL,uR,vR,wR)
        avg(x,y) = @. .5*(x+y)
        Fx = avg(uL,uR),avg(vL,vR),avg(wL,wR)
        Fy = avg(uL,uR),avg(vL,vR),avg(wL,wR)
        return Fx,Fy
    end
    rhs2 = hadamard_sum(ATr,flux_function_tuple,V)
    @test all(rhs .≈ rhs2)

    # function test(u,F)
    #     i,j = 1,2
    #     Fargs = ()
    #     rhs = u
    #     val_i = zeros(eltype(first(rhs)),length(rhs))
    #     ui = getindex.(u,i)
    #     uj = getindex.(u,j)
    #     ATrij = getindex.(ATr,j,i)
    #     Fij = F(ui...,getindex.(Fargs,i)...,uj...,getindex.(Fargs,j)...)
    #     return ATrij,Fij
    # end
    # Aij,Fij1 = test(U,flux_function)
    # _,Fij2 = test(V,flux_function_tuple)
    # _,Fij3 = test(W,flux_function_tuple)
    # # val_i .+= sum.(unzip(bmult.(ATrij,Fij)))
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

    Jskip = hadamard_jacobian(A,df,U,skip_index=skip_index)
    test_dense_eq_sparse = Bool[]
    test_skip = Bool[]
    for i=1:3,j=1:3
        push!(test_dense_eq_sparse, J[i][j] ≈ sparse(Jdense[i][j]))
        push!(test_skip, J[i][j] ≈ sparse(Jskip[i][j]))
    end
    @test all(test_dense_eq_sparse)
    @test all(test_skip)

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
