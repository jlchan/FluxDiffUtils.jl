# FluxDiffUtils Documentation

This package provides utilities for flux differencing and computation of flux differencing Jacobians in terms of derivatives of flux functions. The code based in part on the preprint ["Efficient computation of Jacobian matrices for entropy stable summation-by-parts schemes"](https://arxiv.org/abs/2006.07504).

The routines are meant to be fairly general, but specialize depending on whether the operators are general arrays or `SparseMatrixCSC` (to capitalize on sparsity).

## Example
```
using LinearAlgebra
using FluxDiffUtils
using Test

# make 3-field solution
u = collect(LinRange(-1,1,4))
U = (u,u,u)

avg(x,y) = @. .5*(x+y)
function flux(UL,UR)
    uL,vL,wL = UL
    uR,vR,wR = UR
    Fx = avg(uL,uR),avg(vL,vR),avg(wL,wR)
    Fy = avg(uL,uR),avg(vL,vR),avg(wL,wR)
    return SVector{3}(Fx),SVector{3}(Fy)
end

# jacobians w.r.t. (uR,vR)
df(uL,vL,uR,vR) = ([.5 0 0; 0 .5 0; 0 0 .5], [.5 0 0; 0 .5 0; 0 0 .5])
A_list = (A->A+A').(ntuple(x->randn(4,4),2)) # make symmetric to check formula

rhs = hadamard_sum(A_list,flux,U)

jac = hadamard_jacobian(A_list, df, U)
# jac = hadamard_jacobian(A_list, :sym, df, U) # optimized version

jac11_exact = sum((A->.5*(A + diagm(vec(sum(A,dims=1))))).(A_list))
@test norm(jac11_exact-jac[1,1]) < 1e-12

# converts tuple-block storage of jac to a global matrix
jac_global = hvcat(size(jac,1),jac...)
```

## Conventions

- We assume grouped arguments for both fluxes and derivatives (e.g., `FluxDiffUtils.jl` expects fluxes of the form  `f(U,V)` instead of `f(u1,u2,v1,v2)` for `U=(u1,u2), V=(v1,v2)`).
- We assume the number of outputs from the flux matches the number of operators passed in. In other words, if `f(uL,vL)` has 2 outputs `g,h`, you should provide matrices `(A1, A2)`. `hadamard_sum` will compute `sum(A1.*g + A2.*h, dims = 2)` (`hadamard_jacobian` behaves similarly).
- For Jacobian matrices, we assume derivatives of flux functions `f(uL,uR)` are taken with respect to the second argument `uR`.
- Jacobians are returned as a StaticArray of arrays, and can be concatenated into a global matrix using `hvcat(size(jac,1),jac...)`.
- Jacobian computations can be made more efficient by specifying if the Hadamard product `A.*F` (where `A` is a discretization matrix and `F` is a flux matrix) is symmetric or skew-symmetric by setting `hadamard_product_type` to `:sym` or `:skew`. Otherwise, `FluxDiffUtils.jl` will split the matrix `A` into skew and symmetric parts and compute Jacobians for each.

## Index

```@index
```

## Functions

```@autodocs
Modules = [FluxDiffUtils]
```
