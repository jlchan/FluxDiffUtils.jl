# FluxDiffUtils

[![Build Status](https://travis-ci.com/jlchan/FluxDiffUtils.jl.svg?branch=master)](https://travis-ci.com/jlchan/FluxDiffUtils.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/jlchan/FluxDiffUtils.jl?svg=true)](https://ci.appveyor.com/project/jlchan/FluxDiffUtils-jl)
[![Codecov](https://codecov.io/gh/jlchan/FluxDiffUtils.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jlchan/FluxDiffUtils.jl)

Utilities for flux differencing, as well as Jacobian computations for flux differencing type discretizations (given derivatives of flux functions). Code based in part on this [preprint](https://arxiv.org/abs/2006.07504).

## Performance

The routines are meant to be fairly general, but specialize depending on whether the operators are `AbstractArray` or `SparseMatrixCSC` to capitalize on sparsity. The code also appears to much faster than the old ESDG.jl hand-coded routines -when computing a flux differencing step using fluxes from [EntropyStableEuler.jl](https://github.com/jlchan/EntropyStableEuler.jl), FluxDiffUtils.jl was about 68 times faster on a single core.
```
613.832 μs (6092 allocations: 220.29 KiB) # old ESDG.jl routines
9.060 μs (27 allocations: 3.12 KiB) # FluxDiffUtils.jl
```

## Example
```
using LinearAlgebra
using FluxDiffUtils
using Test

# make 3-field solution
u = collect(LinRange(-1,1,4))
U = (u,u,u)

avg(x,y) = @. .5*(x+y)
function flux(uL,vL,wL,uR,vR,wR)
    Fx = avg(uL,uR),avg(vL,vR),avg(wL,wR)
    Fy = avg(uL,uR),avg(vL,vR),avg(wL,wR)
    return Fx,Fy
end

# jacobians w.r.t. (uR,vR)
df(uL,vL,uR,vR) = ([.5 0 0; 0 .5 0; 0 0 .5], [.5 0 0; 0 .5 0; 0 0 .5])
A_list = (A->A+A').(ntuple(x->randn(4,4),2)) # make symmetric to check formula

# hadamard_sum uses transpose for efficiency
ATr_list = (A->Matrix(transpose(A))).(A_list)
rhs = hadamard_sum(ATr_list,flux,U)

# jacobian computation doesn't need transpose
jac = hadamard_jacobian(A_list, df, U)
# jac = hadamard_jacobian(A_list, :sym, df, U) # faster version

jac11_exact = sum((A->.5*(A + diagm(vec(sum(A,dims=1))))).(A_list))
@test norm(jac11_exact-jac[1][1]) < 1e-12
```

## Conventions:
- Assumes non-grouped arguments for both fluxes and derivatives (e.g., FluxDiffUtils expects fluxes of the form `f(u1,u2,v1,v2)` instead of `f(U,V)` for `U=(u1,u2), V=(v1,v2)`).
- Assumes the number of outputs from the flux matches the number of operators passed in (e.g., if `f(uL,vL)` has 2 outputs `g,h`, you should provide matrices `(A1, A2)` which will then compute `sum(A1.*g + A2.*h, dims = 2)`)
- When computing Jacobian matrices, assumes derivatives of flux functions `f(uL,uR)` are taken with respect to `uR`.
- Jacobians are returned in block form as tuples of tuples (i.e., some assembly required). Number of blocks per dimension is determined by length of input `U = (u1,...,u_Nfields)`
- For efficiency, `hadamard_sum` takes in the transpose of A, while `hadamard_jacobian` takes in A.
- When computing Jacobians, specifying if matrices are symmetric or skew-symmetric by setting `hadamard_product_type` to `:sym` or `:skew` can improve efficiency.
