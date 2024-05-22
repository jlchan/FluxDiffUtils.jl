# This repository is no longer maintained; please see StartUpDG.jl or Trixi.jl for my current Julia-based DG codes.

# FluxDiffUtils
[![Docs-stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jlchan.github.io/FluxDiffUtils.jl/stable)
[![Docs-dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jlchan.github.io/FluxDiffUtils.jl/dev)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/jlchan/FluxDiffUtils.jl?svg=true)](https://ci.appveyor.com/project/jlchan/FluxDiffUtils-jl)
[![Build status](https://github.com/jlchan/FluxDiffUtils.jl/workflows/CI/badge.svg)](https://github.com/jlchan/FluxDiffUtils.jl/actions)
[![Codecov](https://codecov.io/gh/jlchan/FluxDiffUtils.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jlchan/FluxDiffUtils.jl)

This package provides utilities for flux differencing and computation of flux differencing Jacobians in terms of derivatives of flux functions. The code based in part on the preprint ["Efficient computation of Jacobian matrices for entropy stable summation-by-parts schemes"](https://arxiv.org/abs/2006.07504).

The routines are meant to be fairly general, but specialize depending on whether the operators are general arrays or `SparseMatrixCSC` (to capitalize on sparsity).

## Example

```julia
using LinearAlgebra
using FluxDiffUtils
using Test

# make 3-field solution
u = collect(LinRange(-1,1,4))
U = (u,u,u)

# define a flux fS
avg(x,y) = @. .5*(x+y)
function fS(UL,UR)
    uL,vL,wL = UL
    uR,vR,wR = UR
    Fx = avg(uL,uR),avg(vL,vR),avg(wL,wR)
    Fy = avg(uL,uR),avg(vL,vR),avg(wL,wR)
    return SVector{3}(Fx),SVector{3}(Fy)
end

# jacobians w.r.t. (uR,vR)
dfS(UL,UR) = ([.5 0 0; 0 .5 0; 0 0 .5], [.5 0 0; 0 .5 0; 0 0 .5])
A_list = (A->A+A').(ntuple(x->randn(4,4),2)) # make symmetric to check formula

rhs = hadamard_sum(A_list,fS,U)

jac = hadamard_jacobian(A_list, dfS, U)
# jac = hadamard_jacobian(A_list, :sym, df, U) # optimized version

jac11_exact = sum((A->.5*(A + diagm(vec(sum(A,dims=1))))).(A_list))
@test norm(jac11_exact-jac[1,1]) < 1e-12

# converts tuple-block storage of jac to a global matrix
jac_global = blockcat(size(jac,2),jac)
```
