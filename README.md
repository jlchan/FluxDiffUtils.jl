# FluxDiffUtils

[![Build Status](https://travis-ci.com/jlchan/FluxDiffUtils.jl.svg?branch=master)](https://travis-ci.com/jlchan/FluxDiffUtils.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/jlchan/FluxDiffUtils.jl?svg=true)](https://ci.appveyor.com/project/jlchan/FluxDiffUtils-jl)
[![Codecov](https://codecov.io/gh/jlchan/FluxDiffUtils.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jlchan/FluxDiffUtils.jl)

Utilities for flux differencing, as well as Jacobian computations for flux differencing type discretizations (given derivatives of flux functions). Code based in part on this [preprint](https://arxiv.org/abs/2006.07504).

## Example
```
using LinearAlgebra
using FluxDiffUtils

U = (randn(4),randn(4))
flux(uL,vL,uR,vR) = (.5*(uL+uR), .5*(vL+vR)),(.5*(uL+uR), .5*(vL+vR))
df(uL,vL,uR,vR) = ([.5 0; 0 .5], [.5 0; 0 .5]) # jacobians w.r.t. uR,vR
A_list = (A->A+A').(ntuple(x->randn(4,4),2)) # make a list of symmetric matrices

# compute sum(A.*F,dims=2) where Fij = flux(ui,uj)
rhs = hadamard_sum(A_list,flux,U)
jac = hadamard_jacobian(A_list,df,U)

# check against analytical formula
jac11_exact = sum((A->.5*(A - diagm(vec(sum(A,dims=2))))).(A_list))
@assert norm(jac11_exact-jac[1][1]) < 1e-12
```

## Conventions:
- Assumes non-grouped arguments for both fluxes and derivatives (e.g., FluxDiffUtils expects fluxes of the form `f(u1,u2,v1,v2)` instead of `f(U=(u1,u2),V=(v1,v2))`)
- Assumes the number of outputs from the flux matches the number of operators passed in (e.g., if `f(uL,vL)` has 2 outputs `g,h`, you should provide matrices `(A1, A2)` which will then compute `sum(A1.*g + A2.*h, dims = 2)`)
- Assumes derivatives of flux functions `f(uL,uR)` are taken with respect to `uR`
- Jacobians are returned in block form as tuples of tuples (some assembly required). Number of blocks per dimension is determined by length of input `U = (u1,...,u_Nfields)`
