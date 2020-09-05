# FluxDiffJacobians

[![Build Status](https://travis-ci.com/jlchan/ExplicitFluxDiffJacobians.jl.svg?branch=master)](https://travis-ci.com/jlchan/ExplicitFluxDiffJacobians.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/jlchan/ExplicitFluxDiffJacobians.jl?svg=true)](https://ci.appveyor.com/project/jlchan/ExplicitFluxDiffJacobians-jl)
[![Codecov](https://codecov.io/gh/jlchan/ExplicitFluxDiffJacobians.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jlchan/ExplicitFluxDiffJacobians.jl)

Utilities for flux differencing, as well as Jacobian computations for flux differencing type discretizations. Code based in part on this [preprint](https://arxiv.org/abs/2006.07504).

Conventions:
- Assumes non-grouped arguments for both fluxes and derivatives (e.g., FluxDiffUtils expects fluxes of the form `f(u1,u2,v1,v2)` instead of `f(U=(u1,u2),V=(v1,v2))`)
- Jacobians are returned in block form as tuples of tuples (some assembly required). Number of blocks per dimension is determined by length of input `U = (u1,...,u_Nfields)`
