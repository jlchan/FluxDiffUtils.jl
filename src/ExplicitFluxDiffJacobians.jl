"""
Module ExplicitFluxDiffJacobians

Computes explicit Jacobians for discretizations which utilize
flux differencing, e.g., can be cast as sum(Q.*F,dims=2), where
F_ij = f(ui,uj), and f(uL,uR) is a symmetric + consistent numerical flux
"""

module ExplicitFluxDiffJacobians

using LinearAlgebra
using SparseArrays
# using ForwardDiff
# using StaticArrays
# using UnPack

# export columnize
export hadamard_jacobian, accum_hadamard_jacobian!
export hadamard_sum, hadamard_sum!
export banded_matrix_function, banded_matrix_function!

include("sparse_jacobian_functions.jl")

## TODO: specialize for dense matrices too

end
