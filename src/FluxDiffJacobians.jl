"""
Module FluxDiffJacobians

Computes explicit Jacobians for discretizations which utilize
flux differencing, e.g., can be cast as sum(Q.*F,dims=2), where
F_ij = f(ui,uj), and f(uL,uR) is a symmetric + consistent numerical flux
"""

module FluxDiffJacobians

using LinearAlgebra
using SparseArrays

export hadamard_jacobian # add hadamard_jacobian! in next version
export hadamard_sum, hadamard_sum!
export banded_matrix_function, banded_matrix_function!

export accum_hadamard_jacobian! # TODO: remove in next version

include("flux_diff.jl") # new interfaces

# include("sparse_flux_diff.jl") # TODO: remove in next version
include("sparse_jacobian_functions.jl") # TODO: update API

end
