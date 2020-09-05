"""
Module FluxDiffJacobians

Computes explicit Jacobians for discretizations which utilize
flux differencing, e.g., can be cast as sum(Q.*F,dims=2), where
F_ij = f(ui,uj), and f(uL,uR) is a symmetric + consistent numerical flux
"""

module FluxDiffJacobians

using LinearAlgebra
using SparseArrays

export hadamard_jacobian #, hadamard_jacobian! to dispatch over tuple args
export hadamard_sum, hadamard_sum!
export banded_matrix_function, banded_matrix_function!

export accum_hadamard_jacobian! # TODO: remove

include("sparse_jacobian_functions.jl")

end
