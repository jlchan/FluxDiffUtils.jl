"""
Module FluxDiffJacobians

Computes explicit Jacobians for discretizations which utilize
flux differencing, e.g., can be cast as sum(Q.*F,dims=2), where
F_ij = f(ui,uj), and f(uL,uR) is a symmetric + consistent numerical flux
"""

module FluxDiffJacobians

using LinearAlgebra
using SparseArrays

# flux differencing routines
export hadamard_sum, hadamard_sum!

# jacobian matrix routines
export hadamard_jacobian # add hadamard_jacobian! in next version
export banded_matrix_function, banded_matrix_function!
export accum_hadamard_jacobian! # TODO: remove in next version

include("flux_diff.jl") # new interfaces
include("jacobians.jl") # TODO: update API

end
