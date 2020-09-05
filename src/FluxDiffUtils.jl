"""
Module FluxDiffJacobians

Computes explicit Jacobians for discretizations which utilize
flux differencing, e.g., can be cast as sum(Q.*F,dims=2), where
F_ij = f(ui,uj), and f(uL,uR) is a symmetric + consistent numerical flux
"""

module FluxDiffUtils

using LinearAlgebra
using SparseArrays

# flux differencing routines
export hadamard_sum, hadamard_sum!

# jacobian matrix routines
export hadamard_jacobian # add hadamard_jacobian! in next version
export banded_matrix_function, banded_matrix_function!
export accum_hadamard_jacobian! # TODO: remove in next version

#####
##### single-operator dispatch functions
#####
function hadamard_sum(ATr::AbstractArray, F::Fxn, u, Fargs...) where Fxn
    return hadamard_sum(tuple(ATr),F,u,Fargs...)
end
function hadamard_sum!(rhs,ATr::AbstractArray, F::Fxn, u, Fargs...) where Fxn
    return hadamard_sum(rhs,tuple(ATr),F,u,Fargs...)
end
function hadamard_jacobian(A_template::SparseMatrixCSC, dF::Fxn,
                           U, Fargs...; scale = -1) where Fxn
    return hadamard_jacobian(tuple(A_template), dF, U, Fargs...; scale = scale)
end

include("flux_diff.jl")
include("jacobians.jl")

end
