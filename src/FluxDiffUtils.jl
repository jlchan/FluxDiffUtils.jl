module FluxDiffUtils

using LinearAlgebra
using SparseArrays
using StaticArrays

"""
    TupleOrSVector{N}

Either a NTuple or SVector (e.g., fast static container) of length N.
"""
const TupleOrSVector{N} = Union{NTuple{N,T},SVector{N,T}} where {T}


# flux differencing routines
export hadamard_sum, hadamard_sum_ATr!
include("hadamard_sum.jl")

# flux differencing jacobian matrix routines
export hadamard_jacobian, hadamard_jacobian!
export blockcat # flattens matrices
include("hadamard_jacobian.jl")

# for jacobian formulas involving chain rule factors u(v) -> du/dv
export banded_function_evals, banded_function_evals!
include("banded_function_evals.jl")


end
