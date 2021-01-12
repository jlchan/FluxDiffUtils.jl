module FluxDiffUtils

using LinearAlgebra
using SparseArrays
using StaticArrays

# flattens matrices
export blockcat

# flux differencing routines
export hadamard_sum, hadamard_sum!, hadamard_sum_ATr!

# flux differencing jacobian matrix routines
export hadamard_jacobian, hadamard_jacobian!

# for jacobian formulas involving chain rule factors u(v) -> du/dv
export banded_function_evals, banded_function_evals!

include("flux_diff_utils.jl")

end
