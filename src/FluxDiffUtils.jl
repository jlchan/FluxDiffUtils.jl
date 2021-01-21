module FluxDiffUtils

using LinearAlgebra
using SparseArrays
using StaticArrays

# flux differencing routines
export hadamard_sum, hadamard_sum!, hadamard_sum_ATr!
include("hadamard_sum.jl")

# flux differencing jacobian matrix routines
export hadamard_jacobian, hadamard_jacobian!
export blockcat # flattens matrices
include("hadamard_jacobian.jl")

# for jacobian formulas involving chain rule factors u(v) -> du/dv
export banded_function_evals, banded_function_evals!
include("banded_function_evals.jl")


end
