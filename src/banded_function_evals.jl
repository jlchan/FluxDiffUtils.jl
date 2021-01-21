#####
##### other functions for computing Jacobians
#####

"""
    banded_function_evals(mat_fun::Fxn, U, Fargs...)

Computes block-banded matrix whose bands are entries of matrix-valued
function evals (e.g., a Jacobian). Returns SMatrix whose blocks correspond to function
components evaluated at values of U.

## Example:
```julia
julia> mat_fun(U) = [U[1] U[2]; U[2] U[1]]
julia> U = (randn(10),randn(10))
julia> banded_function_evals(mat_fun,U)
```
"""
function banded_function_evals(mat_fun::Fxn,U,Fargs...) where Fxn
    n = length(first(U))
    Nfields = length(U)
    A = SMatrix{Nfields,Nfields}([spzeros(n,n) for i = 1:Nfields, j = 1:Nfields])
    banded_function_evals!(A, mat_fun, U, Fargs...)
    return A
end

"""
    banded_function_evals!(A,mat_fun::Fxn, U, Fargs ...) where Fxn

Mutating version of [`banded_function_evals`](@ref).
"""
function banded_function_evals!(A,mat_fun::Fxn,U,Fargs...) where {Fxn}
    Nfields = size(A,2)
    num_pts = length(first(U))

    for i = 1:num_pts
        mat_i = mat_fun(getindex.(U,i),getindex.(Fargs,i)...)
        for n = 1:Nfields, m = 1:Nfields
            A[m,n][i,i] = mat_i[m,n] # TODO: replace with fast sparse constructor
        end
    end
end
