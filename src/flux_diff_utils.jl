# helper functions
bmult(a,b) = a.*b # broadcasted multiplication

# hadamard function matrix utilities
row_range(j,A_list::NTuple{N,AbstractArray}) where {N} = axes(first(A_list),1)
row_range(j,A_list::NTuple{N,SparseMatrixCSC}) where {N} =
    union(getindex.(rowvals.(A_list),nzrange.(A_list, j))...)

"""
    blockcat(n, A)

Concatenates the n-by-n matrix of matrices A into a global matrix.
If eltype(A) = SparseMatrixCSC, blockcat returns a global sparse matrix.
Otherwise, blockcat returns a global dense array of type Array{eltype(first(A))}.
"""
function blockcat(n, A)
    A0 = first(A)
    nA = size(A0, 1)
    B  = Array{eltype(A0),2}(undef, n * nA, n * nA)
    for a in 1:n, j in 1:n
        B[(a-1)*nA .+ (1:nA), (j-1)*nA .+ (1:nA)] .= A[a,j]
    end
    return B
end

function blockcat(n, A::AbstractMatrix{SparseMatrixCSC{Ti,Tv}}) where {Ti,Tv}
    A0 = first(A)
    nA = size(A0, 1)
    B  = spzeros(n * nA, n * nA)
    for ii in 1:n, jj in 1:n
        for ijv in zip(findnz(A[ii,jj])...)
            i,j,Aij = ijv
            B[(ii-1)*nA + i, (jj-1)*nA + j] = Aij
        end
    end
    return B
end

# # original function courtesy of Mark Kittisopikul via Julia Slack
# function vhcat(cols::Tuple{Vararg{Int64,N}}, xs::AbstractVecOrMat{T}...) where {T,N}
#     cols_cs = cumsum(cols)
#     cols_cat = ( reduce(vcat, xs[ (1:cols[i]) .+ (cols_cs[i] - cols[i])]) for i in 1:last(cols) )
#     return reduce(hcat, cols_cat)
# end
# vhcat(col_dim,A) = hvcat(col_dim,permutedims(A)...) # doesn't work for SMatrix

#####
##### routine works for both dense/sparse matrix routines
#####

"""
    hadamard_sum(A_list::NTuple{N,T}, F::Fxn, u, Fargs...;
                 skip_index=(i,j)->false) where {N,T,Fxn}

computes ∑_i sum(Ai.*Fi,dims=2) where (Fi)_jk = F(uj,uk)[i]

Inputs
- `A_list`: tuple of operators (A1,...,Ad)
- `F`: flux function which outputs a d-tuple of flux vectors
- `u`: collection of solution values (or arrays) at which to evaluate `F`
- `Fargs`: extra arguments to `F(ui,getindex.(Fargs,i)...,
                            uj,getindex.(Fargs,j)...)``
- (optional) `skip_index(i,j)==true` skips computing fluxes for index (i,j)
"""
function hadamard_sum(A_list::NTuple{N,T}, F::Fxn, u, Fargs...;
                      skip_index=(i,j)->false) where {N,T,Fxn}
    rhs = zero.(u)
    hadamard_sum_ATr!(rhs,transpose.(A_list),F,u,Fargs...; skip_index=skip_index)
    return rhs
end

"""
    hadamard_sum!(rhs, A_list::NTuple{N,T}, F::Fxn, u, Fargs...;
                 skip_index=(i,j)->false) where {N,T,Fxn}

Mutating version of [`hadamard_sum`](@ref), where `rhs` is storage for output.
"""
function hadamard_sum!(rhs, A_list::NTuple{N,T}, F::Fxn,
                       u, Fargs...; skip_index=(i,j)->false) where {N,T,Fxn}
    hadamard_sum_ATr!(rhs,transpose.(A_list),F,u,Fargs...; skip_index=skip_index)
end

"""
    hadamard_sum_ATr!(rhs, ATr_list::NTuple{N,T}, F::Fxn, u, Fargs...;
                 skip_index=(i,j)->false) where {N,T,Fxn}

Same as [`hadamard_sum!`](@ref) but `ATr_list` contains transposed matrices.
"""
function hadamard_sum_ATr!(rhs, ATr_list::NTuple{N,T}, F::Fxn,
                        u, Fargs...; skip_index=(i,j)->false) where {N,T,Fxn}
    # cols = rowvals.(ATr_list)
    val_i = zeros(eltype(first(rhs)),length(rhs))
    for i = 1:size(first(ATr_list),2) # all ops should be same length
        ui = getindex.(u,i)
        fill!(val_i,zero(eltype(first(rhs))))

        # loop over nonzero inds for all matrices in ATr_list (dispatched)
        for j in row_range(i,ATr_list)
            if skip_index(i,j)==false
                uj = getindex.(u,j)
                ATrij_list = getindex.(ATr_list,j,i)
                Fij = F(ui,getindex.(Fargs,i)...,uj,getindex.(Fargs,j)...)
                val_i .+= sum(bmult.(ATrij_list,Fij))
            end
        end
        setindex!.(rhs,val_i,i)
    end
end

#####
##### Jacobian functions
#####

function scale_factor(hadamard_product_type::Symbol)
    if hadamard_product_type == :sym
        return I
    elseif hadamard_product_type == :skew
        return -I
    end
end

function hadamard_jacobian(A_list, dF::Fxn,
                           U, Fargs...; skip_index=(i,j)->false) where {N,Fxn}

    # make symmetric and skew parts, call jacobian on each part
    Asym_list  = (A->.5*(A+A')).(A_list)
    Askew_list = (A->.5*(A-A')).(A_list)
    return ((x,y)->x .+ y).(
           hadamard_jacobian(Asym_list,:sym,dF,U,Fargs...;skip_index=skip_index),
           hadamard_jacobian(Askew_list,:skew,dF,U,Fargs...;skip_index=skip_index)
           )
end

"""
    hadamard_jacobian(A_list, dF::Fxn, U, Fargs...;
                      skip_index=(i,j)->false) where {N,Fxn}

    hadamard_jacobian(A_list, hadamard_product_type, dF::Fxn, U,
                      Fargs...; skip_index=(i,j)->false) where {N,Fxn}

Computes Jacobian of the flux differencing term ∑_i sum(Ai.*Fi). Outputs array whose
entries are blocks of the Jacobian matrix corresponding to components of flux vectors.

Inputs:
- `A_list` = tuple of operators
- (optional) `hadamard_product_type` = `:skew`, `:sym`. Specifies if `Ai.*Fi` is skew or symmetric.
- `dF` = Jacobian of the flux function `F(uL,uR)` with respect to `uR`
- `U` = solution at which to evaluate the Jacobian
- `Fargs` = extra args for `dF(uL,uR)`
- (optional) `skip_index(i,j)` = optional function to skip computation of (i,j)th entry
"""
function hadamard_jacobian(A_list, hadamard_product_type, dF::Fxn, U,
                           Fargs...; skip_index=(i,j)->false) where {N,Fxn}
    Nfields = length(U)
    # sum(A_list) = sparse matrix with union of A[i] entries
    A = SMatrix{Nfields,Nfields}([zero(first(A_list)) for i=1:Nfields, j=1:Nfields])
    hadamard_jacobian!(A, A_list, hadamard_product_type, dF, U,
                       Fargs...; skip_index=skip_index)
    return A
end

# handles both dense/sparse matrices
"""
    hadamard_jacobian!(A::SMatrix{N,N},
                       A_list::NTuple{Nd,AbstractArray},
                       hadamard_product_type::Symbol, dF::Fxn, U,
                       Fargs...; skip_index=(i,j)->false) where {N,Nd,Fxn}

Mutating version of [`hadamard_jacobian`](@ref). `A` = matrix for storing Jacobian
output, with each entry storing a block of the Jacobian.

"""
function hadamard_jacobian!(A::SMatrix{N,N},
                            A_list::NTuple{Nd,AbstractArray},
                            hadamard_product_type::Symbol, dF::Fxn, U,
                            Fargs...; skip_index=(i,j)->false) where {N,Nd,Fxn}
    Nfields = length(U)
    num_pts = size(first(A_list),1)

    # accumulator for sum(Q.*dF,1) over jth column
    dFaccum = zeros(eltype(first(A_list)),Nfields,Nfields)

    # loop over cols + non-zero ids
    for j = 1:num_pts
       Uj = getindex.(U,j)
       fill!(dFaccum,zero(eltype(first(A_list))))
       for i in row_range(j,A_list)
           if skip_index(i,j)==false
               Ui = getindex.(U,i)
               A_ij_list = getindex.(A_list,i,j)
               dFij = dF(Ui,getindex.(Fargs,i)...,Uj,getindex.(Fargs,j)...)

               for n = 1:length(U), m=1:length(U)
                   dFijQ = sum(bmult.(getindex.(dFij,m,n),A_ij_list)) # sum result for multiple operators
                   A[m,n][i,j] += dFijQ
                   dFaccum[m,n] += dFijQ # accumulate column sums on-the-fly
               end
           end
       end

       # add diagonal entry for each block
       for n=1:Nfields, m=1:Nfields
           A[m,n][j,j] += scale_factor(hadamard_product_type)*dFaccum[m,n]
       end
    end
end

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
function banded_function_evals(mat_fun::Fxn, U, Fargs ...) where Fxn
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
function banded_function_evals!(A,mat_fun::Fxn, U, Fargs ...) where {Fxn}
    Nfields = size(A,2)
    num_pts = length(first(U))

    for i = 1:num_pts
        mat_i = mat_fun(getindex.(U,i),getindex.(Fargs,i)...)
        for n = 1:Nfields, m = 1:Nfields
            A[m,n][i,i] = mat_i[m,n] # TODO: replace with fast sparse constructor
        end
    end
end
