"""
    blockcat(n, A)
    blockcat(n, A::AbstractMatrix{SparseMatrixCSC{Ti,Tv}}) where {Ti,Tv}

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

#####
##### Jacobian functions
#####

function scale_factor(hadamard_product_type::Symbol)
    if hadamard_product_type == :skew
        return -1.0I
    else
        return 1.0I
    end
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
- `hadamard_product_type` = `:skew`, `:sym`. Specifies if `Ai.*Fi` is skew or symmetric.
- `dF` = Jacobian of the flux function `F(uL,uR)` with respect to `uR`
- `U` = solution at which to evaluate the Jacobian
- `Fargs` = extra args for `dF(uL,uR)`
- (optional) `skip_index(i,j)` = optional function to skip computation of (i,j)th entry
"""
function hadamard_jacobian(A_list,dF::Fxn,U,Fargs...;skip_index=(i,j)->false) where {N,Fxn}

    # make symmetric and skew parts, call jacobian on each part
    Asym_list  = (A->.5*(A+A')).(A_list)
    Askew_list = (A->.5*(A-A')).(A_list)
    return ((x,y)->x .+ y).(
           hadamard_jacobian(Asym_list,:sym,dF,U,Fargs...;skip_index=skip_index),
           hadamard_jacobian(Askew_list,:skew,dF,U,Fargs...;skip_index=skip_index)
           )
end

function hadamard_jacobian(A_list,hadamard_product_type,dF::Fxn,U,Fargs...;
                           skip_index=(i,j)->false) where {Fxn}
    Nfields = length(U)
    A = SMatrix{Nfields,Nfields}([zero(first(A_list)) for i=1:Nfields, j=1:Nfields])
    hadamard_jacobian!(A, A_list, hadamard_product_type, dF, U, Fargs...; skip_index=skip_index)
    return A
end

"""
    hadamard_jacobian!(A::SMatrix{N,N},A_list,hadamard_product_type::Symbol,
                       dF::Fxn,U,Fargs...; skip_index=(i,j)->false) where {N,Nd,Fxn}

Mutating version of [`hadamard_jacobian`](@ref). `A` = matrix for storing Jacobian
output, with each entry storing a block of the Jacobian.
"""
function hadamard_jacobian!(A::SMatrix{Nfields,Nfields},A_list,hadamard_product_type::Symbol,
                            dF::Fxn,U,Fargs...; skip_index=(i,j)->false) where {Fxn,Nfields}

    rows,cols = axes(first(A_list))

    # accumulator for sum(Q.*dF,1) over jth column
    Atype = eltype(first(A_list))
    dFaccum = zeros(Atype,Nfields,Nfields)

    # loop over cols + non-zero ids
    for j in cols
       Uj = getindex.(U,j)
       fill!(dFaccum,zero(Atype))
       for i in rows
           if skip_index(i,j)==false
               Ui = getindex.(U,i)
               A_ij_list = getindex.(A_list,i,j)
               dFij = dF(Ui,Uj,getindex.(Fargs,i)...,getindex.(Fargs,j)...)

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

function hadamard_jacobian!(A::SMatrix{N,N},A_list::NTuple{D,SparseMatrixCSC},
                            hadamard_product_type::Symbol,dF::Fxn,U,Fargs...;
                            skip_index=(i,j)->false) where {N,D,Fxn}
    for (i,Ai) in enumerate(A_list)
        dF_i = (x->getindex(x,i))∘dF
        hadamard_jacobian!(A,Ai,hadamard_product_type,dF_i,U,Fargs...)
    end
end

function hadamard_jacobian!(A::SMatrix{N,N},Amat::SparseMatrixCSC,
                            hadamard_product_type::Symbol,dF::Fxn,U,Fargs...) where {N,Fxn}
    Nfields = length(U)

    rows = rowvals(Amat)
    vals = nonzeros(Amat)

    # accumulator for sum(Q.*dF,1) over jth column
    dFaccum = zeros(eltype(Amat),Nfields,Nfields)

    # loop over cols + non-zero ids
    for j = 1:size(Amat,2)
       Uj = getindex.(U,j)
       fill!(dFaccum,zero(eltype(Amat)))
       for row_id in nzrange(Amat,j)
           i = rows[row_id]
           A_ij = vals[row_id]

           Ui = getindex.(U,i)
           dFij = dF(Ui,Uj,getindex.(Fargs,i)...,getindex.(Fargs,j)...)

           for n = 1:length(U), m=1:length(U)
               dFijQ = dFij[m,n].*A_ij # sum result for single op
               A[m,n][i,j] += dFijQ
               dFaccum[m,n] += dFijQ # accumulate column sums on-the-fly
           end
       end

       # add diagonal entry for each block
       for n=1:Nfields, m=1:Nfields
           A[m,n][j,j] += scale_factor(hadamard_product_type)*dFaccum[m,n]
       end
    end
end
