# helper functions
bmult(a,b) = a.*b # broadcasted multiplication
unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))

# unzip for tuple - a little hacky
unzipsum(a::AbstractArray) = sum(a)
unzipsum(a::Tuple) = sum.(unzip(a))

# hadamard function matrix utilities
row_range(j,A_list::NTuple{N,AbstractArray}) where {N} = axes(first(A_list),1)
row_range(j,A_list::NTuple{N,SparseMatrixCSC}) where {N} =
    union(getindex.(rowvals.(A_list),nzrange.(A_list, j))...)

#####
##### routine works for both dense/sparse matrix routines
#####

# tuple ATr_list dispatch to both dense/sparse
function hadamard_sum(A_list::NTuple{N,T}, F::Fxn, u, Fargs...;
                      skip_index=(i,j)->false) where {N,T,Fxn}
    rhs = zero.(u)
    hadamard_sum_ATr!(rhs,transpose.(A_list),F,u,Fargs...; skip_index=skip_index)
    return rhs
end

function hadamard_sum!(rhs, A_list::NTuple{N,T}, F::Fxn,
                       u, Fargs...; skip_index=(i,j)->false) where {N,T,Fxn}
    hadamard_sum_ATr!(rhs,transpose.(A_list),F,u,Fargs...; skip_index=skip_index)
end

"""
    hadamard_sum_ATr!(rhs, ATr_list::NTuple{N,AbstractArray{T,2}}, F::Fxn,
                      u, Fargs...; skip_index=(i,j)->false) where {N,T,Fxn}

computes ∑_i sum(Ai.*Fi,dims=2) where (Fi)_jk = F(uj,uk)[i]

rhs: output tuple
ATr_list: tuple of operators (A1,...,Ad)
F: flux function which outputs a d-tuple of flux values
u: solution used to evaluate the flux
Fargs: extra arguments to F(ui...,getindex.(Fargs,i)...,
                                uj...,getindex.(Fargs,j)...)
(optional) skip_index(i,j)==true skips computing fluxes for (i,j)
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

"""
    scale_factor(hadamard_product_type::Symbol)

chooses a sign based on the type of hadamard product - :sym,:skew
"""
function scale_factor(hadamard_product_type::Symbol)
    if hadamard_product_type == :sym
        return I
    elseif hadamard_product_type == :skew
        return -I
    end
end

"""
    hadamard_jacobian(A_list::NTuple{N,AbstractArray},
                           dF::Fxn, U,Fargs...; skip_index=(i,j)->false) where {N,Fxn}

    For when the hadamard product type not specified

    A_list = tuple of operators
    dF = Jacobian of the flux function
    U = solution at which to evaluate the Jacobian
    Fargs = extra args for df(uL,uR)
    skip_index(i,j) = optional function specifying whether to skip computation of entries
"""
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

# dispatches for both dense/sparse
"""
    hadamard_jacobian(A_list,
                      hadamard_product_type::Symbol, dF::Fxn, U,
                      Fargs...; skip_index=(i,j)->false) where {N,Fxn}

    A_list = tuple of operators
    hadamard_product_type = :skew, :sym
    dF = Jacobian of the flux function
    U = solution at which to evaluate the Jacobian
    Fargs = extra args for df(uL,uR)
    skip_index(i,j) = optional function to skip computation of (i,j)th entry
"""
function hadamard_jacobian(A_list,
                           hadamard_product_type, dF::Fxn, U,
                           Fargs...; skip_index=(i,j)->false) where {N,Fxn}
    Nfields = length(U)
    # sum(A_list) = sparse matrix with union of A[i] entries
    A = SMatrix{Nfields,Nfields}([zero(first(A_list)) for i=1:Nfields, j=1:Nfields])
    hadamard_jacobian!(A, A_list, hadamard_product_type, dF, U,
                       Fargs...; skip_index=skip_index)
    return A
end

# handles both dense/sparse matrices
"""function hadamard_jacobian!(A::NTuple{N,NTuple{N,AbstractArray}},
                               A_list::NTuple{Nd,AbstractArray},
                               hadamard_product_type::Symbol, dF::Fxn, U,
                               Fargs...; skip_index=(i,j)->false) where {N,Nd,Fxn}

    A = array for storing Jacobian output
    A_list = tuple of operators
    hadamard_product_type = :skew, :sym
    dF = Jacobian of the flux function
    U = solution at which to evaluate the Jacobian
    Fargs = extra args for df(uL,uR)
    skip_index(i,j) = optional function to skip computation of (i,j)th entry
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

computes block-banded matrix whose bands are entries of matrix-valued
function evals (e.g., a Jacobian function).
"""
function banded_function_evals(mat_fun::Fxn, U, Fargs ...) where Fxn
    n = length(first(U))
    Nfields = length(U)
    A = SMatrix{Nfields,Nfields}([spzeros(n,n) for i = 1:Nfields, j = 1:Nfields])
    banded_function_evals!(A, mat_fun, U, Fargs...)
    return A
end

"""
    banded_function_evals!(A::SparseMatrixCSC,mat_fun::Fxn, U, Fargs ...) where Fxn

computes a block-banded matrix whose bands are entries of matrix-valued
function evals (e.g., a Jacobian function) - mutating version.
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
