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
function hadamard_sum(ATr_list::NTuple{N,T}, F::Fxn, u, Fargs...;
                      skip_index=(i,j)->false) where {N,T,Fxn}
    rhs = zero.(u)
    hadamard_sum!(rhs,ATr_list,F,u,Fargs...; skip_index=skip_index)
    return rhs
end

# # specialize for the case when it's just a single array
# function hadamard_sum(ATr_list::AbstractArray{T,2}, F::Fxn, u, Fargs...;
#                       skip_index=(i,j)->false) where {N,T,Fxn}
#     rhs = zero.(u)
#     # wrap ATr_list and output of F in tuples
#     hadamard_sum!(rhs,(ATr_list,),(x->(x,))∘F,u,Fargs...; skip_index=skip_index)
#     return rhs
# end

"function hadamard_sum!(rhs, ATr_list::NTuple{N,AbstractArray{T,2}}, F::Fxn,
                        u, Fargs...; skip_index=(i,j)->false) where {N,T,Fxn}

computes ∑ sum(Ai.*Fi,dims=2) where (Fi)_jk = F(uj,uk)[i]

rhs: output tuple
ATr_list: tuple of operators (A1,...,Ad)
F: flux function which outputs a d-tuple of flux values
u: solution used to evaluate the flux
Fargs: extra arguments to F(ui...,getindex.(Fargs,i)...,
                                uj...,getindex.(Fargs,j)...)
(optional) skip_index(i,j)==true skips computing fluxes for (i,j)
"
function hadamard_sum!(rhs, ATr_list::NTuple{N,AbstractArray{T,2}}, F::Fxn,
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
                Fij = F(ui...,getindex.(Fargs,i)...,uj...,getindex.(Fargs,j)...)
                val_i .+= unzipsum(bmult.(ATrij_list,Fij))
            end
        end
        setindex!.(rhs,val_i,i)
    end
end

#####
##### Jacobian functions
#####

# top-level dispatch based on matrix information - :sym,:skew
function scale_factor(matrix_type::Symbol)
    if matrix_type == :sym
        return I
    elseif matrix_type == :skew
        return -I
    end
end

# matrix type not specified
function hadamard_jacobian(A_template_list::NTuple{N,AbstractArray}, dF::Fxn,
                            U, Fargs...; skip_index=(i,j)->false) where {N,Fxn}

    # make symmetric and skew parts, call jacobian on each part
    Asym_template_list  = (A->.5*(A+A')).(A_template_list)
    Askew_template_list = (A->.5*(A-A')).(A_template_list)
    return ((x,y)->x .+ y).(
           hadamard_jacobian(Asym_template_list,:sym,dF,U,
                             Fargs...;skip_index=skip_index),
           hadamard_jacobian(Askew_template_list,:skew,dF,U,
                             Fargs...;skip_index=skip_index)
           )
end

# dispatches for both dense/sparse
function hadamard_jacobian(A_template_list::NTuple{N,AbstractArray},
                           matrix_type::Symbol, dF::Fxn, U,
                           Fargs...; skip_index=(i,j)->false) where {N,Fxn}
    Nfields = length(U)
    A = ntuple(x->ntuple(x->zero.(first(A_template_list)),Nfields),Nfields)
    hadamard_jacobian!(A,A_template_list, matrix_type, dF, U,
                        Fargs...;skip_index=skip_index)
    return A
end

# handles both dense/sparse matrices
function hadamard_jacobian!(A::NTuple{N,NTuple{N,AbstractArray}},
                            A_template_list::NTuple{Nd,AbstractArray},
                            matrix_type::Symbol, dF::Fxn, U,
                            Fargs...;skip_index=(i,j)->false) where {N,Nd,Fxn}
    Nfields = length(U)
    num_pts = size(first(A_template_list),1)

    # accumulator for sum(Q.*dF,1) over jth column
    dFaccum = zeros(eltype(first(A_template_list)),Nfields,Nfields)

    # loop over cols + non-zero ids
    for j = 1:num_pts
       Uj = getindex.(U,j)
       fill!(dFaccum,zero(eltype(first(A_template_list))))
       for i in row_range(j,A_template_list)
           if skip_index(i,j)==false
               Ui = getindex.(U,i)
               A_ij_list = getindex.(A_template_list,i,j)
               dFij = dF(Ui...,getindex.(Fargs,i)...,
                         Uj...,getindex.(Fargs,j)...)

               for n = 1:length(U), m=1:length(U)
                   dFijQ = sum(bmult.(getindex.(dFij,m,n),A_ij_list)) # sum result for multiple operators
                   A[m][n][i,j] += dFijQ
                   dFaccum[m,n] += dFijQ # accumulate column sums on-the-fly
               end
           end
       end

       # add diagonal entry for each block
       for n=1:Nfields, m=1:Nfields
           A[m][n][j,j] += scale_factor(matrix_type)*dFaccum[m,n]
       end
    end
end

#####
##### other functions for computing Jacobians
#####

"
function banded_matrix_function(mat_fun::Fxn, U, Fargs...)

computes block-banded matrix whose bands are entries of matrix-valued
function evals (e.g., a Jacobian function).
"
function banded_matrix_function(mat_fun::Fxn, U, Fargs ...) where Fxn
    Nfields = length(U)
    n = length(first(U))
    A = ntuple(x->ntuple(x->spzeros(n,n),Nfields),Nfields)
    banded_matrix_function!(A, mat_fun, U, Fargs...)
    return A
end

"
function banded_matrix_function!(A::SparseMatrixCSC,mat_fun::Fxn, U, Fargs ...) where Fxn

computes a block-banded matrix whose bands are entries of matrix-valued
function evals (e.g., a Jacobian function) - mutating version.
"
function banded_matrix_function!(A::NTuple{N,NTuple{N,AbstractArray}},
                                 mat_fun::Fxn, U, Fargs ...) where {N,Fxn}
    Nfields = length(U)
    num_pts = length(first(U))

    for i = 1:num_pts
        mat_i = mat_fun(getindex.(U,i)...,getindex.(Fargs,i)...)
        for n = 1:Nfields, m = 1:Nfields
            A[m][n][i,i] = mat_i[m,n] # TODO: replace with fast sparse constructor
        end
    end
end
