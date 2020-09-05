## sparse hadamard function matrix utilities

"
function hadamard_jacobian(Q::SparseMatrixCSC, dF::Fxn,
                           U, Fargs ...; scale = -1)

Assumes that Q/F is a skew-symmetric/symmetric pair
Can only deal with one coordinate component at a time in higher dimensions.
"
function hadamard_jacobian(A_template_list::NTuple{N,SparseMatrixCSC}, dF::Fxn,
                           U, Fargs...; scale = -1) where {N,Fxn}
    Nfields = length(U)
    n = size(first(A_template_list),2)
    A = ntuple(x->ntuple(x->spzeros(n,n),Nfields),Nfields)
    hadamard_jacobian!(A,A_template_list, dF, U, Fargs ...; scale = scale)
    # accum_hadamard_jacobian!(A,A_template,dF,U,Fargs...; scale=scale)
    return A
end

function hadamard_jacobian!(A::NTuple{N,NTuple{N,SparseMatrixCSC}},
                            A_template_list::NTuple{N,SparseMatrixCSC},
                            dF::Fxn, U, Fargs ...; scale = -1) where {N,Fxn}
    Nfields = length(U)
    num_pts = size(first(A_template_list),1)

    rows = rowvals.(A_template_list)

    # accumulator for sum(Q.*dF,1) over jth column
    dFaccum = zeros(eltype(first(A_template_list)),Nfields,Nfields)

    # loop over cols + non-zero ids
    for j = 1:num_pts
       Uj = getindex.(U,j)
       fill!(dFaccum,zero(eltype(first(A_template_list))))
       for i in union(getindex.(rows,nzrange.(A_template_list, j))...)
           Ui = getindex.(U,i)
           A_ij_list = getindex.(A_template_list,i,j)
           dFij = dF(Ui...,getindex.(Fargs,i)...,
                     Uj...,getindex.(Fargs,j)...)

           for n = 1:length(U), m=1:length(U)
               dFijQ = sum(getindex.(dFij,m,n) .* A_ij_list) # sum result for multiple operators
               A[m][n][i,j] += dFijQ
               dFaccum[m,n] += dFijQ # accumulate column sums on-the-fly
           end
       end

       # add diagonal entry for each block
       for n=1:Nfields, m=1:Nfields
           A[m][n][j,j] += scale*dFaccum[m,n]
       end
    end
end

"
function banded_matrix_function(mat_fun::Fxn, U, Fargs ...)

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
