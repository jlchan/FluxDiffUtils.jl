## sparse hadamard function matrix utilities

# top-level dispatch based on matrix information - :sym,:skew
function scale_factor(matrix_type::Symbol)
    if matrix_type == :sym
        return I
    elseif matrix_type == :skew
        return -I
    end
end

# matrix type not specified
function hadamard_jacobian(A_template_list::NTuple{N,AbstractArray},
                           dF::Fxn, U, Fargs...) where {N,Fxn}

    # make symmetric and skew parts, call jacobian on each part
    Asym_template_list  = (A->.5*(A+A')).(A_template_list)
    Askew_template_list = (A->.5*(A-A')).(A_template_list)
    return ((x,y)->x .+ y).(
           hadamard_jacobian(Asym_template_list,:sym,dF,U,Fargs...),
           hadamard_jacobian(Askew_template_list,:skew,dF,U,Fargs...)
           )
end

# dispatches for both dense/sparse
function hadamard_jacobian(A_template_list::NTuple{N,AbstractArray},
                           matrix_type::Symbol, dF::Fxn,
                           U, Fargs...) where {N,Fxn}
    Nfields = length(U)
    A = ntuple(x->ntuple(x->zero.(first(A_template_list)),Nfields),Nfields)
    hadamard_jacobian!(A,A_template_list, matrix_type, dF, U, Fargs...)
    return A
end

row_range(j,A_list::NTuple{N,AbstractArray}) where N = axes(first(A_list),1)
row_range(j,A_list::NTuple{N,SparseMatrixCSC}) where N =
    union(getindex.(rowvals.(A_list),nzrange.(A_list, j))...)

# handles both dense/sparse matrices
function hadamard_jacobian!(A::NTuple{N,NTuple{N,AbstractArray}},
                            A_template_list::NTuple{Nd,AbstractArray},
                            matrix_type::Symbol, dF::Fxn, U, Fargs ...) where {N,Nd,Fxn}
    Nfields = length(U)
    num_pts = size(first(A_template_list),1)

    # accumulator for sum(Q.*dF,1) over jth column
    dFaccum = zeros(eltype(first(A_template_list)),Nfields,Nfields)

    # loop over cols + non-zero ids
    for j = 1:num_pts
       Uj = getindex.(U,j)
       fill!(dFaccum,zero(eltype(first(A_template_list))))
       for i in row_range(j,A_template_list)
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

       # add diagonal entry for each block
       for n=1:Nfields, m=1:Nfields
           A[m][n][j,j] += scale_factor(matrix_type)*dFaccum[m,n]
       end
    end
end

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
