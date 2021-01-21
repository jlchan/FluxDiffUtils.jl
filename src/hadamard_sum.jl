# helper functions
bmult(a,b) = a.*b # broadcasted multiplication

# hadamard function matrix utilities
row_range(j,A_list::NTuple{N,AbstractArray}) where {N} = axes(first(A_list),1)
row_range(j,A_list::NTuple{N,SparseMatrixCSC}) where {N} =
    union(getindex.(rowvals.(A_list),nzrange.(A_list, j))...)

#####
##### routine works for both dense/sparse matrix routines
#####

"""
    hadamard_sum(A_list, F::Fxn, u, Fargs...;
                 skip_index=(i,j)->false) where {N,T,Fxn}

computes ∑_i sum(Ai.*Fi,dims=2) where (Fi)_jk = F(uj,uk)[i]

Inputs
- `A_list`: tuple (or similar container) of operators (A1,...,Ad)
- `F`: flux function which outputs a d-tuple of flux vectors
- `u`: collection of solution values (or arrays) at which to evaluate `F`
- `Fargs`: extra arguments to `F(ui,uj,getindex.(Fargs,i)...,getindex.(Fargs,j)...)`
- (optional) `skip_index(i,j)==true` skips computing fluxes for index (i,j)

Since this sums over rows of matrices, this function may be slow for column-major
and sparse CSC matrices. If you are using column major/CSC storage, it will be faster
to precompute transposes of `A_list` and pass them to [`hadamard_sum_ATr!`](@ref).
"""
function hadamard_sum(A_list,F::Fxn,u,Fargs...; skip_index=(i,j)->false) where {Fxn}
    rhs = zero.(u)
    hadamard_sum_ATr!(rhs,transpose.(A_list),F,u,Fargs...; skip_index=skip_index)
    return rhs
end

"""
    hadamard_sum_ATr!(rhs,ATr_list,F,u,Fargs...; skip_index=(i,j)->false)
    hadamard_sum_ATr!(rhs,ATr_list::NTuple{N,SparseMatrixCSC},F,u,Fargs...) where {N}
    hadamard_sum_ATr!(rhs,ATr::SparseMatrixCSC,F,u,Fargs...)

Same as [`hadamard_sum!`](@ref) but `ATr_list` contains transposed matrices.
Specializes based on whether `ATr_list` contains SparseMatrixCSC or general arrays.
SparseMatrixCSC works best if all matrices in `ATr_list` have distinct sparsity patterns.
"""
function hadamard_sum_ATr!(rhs,ATr_list,F,u,Fargs...; skip_index=(i,j)->false)
    rhstype = eltype(first(rhs))
    val_i = zeros(rhstype,length(rhs))
    rows,cols = axes(first(ATr_list))
    for i in cols
        ui = getindex.(u,i)
        Fargs_i = getindex.(Fargs,i)
        fill!(val_i,zero(rhstype))
        for j in rows
            if skip_index(i,j)==false
                uj = getindex.(u,j)
                ATrij_list = getindex.(ATr_list,j,i)
                Fij = F(ui,uj,Fargs_i...,getindex.(Fargs,j)...)
                val_i .+= sum(bmult.(ATrij_list,Fij))
            end
        end
        setindex!.(rhs,val_i,i)
    end
end

function hadamard_sum_ATr!(rhs,ATr_list::NTuple{N,SparseMatrixCSC},F,u,Fargs...) where {N}
    for (i,ATr) in enumerate(ATr_list)
        F_i = (x->getindex(x,i)) ∘ F
        hadamard_sum_ATr!(rhs,ATr,F_i,u,Fargs...)
    end
end

function hadamard_sum_ATr!(rhs,ATr::SparseMatrixCSC,F,u,Fargs...)
    rhstype = eltype(first(rhs))
    val_i = zeros(rhstype,length(rhs))
    rows = rowvals(ATr)
    vals = nonzeros(ATr)
    for i = 1:size(ATr,2) # all ops should be same length
        ui = getindex.(u,i)
        # fill!(val_i,zero(rhstype))
        val_i .= getindex.(rhs,i) # accumulate into existing rhs
        Fargs_i = getindex.(Fargs,i)
        for row_id in nzrange(ATr,i)
            j = rows[row_id]
            uj = getindex.(u,j)
            Fij = F(ui,uj,Fargs_i...,getindex.(Fargs,j)...)
            val_i .+= vals[row_id].*Fij
        end
        setindex!.(rhs,val_i,i)
    end
end
