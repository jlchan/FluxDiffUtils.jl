"""
    TupleOrSVector{N}

Either a NTuple or SVector (e.g., fast static container) of length N.
"""
const TupleOrSVector{N} = Union{NTuple{N,T},SVector{N,T}} where {T}

#####
##### routine works for both dense/sparse matrix routines
#####

"""
    hadamard_sum(A_list, F::Fxn, u, Fargs...;
                 skip_index=(i,j)->false) where {N,T,Fxn}

computes ∑_i sum(Ai.*Fi,dims=2) where (Fi)_jk = F(uj,uk)[i]

Inputs
- `A_list`: tuple (or similar container) of operators (A1,...,Ad)
- `F`: flux function which outputs a length `d` container of flux vectors
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

Same as [`hadamard_sum`](@ref) except that `hadamard_sum_ATr!`
- assumes `ATr_list` contains transposed matrices.
- accumulates the result into `rhs`
Specializes based on whether `ATr_list` contains SparseMatrixCSC or general arrays.
The SparseMatrixCSC version works best if the matrices in `ATr_list` have distinct sparsity patterns.
"""
function hadamard_sum_ATr!(rhs,ATr_list,F::Fxn,u,Fargs...; skip_index=(i,j)->false) where {Fxn}
    val_i = zeros(eltype(first(rhs)),length(rhs)) # preallocate array
    rows,cols = axes(first(ATr_list))
    for i in cols
        ui = getindex.(u,i)
        val_i .= getindex.(rhs,i)
        for j in rows
            if skip_index(i,j)==false
                uj = getindex.(u,j)
                ATrij_list = getindex.(ATr_list,j,i)
                Fij = F(ui,uj,getindex.(Fargs,i)...,getindex.(Fargs,j)...)
                val_i .+= sum(map((x,y)->x.*y,ATrij_list,Fij))
            end
        end
        setindex!.(rhs,val_i,i)
    end
end

# version for a list of sparse matrices (loop thru them)
function hadamard_sum_ATr!(rhs,ATr_list::NTuple{N,SparseMatrixCSC},F::Fxn,u,Fargs...) where {N,Fxn}
    for (i,ATr) in enumerate(ATr_list)
        F_i = (x->getindex(x,i)) ∘ F
        hadamard_sum_ATr!(rhs,ATr,F_i,u,Fargs...)
    end
end

# version for a single sparse matrix
function hadamard_sum_ATr!(rhs,ATr::SparseMatrixCSC,F::Fxn,u,Fargs...) where {Fxn}
    val_i = zeros(eltype(first(rhs)),length(rhs))
    rows = rowvals(ATr)
    vals = nonzeros(ATr)
    for i = 1:size(ATr,2) # all ops should be same length
        ui = getindex.(u,i)
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

"""
    hadamard_sum_ATr!(rhs::TupleOrSVector{N},ATr_list::NTuple{D,Array},F::Fxn,u,Fargs...;
                      skip_index=(i,j)->false) where {N,D,Fxn}
    hadamard_sum_ATr!(rhs::NTupleOrSVector{N},ATr::SparseMatrixCSC,F::Fxn,u,Fargs...) where {N,Fxn}

Zero-allocation dense/sparse versions if rhs has a statically inferrable length (e.g., is an NTuple or SVector)
"""
function hadamard_sum_ATr!(rhs::TupleOrSVector{N},ATr_list::NTuple{D,Array},F::Fxn,u,Fargs...;
                           skip_index=(i,j)->false) where {N,D,Fxn}
    rows,cols = axes(first(ATr_list))
    for i in cols
        ui = getindex.(u,i)
        val_i = MVector{N}(getindex.(rhs,i)) # accumulate into existing rhs
        for j in rows
            if skip_index(i,j)==false
                uj = getindex.(u,j)
                ATrij_list = getindex.(ATr_list,j,i)
                Fij = F(ui,uj,getindex.(Fargs,i)...,getindex.(Fargs,j)...)
                val_i .+= sum(map((x,y)->x.*y,ATrij_list,Fij))
            end
        end
        setindex!.(rhs,val_i,i)
    end
end
function hadamard_sum_ATr!(rhs::TupleOrSVector{N},ATr::SparseMatrixCSC,F::Fxn,u,Fargs...) where {N,Fxn}
    rows = rowvals(ATr)
    vals = nonzeros(ATr)
    for i = 1:size(ATr,2)
        ui = getindex.(u,i)
        val_i = MVector{N}(getindex.(rhs,i)) # accumulate into existing rhs
        for row_id in nzrange(ATr,i)
            j = rows[row_id]
            uj = getindex.(u,j)
            Fij = F(ui,uj,getindex.(Fargs,i)...,getindex.(Fargs,j)...)
            val_i .+= vals[row_id].*Fij
        end
        setindex!.(rhs,val_i,i)
    end
end
