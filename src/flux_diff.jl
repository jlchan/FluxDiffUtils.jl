# tuple ATr_list dispatch to both dense/sparse
function hadamard_sum(ATr_list::NTuple{N,T}, F::Fxn, u, Fargs...) where {N,T,Fxn}
    rhs = zero.(u)
    hadamard_sum!(rhs,ATr_list,F,u,Fargs...)
    return rhs
end

# helper functions
unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))
bmult(a,b) = a.*b # broadcasted multiplication

"function hadamard_sum!(rhs, ATr::NTuple{N,Array{T,2}}, F, u, Fargs...) where {N,T}
    general function for dense operators"
function hadamard_sum!(rhs, ATr_list::NTuple{N,AbstractArray{T,2}},
                       F, u, Fargs...) where {N,T}
    n = size(first(ATr_list),2)
    val_i = zeros(eltype(first(rhs)),length(rhs))
    for i = 1:n
        ui = getindex.(u,i)
        fill!(val_i,zero(eltype(first(rhs))))
        for j = 1:n
            uj = getindex.(u,j)
            ATrij = getindex.(ATr_list,j,i)
            Fij = F(ui...,getindex.(Fargs,i)...,uj...,getindex.(Fargs,j)...)
            val_i .+= sum(bmult.(ATrij,Fij))
        end
        setindex!.(rhs,val_i,i)
    end
end


#####
##### general sparse matrix routines
#####
"function hadamard_sum!(rhs, ATr_list::NTuple{N,SparseMatrixCSC}, F::Fxn,
                        u, Fargs...) where {N,Fxn}"
function hadamard_sum!(rhs, ATr_list::NTuple{N,SparseMatrixCSC}, F::Fxn,
                        u, Fargs...) where {N,Fxn}
    cols = rowvals.(ATr_list)
    val_i = zeros(eltype(first(rhs)),length(rhs))
    for i = 1:size(first(ATr_list),2) # all ops should be same length
        ui = getindex.(u,i)
        fill!(val_i,zero(eltype(first(rhs))))

        # loop over union of nonzero inds for all matrices in ATr_list
        for j in union(getindex.(cols,nzrange.(ATr_list, i))...)
            uj = getindex.(u,j)
            ATrij_list = getindex.(ATr_list,j,i)
            Fij = F(ui...,getindex.(Fargs,i)...,uj...,getindex.(Fargs,j)...)
            val_i .+= sum(bmult.(ATrij_list,Fij))
        end
        setindex!.(rhs,val_i,i)
    end
end
