module MPI_VMC_Utils

using MPI
using LinearAlgebra
using Statistics

export MPISession, init_mpi_session
export ObservableBuffer, register_scalar!, register_vector!, register_matrix!
export reset_buffers!, increment_counter!, accumulate_sample!, accumulate_sr_matrix!
export record_scalar! 
export mpi_reduce_all, mpi_gather_scalar 

struct MPISession
    comm::MPI.Comm
    rank::Int
    size::Int
    root::Int
end

function init_mpi_session()
    if !MPI.Initialized()
        MPI.Init()
    end
    comm = MPI.COMM_WORLD
    return MPISession(comm, MPI.Comm_rank(comm), MPI.Comm_size(comm), 0)
end

struct ObservableBuffer{T}
    # 这一部分依然用来存 SR 矩阵和梯度，采用累加方式（省内存）
    scalars::Dict{Symbol, Vector{T}} 
    vectors::Dict{Symbol, Vector{T}}
    matrices::Dict{Symbol, Matrix{T}}
    counter::Vector{Int}
    
    # === 新增：专门用于存能量历史 ===
    scalar_histories::Dict{Symbol, Vector{T}} 
end

function ObservableBuffer(::Type{T}=ComplexF64) where T
    return ObservableBuffer{T}(
        Dict{Symbol, Vector{T}}(),
        Dict{Symbol, Vector{T}}(),
        Dict{Symbol, Matrix{T}}(),
        [0],
        Dict{Symbol, Vector{T}}() # 初始化为空
    )
end

# --- 注册与初始化 ---

function register_scalar!(obs::ObservableBuffer{T}, name::Symbol) where T
    obs.scalars[name] = zeros(T, 1)
end
function register_vector!(obs::ObservableBuffer{T}, name::Symbol, len::Int) where T
    obs.vectors[name] = zeros(T, len)
end
function register_matrix!(obs::ObservableBuffer{T}, name::Symbol, rows::Int, cols::Int) where T
    obs.matrices[name] = zeros(T, rows, cols)
end

function reset_buffers!(obs::ObservableBuffer{T}) where T
    obs.counter[1] = 0
    # 清空累加器
    for v in values(obs.scalars); fill!(v, zero(T)); end
    for v in values(obs.vectors); fill!(v, zero(T)); end
    for v in values(obs.matrices); fill!(v, zero(T)); end
    
    # === 清空能量列表 ===
    for v in values(obs.scalar_histories)
        empty!(v)
    end
end

@inline function increment_counter!(obs::ObservableBuffer)
    obs.counter[1] += 1
end

# --- 累加函数 (用于 SR 矩阵等) ---

@inline function accumulate_sample!(obs::ObservableBuffer, name::Symbol, val::Number)
    @inbounds obs.scalars[name][1] += val
end

@inline function accumulate_sample!(obs::ObservableBuffer, name::Symbol, vec::AbstractVector)
    @inbounds obs.vectors[name] .+= vec
end

@inline function accumulate_sample!(obs::ObservableBuffer, name::Symbol, mat::AbstractMatrix)
    @inbounds obs.matrices[name] .+= mat
end

function accumulate_sr_matrix!(obs::ObservableBuffer{T}, name::Symbol, O_vec::AbstractVector) where T
    m_buff = obs.matrices[name]
    N = length(O_vec)
    @inbounds for c in 1:N
        oc = O_vec[c]
        for r in 1:N
            m_buff[r, c] += conj(O_vec[r]) * oc
        end
    end
end

# --- 新增：记录能量列表 ---

"""
record_energy!: 不做累加，而是 push 到列表里。
这样我们就能保留完整的采样历史，用于 Binning Analysis。
"""

@inline function record_scalar!(obs::ObservableBuffer{T}, name::Symbol, val::Number) where T
    if !haskey(obs.scalar_histories, name)
        obs.scalar_histories[name] = T[]
    end
    push!(obs.scalar_histories[name], T(val))
end

# --- MPI 通信 ---



# 1. 普通 Reduce (用于 SR 矩阵等均值)
function mpi_reduce_all(obs::ObservableBuffer{T}, session::MPISession) where T
    comm = session.comm
    root = session.root
    local_count = obs.counter[1]
    
    # 汇总总样本数
    total_count = MPI.Reduce(local_count, MPI.SUM, root, comm)
    
    results = Dict{Symbol, Any}()
    if session.rank == root && total_count == 0; total_count = 1; end
    
    function reduce_helper(local_data)
        summed = MPI.Reduce(local_data, MPI.SUM, root, comm)
        if session.rank == root
            return summed ./ total_count # 返回平均值
        else
            return nothing
        end
    end

    for (name, val) in obs.scalars
        res = reduce_helper(val)
        if session.rank == session.root
            results[name] = res[1] # 只有 Root 才能解包
        end
    end
    
    for (name, val) in obs.vectors
        res = reduce_helper(val)
        if session.rank == session.root
            results[name] = res
        end
    end
    
    for (name, val) in obs.matrices
        res = reduce_helper(val)
        if session.rank == session.root
            results[name] = res
        end
    end
    
    if session.rank == root
        results[:count] = total_count
        return results
    else
        return nothing
    end
end

# 2. 特殊 Gather (专门用于能量列表)
"""
mpi_gather_energies: 
将所有 Rank 的 scalar_history 拼接起来。
返回: 一个巨大的 Vector{Float64} (仅 Root 有值)，包含所有采样的能量。
"""

function mpi_gather_scalar(obs::ObservableBuffer{T}, session::MPISession, name::Symbol) where T
    comm = session.comm
    root = session.root
    
    # 获取本地数据，如果不存在则为空向量（防止报错）
    local_data = get(obs.scalar_histories, name, T[])
    
    # MPI.Gather 拼接数据
    all_data = MPI.Gather(local_data, root, comm)
    
    if session.rank == root
        # 展平结果
        return reduce(vcat, all_data)
    else
        return nothing
    end
end

end # module
