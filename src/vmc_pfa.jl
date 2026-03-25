function SkewMatrix!(F::AbstractMatrix{T}) where {T<:Number}
    LinearAlgebra.require_one_based_indexing(F)
    n = LinearAlgebra.checksquare(F)

    @inbounds for i in 1:n
        F[i, i] = zero(T)
        for j in 1:i-1
            value_ij = (F[i, j] - F[j, i]) / 2
            F[i, j] = value_ij
            F[j, i] = -value_ij
        end
    end
    return F
end
# ==============================================================================
# WorkSpace
# ==============================================================================
mutable struct PfaR1R2WS{T}
    N::Int
    dr1::Vector{T}   # δr1[i] = gs_F[Rl',elec_locs[i]] - A[l,i]
    dr2::Vector{T}   # δr2[i] = (1-δ_{i,l})(gs_F[Rm',elec_locs[i]] - A[m,i]) + δ_{i,l}(gs_F[Rm',Rl'] - gs_F[elec_locs[m],Rl'])
    col1::Vector{T}  # col1 = Ainv * δr1
    col2::Vector{T}  # col2 = Ainv * δr2
    xi::Vector{T}    # xi = copy(Ainv[:, i])
    xj::Vector{T}    # xj = copy(Ainv[:, j])
    a::T
    b::T
    c::T
    d::T
    e::T
    f::T

    grad_buffer::Vector{T}
end

# ==============================================================================
# Wavefunction: vwf_pfa (Generic T) 
# ==============================================================================
mutable struct vwf_pfa{T,S}
    # -- Matrices --
    gs_F::Matrix{T}

    awf_mat::Matrix{T}
    awf_inv::Matrix{T}
    awf_val::T

    current_ratio::T

    sampler::S

    # -- Workspace --
    ws::PfaR1R2WS{T}
    dF_params::OrderedDict{Symbol,Matrix{T}}
    dF_list::Vector{Matrix{T}}
    param_keys::Vector{Symbol}
end

function vwf_pfa(F::Matrix{T}, sampler) where T
    Nlat = sampler.N_sites
    expected_rows = 2 * Nlat
    expected_cols = 2 * Nlat

    @assert size(F, 1) == expected_rows "F rows $(size(F,1)) != 2*Nlat"
    @assert size(F, 2) == expected_cols "F cols $(size(F,2)) != 2*Nlat"

    dummy_ws = PfaR1R2WS{T}(0, T[], T[], T[], T[], T[], T[], zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), T[])
    #nelec represet particle number conservation if sampler is nonPH
    #nelec represet total sz conservation if sampler is PH
    #one must sepcific one, or the fast update method fail
    nelec = total_elec(sampler)
    @assert nelec % 2 == 0 "total electron must be even!"
    awf_mat = zeros(T, nelec, nelec)
    awf_inv = zeros(T, 1, 1)

    dF_params = OrderedDict{Symbol,Matrix{T}}()
    dF_list = Vector{Matrix{T}}()
    param_keys = Vector{Symbol}()

    return vwf_pfa{T,typeof(sampler)}(
        SkewMatrix!(F),
        awf_mat,      # awf_mat
        awf_inv,        # awf_inv (placeholder)
        T(0),
        one(T),
        sampler,
        dummy_ws,
        dF_params,
        dF_list,
        param_keys
    )
end

function ensure_ws!(v::vwf_pfa{T,S}) where {T,S}
    N = size(v.awf_mat, 1)
    ws = v.ws
    if ws.N != N
        ws = PfaR1R2WS{T}(
            N,
            Vector{T}(undef, N), Vector{T}(undef, N),
            Vector{T}(undef, N), Vector{T}(undef, N),
            Vector{T}(undef, N), Vector{T}(undef, N),
            zero(T), zero(T), zero(T), zero(T), zero(T), zero(T),
            T[]
        )
        v.ws = ws
    end

    n_params = length(v.dF_params) # 注意：是 dF_params 还是 dF_params，请保持变量名一致
    if length(ws.grad_buffer) != n_params
        resize!(ws.grad_buffer, n_params)
    end

    return ws
end


function update_vwf_params!(vwf::vwf_pfa{T}, new_params::OrderedDict{Symbol,Matrix{T}}) where T
    vwf.dF_params = new_params

    empty!(vwf.dF_list)
    empty!(vwf.param_keys)

    for (k, v) in new_params
        push!(vwf.param_keys, k)
        push!(vwf.dF_list, v)
    end

    ensure_ws!(vwf)
    return nothing
end

function init_gswf!(vwf::vwf_pfa{T,S}) where {T,S}
    ss = vwf.sampler
    initialize_lists!(ss)
    total_elec_count = total_elec(ss)

    # elec_locs = zeros(Int, total_elec)    
    # count_found = 0
    # for loc_idx in 1:length(ss.map_spin_to_id)
    #     eid = ss.map_spin_to_id[loc_idx]
    #     if eid != 0
    #         elec_locs[eid] = loc_idx 
    #         count_found += 1
    #     end
    # end

    if size(vwf.awf_mat, 1) != total_elec_count || size(vwf.awf_mat, 2) != total_elec_count
        vwf.awf_mat = zeros(T, total_elec_count, total_elec_count)
    end
    for i in 1:total_elec_count
        row_in_F = ss.electron_locs[i]
        for j in i:total_elec_count
            col_in_F = ss.electron_locs[j]
            element = vwf.gs_F[row_in_F, col_in_F]
            vwf.awf_mat[i, j] = element
            vwf.awf_mat[j, i] = -element
        end
        vwf.awf_mat[i, i] = 0
    end

    #= A_physical = transpose(vwf.awf_mat)
    F = lu(A_physical) =#
    vwf.awf_val = pfaffian(vwf.awf_mat)
    vwf.awf_inv = inv(vwf.awf_mat)
    vwf.current_ratio = one(T)

    ensure_ws!(vwf)
    return nothing
end

function rebuild_inverse!(vwf::vwf_pfa)
    vwf.awf_inv = inv(vwf.awf_mat)
end

"""
用途: 原位执行反对称的 rank-1 更新, 即 `A <- A + alpha * (x*y^T - y*x^T)`.
"""
@inline function skew_rank1_update_blas!(A::Matrix{T}, alpha::T, x::Vector{T}, y::Vector{T}) where {T<:Float64}
    BLAS.ger!(alpha, x, y, A)
    BLAS.ger!(-alpha, y, x, A)
end

"""
用途: 原位执行复数情形下的反对称 rank-1 更新, 即 `A <- A + alpha * (x*y^T - y*x^T)`.
应该用不到这个
"""
@inline function skew_rank1_update_blas!(A::Matrix{T}, alpha::T, x::Vector{T}, y::Vector{T}) where {T<:Complex}
    n_row, n_col = size(A)
    length(x) == n_row || throw(DimensionMismatch("length(x) = $(length(x)) != size(A, 1) = $n_row"))
    length(y) == n_col || throw(DimensionMismatch("length(y) = $(length(y)) != size(A, 2) = $n_col"))

    @inbounds for j in 1:n_col
        x_j = x[j]
        y_j = y[j]
        for i in 1:n_row
            A[i, j] += alpha * (x[i] * y_j - y[i] * x_j)
        end
    end
end

function ratio_rank1(vwf::vwf_pfa{T}, k::Int, new_row_idx_U::Int) where T
    val = zero(T)
    electron_locs = vwf.sampler.electron_locs
    @inbounds @simd for j in eachindex(electron_locs)
        loc = electron_locs[j]
        val += vwf.gs_F[new_row_idx_U, loc] * vwf.awf_inv[j, k]
    end
    return val
end

function update_rank1!(vwf::vwf_pfa{T}, k::Int, new_row_idx_U::Int, ratio::T) where T
    ws = ensure_ws!(vwf)
    A = vwf.awf_mat
    Ainv = vwf.awf_inv
    electron_locs = vwf.sampler.electron_locs

    @inbounds @simd for j in eachindex(electron_locs)
        loc = electron_locs[j]
        ws.dr1[j] = vwf.gs_F[new_row_idx_U, loc] - A[k, j]
    end

    mul!(ws.col1, Ainv, ws.dr1)
    copyto!(ws.xi, @view Ainv[:, k])
    skew_rank1_update_blas!(Ainv, 1 / ratio, ws.xi, ws.col1)

    @inbounds @simd for j in eachindex(electron_locs)
        loc = electron_locs[j]
        if j != k
            A[k, j] = vwf.gs_F[new_row_idx_U, loc]
            A[j, k] = vwf.gs_F[loc, new_row_idx_U]
        else
            A[j, j] = 0
        end
    end

    vwf.awf_val *= ratio
end

function ratio_rank2(vwf::vwf_pfa{T}, k1::Int, k2::Int, new_row1_U::Int, new_row2_U::Int) where T
    ws = ensure_ws!(vwf)
    A = vwf.awf_mat
    Ainv = vwf.awf_inv
    electron_locs = vwf.sampler.electron_locs

    @inbounds @simd for j in eachindex(electron_locs)
        loc = electron_locs[j]
        ws.dr1[j] = vwf.gs_F[new_row1_U, loc] - A[k1, j]
        ws.dr2[j] = j == k1 ? vwf.gs_F[new_row2_U, new_row1_U] - vwf.gs_F[electron_locs[k2], new_row1_U] : vwf.gs_F[new_row2_U, loc] - A[k2, j]
    end
    copyto!(ws.xi, @view Ainv[:, k1])
    copyto!(ws.xj, @view Ainv[:, k2])
    mul!(ws.col2, Ainv, ws.dr2)
    ws.a = 1 + dot(ws.dr1, ws.xi)
    ws.b = dot(ws.dr1, ws.col2)
    ws.c = dot(ws.dr1, ws.xj)
    ws.d = -dot(ws.dr2, ws.xi)
    ws.e = Ainv[k1, k2]
    ws.f = 1 + dot(ws.dr2, ws.xj)
    return ws.a * ws.f - ws.b * ws.e + ws.c * ws.d
end

function update_rank2!(vwf::vwf_pfa{T}, k1::Int, k2::Int, new_row1_U::Int, new_row2_U::Int, ratio::T) where T
    ws = ensure_ws!(vwf)
    A = vwf.awf_mat
    Ainv = vwf.awf_inv
    N = size(A, 1)
    electron_locs = vwf.sampler.electron_locs

    mul!(ws.col1, Ainv, ws.dr1)

    skew_rank1_update_blas!(Ainv, ws.f / ratio, ws.col1, ws.xi)
    skew_rank1_update_blas!(Ainv, -ws.e / ratio, ws.col1, ws.col2)
    skew_rank1_update_blas!(Ainv, ws.d / ratio, ws.col1, ws.xj)
    skew_rank1_update_blas!(Ainv, ws.c / ratio, ws.xi, ws.col2)
    skew_rank1_update_blas!(Ainv, -ws.b / ratio, ws.xi, ws.xj)
    skew_rank1_update_blas!(Ainv, ws.a / ratio, ws.col2, ws.xj)

    @inbounds @simd for j in eachindex(electron_locs)
        loc = electron_locs[j]
        if j == k1
            A[k1, k1] = 0
            A[k1, k2] = vwf.gs_F[new_row1_U, new_row2_U]
            A[k2, k1] = vwf.gs_F[new_row2_U, new_row1_U]
        elseif j == k2
            A[k2, k2] = 0
        else
            A[k1, j] = vwf.gs_F[new_row1_U, loc]
            A[j, k1] = vwf.gs_F[loc, new_row1_U]
            A[k2, j] = vwf.gs_F[new_row2_U, loc]
            A[j, k2] = vwf.gs_F[loc, new_row2_U]
        end
    end

    vwf.awf_val *= ratio
end

function find_stable_config!(vwf::vwf_pfa{T}, kernel::AbstractMCMCKernel, rng::AbstractRNG) where T
    ss = vwf.sampler

    # println("Searching for a numerically stable configuration...")

    max_attempts = 1000
    tol_inv = 1e-5

    for attempt in 1:max_attempts
        init_config_rand!(ss, kernel)

        # === 2. 根据新构型重建矩阵 ===
        # Sampler 已经更新了 electron_locs，直接利用它填充矩阵
        total_elec_count = total_elec(ss)
        for i in 1:total_elec_count
            row_in_F = ss.electron_locs[i]
            for j in i:total_elec_count
                col_in_F = ss.electron_locs[j]
                element = vwf.gs_F[row_in_F, col_in_F]
                vwf.awf_mat[i, j] = element
                vwf.awf_mat[j, i] = -element
            end
            vwf.awf_mat[i, i] = 0
        end

        # === 3. 检查数值稳定性 ===
        # 计算pfaffian
        d = pfaffian(vwf.awf_mat)


        try
            # 计算逆矩阵
            current_inv = inv(vwf.awf_mat)

            # 验证逆矩阵精度
            prod_mat = current_inv * vwf.awf_mat
            diff = norm(prod_mat - I)

            if diff < tol_inv
                # === 成功 ===
                vwf.awf_inv = current_inv
                vwf.awf_val = d
                vwf.current_ratio = one(T)

                # println("Stable config found at attempt $attempt. Det=$d, |A⁻¹A - I|=$diff")
                return
            end

        catch e
            # 奇异异常，继续尝试
        end
    end

    error("Failed to find a stable configuration after $max_attempts attempts. Please check your Trial Wavefunction (U).")
end
# ==============================================================================
# Generic Measurements
# ==============================================================================
function local_energy(ham, vwf::vwf_pfa)
    return 0
end

function compute_grad_log_psi!(vwf::vwf_pfa{T}) where T
    # 1. 准备 Workspace
    ws = ensure_ws!(vwf)
    ss = vwf.sampler

    A_inv = vwf.awf_inv   # Size: (N_orb, N_elec)
    # A_inv[orb, elec] -> 列优先存储，orb 变化最快

    Norb, Nelec = size(A_inv)

    # 2. 获取 Buffer (O_vec)
    O_vec = ws.grad_buffer
    fill!(O_vec, zero(T))

    # 3. 遍历所有可变参数
    for (param_idx, dF) in enumerate(vwf.dF_list)

        total_sum = zero(T)

        # 顺序：外层电子(elec)，内层轨道(orb)
        # 优化理由：dF[orb, r] 和 A_inv[orb, elec] 第一维都是 orb，内存连续
        @inbounds for elec in 1:Nelec
            r = ss.electron_locs[elec]

            col_sum = zero(T)

            # SIMD 内积
            @simd for orb in 1:Norb
                o = ss.electron_locs[orb]
                col_sum += A_inv[orb, elec] * dF[r, o]
            end

            total_sum += 0.5 * col_sum
        end

        # 直接使用 enumerate 的索引，不再依赖计数器变量
        O_vec[param_idx] = total_sum
    end

    # 直接返回 buffer 引用，避免 copy
    return O_vec
end
