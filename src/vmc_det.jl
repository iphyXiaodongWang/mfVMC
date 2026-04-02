# ==============================================================================
# WorkSpace
# ==============================================================================
struct R1R2WS{T}
    N::Int
    dr1::Vector{T}   # δr1 = newF1 - A[i,:]
    dr2::Vector{T}   # δr2 = newF2 - A[j,:]
    col1::Vector{T}  # col1 = Ainv' * δr1
    col2::Vector{T}  # col2 = Ainv' * δr2
    xi::Vector{T}    # xi = copy(Ainv[:, i])
    xj::Vector{T}    # xj = copy(Ainv[:, j])
    s0::Vector{T}    # 1st column of S
    s1::Vector{T}    # 2nd column of S

    grad_buffer::Vector{T}
end

# ==============================================================================
# Wavefunction: vwf_det (Generic T) 
# ==============================================================================
mutable struct vwf_det{T,S}
    # -- Matrices --
    gs_U::Matrix{T}
    gs_U_t::Matrix{T}

    awf_mat_t::Matrix{T}
    awf_inv::Matrix{T}
    awf_val::T

    current_ratio::T

    sampler::S

    # -- Workspace --
    ws::R1R2WS{T}
    dUt_matrix::Array{T,3}
    param_keys::Vector{Symbol}
end

function vwf_det(U::Matrix{T}, sampler) where T
    Nlat = sampler.N_sites
    expected_rows = 2 * Nlat
    expected_cols = total_elec(sampler)

    @assert size(U, 1) == expected_rows "U rows $(size(U,1)) != 2*Nlat"
    @assert size(U, 2) >= expected_cols "U cols $(size(U,2)) < Nelec"

    dummy_ws = R1R2WS{T}(0, T[], T[], T[], T[], T[], T[], T[], T[], T[])

    nelec = expected_cols
    awf_mat_t = zeros(T, nelec, nelec)
    awf_inv = zeros(T, 1, 1)
    dUt_matrix = zeros(T, 1, 1, 1)
    param_keys = Vector{Symbol}()

    return vwf_det{T,typeof(sampler)}(
        U,
        permutedims(U), # gs_U_t
        awf_mat_t,      # awf_mat_t
        awf_inv,        # awf_inv (placeholder)
        T(0),
        one(T),
        sampler,
        dummy_ws,
        dUt_matrix,
        param_keys
    )
end

function ensure_ws!(v::vwf_det{T,S}) where {T,S}
    N = size(v.awf_mat_t, 1)
    ws = v.ws
    if ws.N != N
        ws = R1R2WS{T}(
            N,
            Vector{T}(undef, N), Vector{T}(undef, N),
            Vector{T}(undef, N), Vector{T}(undef, N),
            Vector{T}(undef, N), Vector{T}(undef, N),
            Vector{T}(undef, N), Vector{T}(undef, N),
            T[]
        )
        v.ws = ws
    end

    n_params = length(v.param_keys)
    if length(ws.grad_buffer) != n_params
        resize!(ws.grad_buffer, n_params)
    end

    return ws
end



function update_vwf_params!(vwf::vwf_det{T}, param_names::Vector{Symbol}, dUt_matrix::Array{T,3}) where T

    empty!(vwf.param_keys)

    for name in param_names
        push!(vwf.param_keys, name)
    end

    vwf.dUt_matrix = dUt_matrix
    ensure_ws!(vwf)
    return nothing
end

function init_gswf!(vwf::vwf_det{T,S}) where {T,S}
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

    if size(vwf.awf_mat_t, 1) != total_elec_count
        vwf.awf_mat_t = zeros(T, total_elec_count, total_elec_count)
    end

    for i in 1:total_elec_count
        row_in_U = ss.electron_locs[i]
        copyto!(@view(vwf.awf_mat_t[:, i]), @view(vwf.gs_U_t[:, row_in_U]))
    end

    A_physical = transpose(vwf.awf_mat_t)
    F = lu(A_physical)

    vwf.awf_val = det(F)
    vwf.awf_inv = inv(F)
    vwf.current_ratio = one(T)

    ensure_ws!(vwf)
    return nothing
end

function rebuild_inverse!(vwf::vwf_det)
    vwf.awf_inv = inv(transpose(vwf.awf_mat_t))
end

@inline function rank1_update_blas!(A::Matrix{T}, alpha::T, x::Vector{T}, y::Vector{T}) where T<:Float64
    BLAS.ger!(alpha, x, y, A)
end

@inline function rank1_update_blas!(A::Matrix{T}, alpha::T, x::Vector{T}, y::Vector{T}) where T<:Complex
    y_temp = conj!(y)
    BLAS.ger!(alpha, x, y_temp, A)
end

function ratio_rank1(vwf::vwf_det{T}, k::Int, new_row_idx_U::Int) where T
    val = zero(T)
    N = size(vwf.awf_inv, 1)
    @inbounds @simd for j in 1:N
        val += vwf.gs_U_t[j, new_row_idx_U] * vwf.awf_inv[j, k]
    end
    return val
end

function update_rank1!(vwf::vwf_det{T}, k::Int, new_row_idx_U::Int, ratio::T) where T
    ws = ensure_ws!(vwf)
    A_t = vwf.awf_mat_t
    Ainv = vwf.awf_inv
    N = size(A_t, 1)

    @inbounds @simd for j in 1:N
        ws.dr1[j] = vwf.gs_U_t[j, new_row_idx_U] - A_t[j, k]
    end

    mul!(ws.col1, transpose(Ainv), ws.dr1)
    copyto!(ws.xi, @view Ainv[:, k])
    rank1_update_blas!(Ainv, -1 / ratio, ws.xi, ws.col1)

    @inbounds @simd for j in 1:N
        A_t[j, k] = vwf.gs_U_t[j, new_row_idx_U]
    end

    vwf.awf_val *= ratio
end

function ratio_rank2(vwf::vwf_det{T}, k1::Int, k2::Int, new_row1_U::Int, new_row2_U::Int) where T
    Ainv = vwf.awf_inv
    U_t = vwf.gs_U_t
    N = size(Ainv, 1)

    d11 = zero(T)
    d12 = zero(T)
    d21 = zero(T)
    d22 = zero(T)

    @inbounds @simd for j in 1:N
        u1 = U_t[j, new_row1_U]
        u2 = U_t[j, new_row2_U]
        inv_k1 = Ainv[j, k1]
        inv_k2 = Ainv[j, k2]

        d11 += u1 * inv_k1
        d12 += u1 * inv_k2
        d21 += u2 * inv_k1
        d22 += u2 * inv_k2
    end
    return d11 * d22 - d12 * d21
end

function update_rank2!(vwf::vwf_det{T}, k1::Int, k2::Int, new_row1_U::Int, new_row2_U::Int, ratio::T) where T
    ws = ensure_ws!(vwf)
    A_t = vwf.awf_mat_t
    Ainv = vwf.awf_inv
    N = size(A_t, 1)

    @inbounds @simd for j in 1:N
        ws.dr1[j] = vwf.gs_U_t[j, new_row1_U] - A_t[j, k1]
        ws.dr2[j] = vwf.gs_U_t[j, new_row2_U] - A_t[j, k2]
    end

    mul!(ws.col1, transpose(Ainv), ws.dr1)
    mul!(ws.col2, transpose(Ainv), ws.dr2)

    k_11 = 1.0 + ws.col1[k1]
    k_12 = ws.col1[k2]
    k_21 = ws.col2[k1]
    k_22 = 1.0 + ws.col2[k2]

    detK = k_11 * k_22 - k_12 * k_21
    invDet = 1.0 / detK

    i_11 = k_22 * invDet
    i_12 = -k_12 * invDet
    i_21 = -k_21 * invDet
    i_22 = k_11 * invDet

    copyto!(ws.xi, @view Ainv[:, k1])
    copyto!(ws.xj, @view Ainv[:, k2])

    @inbounds @simd for j in 1:N
        xi_val = ws.xi[j]
        xj_val = ws.xj[j]
        ws.s0[j] = i_11 * xi_val + i_21 * xj_val
        ws.s1[j] = i_12 * xi_val + i_22 * xj_val
    end

    rank1_update_blas!(Ainv, -one(T), ws.s0, ws.col1)
    rank1_update_blas!(Ainv, -one(T), ws.s1, ws.col2)

    @inbounds @simd for j in 1:N
        A_t[j, k1] = vwf.gs_U_t[j, new_row1_U]
        A_t[j, k2] = vwf.gs_U_t[j, new_row2_U]
    end

    vwf.awf_val *= ratio
end

function find_stable_config!(vwf::vwf_det{T}, kernel::AbstractMCMCKernel, rng::AbstractRNG) where T
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
            # electron_locs[i] 存储的是基组索引 (2*site+spin)，对应 gs_U_t 的列
            basis_idx = ss.electron_locs[i]

            # awf_mat_t 是转置存储的 (列是电子，行是轨道)
            copyto!(@view(vwf.awf_mat_t[:, i]), @view(vwf.gs_U_t[:, basis_idx]))
        end

        # === 3. 检查数值稳定性 ===
        # 计算行列式
        F = lu(transpose(vwf.awf_mat_t), check=false)
        d = det(F)


        try
            # 计算逆矩阵
            current_inv = inv(transpose(vwf.awf_mat_t))

            # 验证逆矩阵精度
            prod_mat = current_inv * transpose(vwf.awf_mat_t)
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
function local_energy(ham, vwf::vwf_det)
    return 0
end

function compute_grad_log_psi!(vwf::vwf_det{T}) where T
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
    for idx in eachindex(vwf.param_keys)
        dU_t = vwf.dUt_matrix[idx, :, :]
        total_sum = zero(T)

        # 顺序：外层电子(elec)，内层轨道(orb)
        # 优化理由：dU_t[orb, r] 和 A_inv[orb, elec] 第一维都是 orb，内存连续
        @inbounds for elec in 1:Nelec
            r = ss.electron_locs[elec]

            col_sum = zero(T)

            # SIMD 内积
            @simd for orb in 1:Norb
                col_sum += A_inv[orb, elec] * dU_t[orb, r]
            end

            total_sum += col_sum
        end

        # 直接使用 enumerate 的索引，不再依赖计数器变量
        O_vec[idx] = total_sum
    end

    # 直接返回 buffer 引用，避免 copy
    return O_vec
end
