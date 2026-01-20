module VMC

using Random, LinearAlgebra
using OrderedCollections
using ..Sampler


export vwf_det, VMCRunner, update_vwf_params!
export init_gswf!, mcmc_step!, calc_ham_eng, accept_move!, rebuild_inverse!


export measure_green, measure_SzSz, measure_SplusSminus, measure_SiSj, get_Sz, calc_ratio, compute_grad_log_psi!
export measure_SxSx, measure_SplusSplus
export measure_total_Sz, measure_total_Hole, measure_total_Doublon

# export local_energy 

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
    dUt_params::OrderedDict{Symbol,Matrix{T}}
    dUt_list::Vector{Matrix{T}}
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

    dUt_params = OrderedDict{Symbol,Matrix{T}}()
    dUt_list = Vector{Matrix{T}}()
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
        dUt_params,
        dUt_list,
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

    n_params = length(v.dUt_params) # 注意：是 dU_params 还是 dUt_params，请保持变量名一致
    if length(ws.grad_buffer) != n_params
        resize!(ws.grad_buffer, n_params)
    end

    return ws
end

# function update_dUt_list!(vwf::vwf_det)
#     empty!(vwf.dUt_list)
#     empty!(vwf.param_keys)

#     for (k, v) in vwf.dUt_params
#         push!(vwf.param_keys, k)
#         push!(vwf.dUt_list, v)
#     end

#     ensure_ws!(vwf) 
#     return nothing
# end


function update_vwf_params!(vwf::vwf_det{T}, new_params::OrderedDict{Symbol,Matrix{T}}) where T
    vwf.dUt_params = new_params

    empty!(vwf.dUt_list)
    empty!(vwf.param_keys)

    for (k, v) in new_params
        push!(vwf.param_keys, k)
        push!(vwf.dUt_list, v)
    end

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

function calc_ratio(vwf::vwf_det{T}, p::MoveProposal) where T
    # 如果 Proposal 无效
    if p.site1 == 0
        return one(T)
    end

    # 单电子移动 (Hop, Flip, Flip-Hop) -> Rank 1
    if p.moved_electron_id_2 == 0
        return ratio_rank1(vwf, p.moved_electron_id_1, p.target_map_idx_1)
    else
        # 双电子移动 (Exchange) -> Rank 2
        return ratio_rank2(vwf, p.moved_electron_id_1, p.moved_electron_id_2, p.target_map_idx_1, p.target_map_idx_2)
    end
end

function accept_move!(vwf::vwf_det{T}, p::MoveProposal, ratio::T) where T
    vwf.current_ratio = ratio

    # 1. 更新 Wavefunction 矩阵
    if p.moved_electron_id_2 == 0
        update_rank1!(vwf, p.moved_electron_id_1, p.target_map_idx_1, ratio)
    else
        update_rank2!(vwf, p.moved_electron_id_1, p.moved_electron_id_2, p.target_map_idx_1, p.target_map_idx_2, ratio)
    end

    # 2. 更新 Sampler 状态 (格点, 链表, 计数)
    commit_move!(vwf.sampler, p)
end


# ==============================================================================
# 3. VMCRunner
# ==============================================================================
# 更新：引入 Kernel 类型参数 K，移除 conserve_sz (放在 kernel 内部)
mutable struct VMCRunner{H,W,K<:AbstractMCMCKernel}
    ham::H
    vwf::W
    kernel::K
end

function VMCRunner(ham, vwf; kernel::AbstractMCMCKernel, auto_fix::Bool=true)
    if auto_fix
        rng = Random.default_rng()
        # 此时 vwf.sampler 可能已经初始化过，但为了安全起见，
        # find_stable_config! 会强制重新随机化直到找到良态
        find_stable_config!(vwf, kernel, rng)
    end
    runner = VMCRunner(ham, vwf, kernel)
    return runner
    # return VMCRunner(ham, vwf, kernel)
end


function mcmc_step!(vwf::vwf_det{T}, kernel::AbstractMCMCKernel, rng::AbstractRNG; detailed_balance::Bool=false) where T
    cfg = vwf.sampler
    prop, s1, s2 = propose_move(kernel, cfg, rng)

    if prop.site1 == 0
        return false, 0.0, 1.0, prop
    end

    psi_ratio = calc_ratio(vwf, prop)

    # 3. 计算接受概率 (Metropolis-Hastings)
    # P_acc = |psi_new/psi_old|^2 * (N_forward / N_reverse)
    accept_prob = abs2(psi_ratio)
    # Detailed Balance Correction
    if detailed_balance
        n_fwd = count_choices(kernel, cfg, s1, s2)
        n_rev = count_choices_reserve(kernel, cfg, prop, s1, s2)
        accept_prob *= (Float64(n_fwd) / Float64(n_rev))
    end

    # 4. 接受/拒绝
    if rand(rng) < accept_prob
        accept_move!(vwf, prop, psi_ratio)
        return true, accept_prob, psi_ratio, prop
    else
        return false, accept_prob, psi_ratio, prop
    end
end

function mcmc_step!(runner::VMCRunner, rng::AbstractRNG; detailed_balance::Bool=false)
    vwf = runner.vwf
    kernel = runner.kernel
    cfg = vwf.sampler

    prop, s1, s2 = propose_move(kernel, cfg, rng)

    if prop.site1 == 0
        return false, 0.0, 1.0, prop
    end

    # 2. 计算波函数比值 psi_new / psi_old
    psi_ratio = calc_ratio(vwf, prop)
    # prob_ratio = 

    # 3. 计算接受概率 (Metropolis-Hastings)
    # P_acc = |psi_new/psi_old|^2 * (N_forward / N_reverse)
    accept_prob = abs2(psi_ratio)
    # Detailed Balance Correction
    if detailed_balance
        n_fwd = count_choices(kernel, cfg, s1, s2)
        n_rev = count_choices_reserve(kernel, cfg, prop, s1, s2)
        accept_prob *= (Float64(n_fwd) / Float64(n_rev))
    end

    # 4. 接受/拒绝
    if rand(rng) < accept_prob
        accept_move!(vwf, prop, psi_ratio)
        return true, accept_prob, psi_ratio, prop
    else
        return false, accept_prob, psi_ratio, prop
    end
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


function calc_ham_eng(runner::VMCRunner; Nmc=1000, therm=100, check_every=100)
    rng = Random.default_rng()
    v = runner.vwf
    h = runner.ham
    N_sites = v.sampler.N_sites

    # Therm
    for _ in 1:therm
        for _ in 1:N_sites
            mcmc_step!(runner, rng)
        end
    end

    Engs = Float64[]
    sizehint!(Engs, Nmc)
    steps = 0

    for _ in 1:Nmc
        for _ in 1:N_sites
            mcmc_step!(runner, rng)
        end

        steps += 1
        if steps >= check_every
            rebuild_inverse!(v)
            steps = 0
        end

        # 关键调用：多态分发到具体的 local_energy 实现
        push!(Engs, local_energy(h, v))
    end
    return Engs
end

# ==============================================================================
# Generic Measurements
# ==============================================================================
@inline function get_Sz(st::Int8)
    val = 0.0
    if (st & UP) != 0
        val += 0.5
    end
    if (st & DN) != 0
        val -= 0.5
    end
    return val
end

@inline function get_Hole(st::Int8)
    val = 0.0
    if (st & HOLE) != 0
        val += 1.0
    end
    return val
end

@inline function get_Doublon(st::Int8)
    val = 0.0
    if (st & DB) != 0
        val += 1.0
    end
    return val
end

function measure_green(vwf::vwf_det, i::Int, j::Int, spin_idx::Int8)
    ss = vwf.sampler
    if i == j
        return (ss.state[i] & spin_idx) != 0 ? 1.0 : 0.0
    end

    # c^dag_i c_j: j -> i
    st_i = ss.state[i]
    st_j = ss.state[j]
    if ((st_j & spin_idx) != 0) && ((st_i & spin_idx) == 0)
        prop = build_single_hop(ss, j, i, spin_idx)
        return calc_ratio(vwf, prop)
    end
    return 0.0
end

function measure_green(vwf, i::Int, spin_i::Int8, j::Int, spin_j::Int8)
    ss = vwf.sampler

    st_i = ss.state[i]
    st_j = ss.state[j]

    if (st_j & spin_j) == 0
        return 0.0
    end

    # === Case 1: 同一位置 (i == j) ===
    if i == j
        if spin_i == spin_j
            # 1.1: 粒子数算符 n_{i, sigma}
            # 前面已经检查了 (st_j & spin_j) != 0，所以这里直接返回 1.0
            return 1.0
        else
            # 1.2: 局域自旋翻转 c^dag_{i, up} c_{i, dn}
            # 要求: j(即i) 有 spin_j，且 j(即i) 没有 spin_i
            # (st_j & spin_j) != 0 已检查
            if (st_i & spin_i) != 0
                return 0.0 # 目标自旋已存在，泡利阻塞
            end

            # 构建翻转 Proposal: 把 spin_j 翻转为 spin_i
            prop = build_spin_flip(ss, j, spin_j)
            return calc_ratio(vwf, prop)
        end
    end

    # === Case 2: 不同位置 (i != j) ===
    # 目标位置 i 必须不能有 spin_i (否则泡利阻塞)
    if (st_i & spin_i) != 0
        return 0.0
    end

    if spin_i == spin_j
        # 2.1: 普通跳跃 c^dag_{i, sig} c_{j, sig}
        # j->i, 自旋不变
        prop = build_single_hop(ss, j, i, spin_j)
        return calc_ratio(vwf, prop)
    else
        # 2.2: 自旋翻转跳跃 c^dag_{i, sig'} c_{j, sig}
        # j->i, 自旋从 spin_j 变为 spin_i
        prop = build_spin_flip_hop(ss, j, i, spin_j)
        return calc_ratio(vwf, prop)
    end
end

function measure_total_Sz(vwf::vwf_det)
    return sum(get_Sz(s) for s in vwf.sampler.state)
end

function measure_total_Hole(vwf::vwf_det)
    return sum(get_Hole(s) for s in vwf.sampler.state)
end

function measure_total_Doublon(vwf::vwf_det)
    return sum(get_Doublon(s) for s in vwf.sampler.state)
end

function measure_SzSz(vwf::vwf_det, i::Int, j::Int)
    return get_Sz(vwf.sampler.state[i]) * get_Sz(vwf.sampler.state[j])
end

function measure_SplusSminus(vwf::vwf_det, i::Int, j::Int)
    if i == j
        return 1.0
    end
    ss = vwf.sampler
    if can_exchange(ss, i, j)
        prop = build_exchange(ss, i, j)
        # Operator: 0.5 * (S+S- + h.c.)
        return calc_ratio(vwf, prop)
    end
    return 0.0
end

"""
    measure_SplusSplus(vwf, i, j)
    
计算 <S+_i S+_j>。
物理上对应: i(DN->UP), j(DN->UP)。
"""
function measure_SplusSplus(vwf::vwf_det, i::Int, j::Int)
    if i == j
        return 0.0
    end

    # DN -> UP, 所以 current_spin 是 DN
    prop = build_double_spin_flip(vwf.sampler, i, j, DN)

    # 如果 Proposal 无效 (比如没有 DN 电子)，build 函数会返回 empty，
    # calc_ratio 内部对 site1==0 会直接返回 1.0 (注意! 这里需要小心)
    # 修正: calc_ratio 通常处理的是 Metropolis 接受率，
    # 对于测量，我们需要手动判断 proposal 是否有效。

    if prop.site1 == 0 || prop.moved_electron_id_1 == 0
        return 0.0
    end

    # Rank-2 Update, 系数 +1
    return calc_ratio(vwf, prop)
end

"""
    measure_SminusSminus(vwf, i, j)

计算 <S-_i S-_j>。
物理上对应: i(UP->DN), j(UP->DN)。
"""
function measure_SminusSminus(vwf::vwf_det, i::Int, j::Int)
    if i == j
        return 0.0
    end

    # UP -> DN, 所以 current_spin 是 UP
    prop = build_double_spin_flip(vwf.sampler, i, j, UP)

    if prop.site1 == 0 || prop.moved_electron_id_1 == 0
        return 0.0
    end

    return calc_ratio(vwf, prop)
end

"""
    measure_SxSx(vwf, i, j)

计算 <Sx_i Sx_j> = 0.25 * (S+S- + S-S+ + S+S+ + S-S-)。
"""
function measure_SxSx(vwf::vwf_det, i::Int, j::Int; conserve_sz=true)
    if i == j
        return 0.25
    end

    # 1. Exchange 部分 (S+S- + S-S+)
    # measure_SplusSminus 已经处理了:
    #  - 只有当自旋反平行时才非零
    #  - 已经包含 -1 的交换系数
    val_exchange = measure_SplusSminus(vwf, i, j)

    # 2. Pair Creation/Annihilation 部分 (S+S+ + S-S-)
    # 只有当自旋平行时才非零
    if conserve_sz
        val_plus = 0
        val_minus = 0
    else
        val_plus = measure_SplusSplus(vwf, i, j)
        val_minus = measure_SminusSminus(vwf, i, j)
    end
    # println(val_plus, val_minus)
    # 求和并取实部
    return 0.25 * real(val_exchange + val_plus + val_minus)
end


function measure_SiSj(vwf::vwf_det, i::Int, j::Int)
    return measure_SzSz(vwf, i, j) + 0.5 * real(measure_SplusSminus(vwf, i, j))
end

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
    for (param_idx, dU_t) in enumerate(vwf.dUt_list)

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
        O_vec[param_idx] = total_sum
    end

    # 直接返回 buffer 引用，避免 copy
    return O_vec
end



end # module
