module VMC

using Random, LinearAlgebra
using ..Sampler 

export vwf_det, HeisenbergModel, VMCRunner
export init_gswf!, mcmc_step!, calc_ham_eng, local_energy

# ==============================================================================
# WorkSpace
# ==============================================================================
# R1R2WS的注释不能删除或者修改
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
end

# ==============================================================================
# Wavefunction: vwf_det (Generic T)
# ==============================================================================
mutable struct vwf_det{T, S}
    # -- Matrices --
    gs_U::Matrix{T}            # 原始布局：(2*Nsites) x Nelec (仅保留用于参考或特殊用途)
    gs_U_t::Matrix{T}          # 转置布局：Nelec x (2*Nsites)，用于高性能列读取
    
    awf_mat_t::Matrix{T}       
    awf_inv::Matrix{T}         
    awf_val::T                 # Determinant value
    
    current_ratio::T           # Stored ratio of the last accepted move

    sampler::S                 
    
    ws::R1R2WS{T}        
end

function vwf_det(U::Matrix{T}, sampler) where T
    Nlat = sampler.N_sites
    expected_rows = 2 * Nlat
    expected_cols = sampler.N_up + sampler.N_dn
    
    @assert size(U, 1) == expected_rows "U rows $(size(U,1)) != 2*Nlat"
    @assert size(U, 2) >= expected_cols "U cols $(size(U,2)) < Nelec"

    dummy_ws = R1R2WS{T}(0, T[], T[], T[], T[], T[], T[], T[], T[])
    
    # 初始化时，先生成一个空的 awf_mat_t，大小为 Nelec x Nelec
    # 具体的填充在 init_gswf! 中进行
    nelec = expected_cols
    awf_mat_t = zeros(T, nelec, nelec)
    awf_inv = zeros(T, 1, 1)
    
    return vwf_det{T, typeof(sampler)}(
        U, 
        permutedims(U), # gs_U_t
        awf_mat_t,      # awf_mat_t
        awf_inv,        # awf_inv (placeholder)
        T(0), 
        one(T),     
        sampler,
        dummy_ws
    )
end

function ensure_ws!(v::vwf_det{T, S}) where {T, S}
    N = size(v.awf_mat_t, 1)
    ws = v.ws
    if ws.N != N
        ws = R1R2WS{T}(
            N,
            Vector{T}(undef, N), Vector{T}(undef, N),
            Vector{T}(undef, N), Vector{T}(undef, N),
            Vector{T}(undef, N), Vector{T}(undef, N),
            Vector{T}(undef, N), Vector{T}(undef, N),
        )
        v.ws = ws
    end
    return ws
end

function init_gswf!(vwf::vwf_det{T, S}) where {T, S}
    ss = vwf.sampler
    initialize_lists!(ss)
    
    total_elec = ss.N_up + ss.N_dn
    elec_locs = zeros(Int, total_elec)
    
    count_found = 0
    for loc_idx in 1:length(ss.map_spin_to_id)
        eid = ss.map_spin_to_id[loc_idx]
        if eid != 0
            elec_locs[eid] = loc_idx 
            count_found += 1
        end
    end
    
    # 重新分配或重置 awf_mat_t
    if size(vwf.awf_mat_t, 1) != total_elec
        vwf.awf_mat_t = zeros(T, total_elec, total_elec)
    end
    
    # 填充 Slater 矩阵的转置
    # A[i, :] = gs_U[row, :]  =>  A_t[:, i] = gs_U_t[:, row]
    for i in 1:total_elec
        row_in_U = elec_locs[i]
        # 使用连续内存拷贝
        copyto!(@view(vwf.awf_mat_t[:, i]), @view(vwf.gs_U_t[:, row_in_U]))
    end

    # 计算行列式和逆矩阵
    # 物理矩阵 A = transpose(awf_mat_t)
    # 我们需要对物理矩阵 A 进行 LU 分解
    # 为了避免通过 transpose 创建视图导致 LU 变慢，这里 copy 一份临时的 A 比较安全，
    # 或者直接对 transpose(A_t) 求逆。Julia 的 inv(transpose(...)) 通常有优化。
    
    # 方法：显式构造物理 A 用于初次计算（不影响后续 O(N^2) 更新）
    A_physical = transpose(vwf.awf_mat_t)
    F = lu(A_physical)
    
    vwf.awf_val = det(F)
    vwf.awf_inv = inv(F) # 存储物理 A 的逆
    vwf.current_ratio = one(T)

    ensure_ws!(vwf)
    return nothing
end

function rebuild_inverse!(vwf::vwf_det)
    vwf.awf_inv = inv(transpose(vwf.awf_mat_t))
end

# --- Update Logic Helpers ---
@inline function rank1_update_blas!(A::Matrix{T}, alpha::T, x::Vector{T}, y::Vector{T}) where T <: Float64
    BLAS.ger!(alpha, x, y, A)
end

@inline function rank1_update_blas!(A::Matrix{T}, alpha::T, x::Vector{T}, y::Vector{T}) where T <: Complex
    BLAS.geru!(alpha, x, y, A) 
end

function ratio_rank1(vwf::vwf_det{T}, k::Int, new_row_idx_U::Int) where T
    val = zero(T)
    # N 是电子数
    N = size(vwf.awf_inv, 1) 
    # 计算 <U_new | Ainv[:, k]>
    # U_new 对应 gs_U 的行 new_row_idx_U，即 gs_U_t 的列 new_row_idx_U (连续)
    # Ainv[:, k] 是 Ainv 的第 k 列 (连续)
    @inbounds @simd for j in 1:N
        val += vwf.gs_U_t[j, new_row_idx_U] * vwf.awf_inv[j, k]
    end
    return val
end

function update_rank1!(vwf::vwf_det{T}, k::Int, new_row_idx_U::Int, ratio::T) where T
    ws = ensure_ws!(vwf)
    
    # 这里的 A_t 存储的是转置矩阵
    A_t = vwf.awf_mat_t
    Ainv = vwf.awf_inv
    N = size(A_t, 1) 

    # 1. 计算 dr1 (O(N))
    # 物理: dr = U_new - A_old[k, :]
    # 内存: dr = gs_U_t[:, new] - A_t[:, k] (全连续访问)
    @inbounds @simd for j in 1:N
        ws.dr1[j] = vwf.gs_U_t[j, new_row_idx_U] - A_t[j, k]
    end

    # 2. 准备 col1 = Ainv' * dr1 (O(N^2))
    mul!(ws.col1, transpose(Ainv), ws.dr1)
    
    # 3. 准备 xi = Ainv[:, k] (O(N))
    copyto!(ws.xi, @view Ainv[:, k])
    
    # 4. 更新逆矩阵 (O(N^2))
    rank1_update_blas!(Ainv, -1/ratio, ws.xi, ws.col1)
    
    # 5. 更新 Slater 矩阵 A_t (O(N))
    # 物理: A[k, :] = U_new
    # 内存: A_t[:, k] = gs_U_t[:, new] (全连续访问)
    @inbounds @simd for j in 1:N
        A_t[j, k] = vwf.gs_U_t[j, new_row_idx_U]
    end
    
    vwf.awf_val *= ratio
end

function ratio_rank2(vwf::vwf_det{T}, k1::Int, k2::Int, new_row1_U::Int, new_row2_U::Int) where T
    Ainv = vwf.awf_inv
    # 使用 gs_U_t 进行连续访问
    U_t = vwf.gs_U_t 
    N = size(Ainv, 1)
    
    d11 = zero(T); d12 = zero(T); d21 = zero(T); d22 = zero(T)
    
    @inbounds @simd for j in 1:N
        # 读取 U_t 的列，连续内存
        u1 = U_t[j, new_row1_U]
        u2 = U_t[j, new_row2_U]
        
        # Ainv 列访问，连续内存
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
    A_t = vwf.awf_mat_t  # 使用转置矩阵
    Ainv = vwf.awf_inv
    N = size(A_t, 1)

    # 1. 计算 dr1, dr2 (O(N)) - 全连续内存
    @inbounds @simd for j in 1:N
        # dr1 = U_new1 - A_old_row1
        ws.dr1[j] = vwf.gs_U_t[j, new_row1_U] - A_t[j, k1]
        # dr2 = U_new2 - A_old_row2
        ws.dr2[j] = vwf.gs_U_t[j, new_row2_U] - A_t[j, k2]
    end
    
    # 2. 计算 col1, col2 (O(N^2))
    mul!(ws.col1, transpose(Ainv), ws.dr1)
    mul!(ws.col2, transpose(Ainv), ws.dr2)
    
    # 3. Sherman-Morrison-Woodbury 系数计算
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
    
    # 4. 更新逆矩阵 (O(N^2))
    rank1_update_blas!(Ainv, -one(T), ws.s0, ws.col1)
    rank1_update_blas!(Ainv, -one(T), ws.s1, ws.col2)
    
    # 5. 更新 Slater 矩阵 A_t (O(N)) - 全连续内存
    @inbounds @simd for j in 1:N
        A_t[j, k1] = vwf.gs_U_t[j, new_row1_U]
        A_t[j, k2] = vwf.gs_U_t[j, new_row2_U]
    end

    vwf.awf_val *= ratio
end

function calc_ratio(vwf::vwf_det{T}, p::MoveProposal) where T
    p.site1 == 0 && return zero(T)
    if p.moved_electron_id_2 == 0
        return ratio_rank1(vwf, p.moved_electron_id_1, p.target_map_idx_1)
    else
        return ratio_rank2(vwf, p.moved_electron_id_1, p.moved_electron_id_2, p.target_map_idx_1, p.target_map_idx_2)
    end
end

function accept_move!(vwf::vwf_det{T}, p::MoveProposal, ratio::T) where T
    vwf.current_ratio = ratio 
    
    if p.moved_electron_id_2 == 0
        update_rank1!(vwf, p.moved_electron_id_1, p.target_map_idx_1, ratio)
    else
        update_rank2!(vwf, p.moved_electron_id_1, p.moved_electron_id_2, p.target_map_idx_1, p.target_map_idx_2, ratio)
    end
    commit_move!(vwf.sampler, p)
end

# ==============================================================================
# 3. Hamiltonian / Model
# ==============================================================================
struct HeisenbergModel
    lx::Int; ly::Int; Nlat::Int
    J1::Float64; J2::Float64; J3::Float64
    J1_bonds::Vector{Tuple{Int,Int}}
    J2_bonds::Vector{Tuple{Int,Int}}
    J3_bonds::Vector{Tuple{Int,Int}}
end

function HeisenbergModel(Nlat::Int; model_params=Dict{Symbol,Any}())
    lx = get(model_params, :lx, floor(Int, sqrt(Nlat)))
    ly = get(model_params, :ly, floor(Int, Nlat/lx))
    J1 = get(model_params, :J1, 1.0)
    J2 = get(model_params, :J2, 0.0)
    J3 = get(model_params, :J3, 0.0)
    
    bonds1 = Tuple{Int,Int}[]; bonds2 = Tuple{Int,Int}[]; bonds3 = Tuple{Int,Int}[]
    idx(x, y) = mod(y-1, ly)*lx + mod(x-1, lx) + 1
    
    for y in 1:ly, x in 1:lx
        u = idx(x, y)
        push!(bonds1, (u, idx(x+1, y))); push!(bonds1, (u, idx(x, y+1)))
        if J2 != 0
            push!(bonds2, (u, idx(x+1, y+1))); push!(bonds2, (u, idx(x-1, y+1)))
        end
    end
    return HeisenbergModel(lx, ly, Nlat, J1, J2, J3, bonds1, bonds2, bonds3)
end

function spin_z(ss, site)
    st = ss.state[site]
    return st == UP ? 0.5 : (st == DN ? -0.5 : 0.0)
end

function energy_diagonal(ss, bonds, J)
    E = 0.0
    for (i, j) in bonds
        E += J * spin_z(ss, i) * spin_z(ss, j)
    end
    return E
end

function energy_offdiag(ham::HeisenbergModel, vwf::vwf_det, bonds, J)
    E = 0.0
    ss = vwf.sampler
    for (i, j) in bonds
        if can_exchange(ss, i, j)
            prop = build_exchange(ss, i, j)
            # ratio is complex/real depending on T. J is real.
            ratio = calc_ratio(vwf, prop) 
            E += real(0.5 * J * ratio) # Take real part for Energy
        end
    end
    return E
end

function local_energy(ham::HeisenbergModel, vwf::vwf_det)
    E = 0.0
    if ham.J1 != 0
        E += energy_diagonal(vwf.sampler, ham.J1_bonds, ham.J1)
        E += energy_offdiag(ham, vwf, ham.J1_bonds, ham.J1)
    end
    if ham.J2 != 0
        E += energy_diagonal(vwf.sampler, ham.J2_bonds, ham.J2)
        E += energy_offdiag(ham, vwf, ham.J2_bonds, ham.J2)
    end
    return E
end

# ==============================================================================
# 4. VMCRunner
# ==============================================================================
mutable struct VMCRunner{H, W}
    ham::H
    vwf::W
    conserve_sz::Bool
end

VMCRunner(vwf::vwf_det, ham::HeisenbergModel; conserve_sz::Bool=true) = VMCRunner(ham, vwf, conserve_sz)

function mcmc_step!(runner::VMCRunner, rng::AbstractRNG; tol=1e-12)
    vwf = runner.vwf
    s1 = rand(rng, 1:runner.ham.Nlat)
    s2 = rand(rng, 1:runner.ham.Nlat)
    
    prop = propose_move_Hubbard!(vwf.sampler, s1, s2, runner.conserve_sz, rng)
    if prop.site1 == 0; return nothing; end
    
    ratio = calc_ratio(vwf, prop)
    
    # Metropolis: abs2 works for both Real and Complex
    p_acc = abs2(ratio) 
    
    if rand(rng) < p_acc
        accept_move!(vwf, prop, ratio)
        return prop
    else
        return nothing
    end
end

function calc_ham_eng(runner::VMCRunner; Nmc=1000, therm=100, check_every=100)
    rng = Random.default_rng()
    v = runner.vwf; h = runner.ham
    
    for _ in 1:therm; for _ in 1:h.Nlat; mcmc_step!(runner, rng); end; end
    
    Engs = Float64[]
    sizehint!(Engs, Nmc)
    steps_since_rebuild = 0
    
    for step in 1:Nmc
        for _ in 1:h.Nlat; mcmc_step!(runner, rng); end
        
        steps_since_rebuild += 1
        if steps_since_rebuild >= check_every
            rebuild_inverse!(v)
            steps_since_rebuild = 0
        end
        
        push!(Engs, local_energy(h, v))
    end
    return Engs
end

end # module
