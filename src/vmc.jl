module VMC

using Random, LinearAlgebra
using OrderedCollections
using SkewLinearAlgebra
using ..Sampler
using ..Projector
using ..Backflow

export vwf_det, vwf_pfa, VMCRunner, update_vwf_params!
export set_projector!, set_backflow!, update_vwf_projector_params!, update_vwf_backflow_params!
export get_vwf_projector_param_names, get_vwf_projector_param_values, get_vwf_backflow_param_names, get_vwf_backflow_param_values, get_vwf_total_param_names
export init_gswf!, mcmc_step!, calc_ham_eng, accept_move!, rebuild_inverse!
export measure_green, measure_SzSz, measure_SplusSminus, measure_SiSj, get_Sz, calc_ratio, compute_grad_log_psi!
export measure_SxSx, measure_SplusSplus
export measure_total_Sz, measure_total_Hole, measure_total_Doublon

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

function measure_green(vwf, i::Int, j::Int, spin_idx::Int8)
    ss = vwf.sampler
    if i == j
        return (ss.state[i] & spin_idx) != 0 ? 1.0 : 0.0
    end

    # c^dag_i c_j: j -> i
    st_i = ss.state[i]
    st_j = ss.state[j]
    if ((st_j & spin_idx) != 0) && ((st_i & spin_idx) == 0)
        prop = build_single_hop(ss, j, i, spin_idx)
        ratio_total = calc_total_ratio(vwf, prop)
        return spin_idx == DN && ifPH(ss) ? -ratio_total : ratio_total
    end
    return 0.0
end

#允许spin flip的green暂未做PH修正
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
            return calc_total_ratio(vwf, prop)
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
        return calc_total_ratio(vwf, prop)
    else
        # 2.2: 自旋翻转跳跃 c^dag_{i, sig'} c_{j, sig}
        # j->i, 自旋从 spin_j 变为 spin_i
        prop = build_spin_flip_hop(ss, j, i, spin_j)
        return calc_total_ratio(vwf, prop)
    end
end

function measure_total_Sz(vwf)
    return sum(get_Sz(s) for s in vwf.sampler.state)
end

function measure_total_Hole(vwf)
    return sum(get_Hole(s) for s in vwf.sampler.state)
end

function measure_total_Doublon(vwf)
    return sum(get_Doublon(s) for s in vwf.sampler.state)
end

function measure_SzSz(vwf, i::Int, j::Int)
    return get_Sz(vwf.sampler.state[i]) * get_Sz(vwf.sampler.state[j])
end

function measure_SplusSminus(vwf, i::Int, j::Int)
    if i == j
        return 1.0
    end
    ss = vwf.sampler
    if can_exchange(ss, i, j)
        prop = build_exchange(ss, i, j)
        # 非 Hubbard 的自旋关联测量沿用 Ndefect3 约定
        return calc_total_ratio(vwf, prop)
    end
    return 0.0
end

"""
    measure_SplusSplus(vwf, i, j)
    
计算 <S+_i S+_j>。
物理上对应: i(DN->UP), j(DN->UP)。
"""
function measure_SplusSplus(vwf, i::Int, j::Int)
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
    return calc_total_ratio(vwf, prop)
end

"""
    measure_SminusSminus(vwf, i, j)

计算 <S-_i S-_j>。
物理上对应: i(UP->DN), j(UP->DN)。
"""
function measure_SminusSminus(vwf, i::Int, j::Int)
    if i == j
        return 0.0
    end

    # UP -> DN, 所以 current_spin 是 UP
    prop = build_double_spin_flip(vwf.sampler, i, j, UP)

    if prop.site1 == 0 || prop.moved_electron_id_1 == 0
        return 0.0
    end

    return calc_total_ratio(vwf, prop)
end

"""
    measure_SxSx(vwf, i, j)

计算 <Sx_i Sx_j> = 0.25 * (S+S- + S-S+ + S+S+ + S-S-)。
"""
function measure_SxSx(vwf, i::Int, j::Int; conserve_sz=true)
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


function measure_SiSj(vwf, i::Int, j::Int)
    return measure_SzSz(vwf, i, j) + 0.5 * real(measure_SplusSminus(vwf, i, j))
end

function calc_ratio(vwf, p::MoveProposal)
    # 如果 Proposal 无效
    if p.site1 == 0
        return one(typeof(vwf.awf_val))
    end

    if hasproperty(vwf, :backflow) && Backflow.uses_backflow(getproperty(vwf, :backflow))
        return calc_ratio_rebuild(vwf, p)
    end

    # 单电子移动 (Hop, Flip, Flip-Hop) -> Rank 1
    if p.moved_electron_id_2 == 0
        return ratio_rank1(vwf, p.moved_electron_id_1, p.target_map_idx_1)
    else
        # 双电子移动 (Exchange) -> Rank 2
        return ratio_rank2(vwf, p.moved_electron_id_1, p.moved_electron_id_2, p.target_map_idx_1, p.target_map_idx_2)
    end
end


"""
用途: 计算 projector 对应的比值 `P(C') / P(C)`。

参数:
- `vwf`: 波函数对象。
- `p::MoveProposal`: Monte Carlo proposal。

返回:
- `Number`: projector 比值。若波函数未携带 projector, 返回 1.0。
"""
function calc_projector_ratio(vwf, p::MoveProposal)
    if hasproperty(vwf, :projector)
        return Projector.projector_ratio(getproperty(vwf, :projector), vwf.sampler, p)
    end
    return 1.0
end


"""
用途: 在已知波函数比值 `psi_ratio` 的情况下, 计算总比值。

数学公式:
- `ratio_total = ratio_wf * ratio_projector`

参数:
- `vwf`: 波函数对象。
- `p::MoveProposal`: Monte Carlo proposal。
- `psi_ratio`: 波函数本体比值 `Psi_0(C')/Psi_0(C)`。

返回:
- `Number`: 总比值 `Psi_tot(C')/Psi_tot(C)`。
"""
function calc_total_ratio(vwf, p::MoveProposal, psi_ratio)
    ratio_projector = calc_projector_ratio(vwf, p)
    return psi_ratio * ratio_projector
end


"""
用途: 直接计算总比值 `Psi_tot(C') / Psi_tot(C)`。

参数:
- `vwf`: 波函数对象。
- `p::MoveProposal`: Monte Carlo proposal。

返回:
- `Number`: 总比值。
"""
function calc_total_ratio(vwf, p::MoveProposal)
    psi_ratio = calc_ratio(vwf, p)
    return calc_total_ratio(vwf, p, psi_ratio)
end

function accept_move!(vwf, p::MoveProposal, ratio)
    vwf.current_ratio = ratio

    if hasproperty(vwf, :backflow) && Backflow.uses_backflow(getproperty(vwf, :backflow))
        commit_move!(vwf.sampler, p)
        rebuild_slater_state!(vwf)
        vwf.current_ratio = ratio
        return nothing
    end

    # 1. 更新 Wavefunction 矩阵
    if p.moved_electron_id_2 == 0
        update_rank1!(vwf, p.moved_electron_id_1, p.target_map_idx_1, ratio)
    else
        update_rank2!(vwf, p.moved_electron_id_1, p.moved_electron_id_2, p.target_map_idx_1, p.target_map_idx_2, ratio)
    end

    # 2. 更新 Sampler 状态 (格点, 链表, 计数)
    commit_move!(vwf.sampler, p)
end


function mcmc_step!(vwf, kernel::AbstractMCMCKernel, rng::AbstractRNG; detailed_balance::Bool=false)
    cfg = vwf.sampler
    prop, s1, s2 = propose_move(kernel, cfg, rng)

    if prop.site1 == 0
        return false, 0.0, 1.0, prop
    end

    psi_ratio = calc_ratio(vwf, prop)
    total_ratio = calc_total_ratio(vwf, prop, psi_ratio)

    # 3. 计算接受概率 (Metropolis-Hastings)
    # P_acc = |psi_new/psi_old|^2 * (N_forward / N_reverse)
    accept_prob = abs2(total_ratio)
    # Detailed Balance Correction
    if detailed_balance
        n_fwd = count_choices(kernel, cfg, s1, s2)
        n_rev = count_choices_reserve(kernel, cfg, prop, s1, s2)
        accept_prob *= (Float64(n_fwd) / Float64(n_rev))
    end

    # 4. 接受/拒绝
    if rand(rng) < accept_prob
        accept_move!(vwf, prop, psi_ratio)
        return true, accept_prob, total_ratio, prop
    else
        return false, accept_prob, total_ratio, prop
    end
end

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
    total_ratio = calc_total_ratio(vwf, prop, psi_ratio)
    # prob_ratio = 

    # 3. 计算接受概率 (Metropolis-Hastings)
    # P_acc = |psi_new/psi_old|^2 * (N_forward / N_reverse)
    accept_prob = abs2(total_ratio)
    # Detailed Balance Correction
    if detailed_balance
        n_fwd = count_choices(kernel, cfg, s1, s2)
        n_rev = count_choices_reserve(kernel, cfg, prop, s1, s2)
        accept_prob *= (Float64(n_fwd) / Float64(n_rev))
    end

    # 4. 接受/拒绝
    if rand(rng) < accept_prob
        accept_move!(vwf, prop, psi_ratio)
        return true, accept_prob, total_ratio, prop
    else
        return false, accept_prob, total_ratio, prop
    end
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

include("vmc_det.jl")
include("vmc_pfa.jl")

end # module
