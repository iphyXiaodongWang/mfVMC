module VMC

using Random, LinearAlgebra
using OrderedCollections
using SkewLinearAlgebra
using ..Sampler

export vwf_det, vwf_pfa, VMCRunner, update_vwf_params!
export init_gswf!, mcmc_step!, calc_ham_eng, accept_move!, rebuild_inverse!
export measure_green, measure_SzSz, measure_SplusSminus, measure_SiSj, get_Sz, calc_ratio, compute_grad_log_psi!
export measure_SxSx, measure_SplusSplus
export measure_total_Sz, measure_total_Hole, measure_total_Doublon

@inline function get_Sz(st::Int8)
    return has_up(st) - has_dn(st)
end

@inline function get_Hole(st::Int8)
    return (st == HOLE) ? 1.0 : 0.0
end

@inline function get_Doublon(st::Int8)
    return (st == DB) ? 1.0 : 0.0
end

function measure_green(vwf, i::Int, j::Int, spin_idx::Int8)
    ss = vwf.sampler
    if i == j
        return (ss.state[i] & spin_idx) != 0 ? 1.0 : 0.0
    end

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

    if i == j
        if spin_i == spin_j
            return 1.0
        else
            if (st_i & spin_i) != 0
                return 0.0
            end
            prop = build_spin_flip(ss, j, spin_j)
            return calc_ratio(vwf, prop)
        end
    end

    if (st_i & spin_i) != 0
        return 0.0
    end

    if spin_i == spin_j
        prop = build_single_hop(ss, j, i, spin_j)
        return calc_ratio(vwf, prop)
    else
        prop = build_spin_flip_hop(ss, j, i, spin_j)
        return calc_ratio(vwf, prop)
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
        return calc_ratio(vwf, prop)
    end
    return 0.0
end

function measure_SplusSplus(vwf, i::Int, j::Int)
    if i == j
        return 0.0
    end

    prop = build_double_spin_flip(vwf.sampler, i, j, DN)
    if prop.site1 == 0 || prop.moved_electron_id_1 == 0
        return 0.0
    end

    return calc_ratio(vwf, prop)
end

function measure_SminusSminus(vwf, i::Int, j::Int)
    if i == j
        return 0.0
    end

    prop = build_double_spin_flip(vwf.sampler, i, j, UP)
    if prop.site1 == 0 || prop.moved_electron_id_1 == 0
        return 0.0
    end

    return calc_ratio(vwf, prop)
end

function measure_SxSx(vwf, i::Int, j::Int; conserve_sz=true)
    if i == j
        return 0.25
    end

    val_exchange = measure_SplusSminus(vwf, i, j)

    if conserve_sz
        val_plus = 0
        val_minus = 0
    else
        val_plus = measure_SplusSplus(vwf, i, j)
        val_minus = measure_SminusSminus(vwf, i, j)
    end

    return 0.25 * real(val_exchange + val_plus + val_minus)
end

function measure_SiSj(vwf, i::Int, j::Int)
    return measure_SzSz(vwf, i, j) + 0.5 * real(measure_SplusSminus(vwf, i, j))
end

function calc_ratio(vwf, p::MoveProposal)
    if p.site1 == 0
        return one(typeof(vwf.awf_val))
    end

    if p.moved_electron_id_2 == 0
        return ratio_rank1(vwf, p.moved_electron_id_1, p.target_map_idx_1)
    else
        return ratio_rank2(vwf, p.moved_electron_id_1, p.moved_electron_id_2, p.target_map_idx_1, p.target_map_idx_2)
    end
end

function accept_move!(vwf, p::MoveProposal, ratio)
    vwf.current_ratio = ratio

    if p.moved_electron_id_2 == 0
        update_rank1!(vwf, p.moved_electron_id_1, p.target_map_idx_1, ratio)
    else
        update_rank2!(vwf, p.moved_electron_id_1, p.moved_electron_id_2, p.target_map_idx_1, p.target_map_idx_2, ratio)
    end

    commit_move!(vwf.sampler, p)
end

function mcmc_step!(vwf, kernel::AbstractMCMCKernel, rng::AbstractRNG; detailed_balance::Bool=false)
    cfg = vwf.sampler
    prop, s1, s2 = propose_move(kernel, cfg, rng)

    if prop.site1 == 0
        return false, 0.0, 1.0, prop
    end

    psi_ratio = calc_ratio(vwf, prop)
    accept_prob = abs2(psi_ratio)
    if detailed_balance
        n_fwd = count_choices(kernel, cfg, s1, s2)
        n_rev = count_choices_reserve(kernel, cfg, prop, s1, s2)
        accept_prob *= (Float64(n_fwd) / Float64(n_rev))
    end

    if rand(rng) < accept_prob
        accept_move!(vwf, prop, psi_ratio)
        return true, accept_prob, psi_ratio, prop
    else
        return false, accept_prob, psi_ratio, prop
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
        find_stable_config!(vwf, kernel, rng)
    end
    runner = VMCRunner(ham, vwf, kernel)
    return runner
end

function mcmc_step!(runner::VMCRunner, rng::AbstractRNG; detailed_balance::Bool=false)
    vwf = runner.vwf
    kernel = runner.kernel
    cfg = vwf.sampler

    prop, s1, s2 = propose_move(kernel, cfg, rng)

    if prop.site1 == 0
        return false, 0.0, 1.0, prop
    end

    psi_ratio = calc_ratio(vwf, prop)
    accept_prob = abs2(psi_ratio)
    if detailed_balance
        n_fwd = count_choices(kernel, cfg, s1, s2)
        n_rev = count_choices_reserve(kernel, cfg, prop, s1, s2)
        accept_prob *= (Float64(n_fwd) / Float64(n_rev))
    end

    if rand(rng) < accept_prob
        accept_move!(vwf, prop, psi_ratio)
        return true, accept_prob, psi_ratio, prop
    else
        return false, accept_prob, psi_ratio, prop
    end
end

function calc_ham_eng(runner::VMCRunner; Nmc=1000, therm=100, check_every=100)
    rng = Random.default_rng()

    for _ in 1:therm
        mcmc_step!(runner, rng)
    end

    Engs = Float64[]
    n_accept = 0
    for _ in 1:Nmc
        accepted, _, _, _ = mcmc_step!(runner, rng)
        n_accept += accepted
        v = runner.vwf
        h = runner.ham
        push!(Engs, local_energy(h, v))
    end

    return mean(Engs), n_accept / Nmc
end

include("vmc_det.jl")
include("vmc_pfa.jl")

end # module
