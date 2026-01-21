module Sampler

using Random, LinearAlgebra

export config_Hubbard, config_Heisenberg, MoveProposal, copy_config
export total_elec, ifPH
export init_config_Heisenberg!, initialize_lists!, init_config_Hubbard!, init_config_by_state_char, init_config_rand!
export propose_move, commit_move!, update_site_config!
export build_exchange, build_single_hop, build_spin_flip, build_spin_flip_hop, build_double_spin_flip
export can_hop, can_exchange, can_flip

export get_state_char
export check_consistency, print_config_debug, test_spin_flip_hop, test_config_calculation, print_config
export HOLE, UP, DN, DB
export AbstractMCMCKernel, HubbardKernel, HeisenbergKernel
export has_up, has_dn

const HOLE = Int8(0) # 00
const UP = Int8(1) # 01
const DN = Int8(2) # 10
const DB = Int8(3) # 11 

@inline has_up(s::Int8) = (s & UP) != 0
@inline has_dn(s::Int8) = (s & DN) != 0

function get_state_char(s::Int8)
    s == HOLE && return 'h'
    s == UP && return 'u'
    s == DN && return 'd'
    s == DB && return 'D'
    return 'X'
end

function _parse_char_to_state(c::Char)
    if c == 'u'
        return UP
    elseif c == 'd'
        return DN
    elseif c == 'D'
        return DB
    elseif c == 'h'
        return HOLE
    else
        error("Unknown state character: '$c', char should be u, d, D, or h")
    end
end


# ==============================================================================
# Configuration
# ==============================================================================
mutable struct Configuration
    N_sites::Int
    N_up::Int
    N_dn::Int

    state::Vector{Int8}
    map_spin_to_id::Vector{Int}
    electron_locs::Vector{Int}

    # 加速链表 (用于快速寻找特定状态的格点)
    list_holes::Vector{Int}
    list_ups::Vector{Int}
    list_dns::Vector{Int}
    list_dbs::Vector{Int}

    count_holes::Int
    count_ups::Int
    count_dns::Int
    count_dbs::Int

    site_ptr::Vector{Int}
end
mutable struct ConfigurationPH
    N_sites::Int
    N_up::Int
    N_dn::Int

    state::Vector{Int8}
    map_spin_to_id::Vector{Int}
    electron_locs::Vector{Int}

    # 加速链表 (用于快速寻找特定状态的格点)
    list_holes::Vector{Int}
    list_ups::Vector{Int}
    list_dns::Vector{Int}
    list_dbs::Vector{Int}

    count_holes::Int
    count_ups::Int
    count_dns::Int
    count_dbs::Int

    site_ptr::Vector{Int}
end
@inline ifPH(ss::Union{Configuration,ConfigurationPH}) = ss isa ConfigurationPH
function config_Hubbard(N_sites::Int, N_up::Int, N_dn::Int; ifPH::Bool=false)
    max_N = N_sites
    if ifPH
        ss = ConfigurationPH(
            N_sites, N_up, N_dn,
            zeros(Int8, N_sites),           # state
            zeros(Int, 2 * N_sites),        # map_spin_to_id 
            zeros(Int, N_up - N_dn + N_sites),
            zeros(Int, max_N), # list_holes
            zeros(Int, max_N), # list_ups
            zeros(Int, max_N), # list_dns
            zeros(Int, max_N), # list_dbs
            0, 0, 0, 0,        # counts
            zeros(Int, N_sites) # site_ptr
        )
    else
        ss = Configuration(
            N_sites, N_up, N_dn,
            zeros(Int8, N_sites),           # state
            zeros(Int, 2 * N_sites),        # map_spin_to_id 
            zeros(Int, N_up + N_dn),
            zeros(Int, max_N), # list_holes
            zeros(Int, max_N), # list_ups
            zeros(Int, max_N), # list_dns
            zeros(Int, max_N), # list_dbs
            0, 0, 0, 0,        # counts
            zeros(Int, N_sites) # site_ptr
        )
    end
    return ss
end
function total_elec(cfg::Union{Configuration,ConfigurationPH})
    if ifPH(cfg)
        return cfg.N_up - cfg.N_dn + cfg.N_sites
    else
        return cfg.N_up + cfg.N_dn
    end
end
function config_Heisenberg(N_sites::Int, N_up::Int; ifPH::Bool=false)
    N_dn = N_sites - N_up
    ss = config_Hubbard(N_sites, N_up, N_dn; ifPH=ifPH)
    return ss
end

function copy_config(cfg::Configuration)
    return Configuration(
        cfg.N_sites, cfg.N_up, cfg.N_dn,
        copy(cfg.state), copy(cfg.map_spin_to_id),
        copy(cfg.electron_locs),
        copy(cfg.list_holes), copy(cfg.list_ups), copy(cfg.list_dns), copy(cfg.list_dbs),
        cfg.count_holes, cfg.count_ups, cfg.count_dns, cfg.count_dbs,
        copy(cfg.site_ptr)
    )
end

function update_site_config!(ss, site::Int, new_state::Int8)
    old_state = ss.state[site]
    if old_state == new_state
        return
    end

    delta_up = has_up(new_state) - has_up(old_state) #((new_state & UP) != 0) - ((old_state & UP) != 0)
    ss.N_up += delta_up

    # 计算 DN 变化: (new has DN) - (old has DN)
    delta_dn = has_dn(new_state) - has_dn(old_state) #((new_state & DN) != 0) - ((old_state & DN) != 0)
    ss.N_dn += delta_dn

    if old_state == HOLE
        ss.count_holes = _remove_from_list!(ss.list_holes, ss.count_holes, ss.site_ptr, site)
    elseif old_state == UP
        ss.count_ups = _remove_from_list!(ss.list_ups, ss.count_ups, ss.site_ptr, site)
    elseif old_state == DN
        ss.count_dns = _remove_from_list!(ss.list_dns, ss.count_dns, ss.site_ptr, site)
    else # DB
        ss.count_dbs = _remove_from_list!(ss.list_dbs, ss.count_dbs, ss.site_ptr, site)
    end

    if new_state == HOLE
        ss.count_holes = _add_to_list!(ss.list_holes, ss.count_holes, ss.site_ptr, site)
    elseif new_state == UP
        ss.count_ups = _add_to_list!(ss.list_ups, ss.count_ups, ss.site_ptr, site)
    elseif new_state == DN
        ss.count_dns = _add_to_list!(ss.list_dns, ss.count_dns, ss.site_ptr, site)
    else # DB
        ss.count_dbs = _add_to_list!(ss.list_dbs, ss.count_dbs, ss.site_ptr, site)
    end

    ss.state[site] = new_state
end

@inline function _add_to_list!(list, count, ptr_map, site)
    count += 1
    @inbounds list[count] = site
    @inbounds ptr_map[site] = count
    return count
end

@inline function _remove_from_list!(list, count, ptr_map, site)
    idx = ptr_map[site]

    # 如果 idx == 0，说明站点不在列表中，直接返回
    if idx == 0
        return count
    end

    last_site = list[count]

    # Swap remove
    @inbounds list[idx] = last_site
    @inbounds ptr_map[last_site] = idx

    # 重要：清空被删除站点的指针
    @inbounds ptr_map[site] = 0

    count -= 1
    return count
end

function initialize_lists!(ss)
    ss.count_holes = 0
    ss.count_ups = 0
    ss.count_dns = 0
    ss.count_dbs = 0
    fill!(ss.site_ptr, 0)

    for site in 1:ss.N_sites
        st = ss.state[site]
        if st == HOLE
            ss.count_holes = _add_to_list!(ss.list_holes, ss.count_holes, ss.site_ptr, site)
        elseif st == UP
            ss.count_ups = _add_to_list!(ss.list_ups, ss.count_ups, ss.site_ptr, site)
        elseif st == DN
            ss.count_dns = _add_to_list!(ss.list_dns, ss.count_dns, ss.site_ptr, site)
        else
            ss.count_dbs = _add_to_list!(ss.list_dbs, ss.count_dbs, ss.site_ptr, site)
        end
    end
end

function init_config_Hubbard!(ss::Configuration; n_doublon::Union{Int,Nothing}=nothing)
    fill!(ss.state, HOLE)
    fill!(ss.map_spin_to_id, 0)

    ss.count_holes = 0
    ss.count_ups = 0
    ss.count_dns = 0
    ss.count_dbs = 0
    fill!(ss.site_ptr, 0)

    all_sites = collect(1:ss.N_sites)

    if n_doublon !== nothing
        n_d = n_doublon
        n_u = ss.N_up - n_d
        n_dwn = ss.N_dn - n_d
        n_h = ss.N_sites - (n_d + n_u + n_dwn)

        if n_u < 0 || n_dwn < 0 || n_h < 0
            error("n_doublon: $n_doublon is too large and n_up/n_dn/n_h is negative")
        end

        shuffle!(all_sites)

        current_idx = 1
        for _ in 1:n_d
            site = all_sites[current_idx]
            ss.state[site] = DB
            current_idx += 1
        end

        for _ in 1:n_u
            site = all_sites[current_idx]
            ss.state[site] = UP
            current_idx += 1
        end

        for _ in 1:n_dwn
            site = all_sites[current_idx]
            ss.state[site] = DN
            current_idx += 1
        end

    else
        shuffle!(all_sites)
        for i in 1:ss.N_up
            site = all_sites[i]
            ss.state[site] = UP
        end

        shuffle!(all_sites)
        for i in 1:ss.N_dn
            site = all_sites[i]
            ss.state[site] |= DN
        end
    end

    for site in 1:ss.N_sites
        st = ss.state[site]
        if st == HOLE
            ss.count_holes = _add_to_list!(ss.list_holes, ss.count_holes, ss.site_ptr, site)
        elseif st == UP
            ss.count_ups = _add_to_list!(ss.list_ups, ss.count_ups, ss.site_ptr, site)
        elseif st == DN
            ss.count_dns = _add_to_list!(ss.list_dns, ss.count_dns, ss.site_ptr, site)
        else
            ss.count_dbs = _add_to_list!(ss.list_dbs, ss.count_dbs, ss.site_ptr, site)
        end
    end

    elec_id_counter = 0
    for site in 1:ss.N_sites
        st = ss.state[site]
        if has_up(st)
            elec_id_counter += 1
            idx_global = 2 * (site - 1) + 1
            ss.map_spin_to_id[idx_global] = elec_id_counter
            ss.electron_locs[elec_id_counter] = idx_global
        end
    end

    @assert elec_id_counter == ss.N_up "wrong: up number ($elec_id_counter) is not ($(ss.N_up))"

    for site in 1:ss.N_sites
        st = ss.state[site]
        if has_dn(st)
            elec_id_counter += 1
            idx_global = 2 * (site - 1) + 2 # Spin 2 (Down)
            ss.map_spin_to_id[idx_global] = elec_id_counter
            ss.electron_locs[elec_id_counter] = idx_global
        end
    end
    total_elec = total_elec(ss)
    @assert elec_id_counter == total_elec "total elec number ($elec_id_counter) is not ($total_elec)"

    return ss
end

function init_config_Heisenberg!(ss::Configuration)
    return init_config_Hubbard!(ss; n_doublon=0)
end

function print_config_debug(ss)
    println("=== Current State ===")
    for site in 1:ss.N_sites
        st = ss.state[site]
        up_idx = 2 * (site - 1) + 1
        dn_idx = 2 * (site - 1) + 2
        up_id = ss.map_spin_to_id[up_idx]
        dn_id = ss.map_spin_to_id[dn_idx]

        print("Site $site [$(get_state_char(st))]: ")
        if has_up(st)
            print("UP(elec=$(up_id)) ")
        else
            print("UP(---) ")
        end
        if has_dn(st)
            println("DN(elec=$(dn_id))")
        else
            println("DN(---)")
        end
    end

    # 检查映射一致性
    println("\n=== Mapping Check ===")
    for i in 1:length(ss.map_spin_to_id)
        if ss.map_spin_to_id[i] != 0
            site = div(i - 1, 2) + 1
            spin = ((i - 1) % 2) == 0 ? "UP" : "DN"
            println("Elec $(ss.map_spin_to_id[i]) mapped to Site $site ($spin)")
        end
    end

    # 列表计数
    println("\n=== List Counts ===")
    println("Holes: $(ss.count_holes), Ups: $(ss.count_ups), Dns: $(ss.count_dns), DBs: $(ss.count_dbs)")
end

function print_config(ss::Configuration; chunk_size::Int=20)
    N = ss.N_sites
    println("="^60)
    println("Configuration Summary:")
    println("  Sites: $N | Up: $(ss.N_up) | Dn: $(ss.N_dn)")
    println("  Counts -> H:$(ss.count_holes) | U:$(ss.count_ups) | D:$(ss.count_dns) | DB:$(ss.count_dbs)")
    println("-"^60)

    # 分块打印，防止 N 很大时换行混乱
    for i in 1:chunk_size:N
        range_end = min(i + chunk_size - 1, N)
        sites = i:range_end

        # 1. 打印格点索引 (占位符宽度设为 4)
        print("Site |")
        for site in sites
            print(lpad(site, 4))
        end
        println("|")

        # 2. 打印状态字符
        print("Stat |")
        for site in sites
            st = ss.state[site]
            char = get_state_char(st)

            # 根据状态设置颜色 (如果终端支持)
            # 简单起见这里只用字符区分，想要颜色可以用 Crayons.jl
            str_rep = string(char)

            print(lpad(str_rep, 4))
        end
        println("|")
        println("-"^60)
    end
end

function init_config_Hubbard_by_state_char(chars::Vector{Char})
    L = length(chars)

    n_up = 0
    n_dn = 0

    temp_states = zeros(Int8, L)

    for (i, c) in enumerate(chars)
        st = _parse_char_to_state(c)
        temp_states[i] = st

        if has_up(st)
            n_up += 1
        end
        if has_dn(st)
            n_dn += 1
        end
    end

    ss = config_Hubbard(L, n_up, n_dn)
    ss.state .= temp_states

    initialize_lists!(ss)

    fill!(ss.map_spin_to_id, 0)
    elec_id_counter = 0

    # Pass 1: UP electrons
    for site in 1:L
        if has_up(ss.state[site])
            elec_id_counter += 1
            idx_global = 2 * (site - 1) + 1
            ss.map_spin_to_id[idx_global] = elec_id_counter
            ss.electron_locs[elec_id_counter] = idx_global
        end
    end

    # Pass 2: DN electrons
    for site in 1:L
        if has_dn(ss.state[site])
            elec_id_counter += 1
            idx_global = 2 * (site - 1) + 2
            ss.map_spin_to_id[idx_global] = elec_id_counter
            ss.electron_locs[elec_id_counter] = idx_global
        end
    end

    @assert elec_id_counter == (n_up + n_dn) "ID assignment mismatch"

    return ss
end

init_config_Hubbard_by_state_char(s::String) = init_config_Hubbard_by_state_char(collect(s))

function init_config_Heisenberg_by_state_char(chars::Vector{Char})
    L = length(chars)

    n_up = 0
    temp_states = zeros(Int8, L)

    for (i, c) in enumerate(chars)
        st = _parse_char_to_state(c)

        if st == HOLE || st == DB
            error("Heisenberg config cannot contain Holes ('h') or Doublons ('D') at index $i")
        end

        temp_states[i] = st
        if st == UP
            n_up += 1
        end
    end

    # 此时 N_dn 会自动被计算为 L - n_up
    ss = config_Heisenberg(L, n_up)

    # 再次检查 n_dn 是否符合预期 (虽然逻辑上肯定符合，但为了双重保险)
    n_dn_actual = count(c -> _parse_char_to_state(c) == DN, chars)
    if ss.N_dn != n_dn_actual
        error("Heisenberg init mismatch: N_dn calculated $(ss.N_dn) vs parsed $n_dn_actual")
    end

    ss.state .= temp_states

    # 初始化链表 (虽然 Heisenberg 主要是 UP/DN 表，但保持通用性)
    initialize_lists!(ss)

    # 重建 map_spin_to_id
    fill!(ss.map_spin_to_id, 0)
    elec_id_counter = 0

    # Pass 1: UP
    for site in 1:L
        if ss.state[site] == UP
            elec_id_counter += 1
            ss.map_spin_to_id[2*(site-1)+1] = elec_id_counter
            ss.electron_locs[elec_id_counter] = idx_global
        end
    end

    # Pass 2: DN
    for site in 1:L
        if ss.state[site] == DN
            elec_id_counter += 1
            ss.map_spin_to_id[2*(site-1)+2] = elec_id_counter
            ss.electron_locs[elec_id_counter] = idx_global
        end
    end

    return ss
end

init_config_Heisenberg_by_state_char(s::String) = init_config_Heisenberg_by_state_char(collect(s))

# ==============================================================================
# ConfigurationPH
# ==============================================================================
function copy_config(cfg::ConfigurationPH)
    return ConfigurationPH(
        cfg.N_sites, cfg.N_up, cfg.N_dn,
        copy(cfg.state), copy(cfg.map_spin_to_id),
        copy(cfg.electron_locs),
        copy(cfg.list_holes), copy(cfg.list_ups), copy(cfg.list_dns), copy(cfg.list_dbs),
        cfg.count_holes, cfg.count_ups, cfg.count_dns, cfg.count_dbs,
        copy(cfg.site_ptr)
    )
end
function init_config_Hubbard!(ss::ConfigurationPH; n_doublon::Union{Int,Nothing}=nothing)
    fill!(ss.state, HOLE)
    fill!(ss.map_spin_to_id, 0)

    ss.count_holes = 0
    ss.count_ups = 0
    ss.count_dns = 0
    ss.count_dbs = 0
    fill!(ss.site_ptr, 0)

    all_sites = collect(1:ss.N_sites)

    if n_doublon !== nothing
        n_d = n_doublon
        n_u = ss.N_up - n_d
        n_dwn = ss.N_dn - n_d
        n_h = ss.N_sites - (n_d + n_u + n_dwn)

        if n_u < 0 || n_dwn < 0 || n_h < 0
            error("n_doublon: $n_doublon is too large and n_up/n_dn/n_h is negative")
        end

        shuffle!(all_sites)

        current_idx = 1
        for _ in 1:n_d
            site = all_sites[current_idx]
            ss.state[site] = DB
            current_idx += 1
        end

        for _ in 1:n_u
            site = all_sites[current_idx]
            ss.state[site] = UP
            current_idx += 1
        end

        for _ in 1:n_dwn
            site = all_sites[current_idx]
            ss.state[site] = DN
            current_idx += 1
        end

    else
        shuffle!(all_sites)
        for i in 1:ss.N_up
            site = all_sites[i]
            ss.state[site] = UP
        end

        shuffle!(all_sites)
        for i in 1:ss.N_dn
            site = all_sites[i]
            ss.state[site] |= DN
        end
    end

    for site in 1:ss.N_sites
        st = ss.state[site]
        if st == HOLE
            ss.count_holes = _add_to_list!(ss.list_holes, ss.count_holes, ss.site_ptr, site)
        elseif st == UP
            ss.count_ups = _add_to_list!(ss.list_ups, ss.count_ups, ss.site_ptr, site)
        elseif st == DN
            ss.count_dns = _add_to_list!(ss.list_dns, ss.count_dns, ss.site_ptr, site)
        else
            ss.count_dbs = _add_to_list!(ss.list_dbs, ss.count_dbs, ss.site_ptr, site)
        end
    end

    elec_id_counter = 0
    for site in 1:ss.N_sites
        st = ss.state[site]
        if has_up(st)
            elec_id_counter += 1
            idx_global = 2 * (site - 1) + 1
            ss.map_spin_to_id[idx_global] = elec_id_counter
            ss.electron_locs[elec_id_counter] = idx_global
        end
    end

    @assert elec_id_counter == ss.N_up "wrong: up number ($elec_id_counter) is not ($(ss.N_up))"

    for site in 1:ss.N_sites
        st = ss.state[site]
        if !has_dn(st)
            elec_id_counter += 1
            idx_global = 2 * (site - 1) + 2 # Spin 2 (Down)
            ss.map_spin_to_id[idx_global] = elec_id_counter
            ss.electron_locs[elec_id_counter] = idx_global
        end
    end
    @assert elec_id_counter == total_elec(ss) "total elec number ($elec_id_counter) is not ($(total_elec(ss)))"



    return ss
end

function init_config_Heisenberg!(ss::ConfigurationPH)
    return init_config_Hubbard!(ss; n_doublon=0)
end
# ==============================================================================
# Moves & Proposals
# ==============================================================================
struct MoveProposal
    site1::Int
    site2::Int

    old_state1::Int8
    old_state2::Int8

    new_state1::Int8
    new_state2::Int8

    moved_electron_id_1::Int
    source_map_idx_1::Int
    target_map_idx_1::Int # 2*site + spin
    # target_site_spin_idx_1::Int 

    moved_electron_id_2::Int
    source_map_idx_2::Int
    target_map_idx_2::Int
    # target_site_spin_idx_2::Int

    delta_doublon::Int
end

@inline function empty_proposal(s1, s2, st1, st2)
    MoveProposal(s1, s2, Int8(st1), Int8(st2), Int8(st1), Int8(st2), 0, 0, 0, 0, 0, 0, 0)
end

@inline function check_hop(st1::Int8, st2::Int8, spin::Int8)
    # can_hop logic: s1 has spin AND s2 does not have spin
    return ((st1 & spin) != 0) && ((st2 & spin) == 0)
end

@inline function check_exchange(st1::Int8, st2::Int8)
    # can_exchange logic: XOR sum is 3 (11 binary), covers (0,3), (3,0), (1,2), (2,1)
    return (st1 ⊻ st2) == 3
end

@inline function check_flip(st::Int8)
    # can_flip logic: UP(1) or DN(2)
    return (st == UP) || (st == DN)
end

@inline can_hop(ss, s1::Int, s2::Int, spin::Int8) = check_hop(ss.state[s1], ss.state[s2], spin)
@inline can_exchange(ss, s1::Int, s2::Int) = check_exchange(ss.state[s1], ss.state[s2])
@inline can_flip(ss, s1::Int) = check_flip(ss.state[s1])

function count_choices_from_states(st1::Int8, st2::Int8, conserve_sz::Bool, is_same_site::Bool)
    c = 0

    # 如果随机选到了同一个点 (s1 == s2)
    if is_same_site
        if !conserve_sz && check_flip(st1)
            # 只有 Flip s1 一种可能 (spin up->dn or dn->up)
            # 在 s1==s2 的情况下，通常只有这一个操作是有效的
            return 1
        else
            return 0
        end
    end

    # --- 1. Hop ---
    if check_hop(st1, st2, UP)
        c += 1
    end # s1->s2 UP
    if check_hop(st2, st1, UP)
        c += 1
    end # s2->s1 UP
    if check_hop(st1, st2, DN)
        c += 1
    end # s1->s2 DN
    if check_hop(st2, st1, DN)
        c += 1
    end # s2->s1 DN

    # --- 2. Exchange ---
    if check_exchange(st1, st2)
        c += 1
    end

    # --- 3. Non-conserving Sz ---
    if !conserve_sz
        # Flip
        if check_flip(st1)
            c += 1
        end
        if check_flip(st2)
            c += 1
        end

        # Flip-Hop (Explicit check)
        # s1 -> s2
        if (st1 & UP) != 0 && (st2 & DN) == 0
            c += 1
        end # s1 UP -> s2 DN
        if (st1 & DN) != 0 && (st2 & UP) == 0
            c += 1
        end # s1 DN -> s2 UP
        # s2 -> s1
        if (st2 & UP) != 0 && (st1 & DN) == 0
            c += 1
        end # s2 UP -> s1 DN
        if (st2 & DN) != 0 && (st1 & UP) == 0
            c += 1
        end # s2 DN -> s1 UP
    end

    return c
end

function count_choices_heisenberg_from_states(st1::Int8, st2::Int8, conserve_sz::Bool, is_same_site::Bool)
    if is_same_site
        if !conserve_sz && check_flip(st1)
            return 1
        else
            return 0
        end
    end

    c = 0
    if check_exchange(st1, st2)
        c += 1
    end

    if !conserve_sz
        if check_flip(st1)
            c += 1
        end
        if check_flip(st2)
            c += 1
        end
    end
    return c
end

@inline function build_single_hop(ss, s1::Int, s2::Int, spin_idx::Int8)
    st1 = ss.state[s1]
    st2 = ss.state[s2]

    # 1 for up and 2 for dn    
    has_spin_s1 = (st1 & spin_idx) != 0
    no_spin_s2 = (st2 & spin_idx) == 0

    if !(has_spin_s1 && no_spin_s2)
        return empty_proposal(s1, s2, st1, st2)
    end

    new_st1 = st1 ⊻ spin_idx
    new_st2 = st2 | spin_idx
    if (spin_idx == DN && ifPH(ss))
        src_idx = 2 * (s2 - 1) + spin_idx
        tgt_idx = 2 * (s1 - 1) + spin_idx
    else
        src_idx = 2 * (s1 - 1) + spin_idx
        tgt_idx = 2 * (s2 - 1) + spin_idx
    end
    elec_id = ss.map_spin_to_id[src_idx]
    d_change = (new_st1 == DB) + (new_st2 == DB) - (st1 == DB) - (st2 == DB)

    return MoveProposal(s1, s2, st1, st2, new_st1, new_st2,
        elec_id, src_idx, tgt_idx,
        0, 0, 0, d_change)
end

"""
exchange (UP, DN), (DN, UP), (HOLE, DB), (DB, HOLE)
"""
@inline function build_exchange(ss, s1::Int, s2::Int)
    st1 = ss.state[s1]
    st2 = ss.state[s2]

    # 快速检查：异或必须为 3 (11二进制)，覆盖了 1<->2 和 0<->3
    if (st1 ⊻ st2) != 3
        return empty_proposal(s1, s2, st1, st2)
    end

    new_st1 = st2
    new_st2 = st1

    # 临时变量用于填充 Proposal 的两个电子槽位
    e1_id, e1_src, e1_tgt = 0, 0, 0
    e2_id, e2_src, e2_tgt = 0, 0, 0
    slot = 1

    # 检查 UP 自旋的移动
    # UP = 1
    has_u1 = (st1 & UP) != 0
    has_u2 = (st2 & UP) != 0

    if has_u1 && !has_u2
        # UP 从 s1 -> s2
        src = 2 * (s1 - 1) + 1
        tgt = 2 * (s2 - 1) + 1
        id = ss.map_spin_to_id[src]
        if slot == 1
            e1_id, e1_src, e1_tgt = id, src, tgt
        else
            e2_id, e2_src, e2_tgt = id, src, tgt
        end
        slot += 1
    elseif !has_u1 && has_u2
        # UP 从 s2 -> s1
        src = 2 * (s2 - 1) + 1
        tgt = 2 * (s1 - 1) + 1
        id = ss.map_spin_to_id[src]
        if slot == 1
            e1_id, e1_src, e1_tgt = id, src, tgt
        else
            e2_id, e2_src, e2_tgt = id, src, tgt
        end
        slot += 1
    end

    # 检查 DN 自旋的移动
    # DN = 2
    has_d1 = (st1 & DN) != 0
    has_d2 = (st2 & DN) != 0

    if has_d1 && !has_d2
        # DN 从 s1 -> s2
        if ifPH(ss)
            src = 2 * (s2 - 1) + 2
            tgt = 2 * (s1 - 1) + 2
        else
            src = 2 * (s1 - 1) + 2
            tgt = 2 * (s2 - 1) + 2
        end
        id = ss.map_spin_to_id[src]
        if slot == 1
            e1_id, e1_src, e1_tgt = id, src, tgt
        else
            e2_id, e2_src, e2_tgt = id, src, tgt
        end
        slot += 1
    elseif !has_d1 && has_d2
        # DN 从 s2 -> s1
        if ifPH(ss)
            src = 2 * (s1 - 1) + 2
            tgt = 2 * (s2 - 1) + 2
        else
            src = 2 * (s2 - 1) + 2
            tgt = 2 * (s1 - 1) + 2
        end
        id = ss.map_spin_to_id[src]
        if slot == 1
            e1_id, e1_src, e1_tgt = id, src, tgt
        else
            e2_id, e2_src, e2_tgt = id, src, tgt
        end
        slot += 1
    end

    return MoveProposal(s1, s2, st1, st2, new_st1, new_st2,
        e1_id, e1_src, e1_tgt,
        e2_id, e2_src, e2_tgt, 0)
end

@inline function build_spin_flip(ss, s1::Int, current_spin::Int8)
    st1 = ss.state[s1]
    target_spin = Int8(3 - current_spin) # 1->2, 2->1

    if ((st1 & current_spin) == 0) || ((st1 & target_spin) != 0)
        return empty_proposal(s1, 0, st1, Int8(0))
    end

    new_st1 = (st1 ⊻ current_spin) | target_spin
    #这种情况下PH和nonPH会有很大区别，一个是rank2一个是rank1,只有pfaffian需要调用PH的情况才对
    if ifPH(ss)
        error("Wrong PH spin flip")
    else
        src_idx = 2 * (s1 - 1) + current_spin
        tgt_idx = 2 * (s1 - 1) + target_spin
        elec_id = ss.map_spin_to_id[src_idx]
        return MoveProposal(s1, s1, st1, Int8(0), new_st1, Int8(0), elec_id, src_idx, tgt_idx, 0, 0, 0, 0)
    end

    # return MoveProposal(s1, s1, st1, 0, new_st1, 0, elec_id, src_idx, tgt_idx, 0, 0, 0, 0)
end

@inline function build_double_spin_flip(ss, s1::Int, s2::Int, current_spin::Int8)
    st1 = ss.state[s1]
    st2 = ss.state[s2]

    if st1 != current_spin || st2 != current_spin
        return empty_proposal(s1, s2, st1, st2)
    end

    target_spin = Int8(3 - current_spin) # 1->2, 2->1

    new_st1 = target_spin
    new_st2 = target_spin

    # 查找电子 ID
    # current_spin 本身就是 offset (UP=1, DN=2)
    src_idx_1 = 2 * (s1 - 1) + current_spin
    tgt_idx_1 = 2 * (s1 - 1) + target_spin
    elec_id_1 = ss.map_spin_to_id[src_idx_1]

    src_idx_2 = 2 * (s2 - 1) + current_spin
    tgt_idx_2 = 2 * (s2 - 1) + target_spin
    elec_id_2 = ss.map_spin_to_id[src_idx_2]

    if elec_id_1 == 0 || elec_id_2 == 0
        return empty_proposal(s1, s2, st1, st2)
    end

    d_change = 0

    return MoveProposal(
        s1, s2,
        st1, st2,
        new_st1, new_st2,
        elec_id_1, src_idx_1, tgt_idx_1,
        elec_id_2, src_idx_2, tgt_idx_2,
        d_change
    )
end


@inline function build_spin_flip_hop(ss, s1::Int, s2::Int, src_spin::Int8)
    st1 = ss.state[s1]
    st2 = ss.state[s2]
    tgt_spin = Int8(3 - src_spin)

    if ((st1 & src_spin) == 0) || ((st2 & tgt_spin) != 0)
        return empty_proposal(s1, s2, st1, st2)
    end

    new_st1 = st1 ⊻ src_spin
    new_st2 = st2 | tgt_spin
    if ifPH(ss)
        error("Wrong PH spin flip")
    else
        src_idx = 2 * (s1 - 1) + src_spin
        tgt_idx = 2 * (s2 - 1) + tgt_spin
        elec_id = ss.map_spin_to_id[src_idx]

        d_change = (new_st1 == DB) + (new_st2 == DB) - (st1 == DB) - (st2 == DB)

        return MoveProposal(s1, s2, st1, st2, new_st1, new_st2, elec_id, src_idx, tgt_idx, 0, 0, 0, d_change)
    end
end

function propose_move_Hubbard!(ss, s1::Int, s2::Int, conserve_sz::Bool, rng::AbstractRNG)
    # 定义 Move 代号 (用于 dispatch)
    # 1: Hop UP s1->s2
    # 2: Hop UP s2->s1
    # 3: Hop DN s1->s2
    # 4: Hop DN s2->s1
    # 5: Exchange
    # 6: Flip s1
    # 7: Flip s2
    # 8: FlipHop s1->s2 (UP->DN)
    # 9: FlipHop s1->s2 (DN->UP)
    # 10: FlipHop s2->s1 (UP->DN)
    # 11: FlipHop s2->s1 (DN->UP)

    # 预分配一个小数组存放可行步的代号 (最大可能只有~12种情况)
    choices = Int[]
    sizehint!(choices, 12)

    # --- 1. Hop Check ---
    if can_hop(ss, s1, s2, UP)
        push!(choices, 1)
    end
    if can_hop(ss, s2, s1, UP)
        push!(choices, 2)
    end
    if can_hop(ss, s1, s2, DN)
        push!(choices, 3)
    end
    if can_hop(ss, s2, s1, DN)
        push!(choices, 4)
    end

    # --- 2. Exchange Check ---
    if can_exchange(ss, s1, s2)
        push!(choices, 5)
    end

    # --- 3. Non-conserving Sz Check ---
    if !conserve_sz
        # Flip
        if can_flip(ss, s1)
            push!(choices, 6)
        end
        if can_flip(ss, s2)
            push!(choices, 7)
        end

        # Flip-Hop (Strictly requires s1 != s2 to avoid redundancy with Flip)
        if s1 != s2
            st1, st2 = ss.state[s1], ss.state[s2]

            # s1 -> s2
            # UP -> DN (s1 has UP, s2 no DN)
            if (st1 & UP) != 0 && (st2 & DN) == 0
                push!(choices, 8)
            end
            # DN -> UP (s1 has DN, s2 no UP)
            if (st1 & DN) != 0 && (st2 & UP) == 0
                push!(choices, 9)
            end

            # s2 -> s1
            # UP -> DN (s2 has UP, s1 no DN)
            if (st2 & UP) != 0 && (st1 & DN) == 0
                push!(choices, 10)
            end
            # DN -> UP (s2 has DN, s1 no UP)
            if (st2 & DN) != 0 && (st1 & UP) == 0
                push!(choices, 11)
            end
        end
    end

    # 如果没有可行步，返回空 Proposal
    if isempty(choices)
        return MoveProposal(0, 0, Int8(0), Int8(0), Int8(0), Int8(0), 0, 0, 0, 0, 0, 0, 0)
        # return MoveProposal(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    end

    # --- 随机选择并构建 ---
    # 直接从可行列表中随机选取一个代号
    chosen_code = rand(rng, choices)


    if chosen_code == 1
        return build_single_hop(ss, s1, s2, UP)
    elseif chosen_code == 2
        return build_single_hop(ss, s2, s1, UP)
    elseif chosen_code == 3
        return build_single_hop(ss, s1, s2, DN)
    elseif chosen_code == 4
        return build_single_hop(ss, s2, s1, DN)
    elseif chosen_code == 5
        return build_exchange(ss, s1, s2)
    elseif chosen_code == 6
        spin = ((ss.state[s1] & UP) != 0) ? UP : DN
        return build_spin_flip(ss, s1, spin)
    elseif chosen_code == 7
        spin = ((ss.state[s2] & UP) != 0) ? UP : DN
        return build_spin_flip(ss, s2, spin)
    elseif chosen_code == 8
        return build_spin_flip_hop(ss, s1, s2, UP)
    elseif chosen_code == 9
        return build_spin_flip_hop(ss, s1, s2, DN)
    elseif chosen_code == 10
        return build_spin_flip_hop(ss, s2, s1, UP)
    elseif chosen_code == 11
        return build_spin_flip_hop(ss, s2, s1, DN)
    end

    # Should technically never reach here
    # return MoveProposal(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    return MoveProposal(0, 0, Int8(0), Int8(0), Int8(0), Int8(0), 0, 0, 0, 0, 0, 0, 0)
end

function propose_move_Heisenberg!(ss, s1::Int, s2::Int, conserve_sz::Bool, rng::AbstractRNG)
    @assert ss.count_holes == 0 && ss.count_dbs == 0 "Heisenberg model expects no holes/doublons"

    # 代号定义:
    # 1: Exchange
    # 2: Flip s1
    # 3: Flip s2

    choices = Int[]
    sizehint!(choices, 3)

    if can_exchange(ss, s1, s2)
        push!(choices, 1)
    end

    if !conserve_sz
        if can_flip(ss, s1)
            push!(choices, 2)
        end
        if can_flip(ss, s2)
            push!(choices, 3)
        end
    end

    if isempty(choices)
        # return MoveProposal(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        return MoveProposal(0, 0, Int8(0), Int8(0), Int8(0), Int8(0), 0, 0, 0, 0, 0, 0, 0)
    end

    chosen_code = rand(rng, choices)


    if chosen_code == 1
        return build_exchange(ss, s1, s2)
    elseif chosen_code == 2
        spin = ((ss.state[s1] & UP) != 0) ? UP : DN
        return build_spin_flip(ss, s1, spin)
    elseif chosen_code == 3
        spin = ((ss.state[s2] & UP) != 0) ? UP : DN
        return build_spin_flip(ss, s2, spin)
    end

    # return MoveProposal(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    return MoveProposal(0, 0, Int8(0), Int8(0), Int8(0), Int8(0), 0, 0, 0, 0, 0, 0, 0)
end

@inline function rand_from_union(rng, list1, n1, list2, n2)
    tot = n1 + n2
    if tot == 0
        return 0
    end # Should not happen in standard Hubbard
    r = rand(rng, 1:tot)
    if r <= n1
        return @inbounds list1[r]
    else
        return @inbounds list2[r-n1]
    end
end

"""
conserve_sz = true: only Hop and Exchange
conserve_sz = false: Hop, Exchange, Flip, FlipHop
"""
function count_choices_Hubbard(ss, s1::Int, s2::Int, conserve_sz::Bool)
    c = 0

    # hop s1->s2 Up, s2->s1 Up, s1->s2 Dn, s2->s1 Dn
    if can_hop(ss, s1, s2, UP)
        c += 1
    end
    if can_hop(ss, s2, s1, UP)
        c += 1
    end
    if can_hop(ss, s1, s2, DN)
        c += 1
    end
    if can_hop(ss, s2, s1, DN)
        c += 1
    end

    if can_exchange(ss, s1, s2)
        c += 1
    end

    if !conserve_sz
        if can_flip(ss, s1)
            c += 1
        end
        if can_flip(ss, s2)
            c += 1
        end

        # Flip-Hop         
        # s1 -> s2
        st1, st2 = ss.state[s1], ss.state[s2]
        if (st1 & UP) != 0 && (st2 & DN) == 0
            c += 1
        end # s1 Up -> s2 Dn
        if (st1 & DN) != 0 && (st2 & UP) == 0
            c += 1
        end # s1 Dn -> s2 Up
        # s2 -> s1
        if (st2 & UP) != 0 && (st1 & DN) == 0
            c += 1
        end # s2 Up -> s1 Dn
        if (st2 & DN) != 0 && (st1 & UP) == 0
            c += 1
        end # s2 Dn -> s1 Up
    end

    return c
end

function count_choices_Heisenberg(ss, s1::Int, s2::Int, conserve_sz::Bool)
    c = 0

    if can_exchange(ss, s1, s2)
        c += 1
    end

    if !conserve_sz
        if can_flip(ss, s1)
            c += 1
        end
        if can_flip(ss, s2)
            c += 1
        end
    end

    return c
end

function commit_move!(ss, p::MoveProposal)
    # 0. 安全检查：如果 proposal 无效，直接返回
    if p.site1 == 0
        return
    end

    # 1. 首先更新电子ID映射表
    old_id1 = p.moved_electron_id_1
    old_id2 = p.moved_electron_id_2
    old_src1 = p.source_map_idx_1
    old_src2 = p.source_map_idx_2
    new_tgt1 = p.target_map_idx_1
    new_tgt2 = p.target_map_idx_2

    # 清空所有旧位置
    if old_id1 != 0 && old_src1 > 0
        ss.map_spin_to_id[old_src1] = 0
    end
    if old_id2 != 0 && old_src2 > 0
        ss.map_spin_to_id[old_src2] = 0
    end

    # 设置新位置
    if old_id1 != 0 && new_tgt1 > 0
        ss.map_spin_to_id[new_tgt1] = old_id1
        ss.electron_locs[old_id1] = new_tgt1
    end
    if old_id2 != 0 && new_tgt2 > 0
        ss.map_spin_to_id[new_tgt2] = old_id2
        ss.electron_locs[old_id2] = new_tgt2
    end

    # 2. 记录移动前的状态（用于计算自旋数变化）
    old_st1 = ss.state[p.site1]
    old_st2 = p.site2 != 0 ? ss.state[p.site2] : Int8(0)

    # 3. 更新网格状态（这会自动更新链表）
    update_site_config!(ss, p.site1, p.new_state1)
    if p.site2 != 0 && p.site2 != p.site1
        update_site_config!(ss, p.site2, p.new_state2)
    end
end

# ==============================================================================
# MCMC kernel tags
# ==============================================================================
abstract type AbstractMCMCKernel end

struct HubbardKernel <: AbstractMCMCKernel
    conserve_sz::Bool
    HubbardKernel(; conserve_sz=true) = new(conserve_sz)
end

struct HeisenbergKernel <: AbstractMCMCKernel
    conserve_sz::Bool
    HeisenbergKernel(; conserve_sz=true) = new(conserve_sz)
end

function count_choices(kernel::HubbardKernel, cfg::Configuration, s1::Int, s2::Int)
    return count_choices_from_states(cfg.state[s1], cfg.state[s2], kernel.conserve_sz, s1 == s2)
end

function count_choices(kernel::HeisenbergKernel, cfg::Configuration, s1::Int, s2::Int)
    return count_choices_heisenberg_from_states(cfg.state[s1], cfg.state[s2], kernel.conserve_sz, s1 == s2)
end

function count_choices_reserve(kernel::AbstractMCMCKernel, cfg::Configuration, prop::MoveProposal, s1_orig::Int, s2_orig::Int)
    # -----------------------------------------------------------
    # s1_orig, s2_orig 是 MCMC 这一步随机选中的两个点。
    # prop 包含了这次移动导致的某些点的状态变化。
    # 我们必须根据 prop.site1 / prop.site2 到底等于 s1_orig 还是 s2_orig
    # -----------------------------------------------------------

    # 1. 确定 s1_orig 在移动后的状态
    nxt_st1 = cfg.state[s1_orig] # 默认保持原样
    if s1_orig == prop.site1
        nxt_st1 = prop.new_state1
    elseif s1_orig == prop.site2
        nxt_st1 = prop.new_state2
    end

    # 2. 确定 s2_orig 在移动后的状态
    # 注意：s2_orig 可能等于 s1_orig (虽然在 Hubbard 随机对中一般是不重复的，但为了通用性)
    nxt_st2 = cfg.state[s2_orig] # 默认保持原样
    if s2_orig == prop.site1
        nxt_st2 = prop.new_state1
    elseif s2_orig == prop.site2
        nxt_st2 = prop.new_state2
    end

    # 3. 计算基于新状态的选择数
    if isa(kernel, HubbardKernel)
        return count_choices_from_states(nxt_st1, nxt_st2, kernel.conserve_sz, s1_orig == s2_orig)
    else
        return count_choices_heisenberg_from_states(nxt_st1, nxt_st2, kernel.conserve_sz, s1_orig == s2_orig)
    end
end

@inline function rand_distinct_pair(rng::AbstractRNG, N::Int)
    # 1. 选第一个点
    s1 = rand(rng, 1:N)

    # 2. 选第二个点 (范围减 1)
    s2 = rand(rng, 1:N-1)

    # 3. 如果 s2 落在了 s1 及其之后，由于我们需要跳过 s1，所以+1
    if s2 >= s1
        s2 += 1
    end

    return s1, s2
end

function propose_move(kernel::HubbardKernel, cfg::Union{Configuration,ConfigurationPH}, rng::AbstractRNG)
    # 随机选择两个格点
    # s1, s2 = rand(rng, 1:cfg.N_sites, 2)
    s1, s2 = rand_distinct_pair(rng, cfg.N_sites)
    prop = propose_move_Hubbard!(cfg, s1, s2, kernel.conserve_sz, rng)
    return prop, s1, s2
end

function propose_move(kernel::HeisenbergKernel, cfg::Union{Configuration,ConfigurationPH}, rng::AbstractRNG)
    # 随机选择两个格点
    # s1, s2 = rand(rng, 1:cfg.N_sites, 2)
    s1, s2 = rand_distinct_pair(rng, cfg.N_sites)
    prop = propose_move_Heisenberg!(cfg, s1, s2, kernel.conserve_sz, rng)
    return prop, s1, s2
end

function init_config_rand!(ss, kernel::HeisenbergKernel)
    init_config_Heisenberg!(ss)
end

function init_config_rand!(ss, kernel::HubbardKernel; n_doublon::Union{Int,Nothing}=nothing)
    init_config_Hubbard!(ss; n_doublon)
end

function init_config_by_state_char(kernel::HeisenbergKernel, chars::Vector{Char})
    return init_config_Heisenberg_by_state_char(chars)
end

function init_config_by_state_char(kernel::HeisenbergKernel, s::String)
    return init_config_Heisenberg_by_state_char(s)
end

function init_config_by_state_char(kernel::HubbardKernel, chars::Vector{Char})
    return init_config_Hubbard_by_state_char(chars)
end

function init_config_by_state_char(kernel::HubbardKernel, s::String)
    return init_config_Hubbard_by_state_char(s)
end

kernel_name(kernel::HubbardKernel) = "Hubbard"
kernel_name(kernel::HeisenbergKernel) = "Heisenberg"


function check_consistency(ss)
    # 1. 检查状态和映射的一致性
    for site in 1:ss.N_sites
        st = ss.state[site]

        # 检查UP自旋
        if (st & UP) != 0
            idx = 2 * (site - 1) + 1
            if ss.map_spin_to_id[idx] == 0
                return false, "Site $site has UP, but map[$idx] is 0"
            end
        end

        # 检查DN自旋
        if (st & DN) != 0
            idx = 2 * (site - 1) + 2
            if ss.map_spin_to_id[idx] == 0
                return false, "Site $site has DN, but map[$idx] is 0"
            end
        end
    end

    # 2. 反向检查映射：确保每个非零映射对应的站点有正确的自旋
    for i in 1:length(ss.map_spin_to_id)
        elec_id = ss.map_spin_to_id[i]
        if elec_id != 0
            site = div(i - 1, 2) + 1
            spin_is_up = ((i - 1) % 2) == 0  # 0->UP, 1->DN

            st = ss.state[site]
            if spin_is_up
                if (st & UP) == 0
                    return false, "Map says Elec $elec_id at Site $site (UP), but grid is $(get_state_char(st))"
                end
            else
                if (st & DN) == 0
                    return false, "Map says Elec $elec_id at Site $site (DN), but grid is $(get_state_char(st))"
                end
            end
        end
    end

    # 3. 检查电子ID的唯一性（可选但推荐）
    id_set = Set{Int}()
    for i in 1:length(ss.map_spin_to_id)
        elec_id = ss.map_spin_to_id[i]
        if elec_id != 0
            if elec_id in id_set
                return false, "Duplicate electron ID: $elec_id"
            end
            push!(id_set, elec_id)
        end
    end

    # 4. 检查状态中的自旋数与N_up/N_dn的一致性
    state_up_count = 0
    state_dn_count = 0
    for site in 1:ss.N_sites
        st = ss.state[site]
        if (st & UP) != 0
            state_up_count += 1
        end
        if (st & DN) != 0
            state_dn_count += 1
        end
    end

    if state_up_count != ss.N_up
        return false, "UP count mismatch: state has $state_up_count, N_up is $(ss.N_up)"
    end
    if state_dn_count != ss.N_dn
        return false, "DN count mismatch: state has $state_dn_count, N_dn is $(ss.N_dn)"
    end

    # 5. 检查映射中的电子总数
    total_elec_in_map = count(x -> x != 0, ss.map_spin_to_id)
    total_elec_expected = ss.N_up + ss.N_dn

    if total_elec_in_map != total_elec_expected
        return false, "Total electron mismatch: map has $total_elec_in_map, expected $total_elec_expected"
    end

    return true, ""
end

function test_spin_flip_hop()
    println("="^60)
    println("TEST: Fixed Flip-Hop Operation")
    println("="^60)

    # 创建一个简单系统：4个站点，2个上电子，1个下电子
    ss = config_Hubbard(4, 2, 1)

    # 手动初始化状态
    fill!(ss.state, HOLE)
    ss.state[1] = UP
    ss.state[2] = HOLE
    ss.state[3] = DN
    ss.state[4] = UP

    # 手动设置电子ID映射
    fill!(ss.map_spin_to_id, 0)
    ss.map_spin_to_id[2*(1-1)+1] = 101  # 电子101: 站点1的UP自旋
    ss.map_spin_to_id[2*(3-1)+2] = 102  # 电子102: 站点3的DN自旋  
    ss.map_spin_to_id[2*(4-1)+1] = 103  # 电子103: 站点4的UP自旋

    # 初始化链表
    initialize_lists!(ss)

    println("\nInitial state:")
    println("N_up = $(ss.N_up), N_dn = $(ss.N_dn)")
    print_config_debug(ss)

    # 测试 Flip-Hop: UP at site1 -> DN at site2
    println("\n\nExecuting Flip-Hop: UP at site1 -> DN at site2")
    prop = build_spin_flip_hop(ss, 1, 2, UP)

    if prop.site1 != 0
        println("Proposal valid:")
        println("  Electron ID: $(prop.moved_electron_id_1)")
        println("  From: site $(prop.site1), state $(get_state_char(prop.old_state1))")
        println("  To:   site $(prop.site2), state $(get_state_char(prop.old_state2))")
        println("  New states: $(get_state_char(prop.new_state1)) @$(prop.site1), $(get_state_char(prop.new_state2)) @$(prop.site2)")

        # 执行
        commit_move!(ss, prop)

        println("\nAfter Flip-Hop:")
        println("N_up = $(ss.N_up), N_dn = $(ss.N_dn)")
        print_config_debug(ss)

        ok, msg = check_consistency(ss)
        println("\nConsistency check: $(ok ? "PASS" : "FAIL")")
        if !ok
            println("Error: $msg")
        else
            println("SUCCESS: Flip-Hop operation completed correctly!")

            # 验证计数
            expected_N_up = 1  # 原来2个UP，1个翻转为DN
            expected_N_dn = 2  # 原来1个DN，加上翻转的1个
            if ss.N_up == expected_N_up && ss.N_dn == expected_N_dn
                println("Counts correct: UP=$(ss.N_up), DN=$(ss.N_dn)")
            else
                println("Counts incorrect: expected UP=$expected_N_up, DN=$expected_N_dn")
            end
        end
    else
        println("Invalid proposal!")
    end

    return ss
end

function test_config_calculation()
    println("="^60)
    println("Testing State Calculation")
    println("="^60)

    # 测试 UP -> DN 翻转跳跃
    st1 = UP  # 01
    st2 = HOLE  # 00
    src_spin = UP  # 01
    tgt_spin = DN  # 10

    new_st1 = st1 ⊻ src_spin  # 01 ⊻ 01 = 00 (HOLE)
    new_st2 = st2 | tgt_spin  # 00 | 10 = 10 (DN)

    println("UP -> DN flip-hop:")
    println("  st1: $(get_state_char(st1)) ($(Int(st1))) -> new_st1: $(get_state_char(new_st1)) ($(Int(new_st1)))")
    println("  st2: $(get_state_char(st2)) ($(Int(st2))) -> new_st2: $(get_state_char(new_st2)) ($(Int(new_st2)))")
    println("  Expected: site1 becomes HOLE, site2 becomes DN")

    # 测试 DN -> UP 翻转跳跃
    st1 = DN  # 10
    st2 = HOLE  # 00
    src_spin = DN  # 10
    tgt_spin = UP  # 01

    new_st1 = st1 ⊻ src_spin  # 10 ⊻ 10 = 00 (HOLE)
    new_st2 = st2 | tgt_spin  # 00 | 01 = 01 (UP)

    println("\nDN -> UP flip-hop:")
    println("  st1: $(get_state_char(st1)) ($(Int(st1))) -> new_st1: $(get_state_char(new_st1)) ($(Int(new_st1)))")
    println("  st2: $(get_state_char(st2)) ($(Int(st2))) -> new_st2: $(get_state_char(new_st2)) ($(Int(new_st2)))")
    println("  Expected: site1 becomes HOLE, site2 becomes UP")

    # 测试从 DB 的翻转跳跃
    st1 = DB  # 11
    st2 = HOLE  # 00
    src_spin = UP  # 01
    tgt_spin = DN  # 10

    new_st1 = st1 ⊻ src_spin  # 11 ⊻ 01 = 10 (DN)
    new_st2 = st2 | tgt_spin  # 00 | 10 = 10 (DN)

    println("\nDB (UP) -> DN flip-hop:")
    println("  st1: $(get_state_char(st1)) ($(Int(st1))) -> new_st1: $(get_state_char(new_st1)) ($(Int(new_st1)))")
    println("  st2: $(get_state_char(st2)) ($(Int(st2))) -> new_st2: $(get_state_char(new_st2)) ($(Int(new_st2)))")
    println("  Expected: site1 becomes DN, site2 becomes DN")

    return true
end


# ==============================================================================
# MCMC kernel tags
# ==============================================================================
# mutable struct VMCRunner{H, W, K}
#     ham::H                 # Hamiltonian (Hubbard / Heisenberg / ...)
#     vwf::W                 # variational wavefunction (contains sampler)
#     kernel::K              # MCMC kernel (HubbardKernel / HeisenbergKernel)
#     conserve_sz::Bool      # whether Sz is conserved
# end




end # module

