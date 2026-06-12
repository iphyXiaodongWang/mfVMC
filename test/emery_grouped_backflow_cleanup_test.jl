using Test

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "Emery.jl"))
using mfVMC.Sampler

function has_method_signature_containing(function_object, type_name::String)::Bool
    return any(method -> occursin(type_name, sprint(show, method.sig)), methods(function_object))
end

"""
用途: 构造一个最小 `BackflowEpsilonTerm`, 用于测试 group_names 字段.

参数:
- 无.

返回:
- `mfVMC.BackflowEpsilonTerm`: 带 group_names 的 epsilon term.
"""
function build_test_epsilon_term()
    return mfVMC.Backflow.BackflowEpsilonTerm(
        param_name=:bf_epsilon_test,
        epsilon_bf=1.2,
        group_names=Symbol[:dd, :dp],
    )
end

@testset "Emery grouped backflow cleanup" begin
    @test !isdefined(mfVMC.Backflow, Symbol("fill_backflow_site_row_after_proposal_by_terms!"))
    @test !isdefined(mfVMC.Backflow, Symbol("fill_backflow_site_block_after_proposal_by_terms!"))

    # 验证已删除的旧 epsilon 专用函数不再存在.
    old_epsilon_helpers = [
        Symbol("add_backflow_correction_site_block_after_proposal!"),
        Symbol("add_backflow_correction_site_row_after_proposal!"),
        Symbol("compute_backflow_epsilon_row_mask"),
        Symbol("compute_backflow_epsilon_neighbor_data_signature"),
        Symbol("build_backflow_epsilon_neighbor_cache"),
        Symbol("add_backflow_correction_chain_rule_row!"),
        Symbol("add_backflow_correction_chain_rule_source_weights!"),
    ]
    for helper_name in old_epsilon_helpers
        @test !isdefined(mfVMC.Backflow, helper_name)
    end

    # 验证新的 eta-driven 辅助函数存在.
    new_helpers = [
        Symbol("add_source_group_eta_contributions_and_track_activation!"),
        Symbol("add_epsilon_contributions_from_active_groups!"),
        Symbol("add_source_group_chain_rule_source_weights_and_track!"),
    ]
    for helper_name in new_helpers
        @test isdefined(mfVMC.Backflow, helper_name)
    end

    dd_source_bonds,
    dd_source_amplitudes,
    dp_source_bonds,
    dp_source_amplitudes,
    pd_source_bonds,
    pd_source_amplitudes,
    pp_source_bonds,
    pp_source_amplitudes = build_emery_backflow_source_data_by_directed_orbital_type(
        2,
        2;
        tpd=1.0,
        tpp=0.2,
    )

    backflow = build_column_directed_emery_backflow(
        dd_source_bonds,
        dd_source_amplitudes,
        dp_source_bonds,
        dp_source_amplitudes,
        pd_source_bonds,
        pd_source_amplitudes,
        pp_source_bonds,
        pp_source_amplitudes,
        1.1,
        0.9,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
        1.3,
        1.4,
        1.5,
        1.6,
    )

    n_sites = emery_n_sites(2, 2)
    base_orbitals = reshape(collect(1.0:(8.0 * n_sites)), 2 * n_sites, 4)

    # 构造一个 site1 有 eta 贡献的状态:
    # site1=DB (3), site2=HOLE (0) -> eta1 通过 dd bond (1->2) 激活.
    state_vector = fill(Int8(0), n_sites)
    state_vector[1] = Int8(3)
    state_vector[2] = Int8(0)
    state_vector[3] = Int8(1)
    state_vector[4] = Int8(2)

    proposal = MoveProposal(
        1,
        2,
        Int8(3),
        Int8(0),
        Int8(1),
        Int8(2),
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )

    state_after = copy(state_vector)
    state_after[1] = Int8(1)
    state_after[2] = Int8(2)

    full_orbitals_after = mfVMC.build_backflow_orbitals(base_orbitals, state_after, backflow)
    site_block_buffer = zeros(Float64, 2, size(base_orbitals, 2))
    mfVMC.Backflow.fill_backflow_site_block_after_proposal!(
        site_block_buffer,
        base_orbitals,
        state_vector,
        backflow,
        proposal,
        1,
    )

    @test site_block_buffer == full_orbitals_after[1:2, :]
end

@testset "Eta-driven epsilon term structure" begin
    epsilon_term = build_test_epsilon_term()

    # BackflowEpsilonTerm 不再存储 site-neighbor mask 字段.
    @test !(:target_neighbors_by_source_site in fieldnames(typeof(epsilon_term)))
    @test !(:source_sites_by_target_neighbor in fieldnames(typeof(epsilon_term)))
    @test !(:source_sites in fieldnames(typeof(epsilon_term)))
    @test !(:neighbor_data_signature in fieldnames(typeof(epsilon_term)))
    @test !(:source_bonds in fieldnames(typeof(epsilon_term)))
    @test !(:source_amplitudes in fieldnames(typeof(epsilon_term)))

    # 验证新字段 group_names.
    @test epsilon_term.group_names == Symbol[:dd, :dp]
    @test epsilon_term.param_name == :bf_epsilon_test
    @test epsilon_term.epsilon_bf ≈ 1.2

    # 验证旧 keyword 构造函数 (source_bonds) 不再有效.
    @test_throws MethodError mfVMC.Backflow.BackflowEpsilonTerm(
        param_name=:bf_epsilon_test,
        epsilon_bf=1.2,
        source_bonds=Tuple{Int,Int}[(1, 2)],
    )
    @test_throws MethodError mfVMC.Backflow.BackflowEpsilonTerm(
        param_name=:bf_epsilon_test,
        epsilon_bf=1.2,
        source_bonds=Tuple{Int,Int}[(1, 2)],
        source_amplitudes=[1.0],
    )
end

@testset "Eta-driven epsilon activation" begin
    n_sites = 2
    n_orbitals = 2

    dd_bonds = Tuple{Int,Int}[(1, 2)]
    dd_amplitudes = [1.0]
    empty_bonds = Tuple{Int,Int}[]
    empty_amplitudes = Float64[]

    base_orbitals = ones(Float64, 2 * n_sites, n_orbitals)

    # Test 1: epsilon inactive when all eta parameters are zero.
    # state: site1=DB, site2=HOLE -> occupation allows hopping, but eta params are 0.
    bf_epsilon_d = 2.0
    backflow_zero_eta = build_column_directed_emery_backflow(
        dd_bonds, dd_amplitudes,
        empty_bonds, empty_amplitudes,
        empty_bonds, empty_amplitudes,
        empty_bonds, empty_amplitudes,
        bf_epsilon_d, 1.0,
        0.0, 0.0, 0.0, 0.0,  # dd: 全部为零
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    )

    state_db_hole = Int8[3, 0]
    backflow_orbitals = mfVMC.build_backflow_orbitals(base_orbitals, state_db_hole, backflow_zero_eta)
    @test backflow_orbitals ≈ base_orbitals  # 无 eta → 无 epsilon.

    # Test 2: epsilon active when eta contribution is nonzero.
    eta4_value = 0.5
    backflow_with_eta = build_column_directed_emery_backflow(
        dd_bonds, dd_amplitudes,
        empty_bonds, empty_amplitudes,
        empty_bonds, empty_amplitudes,
        empty_bonds, empty_amplitudes,
        bf_epsilon_d, 1.0,
        0.0, 0.0, 0.0, eta4_value,  # dd: 仅 eta4 非零
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    )

    # state: site1=UP, site2=HOLE -> eta4 (single-hole) 对于 UP row 激活.
    state_up_hole = Int8[1, 0]
    backflow_orbitals2 = mfVMC.build_backflow_orbitals(base_orbitals, state_up_hole, backflow_with_eta)
    # UP row (row 1): U_b = U_0 + eta4*t_ij*U_0(j) + (epsilon_d-1)*U_0(i)
    # = 1.0 + 0.5*1.0*1.0 + (2.0-1.0)*1.0 = 2.5
    @test backflow_orbitals2[1, 1] ≈ 2.5
    # DN row (row 2): n_dn=0, 无 eta 也无 epsilon.
    @test backflow_orbitals2[2, 1] ≈ 1.0

    # Test 3: epsilon inactive when eta parameter nonzero but bond amplitude is zero.
    zero_amp_bonds = Tuple{Int,Int}[(1, 2)]
    zero_amps = [0.0]
    backflow_zero_amp = build_column_directed_emery_backflow(
        zero_amp_bonds, zero_amps,
        empty_bonds, empty_amplitudes,
        empty_bonds, empty_amplitudes,
        empty_bonds, empty_amplitudes,
        bf_epsilon_d, 1.0,
        0.0, 0.0, 0.0, eta4_value,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    )
    backflow_orbitals3 = mfVMC.build_backflow_orbitals(base_orbitals, state_up_hole, backflow_zero_amp)
    @test backflow_orbitals3 ≈ base_orbitals

    # Test 4: epsilon sharing — bf_epsilon_d 响应 :dd 和 :dp, 但不响应 :pd 或 :pp.
    # 仅使用 dd group (包含 bf_epsilon_d).
    backflow_epsilon_d_active = build_column_directed_emery_backflow(
        dd_bonds, dd_amplitudes,
        empty_bonds, empty_amplitudes,  # dp: 空
        empty_bonds, empty_amplitudes,  # pd: 空
        empty_bonds, empty_amplitudes,  # pp: 空
        bf_epsilon_d, 1.0,
        0.0, 0.0, 0.0, eta4_value,  # dd: eta4 激活
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    )
    backflow_eps_d = mfVMC.build_backflow_orbitals(base_orbitals, state_up_hole, backflow_epsilon_d_active)
    @test backflow_eps_d[1, 1] ≈ 2.5  # epsilon_d 通过 dd 组激活

    # Test 5: bf_epsilon_p 不响应 :dd 或 :dp (group_names=[:pd, :pp]).
    backflow_epsilon_p_only = build_column_directed_emery_backflow(
        dd_bonds, dd_amplitudes,
        empty_bonds, empty_amplitudes,
        empty_bonds, empty_amplitudes,
        empty_bonds, empty_amplitudes,
        1.0, bf_epsilon_d,              # epsilon_d=1.0 (无效果), epsilon_p=2.0
        0.0, 0.0, 0.0, eta4_value,     # dd: eta4 激活, 但 epsilon_p 不应响应
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    )
    backflow_eps_p = mfVMC.build_backflow_orbitals(base_orbitals, state_up_hole, backflow_epsilon_p_only)
    # 仅 eta4 贡献 (无 epsilon), 因为 epsilon_p 只响应 :pd/:pp.
    # eta4 coefficient = 1.0 * (0.0*...) = 0.0... wait let me re-check.
    # bond_amplitude = 1.0, eta4_factor = n_up * h_dn * H_j = 1*1*1 = 1, eta4_value = 0.5
    # coefficient = 1.0 * 0.5 * 1.0 = 0.5
    # U_b(1,:) = U_0(1,:) + 0.5*U_0(3,:) = 1.0 + 0.5 = 1.5
    @test backflow_eps_p[1, 1] ≈ 1.5  # epsilon_p 不应激活 (dd 不在 [:pd,:pp] 中)
end

@testset "Eta-driven epsilon derivative activation" begin
    n_sites = 2
    n_orbitals = 2
    dd_bonds = Tuple{Int,Int}[(1, 2)]
    dd_amplitudes = [1.0]
    empty_bonds = Tuple{Int,Int}[]
    empty_amplitudes = Float64[]
    base_orbitals = ones(Float64, 2 * n_sites, n_orbitals)
    state_up_hole = Int8[1, 0]

    backflow_zero_eta = build_column_directed_emery_backflow(
        dd_bonds, dd_amplitudes,
        empty_bonds, empty_amplitudes,
        empty_bonds, empty_amplitudes,
        empty_bonds, empty_amplitudes,
        2.0, 1.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    )
    zero_eta_derivatives = mfVMC.Backflow.build_backflow_derivative_orbitals(
        base_orbitals,
        state_up_hole,
        backflow_zero_eta,
    )
    epsilon_d_index = findfirst(pair -> first(pair) == :bf_epsilon_d, zero_eta_derivatives)
    eta4_dd_index = findfirst(pair -> first(pair) == :bf_eta4_dd, zero_eta_derivatives)
    @test epsilon_d_index !== nothing
    @test eta4_dd_index !== nothing
    @test zero_eta_derivatives[epsilon_d_index].second == zeros(Float64, size(base_orbitals))
    @test zero_eta_derivatives[eta4_dd_index].second[1, :] == base_orbitals[3, :]

    backflow_active_eta = build_column_directed_emery_backflow(
        dd_bonds, dd_amplitudes,
        empty_bonds, empty_amplitudes,
        empty_bonds, empty_amplitudes,
        empty_bonds, empty_amplitudes,
        2.0, 1.0,
        0.0, 0.0, 0.0, 0.5,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    )
    active_eta_derivatives = mfVMC.Backflow.build_backflow_derivative_orbitals(
        base_orbitals,
        state_up_hole,
        backflow_active_eta,
    )
    epsilon_d_active_index = findfirst(pair -> first(pair) == :bf_epsilon_d, active_eta_derivatives)
    @test active_eta_derivatives[epsilon_d_active_index].second[1, :] == base_orbitals[1, :]
    @test active_eta_derivatives[epsilon_d_active_index].second[2:end, :] == zeros(Float64, 3, n_orbitals)

    zero_amp_backflow = build_column_directed_emery_backflow(
        dd_bonds, [0.0],
        empty_bonds, empty_amplitudes,
        empty_bonds, empty_amplitudes,
        empty_bonds, empty_amplitudes,
        2.0, 1.0,
        0.0, 0.0, 0.0, 0.5,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    )
    zero_amp_derivatives = mfVMC.Backflow.build_backflow_derivative_orbitals(
        base_orbitals,
        state_up_hole,
        zero_amp_backflow,
    )
    epsilon_d_zero_amp_index = findfirst(pair -> first(pair) == :bf_epsilon_d, zero_amp_derivatives)
    @test zero_amp_derivatives[epsilon_d_zero_amp_index].second == zeros(Float64, size(base_orbitals))
end

@testset "Proposal row-block consistency" begin
    n_sites = 2
    n_orbitals = 2
    dd_bonds = Tuple{Int,Int}[(1, 2)]
    dd_amplitudes = [1.0]
    empty_bonds = Tuple{Int,Int}[]
    empty_amplitudes = Float64[]

    # 构造 eta 和 epsilon 同时激活的配置.
    backflow = build_column_directed_emery_backflow(
        dd_bonds, dd_amplitudes,
        empty_bonds, empty_amplitudes,
        empty_bonds, empty_amplitudes,
        empty_bonds, empty_amplitudes,
        1.5, 1.0,                       # epsilon_d=1.5, epsilon_p=1.0 (无效果)
        0.0, 0.0, 0.0, 0.5,            # dd: eta4=0.5
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    )

    base_orbitals = ones(Float64, 2 * n_sites, n_orbitals)
    state_vector = Int8[1, 0]  # site1=UP, site2=HOLE

    # proposal: site1 UP->UP, site2 HOLE->DN.
    # proposal 后 state_after: site1=UP, site2=DN.
    # For UP row at site 1: n_up=1, site2 is DN (not HOLE), so eta4 factor = n_up*h_dn*H_j = 1*1*0 = 0.
    # 也就是说 proposal 后 eta 不激活, epsilon 也不激活.
    # 那么全重建应该等于 base.
    proposal_on = MoveProposal(
        1, 2,
        Int8(1), Int8(1),    # site1: current=UP, new=UP (不变)
        Int8(0), Int8(2),    # site2: current=HOLE, new=DN
        0, 0, 0, 0, 0, 0, 0,
    )

    state_after = copy(state_vector)
    state_after[1] = proposal_on.new_state1
    state_after[2] = proposal_on.new_state2

    full_orbitals_after = mfVMC.build_backflow_orbitals(base_orbitals, state_after, backflow)
    site_block_buffer = zeros(Float64, 2, size(base_orbitals, 2))
    mfVMC.Backflow.fill_backflow_site_block_after_proposal!(
        site_block_buffer, base_orbitals, state_vector, backflow, proposal_on, 1,
    )
    @test site_block_buffer[1, :] == full_orbitals_after[1, :]
    @test site_block_buffer[2, :] == full_orbitals_after[2, :]

    # 另一个 proposal: site1 UP->UP, site2 HOLE->HOLE (不变),
    # 此时 eta 应激活 (eta4: UP + HOLE)
    proposal_eta = MoveProposal(
        1, 2,
        Int8(1), Int8(1),    # site1: UP -> UP (不变)
        Int8(0), Int8(0),    # site2: HOLE -> HOLE (不变)
        0, 0, 0, 0, 0, 0, 0,
    )

    state_after_eta = copy(state_vector)
    state_after_eta[1] = proposal_eta.new_state1
    state_after_eta[2] = proposal_eta.new_state2

    full_orbitals_after_eta = mfVMC.build_backflow_orbitals(base_orbitals, state_after_eta, backflow)
    site_block_buffer_eta = zeros(Float64, 2, size(base_orbitals, 2))
    mfVMC.Backflow.fill_backflow_site_block_after_proposal!(
        site_block_buffer_eta, base_orbitals, state_vector, backflow, proposal_eta, 1,
    )
    @test site_block_buffer_eta[1, :] == full_orbitals_after_eta[1, :]
    @test site_block_buffer_eta[2, :] == full_orbitals_after_eta[2, :]
end
