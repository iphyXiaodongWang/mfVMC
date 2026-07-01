using Test

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using mfVMC
using mfVMC.Sampler
using mfVMC.Backflow

function build_test_backflow(; particle_hole_lower_block::Bool)
    source_bonds = [(1, 2), (2, 1)]
    source_amplitudes = [1.0, 1.0]
    group = mfVMC.Backflow.build_directed_backflow_source_group(
        :hubbard,
        source_bonds,
        source_amplitudes,
        BackflowEta1DoublonHoleTerm(param_name=:bf_eta1, eta1_bf=0.7),
        BackflowEta2SpinExchangeTerm(param_name=:bf_eta2, eta2_bf=0.0),
        BackflowEta3DoublonSingleTerm(param_name=:bf_eta3, eta3_bf=0.0),
        BackflowEta4SingleHoleTerm(param_name=:bf_eta4, eta4_bf=0.0),
    )
    return CompositeBackflowTerm(
        [BackflowEpsilonTerm(param_name=:bf_epsilon, epsilon_bf=1.0, group_names=Symbol[:hubbard])],
        [group];
        particle_hole_lower_block=particle_hole_lower_block,
    )
end

function build_split_ph_test_backflow(;
    up_eta2::Float64=0.0,
    up_eta3::Float64=0.0,
    up_eta4::Float64=0.0,
    dn_hole_eta2::Float64=0.0,
    dn_hole_eta3::Float64=0.0,
    dn_hole_eta4::Float64=0.0,
)
    source_bonds = [(1, 2), (2, 1)]
    source_amplitudes = [1.0, 1.0]
    upper_group = mfVMC.Backflow.build_directed_backflow_source_group(
        :hubbard,
        source_bonds,
        source_amplitudes,
        BackflowEta1DoublonHoleTerm(param_name=:bf_eta1_up, eta1_bf=0.0),
        BackflowEta2SpinExchangeTerm(param_name=:bf_eta2_up, eta2_bf=up_eta2),
        BackflowEta3DoublonSingleTerm(param_name=:bf_eta3_up, eta3_bf=up_eta3),
        BackflowEta4SingleHoleTerm(param_name=:bf_eta4_up, eta4_bf=up_eta4),
    )
    lower_group = mfVMC.Backflow.build_directed_backflow_source_group(
        :hubbard,
        source_bonds,
        source_amplitudes,
        BackflowEta1DoublonHoleTerm(param_name=:bf_eta1_dn_hole, eta1_bf=0.0),
        BackflowEta2SpinExchangeTerm(param_name=:bf_eta2_dn_hole, eta2_bf=dn_hole_eta2),
        BackflowEta3DoublonSingleTerm(param_name=:bf_eta3_dn_hole, eta3_bf=dn_hole_eta3),
        BackflowEta4SingleHoleTerm(param_name=:bf_eta4_dn_hole, eta4_bf=dn_hole_eta4),
    )
    return CompositeBackflowTerm(
        [BackflowEpsilonTerm(param_name=:bf_epsilon_up, epsilon_bf=1.0, group_names=Symbol[:hubbard])],
        [upper_group];
        particle_hole_lower_block=true,
        lower_epsilon_terms=[
            BackflowEpsilonTerm(
                param_name=:bf_epsilon_dn_hole,
                epsilon_bf=1.0,
                group_names=Symbol[:hubbard],
            ),
        ],
        lower_source_groups=[lower_group],
    )
end

function collect_source_weight_map(state_vector, backflow_term, row_index)
    source_rows = Vector{Int}(undef, 2 * length(state_vector))
    source_weights = Vector{Float64}(undef, 2 * length(state_vector))
    source_count = mfVMC.Backflow.fill_backflow_chain_rule_source_weights!(
        source_rows,
        source_weights,
        state_vector,
        backflow_term,
        row_index,
    )
    return Dict(source_rows[index] => source_weights[index] for index in 1:source_count)
end

"""
用途: 读取当前全局 timing 统计中某个 label 的调用次数。

参数:
- `label::String`: timing label 名称。

返回:
- `Int`: label 的调用次数; 若 label 尚未出现, 返回 `0`。
"""
function timing_label_call_count(label::String)::Int
    label_index = findfirst(==(label), mfVMC.Timing._global_timing.labels)
    if label_index === nothing
        return 0
    end
    return mfVMC.Timing._global_timing.call_counts[label_index]
end

"""
用途: 构造一个最小 PH backflow determinant 波函数和一个有效的 up-spin hop proposal。

参数:
- 无。内部固定使用 `2x2` Hubbard PH 构型 `DB, HOLE, UP, DN`。

返回:
- `Tuple`: `(vwf, proposal)`, 其中 `vwf` 是已初始化 inverse 的 determinant 波函数,
  `proposal` 是从 site 1 到 site 2 的 up-spin hop。
"""
function build_minimal_ph_backflow_vwf_and_hop_proposal()
    lx = 2
    ly = 2
    n_sites = lx * ly
    sampler = config_Hubbard(n_sites, 2, 2; ifPH=true)
    sampler.state .= Int8[DB, HOLE, UP, DN]
    initialize_lists!(sampler)
    fill!(sampler.map_spin_to_id, 0)
    fill!(sampler.electron_locs, 0)
    electron_id_counter = 0
    for site in 1:n_sites
        if has_up(sampler.state[site])
            electron_id_counter += 1
            row_index = 2 * (site - 1) + UP
            sampler.map_spin_to_id[row_index] = electron_id_counter
            sampler.electron_locs[electron_id_counter] = row_index
        end
    end
    for site in 1:n_sites
        if !has_dn(sampler.state[site])
            electron_id_counter += 1
            row_index = 2 * (site - 1) + DN
            sampler.map_spin_to_id[row_index] = electron_id_counter
            sampler.electron_locs[electron_id_counter] = row_index
        end
    end

    hopping_bonds = build_twist_nearest_neighbor_bonds(lx, ly)
    source_bonds, source_amplitudes = build_twist_backflow_source_data(hopping_bonds, 1.0, 1.0, 0.0)
    backflow = build_twist_optional_backflow(
        true,
        source_bonds,
        source_amplitudes,
        1.0,
        0.2,
        0.0,
        0.0,
        0.0;
        particle_hole_lower_block=true,
    )

    base_orbitals = [
        1.0 1.0 1.0 1.0
        1.0 2.0 4.0 8.0
        1.0 3.0 9.0 27.0
        1.0 4.0 16.0 64.0
        1.0 5.0 25.0 125.0
        1.0 6.0 36.0 216.0
        1.0 7.0 49.0 343.0
        1.0 8.0 64.0 512.0
    ]
    vwf = mfVMC.VMC.vwf_det(base_orbitals, sampler; backflow=backflow)
    mfVMC.VMC.init_gswf!(vwf)
    proposal = build_single_hop(sampler, 1, 2, UP)
    return vwf, proposal
end

@testset "PH backflow eta1 uses down-hole lower block" begin
    state_vector = Int8[HOLE, DB]
    ph_backflow = build_test_backflow(particle_hole_lower_block=true)

    lower_row_site1 = 2
    lower_row_site2 = 4
    weights = collect_source_weight_map(state_vector, ph_backflow, lower_row_site1)

    @test weights[lower_row_site1] ≈ 1.0
    @test weights[lower_row_site2] ≈ 0.7
end

@testset "nonPH backflow keeps down-electron eta1 direction" begin
    state_vector = Int8[DB, HOLE]
    nonph_backflow = build_test_backflow(particle_hole_lower_block=false)

    lower_row_site1 = 2
    lower_row_site2 = 4
    weights = collect_source_weight_map(state_vector, nonph_backflow, lower_row_site1)

    @test weights[lower_row_site1] ≈ 1.0
    @test weights[lower_row_site2] ≈ 0.7
end

@testset "PH backflow eta1 derivative uses down-hole direction" begin
    state_vector = Int8[HOLE, DB]
    ph_backflow = build_test_backflow(particle_hole_lower_block=true)
    base_orbitals = reshape(collect(1.0:16.0), 4, 4)

    derivative_pairs = mfVMC.Backflow.build_backflow_derivative_orbitals(
        base_orbitals,
        state_vector,
        ph_backflow,
    )
    derivative_map = Dict(first(pair) => last(pair) for pair in derivative_pairs)

    lower_row_site1 = 2
    lower_row_site2 = 4
    @test derivative_map[:bf_eta1][lower_row_site1, :] ≈ base_orbitals[lower_row_site2, :]
end

@testset "split PH backflow lower eta2 uses dn-hole parameter" begin
    state_vector = Int8[UP, DN]
    ph_backflow = build_split_ph_test_backflow(up_eta2=0.0, dn_hole_eta2=0.23)

    upper_row_site1 = 1
    lower_row_site1 = 2
    lower_row_site2 = 4
    upper_weights = collect_source_weight_map(state_vector, ph_backflow, upper_row_site1)
    lower_weights = collect_source_weight_map(state_vector, ph_backflow, lower_row_site1)

    @test length(upper_weights) == 1
    @test upper_weights[upper_row_site1] ≈ 1.0
    @test lower_weights[lower_row_site1] ≈ 1.0
    @test lower_weights[lower_row_site2] ≈ 0.23

    base_orbitals = reshape(collect(1.0:16.0), 4, 4)
    derivative_pairs = mfVMC.Backflow.build_backflow_derivative_orbitals(
        base_orbitals,
        state_vector,
        ph_backflow,
    )
    derivative_map = Dict(first(pair) => last(pair) for pair in derivative_pairs)

    @test derivative_map[:bf_eta2_up][lower_row_site1, :] ≈ zeros(4)
    @test derivative_map[:bf_eta2_dn_hole][lower_row_site1, :] ≈ base_orbitals[lower_row_site2, :]
end

@testset "split PH backflow lower eta3 uses dn-hole swapped-sites factor" begin
    state_vector = Int8[UP, DB]
    ph_backflow = build_split_ph_test_backflow(up_eta3=0.0, dn_hole_eta3=0.31)

    lower_row_site1 = 2
    lower_row_site2 = 4
    lower_weights = collect_source_weight_map(state_vector, ph_backflow, lower_row_site1)

    @test lower_weights[lower_row_site1] ≈ 1.0
    @test lower_weights[lower_row_site2] ≈ 0.31

    base_orbitals = reshape(collect(1.0:16.0), 4, 4)
    derivative_pairs = mfVMC.Backflow.build_backflow_derivative_orbitals(
        base_orbitals,
        state_vector,
        ph_backflow,
    )
    derivative_map = Dict(first(pair) => last(pair) for pair in derivative_pairs)

    @test derivative_map[:bf_eta3_up][lower_row_site1, :] ≈ zeros(4)
    @test derivative_map[:bf_eta3_dn_hole][lower_row_site1, :] ≈ base_orbitals[lower_row_site2, :]
end

@testset "split PH backflow lower eta4 uses dn-hole swapped-sites factor" begin
    state_vector = Int8[HOLE, DN]
    ph_backflow = build_split_ph_test_backflow(up_eta4=0.0, dn_hole_eta4=0.41)

    lower_row_site1 = 2
    lower_row_site2 = 4
    lower_weights = collect_source_weight_map(state_vector, ph_backflow, lower_row_site1)

    @test lower_weights[lower_row_site1] ≈ 1.0
    @test lower_weights[lower_row_site2] ≈ 0.41

    base_orbitals = reshape(collect(1.0:16.0), 4, 4)
    derivative_pairs = mfVMC.Backflow.build_backflow_derivative_orbitals(
        base_orbitals,
        state_vector,
        ph_backflow,
    )
    derivative_map = Dict(first(pair) => last(pair) for pair in derivative_pairs)

    @test derivative_map[:bf_eta4_up][lower_row_site1, :] ≈ zeros(4)
    @test derivative_map[:bf_eta4_dn_hole][lower_row_site1, :] ≈ base_orbitals[lower_row_site2, :]
end

include(joinpath(@__DIR__, "..", "twist_Hubbard_PH.jl"))

@testset "twist Hubbard PH constructs PH-mode backflow" begin
    hopping_bonds = build_twist_nearest_neighbor_bonds(2, 2)
    source_bonds, source_amplitudes = build_twist_backflow_source_data(hopping_bonds, 1.0, 1.0, 0.0)
    backflow = build_twist_optional_backflow(
        true,
        source_bonds,
        source_amplitudes,
        1.0,
        0.2,
        0.0,
        0.0,
        0.0;
        particle_hole_lower_block=true,
    )

    @test mfVMC.Backflow.uses_backflow(backflow)
    @test backflow.particle_hole_lower_block
    @test mfVMC.Backflow.backflow_param_names(backflow) == [:bf_epsilon, :bf_eta1, :bf_eta2, :bf_eta3, :bf_eta4]
end

@testset "twist Hubbard PH constructs split up and dn-hole backflow parameters" begin
    hopping_bonds = build_twist_nearest_neighbor_bonds(2, 2)
    source_bonds, source_amplitudes = build_twist_backflow_source_data(hopping_bonds, 1.0, 1.0, 0.0)
    backflow = build_twist_ph_split_backflow(
        source_bonds,
        source_amplitudes,
        1.0,
        0.2,
        0.0,
        0.0,
        0.0,
        1.0,
        0.3,
        0.4,
        0.5,
        0.6,
    )

    @test mfVMC.Backflow.backflow_param_names(backflow) == [
        :bf_epsilon_up,
        :bf_eta1_up,
        :bf_eta2_up,
        :bf_eta3_up,
        :bf_eta4_up,
        :bf_epsilon_dn_hole,
        :bf_eta1_dn_hole,
        :bf_eta2_dn_hole,
        :bf_eta3_dn_hole,
        :bf_eta4_dn_hole,
    ]
    @test mfVMC.Backflow.backflow_param_values(backflow) ≈ [1.0, 0.2, 0.0, 0.0, 0.0, 1.0, 0.3, 0.4, 0.5, 0.6]
end

@testset "PH backflow zero parameters degenerates to identity rows" begin
    hopping_bonds = build_twist_nearest_neighbor_bonds(2, 2)
    source_bonds, source_amplitudes = build_twist_backflow_source_data(hopping_bonds, 1.0, 1.0, 0.0)
    backflow = build_twist_optional_backflow(
        true,
        source_bonds,
        source_amplitudes,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0;
        particle_hole_lower_block=true,
    )
    state_vector = Int8[DB, HOLE, UP, DN]

    for row_index in 1:(2 * length(state_vector))
        weights = collect_source_weight_map(state_vector, backflow, row_index)
        @test length(weights) == 1
        @test weights[row_index] ≈ 1.0
    end
end

@testset "PH backflow ratio path uses ratio-only rank-k before accept" begin
    vwf, proposal = build_minimal_ph_backflow_vwf_and_hop_proposal()
    vwf.backflow_debug_verify = true
    previous_timing_flag = mfVMC.Timing.ENABLE_TIMING[]
    mfVMC.Timing.timing_reset!()
    mfVMC.Timing.ENABLE_TIMING[] = true
    try
        ratio = mfVMC.VMC.calc_backflow_ratio_local_update(vwf, proposal)

        @test timing_label_call_count("backflow_rankk_ratio_only") == 1
        @test timing_label_call_count("backflow_rankk_ratio_blas") == 1
        @test timing_label_call_count("backflow_rankk_factor") == 0

        mfVMC.VMC.accept_backflow_local_update!(vwf, proposal, ratio)

        @test timing_label_call_count("backflow_rankk_factor") == 1
    finally
        mfVMC.Timing.ENABLE_TIMING[] = previous_timing_flag
        mfVMC.Timing.timing_reset!()
    end
end

@testset "PH backflow local ratio matches rebuild" begin
    vwf, proposal = build_minimal_ph_backflow_vwf_and_hop_proposal()
    vwf.backflow_debug_verify = true

    fast_ratio = mfVMC.VMC.calc_backflow_ratio_local_update(vwf, proposal)
    rebuild_ratio = mfVMC.VMC.calc_ratio_rebuild(vwf, proposal)

    @test fast_ratio ≈ rebuild_ratio
end
