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

@testset "PH backflow local ratio matches rebuild" begin
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
    vwf.backflow_debug_verify = true
    mfVMC.VMC.init_gswf!(vwf)

    proposal = build_single_hop(sampler, 1, 2, UP)
    fast_ratio = mfVMC.VMC.calc_backflow_ratio_local_update(vwf, proposal)
    rebuild_ratio = mfVMC.VMC.calc_ratio_rebuild(vwf, proposal)

    @test fast_ratio ≈ rebuild_ratio
end
