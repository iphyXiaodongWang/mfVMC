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
