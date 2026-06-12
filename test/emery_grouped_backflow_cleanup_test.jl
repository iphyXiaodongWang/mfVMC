using Test

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "Emery.jl"))
using mfVMC.Sampler

function has_method_signature_containing(function_object, type_name::String)::Bool
    return any(method -> occursin(type_name, sprint(show, method.sig)), methods(function_object))
end

@testset "Emery grouped backflow cleanup" begin
    @test !isdefined(mfVMC.Backflow, Symbol("fill_backflow_site_row_after_proposal_by_terms!"))
    @test !isdefined(mfVMC.Backflow, Symbol("fill_backflow_site_block_after_proposal_by_terms!"))

    eta_type_names = [
        "BackflowEta1DoublonHoleTerm",
        "BackflowEta2SpinExchangeTerm",
        "BackflowEta3DoublonSingleTerm",
        "BackflowEta4SingleHoleTerm",
    ]
    source_dependent_helper_names = [
        Symbol("add_backflow_correction_site_block_after_proposal!"),
        Symbol("add_backflow_correction_site_row_after_proposal!"),
        Symbol("add_backflow_correction_orbitals!"),
        Symbol("add_backflow_correction_chain_rule_row!"),
        Symbol("add_backflow_correction_chain_rule_source_weights!"),
        Symbol("add_backflow_correction_derivative_orbitals!"),
    ]
    for helper_name in source_dependent_helper_names
        helper_function = getproperty(mfVMC.Backflow, helper_name)
        for eta_type_name in eta_type_names
            @test !has_method_signature_containing(helper_function, eta_type_name)
        end
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
