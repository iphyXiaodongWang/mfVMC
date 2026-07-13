using Test

include(joinpath(@__DIR__, "..", "twist_Hubbard.jl"))

struct StrongCouplingTestSampler
    state::Vector{Int8}
end

struct StrongCouplingTestWaveFunction
    sampler::StrongCouplingTestSampler
end

@testset "twist Hubbard strong-coupling observables expose local estimators" begin
    observables = definition_twist_observables(
        2,
        2;
        strong_coupling_bonds=build_twist_nearest_neighbor_bonds(2, 2),
        tx=1.0,
        ty=0.8,
        t2=-0.2,
        onsite_u=8.0,
    )

    expected_keys = [
        :E_pert_t_local_proj,
        :E_pert_t_local_proj_x,
        :E_pert_t_local_proj_y,
        :E_pert_t_local_proj_t2,
        :E_pert_J_local_proj,
        :E_pert_J_local_proj_x,
        :E_pert_J_local_proj_y,
        :E_pert_J_local_proj_t2,
    ]

    @test all(key -> haskey(observables, key), expected_keys)
    @test !haskey(observables, :P_no_doublon)
    @test !haskey(observables, :E_pert_t_proj_num)
    @test !haskey(observables, :E_pert_J_proj_num)
end

@testset "twist Hubbard local projector only checks requested bond endpoints" begin
    wavefunction = StrongCouplingTestWaveFunction(
        StrongCouplingTestSampler(Int8[DB, HOLE, UP]),
    )

    @test measure_twist_local_no_doublon_indicator(wavefunction, 2, 3) == 1.0
    @test measure_twist_local_no_doublon_indicator(wavefunction, 1, 2) == 0.0
end
