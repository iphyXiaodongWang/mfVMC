using Test

include(joinpath(@__DIR__, "..", "twist_Hubbard.jl"))

struct StrongCouplingTestSampler
    state::Vector{Int8}
end

struct StrongCouplingTestWaveFunction
    sampler::StrongCouplingTestSampler
end

@testset "twist Hubbard strong-coupling observables expose numerator and denominator" begin
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
        :P_no_doublon,
        :E_pert_t_proj_num,
        :E_pert_t_proj_x_num,
        :E_pert_t_proj_y_num,
        :E_pert_t_proj_t2_num,
        :E_pert_J_proj_num,
        :E_pert_J_proj_x_num,
        :E_pert_J_proj_y_num,
        :E_pert_J_proj_t2_num,
    ]

    @test all(key -> haskey(observables, key), expected_keys)
end

@testset "twist Hubbard strong-coupling numerator is zero outside no-doublon sector" begin
    wavefunction = StrongCouplingTestWaveFunction(
        StrongCouplingTestSampler(Int8[DB, HOLE]),
    )

    @test measure_twist_no_doublon_indicator(wavefunction) == 0.0
    @test measure_twist_projected_hopping_energy_sum([(1, 2)], 1.0, wavefunction) == 0.0
    @test measure_twist_projected_exchange_energy_sum([(1, 2)], 1.0, 8.0, wavefunction) == 0.0
end
