using Test

include(joinpath(@__DIR__, "..", "bench_twist_hubbard_ph_nonph.jl"))

@testset "twist Hubbard PH/nonPH U=0 benchmark" begin
    result = run_twist_hubbard_ph_nonph_u0_benchmark(
        lx=2,
        ly=2,
        tx=1.0,
        ty=1.0,
        t2=0.0,
        doping=0.0,
        target_sz=0,
    )

    @test result.nup == 2
    @test result.ndn == 2
    @test result.nonph_occupied_orbitals == 4
    @test result.ph_occupied_orbitals == 4
    @test result.nonph_energy ≈ result.exact_energy atol=1.0e-8
    @test result.ph_energy ≈ result.exact_energy atol=1.0e-8
    @test result.ph_energy ≈ result.nonph_energy atol=1.0e-8
end
