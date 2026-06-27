using Test
using LinearAlgebra

include(joinpath(@__DIR__, "..", "twist_Hubbard_PH.jl"))

@testset "twist Hubbard PH no-backflow setup" begin
    afm_setup = build_twist_ph_mean_field_parameter_setup(Dict(
        "ansatz" => "AFM",
        "tx" => 2.0,
        "ty" => 1.0,
        "t2" => -0.4,
        "etax" => 0.11,
        "etay" => -0.07,
        "Delta_AF" => 0.3,
    ))
    @test afm_setup.wf_param_names == [:chi1y, :chi2, :etax, :etay, :Delta_AF]
    @test afm_setup.wf_init_params == [0.5, -0.2, 0.11, -0.07, 0.3]
    @test afm_setup.stripe_wavevector == 0.0
    @test afm_setup.stripe_center_offset == 0.0

    stripe_setup = build_twist_ph_mean_field_parameter_setup(Dict(
        "ansatz" => "Stripe",
        "tx" => 1.0,
        "ty" => 0.8,
        "t2" => 0.2,
        "etax" => 0.13,
        "etay" => 0.17,
        "Delta_c" => 0.4,
        "Delta_s" => 0.5,
        "lambda" => 4,
        "stripe_center" => "bond",
    ))
    @test stripe_setup.wf_param_names == [:chi1y, :chi2, :etax, :etay, :Delta_c, :Delta_s]
    @test stripe_setup.wf_init_params == [0.8, 0.2, 0.13, 0.17, 0.4, 0.5]
    @test stripe_setup.stripe_wavevector ≈ π / 2
    @test stripe_setup.stripe_center_offset == 0.5
end

@testset "twist Hubbard PH stripe pairing modulation" begin
    lx = 6
    ly = 5
    wavevector = π / 2
    stripe_center_offset = 0.5
    etax = 2.0
    etay = 3.0
    params = TwistHubbardPHParams(
        lx=lx,
        ly=ly,
        bcx=1.0,
        bcy=1.0,
        chi_x=0.0,
        chi_y=0.0,
        chi2=0.0,
        etax=etax,
        etay=etay,
        delta_af=0.0,
        delta_c=0.0,
        delta_s=0.0,
        stripe_wavevector=wavevector,
        stripe_center_offset=stripe_center_offset,
    )
    hamiltonian = Matrix(build_twist_hubbard_ph_hamiltonian(params))

    x = 2
    y = 3
    site_i = twist_site_index(x, y, ly)
    site_x = twist_site_index(x + 1, y, ly)
    site_y = twist_site_index(x, y + 1, ly)
    row_up_i = 2 * site_i - 1
    row_dn_hole_x = 2 * site_x
    row_dn_hole_y = 2 * site_y

    expected_etax0 = etax * abs(cos(wavevector / 2 * (x + 0.5 - stripe_center_offset)))
    expected_etay0 = -etay * abs(cos(wavevector / 2 * (x - stripe_center_offset)))

    @test hamiltonian[row_up_i, row_dn_hole_x] ≈ expected_etax0
    @test hamiltonian[row_up_i, row_dn_hole_y] ≈ expected_etay0
    @test ishermitian(hamiltonian)
end
