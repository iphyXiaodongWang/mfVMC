using Test
using LinearAlgebra
using Random

include(joinpath(@__DIR__, "..", "twist_Hubbard_PH.jl"))

function parse_with_temporary_args(parse_func, temporary_args::Vector{String})
    saved_args = copy(ARGS)
    empty!(ARGS)
    append!(ARGS, temporary_args)
    try
        return parse_func()
    finally
        empty!(ARGS)
        append!(ARGS, saved_args)
    end
end

@testset "twist Hubbard timing flag defaults off and can be enabled" begin
    nonph_default_args = parse_with_temporary_args(parse_twist_commandline, String[])
    ph_default_args = parse_with_temporary_args(parse_twist_ph_commandline, String[])
    nonph_enabled_args = parse_with_temporary_args(parse_twist_commandline, ["--enable_timing", "true"])
    ph_enabled_args = parse_with_temporary_args(parse_twist_ph_commandline, ["--enable_timing", "true"])

    @test nonph_default_args["enable_timing"] == "false"
    @test ph_default_args["enable_timing"] == "false"
    @test parse_twist_bool_flag(nonph_enabled_args["enable_timing"], "--enable_timing")
    @test parse_twist_bool_flag(ph_enabled_args["enable_timing"], "--enable_timing")
end

"""
用途: 构造最近邻 `V` observable 测试使用的小尺寸 non-PH determinant 波函数。

参数:
- `lx, ly::Int`: 二维晶格尺寸。

返回:
- `vwf_det`: 已初始化 mean-field 轨道和 identity projector 的测试波函数。
"""
function build_twist_v_test_wavefunction(lx::Int, ly::Int)
    n_sites = lx * ly
    nelec = n_sites
    nup = nelec ÷ 2
    ndn = nelec - nup
    Random.seed!(20260716)
    sampler = config_Hubbard(n_sites, nup, ndn; ifPH=false)
    init_config_Hubbard!(sampler)
    projector = build_twist_projector(lx, ly, 0.0; jastrow_dx_max=0, jastrow_dy_max=0)
    projector_names = projector_param_names(projector)
    vwf = vwf_det(zeros(Float64, 2 * n_sites, nelec), sampler)
    set_projector!(vwf, projector)
    update_twist_ansatz!(
        vwf,
        vcat([:chi1y, :chi2, :Delta_AF], projector_names),
        vcat([0.8, -0.2, 0.0], projector_param_values(projector)),
        lx,
        ly,
        1.0,
        1.0,
        1.0,
        0.8,
        nelec;
        nparams_proj=length(projector_names),
    )
    return vwf
end

@testset "twist Hubbard nearest-neighbor V command-line parameter" begin
    default_args = parse_with_temporary_args(parse_twist_commandline, String[])
    explicit_args = parse_with_temporary_args(parse_twist_commandline, ["--V", "1.5"])

    @test default_args["V"] == 0.0
    @test explicit_args["V"] == 1.5
end

@testset "twist Hubbard nearest-neighbor V Hamiltonian terms" begin
    lx = 4
    ly = 3
    onsite_u = 8.0
    nearest_neighbor_v = 1.25
    n_sites = lx * ly

    zero_v_setup = build_twist_hamiltonian_terms(lx, ly, 1.0, 0.8, -0.2, onsite_u)
    finite_v_setup = build_twist_hamiltonian_terms(
        lx,
        ly,
        1.0,
        0.8,
        -0.2,
        onsite_u;
        nearest_neighbor_v=nearest_neighbor_v,
    )
    bonds = build_twist_nearest_neighbor_bonds(lx, ly)
    expected_nearest_neighbor_bonds = Set(vcat(bonds.x_bonds, bonds.y_bonds))
    diagonal_bonds = Set(bonds.diagonal_bonds)
    actual_v_bonds = Set(Tuple(term.sites) for term in finite_v_setup.nearest_neighbor_interaction_terms)

    @test isempty(zero_v_setup.nearest_neighbor_interaction_terms)
    @test zero_v_setup.interaction_terms == zero_v_setup.onsite_interaction_terms
    @test length(finite_v_setup.onsite_interaction_terms) == n_sites
    @test length(finite_v_setup.nearest_neighbor_interaction_terms) == 2 * n_sites
    @test length(finite_v_setup.interaction_terms) == 3 * n_sites
    @test all(term.ops == [:n, :n] for term in finite_v_setup.nearest_neighbor_interaction_terms)
    @test all(term.coef == nearest_neighbor_v for term in finite_v_setup.nearest_neighbor_interaction_terms)
    @test actual_v_bonds == expected_nearest_neighbor_bonds
    @test isempty(intersect(actual_v_bonds, diagonal_bonds))
end

@testset "twist Hubbard nearest-neighbor V interaction observables" begin
    lx = 4
    ly = 3
    onsite_u = 8.0
    nearest_neighbor_v = 1.25
    zero_v_setup = build_twist_hamiltonian_terms(lx, ly, 1.0, 0.8, -0.2, onsite_u)
    finite_v_setup = build_twist_hamiltonian_terms(
        lx,
        ly,
        1.0,
        0.8,
        -0.2,
        onsite_u;
        nearest_neighbor_v=nearest_neighbor_v,
    )
    zero_v_observables = definition_twist_observables(
        lx,
        ly;
        interaction_terms=zero_v_setup.interaction_terms,
        onsite_interaction_terms=zero_v_setup.onsite_interaction_terms,
        nearest_neighbor_interaction_terms=zero_v_setup.nearest_neighbor_interaction_terms,
        onsite_u=onsite_u,
    )
    finite_v_observables = definition_twist_observables(
        lx,
        ly;
        interaction_terms=finite_v_setup.interaction_terms,
        onsite_interaction_terms=finite_v_setup.onsite_interaction_terms,
        nearest_neighbor_interaction_terms=finite_v_setup.nearest_neighbor_interaction_terms,
        onsite_u=onsite_u,
    )

    @test !haskey(zero_v_observables, :E_int_U)
    @test !haskey(zero_v_observables, :E_int_V)
    @test haskey(finite_v_observables, :E_int_U)
    @test haskey(finite_v_observables, :E_int_V)

    vwf = build_twist_v_test_wavefunction(lx, ly)
    ham = GeneralModel(lx * ly, finite_v_setup.all_terms)
    site_occupations = [
        Float64((site_state & UP) != 0) + Float64((site_state & DN) != 0)
        for site_state in vwf.sampler.state
    ]
    bonds = build_twist_nearest_neighbor_bonds(lx, ly)
    expected_v_energy = nearest_neighbor_v * sum(
        site_occupations[site_i] * site_occupations[site_j]
        for (site_i, site_j) in vcat(bonds.x_bonds, bonds.y_bonds)
    )

    @test finite_v_observables[:E_int_V](ham, vwf) ≈ expected_v_energy
    @test finite_v_observables[:E_int_U](ham, vwf) ≈
          finite_v_observables[:E_int_charge](ham, vwf) +
          finite_v_observables[:E_int_spin](ham, vwf)
    @test finite_v_observables[:E_int](ham, vwf) ≈
          finite_v_observables[:E_int_U](ham, vwf) +
          finite_v_observables[:E_int_V](ham, vwf)
end

@testset "twist Hubbard V-dependent measurement history and main wiring" begin
    zero_v_history = build_twist_measure_history_observables(0.0)
    finite_v_history = build_twist_measure_history_observables(1.25)

    @test :E_int_U ∉ zero_v_history
    @test :E_int_V ∉ zero_v_history
    @test :E_int_U ∈ finite_v_history
    @test :E_int_V ∈ finite_v_history
    @test length(finite_v_history) == length(zero_v_history) + 2

    nonph_source = read(joinpath(@__DIR__, "..", "twist_Hubbard.jl"), String)
    @test occursin("nearest_neighbor_v = args[\"V\"]", nonph_source)
    @test occursin("nearest_neighbor_v=nearest_neighbor_v", nonph_source)
    @test occursin("onsite_interaction_terms=term_setup.onsite_interaction_terms", nonph_source)
    @test occursin(
        "nearest_neighbor_interaction_terms=term_setup.nearest_neighbor_interaction_terms",
        nonph_source,
    )
    @test occursin("history_observables=build_twist_measure_history_observables(nearest_neighbor_v)", nonph_source)
end

@testset "twist Hubbard PH no-backflow setup" begin
    afm_setup = build_twist_ph_mean_field_parameter_setup(Dict(
        "ansatz" => "AFM",
        "tx" => 2.0,
        "ty" => 1.0,
        "t2" => -0.4,
        "etax" => 0.11,
        "etay" => -0.07,
        "Delta_AF" => 0.3,
        "mu" => -1.2,
    ))
    @test afm_setup.wf_param_names == [:chi1y, :chi2, :etax, :etay, :Delta_AF, :mu]
    @test afm_setup.wf_init_params == [0.5, -0.2, 0.11, -0.07, 0.3, -1.2]
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
        "mu" => -0.9,
        "lambda" => 4,
        "stripe_center" => "bond",
    ))
    @test stripe_setup.wf_param_names == [:chi1y, :chi2, :etax, :etay, :Delta_c, :Delta_s, :mu]
    @test stripe_setup.wf_init_params == [0.8, 0.2, 0.13, 0.17, 0.4, 0.5, -0.9]
    @test stripe_setup.stripe_wavevector ≈ π / 2
    @test stripe_setup.stripe_center_offset == 0.5
end

@testset "twist Hubbard PH measurement includes directional hopping observables" begin
    ph_source = read(joinpath(@__DIR__, "..", "twist_Hubbard_PH.jl"), String)

    @test occursin("tx_hopping_terms=term_setup.tx_hopping_terms", ph_source)
    @test occursin("ty_hopping_terms=term_setup.ty_hopping_terms", ph_source)
    @test occursin("t2_hopping_terms=term_setup.t2_hopping_terms", ph_source)
    @test occursin(":E_hop_tx", ph_source)
    @test occursin(":E_hop_ty", ph_source)
    @test occursin(":E_hop_t2", ph_source)
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
        mu=0.0,
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

@testset "twist Hubbard PH chemical potential onsite field" begin
    lx = 4
    ly = 3
    wavevector = π / 2
    stripe_center_offset = 0.5
    mu = -1.3
    delta_c = 0.4
    params = TwistHubbardPHParams(
        lx=lx,
        ly=ly,
        bcx=1.0,
        bcy=1.0,
        chi_x=0.0,
        chi_y=0.0,
        chi2=0.0,
        etax=0.0,
        etay=0.0,
        mu=mu,
        delta_af=0.0,
        delta_c=delta_c,
        delta_s=0.0,
        stripe_wavevector=wavevector,
        stripe_center_offset=stripe_center_offset,
    )
    hamiltonian = Matrix(build_twist_hubbard_ph_hamiltonian(params))

    x = 2
    y = 1
    site_i = twist_site_index(x, y, ly)
    row_up = 2 * site_i - 1
    row_down_hole = 2 * site_i
    charge_field_x = mu + delta_c * cos(wavevector * (x - stripe_center_offset))

    @test hamiltonian[row_up, row_up] ≈ charge_field_x
    @test hamiltonian[row_down_hole, row_down_hole] ≈ -charge_field_x

    dh_dmu = build_twist_hubbard_ph_dh_dparam(params, :mu)
    @test dh_dmu[row_up, row_up] ≈ 1.0
    @test dh_dmu[row_down_hole, row_down_hole] ≈ -1.0
end
