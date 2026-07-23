using Test
using LinearAlgebra
using Random

include(joinpath(@__DIR__, "..", "twist_Emery.jl"))

@testset "twist Emery PBC site mapping and coordinates" begin
    lx = 3
    ly = 2

    @test twist_emery_n_sites(lx, ly) == 18

    sites = [
        twist_emery_xyo_to_site_index(x, y, orbital, lx, ly)
        for x in 1:lx, y in 1:ly, orbital in (EMERY_ORB_D, EMERY_ORB_PY, EMERY_ORB_PX)
    ]
    @test sort(vec(sites)) == collect(1:18)
    @test twist_emery_xyo_to_site_index(lx + 1, 1, EMERY_ORB_D, lx, ly) ==
          twist_emery_xyo_to_site_index(1, 1, EMERY_ORB_D, lx, ly)
    @test twist_emery_xyo_to_site_index(1, ly + 1, EMERY_ORB_D, lx, ly) ==
          twist_emery_xyo_to_site_index(1, 1, EMERY_ORB_D, lx, ly)
    @test twist_emery_orbital_coordinate(2, 1, EMERY_ORB_D) == (2.0, 1.0)
    @test twist_emery_orbital_coordinate(2, 1, EMERY_ORB_PX) == (2.5, 1.0)
    @test twist_emery_orbital_coordinate(2, 1, EMERY_ORB_PY) == (2.0, 1.5)
end

"""
构造仅替换一个 mean-field 参数的测试参数对象.

参数:
- `params::TwistEmeryNonPHParams`: 原始参数.
- `param_name::Symbol`: 需要替换的参数名.
- `value::Real`: 新参数值.

返回:
- `TwistEmeryNonPHParams`: 替换指定参数后的新对象.
"""
function replace_twist_emery_test_param(
    params::TwistEmeryNonPHParams,
    param_name::Symbol,
    value::Real,
)
    parameter_values = Dict{Symbol, Any}(
        field_name => getfield(params, field_name)
        for field_name in fieldnames(TwistEmeryNonPHParams)
    )
    parameter_values[param_name] = Float64(value)
    return TwistEmeryNonPHParams(; parameter_values...)
end

@testset "twist Emery AFM and Stripe mean-field Hamiltonian" begin
    stripe_params = TwistEmeryNonPHParams(
        lx=4,
        ly=2,
        bcx=1.0,
        bcy=1.0,
        chi1_dd=0.0,
        chi1_dp_x=1.0,
        chi1_dp_y=0.8,
        chi1_pp=0.4,
        mu_px=2.0,
        mu_py=3.0,
        delta_af_d=0.0,
        delta_c_d=0.5,
        delta_c_px=0.6,
        delta_c_py=0.7,
        delta_s_d=0.9,
        stripe_wavevector=pi / 2,
        stripe_center_offset=0.0,
    )
    stripe_hamiltonian = Matrix(build_twist_emery_nonph_hamiltonian(stripe_params))

    d_site = twist_emery_xyo_to_site_index(1, 1, EMERY_ORB_D, 4, 2)
    px_site = twist_emery_xyo_to_site_index(1, 1, EMERY_ORB_PX, 4, 2)
    py_site = twist_emery_xyo_to_site_index(1, 1, EMERY_ORB_PY, 4, 2)
    expected_px_charge = 2.0 - 0.6 * cos((pi / 2) * 1.5)
    expected_py_charge = 3.0 - 0.7 * cos((pi / 2) * 1.0)
    expected_d_charge = -0.5 * cos((pi / 2) * 1.0)
    expected_d_spin = 0.9 * sin((pi / 4) * 1.0)

    @test stripe_hamiltonian[emery_spin_index(px_site, 1), emery_spin_index(px_site, 1)] ≈
          expected_px_charge
    @test stripe_hamiltonian[emery_spin_index(py_site, 1), emery_spin_index(py_site, 1)] ≈
          expected_py_charge
    @test stripe_hamiltonian[emery_spin_index(d_site, 1), emery_spin_index(d_site, 1)] ≈
          expected_d_charge + expected_d_spin
    @test stripe_hamiltonian[emery_spin_index(d_site, 2), emery_spin_index(d_site, 2)] ≈
          expected_d_charge - expected_d_spin
    @test stripe_hamiltonian[emery_spin_index(d_site, 1), emery_spin_index(px_site, 1)] ≈ -1.0
    @test stripe_hamiltonian[emery_spin_index(d_site, 1), emery_spin_index(py_site, 1)] ≈ 0.8

    afm_params = TwistEmeryNonPHParams(
        lx=2,
        ly=2,
        delta_af_d=0.4,
    )
    afm_hamiltonian = Matrix(build_twist_emery_nonph_hamiltonian(afm_params))
    afm_d_site = twist_emery_xyo_to_site_index(1, 1, EMERY_ORB_D, 2, 2)
    afm_px_site = twist_emery_xyo_to_site_index(1, 1, EMERY_ORB_PX, 2, 2)
    afm_py_site = twist_emery_xyo_to_site_index(1, 1, EMERY_ORB_PY, 2, 2)

    @test afm_hamiltonian[emery_spin_index(afm_d_site, 1), emery_spin_index(afm_d_site, 1)] ≈ 0.4
    @test afm_hamiltonian[emery_spin_index(afm_d_site, 2), emery_spin_index(afm_d_site, 2)] ≈ -0.4
    @test afm_hamiltonian[emery_spin_index(afm_px_site, 1), emery_spin_index(afm_px_site, 1)] ≈
          afm_hamiltonian[emery_spin_index(afm_px_site, 2), emery_spin_index(afm_px_site, 2)]
    @test afm_hamiltonian[emery_spin_index(afm_py_site, 1), emery_spin_index(afm_py_site, 1)] ≈
          afm_hamiltonian[emery_spin_index(afm_py_site, 2), emery_spin_index(afm_py_site, 2)]
end

@testset "twist Emery mean-field analytic derivatives" begin
    params = TwistEmeryNonPHParams(
        lx=3,
        ly=2,
        bcx=-0.7,
        bcy=0.8,
        chi1_dd=0.13,
        chi1_dp_x=1.0,
        chi1_dp_y=0.77,
        chi1_pp=0.31,
        mu_px=2.2,
        mu_py=2.7,
        delta_af_d=0.4,
        delta_c_d=0.21,
        delta_c_px=0.18,
        delta_c_py=0.16,
        delta_s_d=0.35,
        stripe_wavevector=2 * pi / 3,
        stripe_center_offset=0.5,
    )
    public_to_field = (
        (:chi1_dp_y, :chi1_dp_y),
        (:chi1_pp, :chi1_pp),
        (:chi1_dd, :chi1_dd),
        (:mu_px, :mu_px),
        (:mu_py, :mu_py),
        (:Delta_AF_d, :delta_af_d),
        (:Delta_c_d, :delta_c_d),
        (:Delta_c_px, :delta_c_px),
        (:Delta_c_py, :delta_c_py),
        (:Delta_s_d, :delta_s_d),
    )
    finite_difference_step = 1.0e-6

    for (public_name, field_name) in public_to_field
        parameter_value = getfield(params, field_name)
        plus_params = replace_twist_emery_test_param(
            params,
            field_name,
            parameter_value + finite_difference_step,
        )
        minus_params = replace_twist_emery_test_param(
            params,
            field_name,
            parameter_value - finite_difference_step,
        )
        finite_difference = (
            Matrix(build_twist_emery_nonph_hamiltonian(plus_params)) -
            Matrix(build_twist_emery_nonph_hamiltonian(minus_params))
        ) / (2 * finite_difference_step)
        analytic_derivative = Matrix(
            build_twist_emery_nonph_dh_dparam(params, public_name),
        )
        @test analytic_derivative ≈ finite_difference atol=1.0e-8 rtol=1.0e-8
    end
end

@testset "twist Emery AFM and Stripe parameter setup" begin
    common_args = Dict{String, Any}(
        "lx" => 6,
        "ly" => 2,
        "bcx" => -1.0,
        "bcy" => 1.0,
        "tpd_x" => 2.0,
        "tpd_y" => 1.2,
        "tpp" => 0.6,
        "ep_x" => 4.0,
        "ep_y" => 3.0,
        "chi1_dp_y" => NaN,
        "chi1_pp" => NaN,
        "chi1_dd" => 0.1,
        "mu_px" => NaN,
        "mu_py" => NaN,
        "Delta_AF_d" => 0.4,
        "Delta_c_d" => 0.2,
        "Delta_c_px" => 0.3,
        "Delta_c_py" => 0.5,
        "Delta_s_d" => 0.7,
        "lambda" => 3,
        "stripe_center" => "bond",
    )

    stripe_args = copy(common_args)
    stripe_args["ansatz"] = "Stripe"
    stripe_setup = build_twist_emery_mean_field_parameter_setup(stripe_args)
    @test stripe_setup.param_names == [
        :chi1_dp_y,
        :chi1_pp,
        :chi1_dd,
        :mu_px,
        :mu_py,
        :Delta_c_d,
        :Delta_c_px,
        :Delta_c_py,
        :Delta_s_d,
    ]
    @test stripe_setup.param_values[1:5] ≈ [0.6, 0.3, 0.1, 2.0, 1.5]
    @test stripe_setup.stripe_wavevector ≈ 2 * pi / 3
    @test stripe_setup.stripe_center_offset == 0.5

    afm_args = copy(common_args)
    afm_args["ansatz"] = "AFM"
    afm_args["chi1_dp_y"] = 0.9
    afm_setup = build_twist_emery_mean_field_parameter_setup(afm_args)
    @test afm_setup.param_names[end] == :Delta_AF_d
    @test afm_setup.param_values[1] == 0.9
    @test afm_setup.stripe_wavevector == 0.0
    @test afm_setup.stripe_center_offset == 0.0
end

@testset "twist Emery noninteracting one-body parity" begin
    lx = 3
    ly = 2
    tpd_x = 2.0
    setup = build_twist_emery_physical_term_groups(
        lx,
        ly;
        tpd_x=tpd_x,
        tpd_y=1.2,
        tpp=0.6,
        ep_x=4.0,
        ep_y=3.0,
        Udd=0.0,
        Up=0.0,
        Vpd_x=0.0,
        Vpd_y=0.0,
        Vpp=0.0,
    )
    physical_up = zeros(Float64, twist_emery_n_sites(lx, ly), twist_emery_n_sites(lx, ly))
    for term in setup.all_terms
        if term.ops == [:cdag_up, :c_up]
            physical_up[term.sites[1], term.sites[2]] += term.coef
        elseif term.ops == [:n]
            physical_up[term.sites[1], term.sites[1]] += term.coef
        end
    end

    params = TwistEmeryNonPHParams(
        lx=lx,
        ly=ly,
        chi1_dp_x=1.0,
        chi1_dp_y=1.2 / tpd_x,
        chi1_pp=0.6 / tpd_x,
        mu_px=4.0 / tpd_x,
        mu_py=3.0 / tpd_x,
    )
    mean_field = Matrix(build_twist_emery_nonph_hamiltonian(params))
    up_indices = [emery_spin_index(site, 1) for site in 1:twist_emery_n_sites(lx, ly)]
    @test mean_field[up_indices, up_indices] ≈ physical_up / tpd_x
end

@testset "twist Emery directional Cu-O PBC bonds" begin
    lx = 3
    ly = 2
    groups = build_twist_emery_pd_bond_groups(
        lx,
        ly;
        amplitude_x=2.0,
        amplitude_y=3.0,
        bcx=-1.0,
        bcy=0.5,
    )

    @test length(groups.x_bonds) == 2 * lx * ly
    @test length(groups.y_bonds) == 2 * lx * ly

    d_first = twist_emery_xyo_to_site_index(1, 1, EMERY_ORB_D, lx, ly)
    px_first = twist_emery_xyo_to_site_index(1, 1, EMERY_ORB_PX, lx, ly)
    py_first = twist_emery_xyo_to_site_index(1, 1, EMERY_ORB_PY, lx, ly)
    px_boundary = twist_emery_xyo_to_site_index(lx, 1, EMERY_ORB_PX, lx, ly)
    py_boundary = twist_emery_xyo_to_site_index(1, ly, EMERY_ORB_PY, lx, ly)

    @test any(
        bond -> bond.i == d_first && bond.j == px_first && bond.coef == -2.0,
        groups.x_bonds,
    )
    @test any(
        bond -> bond.i == d_first && bond.j == py_first && bond.coef == 3.0,
        groups.y_bonds,
    )
    @test any(
        bond -> bond.i == px_boundary && bond.j == d_first && bond.coef == -2.0,
        groups.x_bonds,
    )
    @test any(
        bond -> bond.i == py_boundary && bond.j == d_first && bond.coef == -1.5,
        groups.y_bonds,
    )
end

@testset "twist Emery PBC O-O and Cu-Cu bonds" begin
    lx = 3
    ly = 2
    pp_bonds = build_twist_emery_pp_bonds(
        lx,
        ly;
        amplitude=4.0,
        bcx=-1.0,
        bcy=0.5,
    )
    dd_groups = build_twist_emery_dd_bond_groups(
        lx,
        ly;
        amplitude=0.7,
        bcx=-1.0,
        bcy=0.5,
    )

    @test length(pp_bonds) == 4 * lx * ly
    @test length(dd_groups.x_bonds) == lx * ly
    @test length(dd_groups.y_bonds) == lx * ly

    px_corner = twist_emery_xyo_to_site_index(lx, 1, EMERY_ORB_PX, lx, ly)
    py_corner = twist_emery_xyo_to_site_index(1, ly, EMERY_ORB_PY, lx, ly)
    d_corner = twist_emery_xyo_to_site_index(lx, 1, EMERY_ORB_D, lx, ly)
    d_first_x = twist_emery_xyo_to_site_index(1, 1, EMERY_ORB_D, lx, ly)
    d_first_y = twist_emery_xyo_to_site_index(lx, 2, EMERY_ORB_D, lx, ly)

    @test any(
        bond -> bond.i == px_corner && bond.j == py_corner && bond.coef == -2.0,
        pp_bonds,
    )
    @test any(
        bond -> bond.i == d_corner && bond.j == d_first_x && bond.coef == -0.7,
        dd_groups.x_bonds,
    )
    @test any(
        bond -> bond.i == d_first_y && bond.j == d_corner && bond.coef == 0.35,
        dd_groups.y_bonds,
    )
end

@testset "twist Emery anisotropic physical term groups" begin
    lx = 3
    ly = 2
    number_of_cells = lx * ly
    setup = build_twist_emery_physical_term_groups(
        lx,
        ly;
        tpd_x=1.2,
        tpd_y=0.8,
        tpp=0.4,
        ep_x=3.1,
        ep_y=2.9,
        Udd=8.0,
        Up=3.0,
        Vpd_x=1.1,
        Vpd_y=0.7,
        Vpp=0.5,
    )

    @test length(setup.tpd_x_terms) == 8 * number_of_cells
    @test length(setup.tpd_y_terms) == 8 * number_of_cells
    @test length(setup.tpp_terms) == 16 * number_of_cells
    @test length(setup.ep_x_terms) == number_of_cells
    @test length(setup.ep_y_terms) == number_of_cells
    @test length(setup.udd_terms) == number_of_cells
    @test length(setup.up_terms) == 2 * number_of_cells
    @test length(setup.vpd_x_terms) == 2 * number_of_cells
    @test length(setup.vpd_y_terms) == 2 * number_of_cells
    @test length(setup.vpp_terms) == 4 * number_of_cells

    grouped_term_count = sum(
        length(getproperty(setup, group_name))
        for group_name in (
            :tpd_x_terms,
            :tpd_y_terms,
            :tpp_terms,
            :ep_x_terms,
            :ep_y_terms,
            :udd_terms,
            :up_terms,
            :vpd_x_terms,
            :vpd_y_terms,
            :vpp_terms,
        )
    )
    @test length(setup.all_terms) == grouped_term_count

    first_px = twist_emery_xyo_to_site_index(1, 1, EMERY_ORB_PX, lx, ly)
    first_py = twist_emery_xyo_to_site_index(1, 1, EMERY_ORB_PY, lx, ly)
    first_d = twist_emery_xyo_to_site_index(1, 1, EMERY_ORB_D, lx, ly)
    @test any(
        term -> term.ops == [:n] && term.sites == [first_px] && term.coef == 3.1,
        setup.ep_x_terms,
    )
    @test any(
        term -> term.ops == [:n] && term.sites == [first_py] && term.coef == 2.9,
        setup.ep_y_terms,
    )
    @test any(
        term -> term.ops == [:n_up, :n_dn] &&
                term.sites == [first_d, first_d] &&
                term.coef == 8.0,
        setup.udd_terms,
    )
end

@testset "twist Emery directional energy observables" begin
    lx = 2
    ly = 2
    number_of_sites = twist_emery_n_sites(lx, ly)
    number_of_electrons = lx * ly
    setup = build_twist_emery_physical_term_groups(
        lx,
        ly;
        tpd_x=1.2,
        tpd_y=0.8,
        tpp=0.4,
        ep_x=3.1,
        ep_y=2.9,
        Udd=8.0,
        Up=3.0,
        Vpd_x=1.1,
        Vpd_y=0.7,
        Vpp=0.5,
    )
    model = build_twist_emery_general_model(lx, ly, setup)
    observables = build_twist_emery_observables(lx, ly, setup)
    energy_component_names = (
        :E_tpd_x,
        :E_tpd_y,
        :E_tpp,
        :E_ep_x,
        :E_ep_y,
        :E_Udd,
        :E_Up,
        :E_Vpd_x,
        :E_Vpd_y,
        :E_Vpp,
    )

    @test model.Nsite == number_of_sites
    @test all(haskey(observables, name) for name in (:E, energy_component_names...))
    @test haskey(observables, :n_d_1_1)
    @test haskey(observables, :n_px_2_2)
    @test haskey(observables, :n_py_2_2)
    @test !haskey(observables, :n_px_0_1)
    @test haskey(observables, :Szzq_0_0)

    Random.seed!(20260723)
    sampler = config_Hubbard(
        number_of_sites,
        number_of_electrons ÷ 2,
        number_of_electrons ÷ 2;
        ifPH=false,
    )
    init_config_Hubbard!(sampler)
    base_orbitals = randn(Float64, 2 * number_of_sites, number_of_electrons)
    vwf = vwf_det(base_orbitals, sampler)
    set_projector!(
        vwf,
        CompositeProjector([
            GutzwillerProjectorTerm(param_name=:g, g=0.0),
        ]),
    )
    init_gswf!(vwf)

    component_energy = sum(
        observables[name](model, vwf)
        for name in energy_component_names
    )
    @test component_energy ≈ observables[:E](model, vwf)
    @test build_twist_emery_history_observables() ==
          [:E, energy_component_names...]
end
