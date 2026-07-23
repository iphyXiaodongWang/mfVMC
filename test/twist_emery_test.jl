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
