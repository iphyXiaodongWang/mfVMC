using Test
using LinearAlgebra

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
