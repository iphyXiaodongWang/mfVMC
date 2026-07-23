# This file is used for PBC Emery model VMC for AFM and stripe states.

include(joinpath(@__DIR__, "Emery.jl"))

"""
用途: 返回 x/y 双向 PBC Emery 三带模型的总 site 数.

参数:
- `lx, ly::Int`: Cu unit cell 在 x/y 方向的数量, 必须为正整数.

返回:
- `Int`: 总 site 数, 公式为 `N_sites = 3 * lx * ly`.
"""
function twist_emery_n_sites(lx::Int, ly::Int)::Int
    lx > 0 || error("lx must be positive, got $(lx).")
    ly > 0 || error("ly must be positive, got $(ly).")
    return 3 * lx * ly
end

"""
用途: 将 PBC Emery orbital 坐标 `(x,y,orbital)` 映射为一维 site index.

参数:
- `x, y::Int`: unit cell 坐标, 函数内部使用 `mod1` 按 x/y 双向 PBC wrap.
- `orbital::Int`: orbital 编号, 支持 `EMERY_ORB_D`, `EMERY_ORB_PY`, `EMERY_ORB_PX`.
- `lx, ly::Int`: Cu unit cell 在 x/y 方向的数量.

返回:
- `Int`: 1-based site index, 每个 unit cell 内按 `d, p_y, p_x` 排列.
"""
function twist_emery_xyo_to_site_index(
    x::Int,
    y::Int,
    orbital::Int,
    lx::Int,
    ly::Int,
)::Int
    twist_emery_n_sites(lx, ly)
    orbital in (EMERY_ORB_D, EMERY_ORB_PY, EMERY_ORB_PX) ||
        error("Unknown Emery orbital $(orbital). Expected 1=d, 2=p_y, 3=p_x.")
    cell_index = (mod1(x, lx) - 1) * ly + mod1(y, ly)
    return 3 * (cell_index - 1) + orbital
end

"""
用途: 返回 Emery orbital 相对 Cu unit cell 的实际二维坐标.

参数:
- `x, y::Int`: orbital 所属 unit cell 的整数坐标.
- `orbital::Int`: orbital 编号, 支持 `d`, `p_y`, `p_x`.

返回:
- `Tuple{Float64, Float64}`: 实际坐标 `(x_coordinate, y_coordinate)`.

公式:
- `r_d=(x,y)`.
- `r_px=(x+1/2,y)`.
- `r_py=(x,y+1/2)`.
"""
function twist_emery_orbital_coordinate(
    x::Int,
    y::Int,
    orbital::Int,
)::Tuple{Float64,Float64}
    orbital == EMERY_ORB_D && return (Float64(x), Float64(y))
    orbital == EMERY_ORB_PX && return (Float64(x) + 0.5, Float64(y))
    orbital == EMERY_ORB_PY && return (Float64(x), Float64(y) + 0.5)
    error("Unknown Emery orbital $(orbital). Expected 1=d, 2=p_y, 3=p_x.")
end

"""
用途: 构造方向分辨的 PBC Emery Cu-O hopping 代表 bonds.

参数:
- `lx, ly::Int`: Cu unit cell 在 x/y 方向的数量.
- `amplitude_x, amplitude_y::Real`: 水平 Cu-p_x 和垂直 Cu-p_y hopping 振幅.
- `bcx, bcy::Real`: mean-field 跨 x/y 边界 bond 的实数因子.

返回:
- `NamedTuple`: 包含 `x_bonds` 和 `y_bonds`, 每项为 `Vector{EmeryBond}`.

公式:
- x 方向为 `d(x,y)->p_x(x,y): -t_x`,
  `p_x(x,y)->d(x+1,y): +t_x`.
- y 方向为 `d(x,y)->p_y(x,y): +t_y`,
  `p_y(x,y)->d(x,y+1): -t_y`.
"""
function build_twist_emery_pd_bond_groups(
    lx::Int,
    ly::Int;
    amplitude_x::Real,
    amplitude_y::Real,
    bcx::Real=1.0,
    bcy::Real=1.0,
)
    twist_emery_n_sites(lx, ly)
    t_x = Float64(amplitude_x)
    t_y = Float64(amplitude_y)
    boundary_x = Float64(bcx)
    boundary_y = Float64(bcy)
    x_bonds = EmeryBond[]
    y_bonds = EmeryBond[]
    sizehint!(x_bonds, 2 * lx * ly)
    sizehint!(y_bonds, 2 * lx * ly)

    for x in 1:lx, y in 1:ly
        d_site = twist_emery_xyo_to_site_index(x, y, EMERY_ORB_D, lx, ly)
        px_site = twist_emery_xyo_to_site_index(x, y, EMERY_ORB_PX, lx, ly)
        py_site = twist_emery_xyo_to_site_index(x, y, EMERY_ORB_PY, lx, ly)
        right_d_site = twist_emery_xyo_to_site_index(x + 1, y, EMERY_ORB_D, lx, ly)
        upper_d_site = twist_emery_xyo_to_site_index(x, y + 1, EMERY_ORB_D, lx, ly)
        x_factor = x == lx ? boundary_x : 1.0
        y_factor = y == ly ? boundary_y : 1.0

        push!(x_bonds, EmeryBond(d_site, px_site, -t_x))
        push!(x_bonds, EmeryBond(px_site, right_d_site, t_x * x_factor))
        push!(y_bonds, EmeryBond(d_site, py_site, t_y))
        push!(y_bonds, EmeryBond(py_site, upper_d_site, -t_y * y_factor))
    end
    return (; x_bonds=x_bonds, y_bonds=y_bonds)
end

"""
用途: 构造 x/y 双向 PBC Emery 模型的 O-O hopping 代表 bonds.

参数:
- `lx, ly::Int`: Cu unit cell 在 x/y 方向的数量.
- `amplitude::Real`: 各向同性 O-O hopping 振幅.
- `bcx, bcy::Real`: mean-field 跨 x/y 边界 bond 的实数因子.

返回:
- `Vector{EmeryBond}`: 每个 unit cell 四条带 Emery 标准符号的 O-O bonds.

公式:
- `p_y(x,y)->p_x(x,y): +t_pp`.
- `p_y(x,y)->p_x(x,y+1): -t_pp`.
- `p_x(x,y)->p_y(x+1,y): -t_pp`.
- `p_x(x,y)->p_y(x+1,y-1): +t_pp`.
"""
function build_twist_emery_pp_bonds(
    lx::Int,
    ly::Int;
    amplitude::Real,
    bcx::Real=1.0,
    bcy::Real=1.0,
)::Vector{EmeryBond}
    twist_emery_n_sites(lx, ly)
    hopping = Float64(amplitude)
    boundary_x = Float64(bcx)
    boundary_y = Float64(bcy)
    bonds = EmeryBond[]
    sizehint!(bonds, 4 * lx * ly)

    for x in 1:lx, y in 1:ly
        py_site = twist_emery_xyo_to_site_index(x, y, EMERY_ORB_PY, lx, ly)
        px_site = twist_emery_xyo_to_site_index(x, y, EMERY_ORB_PX, lx, ly)
        px_upper_site = twist_emery_xyo_to_site_index(x, y + 1, EMERY_ORB_PX, lx, ly)
        py_right_site = twist_emery_xyo_to_site_index(x + 1, y, EMERY_ORB_PY, lx, ly)
        py_right_lower_site = twist_emery_xyo_to_site_index(
            x + 1,
            y - 1,
            EMERY_ORB_PY,
            lx,
            ly,
        )
        x_factor = x == lx ? boundary_x : 1.0
        upper_y_factor = y == ly ? boundary_y : 1.0
        lower_y_factor = y == 1 ? boundary_y : 1.0

        push!(bonds, EmeryBond(py_site, px_site, hopping))
        push!(bonds, EmeryBond(py_site, px_upper_site, -hopping * upper_y_factor))
        push!(bonds, EmeryBond(px_site, py_right_site, -hopping * x_factor))
        push!(
            bonds,
            EmeryBond(
                px_site,
                py_right_lower_site,
                hopping * x_factor * lower_y_factor,
            ),
        )
    end
    return bonds
end

"""
用途: 构造方向分辨的 PBC Emery Cu-Cu 最近邻代表 bonds.

参数:
- `lx, ly::Int`: Cu unit cell 在 x/y 方向的数量.
- `amplitude::Real`: Cu-Cu auxiliary mean-field hopping 振幅.
- `bcx, bcy::Real`: mean-field 跨 x/y 边界 bond 的实数因子.

返回:
- `NamedTuple`: 包含 `x_bonds` 和 `y_bonds`, 每个 cell 在每个方向各一条 bond.
"""
function build_twist_emery_dd_bond_groups(
    lx::Int,
    ly::Int;
    amplitude::Real,
    bcx::Real=1.0,
    bcy::Real=1.0,
)
    twist_emery_n_sites(lx, ly)
    hopping = Float64(amplitude)
    boundary_x = Float64(bcx)
    boundary_y = Float64(bcy)
    x_bonds = EmeryBond[]
    y_bonds = EmeryBond[]
    sizehint!(x_bonds, lx * ly)
    sizehint!(y_bonds, lx * ly)

    for x in 1:lx, y in 1:ly
        d_site = twist_emery_xyo_to_site_index(x, y, EMERY_ORB_D, lx, ly)
        right_d_site = twist_emery_xyo_to_site_index(x + 1, y, EMERY_ORB_D, lx, ly)
        upper_d_site = twist_emery_xyo_to_site_index(x, y + 1, EMERY_ORB_D, lx, ly)
        x_factor = x == lx ? boundary_x : 1.0
        y_factor = y == ly ? boundary_y : 1.0
        push!(x_bonds, EmeryBond(d_site, right_d_site, hopping * x_factor))
        push!(y_bonds, EmeryBond(d_site, upper_d_site, hopping * y_factor))
    end
    return (; x_bonds=x_bonds, y_bonds=y_bonds)
end
