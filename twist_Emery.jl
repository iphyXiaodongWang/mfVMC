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
用途: 保存 PBC Emery 无 pairing mean-field Hamiltonian 的有限组参数.

字段:
- `lx, ly::Int`: Cu unit cell 尺寸.
- `bcx, bcy::Float64`: 跨 x/y 边界 hopping 的实数因子.
- `chi1_dd, chi1_dp_x, chi1_dp_y, chi1_pp::Float64`: 四类 hopping 振幅.
- `mu_px, mu_py::Float64`: `p_x/p_y` orbital 的均匀 onsite energy, `mu_d=0` 为固定 gauge.
- `delta_af_d::Float64`: 仅作用在 Cu d orbital 上的 AFM 振幅.
- `delta_c_d, delta_c_px, delta_c_py::Float64`: orbital-resolved charge-stripe 振幅.
- `delta_s_d::Float64`: 仅作用在 Cu d orbital 上的 spin-stripe 振幅.
- `stripe_wavevector::Float64`: charge stripe wavevector `Q=2π/lambda`.
- `stripe_center_offset::Float64`: stripe center 的 x 方向 offset.
"""
struct TwistEmeryNonPHParams
    lx::Int
    ly::Int
    bcx::Float64
    bcy::Float64
    chi1_dd::Float64
    chi1_dp_x::Float64
    chi1_dp_y::Float64
    chi1_pp::Float64
    mu_px::Float64
    mu_py::Float64
    delta_af_d::Float64
    delta_c_d::Float64
    delta_c_px::Float64
    delta_c_py::Float64
    delta_s_d::Float64
    stripe_wavevector::Float64
    stripe_center_offset::Float64
end

"""
用途: 构造并验证 PBC Emery 无 pairing mean-field 参数.

参数:
- 所有 keyword 参数与 `TwistEmeryNonPHParams` 字段同名.
- `lx, ly::Int`: 必须为正整数.
- 其余参数均为实数, 默认给出无序态且固定 `chi1_dp_x=1`.

返回:
- `TwistEmeryNonPHParams`: 转换为 `Float64` 后的不可变参数对象.
"""
function TwistEmeryNonPHParams(;
    lx::Int,
    ly::Int,
    bcx::Real=1.0,
    bcy::Real=1.0,
    chi1_dd::Real=0.0,
    chi1_dp_x::Real=1.0,
    chi1_dp_y::Real=1.0,
    chi1_pp::Real=0.0,
    mu_px::Real=0.0,
    mu_py::Real=0.0,
    delta_af_d::Real=0.0,
    delta_c_d::Real=0.0,
    delta_c_px::Real=0.0,
    delta_c_py::Real=0.0,
    delta_s_d::Real=0.0,
    stripe_wavevector::Real=0.0,
    stripe_center_offset::Real=0.0,
)::TwistEmeryNonPHParams
    twist_emery_n_sites(lx, ly)
    return TwistEmeryNonPHParams(
        lx,
        ly,
        Float64(bcx),
        Float64(bcy),
        Float64(chi1_dd),
        Float64(chi1_dp_x),
        Float64(chi1_dp_y),
        Float64(chi1_pp),
        Float64(mu_px),
        Float64(mu_py),
        Float64(delta_af_d),
        Float64(delta_c_d),
        Float64(delta_c_px),
        Float64(delta_c_py),
        Float64(delta_s_d),
        Float64(stripe_wavevector),
        Float64(stripe_center_offset),
    )
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

"""
用途: 构造 PBC Emery AFM 或 Stripe 态的无 pairing spinful mean-field Hamiltonian.

数学公式:
- charge field:
  `mu_d(x)=-Delta_c_d*cos(Q*(x-x0))`,
  `mu_px(x)=mu_px-Delta_c_px*cos(Q*(x+1/2-x0))`,
  `mu_py(x)=mu_py-Delta_c_py*cos(Q*(x-x0))`.
- Cu spin field:
  `m_d(x)=Delta_AF_d+Delta_s_d*sin((Q/2)*(x-x0))`.
- onsite spin splitting:
  `H_up=mu+(-1)^(x+y)*m`, `H_dn=mu-(-1)^(x+y)*m`.

参数:
- `params::TwistEmeryNonPHParams`: lattice、boundary、hopping 与有序场参数.

返回:
- `Hermitian{Float64, Matrix{Float64}}`: 维度为 `2N_sites x 2N_sites` 的
  实对称 Hamiltonian, spin index 使用现有交错排列 `(site_up, site_down)`.
"""
function build_twist_emery_nonph_hamiltonian(
    params::TwistEmeryNonPHParams,
)
    lx = params.lx
    ly = params.ly
    n_sites = twist_emery_n_sites(lx, ly)
    hamiltonian = zeros(Float64, 2 * n_sites, 2 * n_sites)

    dd_groups = build_twist_emery_dd_bond_groups(
        lx,
        ly;
        amplitude=params.chi1_dd,
        bcx=params.bcx,
        bcy=params.bcy,
    )
    pd_groups = build_twist_emery_pd_bond_groups(
        lx,
        ly;
        amplitude_x=params.chi1_dp_x,
        amplitude_y=params.chi1_dp_y,
        bcx=params.bcx,
        bcy=params.bcy,
    )
    pp_bonds = build_twist_emery_pp_bonds(
        lx,
        ly;
        amplitude=params.chi1_pp,
        bcx=params.bcx,
        bcy=params.bcy,
    )
    all_bonds = vcat(
        dd_groups.x_bonds,
        dd_groups.y_bonds,
        pd_groups.x_bonds,
        pd_groups.y_bonds,
        pp_bonds,
    )
    for bond in all_bonds
        add_emery_spinful_hopping!(hamiltonian, bond.i, bond.j, bond.coef)
    end

    wavevector = params.stripe_wavevector
    center_offset = params.stripe_center_offset
    for x in 1:lx, y in 1:ly
        staggered_sign = Float64((-1)^(x + y))
        d_site = twist_emery_xyo_to_site_index(x, y, EMERY_ORB_D, lx, ly)
        px_site = twist_emery_xyo_to_site_index(x, y, EMERY_ORB_PX, lx, ly)
        py_site = twist_emery_xyo_to_site_index(x, y, EMERY_ORB_PY, lx, ly)

        d_coordinate_x = twist_emery_orbital_coordinate(x, y, EMERY_ORB_D)[1]
        px_coordinate_x = twist_emery_orbital_coordinate(x, y, EMERY_ORB_PX)[1]
        py_coordinate_x = twist_emery_orbital_coordinate(x, y, EMERY_ORB_PY)[1]
        d_charge = -params.delta_c_d * cos(wavevector * (d_coordinate_x - center_offset))
        px_charge = params.mu_px -
                    params.delta_c_px * cos(wavevector * (px_coordinate_x - center_offset))
        py_charge = params.mu_py -
                    params.delta_c_py * cos(wavevector * (py_coordinate_x - center_offset))
        d_spin = params.delta_af_d +
                 params.delta_s_d * sin(0.5 * wavevector * (d_coordinate_x - center_offset))

        add_emery_onsite!(hamiltonian, d_site, d_charge, d_spin, staggered_sign)
        add_emery_onsite!(hamiltonian, px_site, px_charge, 0.0, staggered_sign)
        add_emery_onsite!(hamiltonian, py_site, py_charge, 0.0, staggered_sign)
    end
    return Hermitian(hamiltonian)
end

"""
用途: 构造 PBC Emery mean-field Hamiltonian 对指定优化参数的解析导数.

数学公式:
- Hamiltonian 对每个有限振幅参数均为线性, 因而
  `dH/dp = H(p=1, 其余可优化振幅=0, chi1_dp_x=0)`.
- lattice、boundary、`Q` 与 stripe center 保持不变.

参数:
- `params::TwistEmeryNonPHParams`: 当前 mean-field 参数和几何信息.
- `param_name::Symbol`: 支持 `chi1_dp_y`, `chi1_pp`, `chi1_dd`, `mu_px`,
  `mu_py`, `Delta_AF_d`, `Delta_c_d`, `Delta_c_px`, `Delta_c_py`, `Delta_s_d`.

返回:
- `Hermitian{Float64, Matrix{Float64}}`: `dH/dp` 解析导数矩阵.
"""
function build_twist_emery_nonph_dh_dparam(
    params::TwistEmeryNonPHParams,
    param_name::Symbol,
)
    supported_names = (
        :chi1_dp_y,
        :chi1_pp,
        :chi1_dd,
        :mu_px,
        :mu_py,
        :Delta_AF_d,
        :Delta_c_d,
        :Delta_c_px,
        :Delta_c_py,
        :Delta_s_d,
    )
    param_name in supported_names ||
        error("Unsupported twist Emery mean-field parameter $(param_name).")

    return build_twist_emery_nonph_hamiltonian(
        TwistEmeryNonPHParams(
            lx=params.lx,
            ly=params.ly,
            bcx=params.bcx,
            bcy=params.bcy,
            chi1_dd=param_name == :chi1_dd ? 1.0 : 0.0,
            chi1_dp_x=0.0,
            chi1_dp_y=param_name == :chi1_dp_y ? 1.0 : 0.0,
            chi1_pp=param_name == :chi1_pp ? 1.0 : 0.0,
            mu_px=param_name == :mu_px ? 1.0 : 0.0,
            mu_py=param_name == :mu_py ? 1.0 : 0.0,
            delta_af_d=param_name == :Delta_AF_d ? 1.0 : 0.0,
            delta_c_d=param_name == :Delta_c_d ? 1.0 : 0.0,
            delta_c_px=param_name == :Delta_c_px ? 1.0 : 0.0,
            delta_c_py=param_name == :Delta_c_py ? 1.0 : 0.0,
            delta_s_d=param_name == :Delta_s_d ? 1.0 : 0.0,
            stripe_wavevector=params.stripe_wavevector,
            stripe_center_offset=params.stripe_center_offset,
        ),
    )
end

"""
用途: 从 ArgParse 风格字典中读取参数, 同时兼容 String 和 Symbol key.

参数:
- `args::AbstractDict`: 参数字典.
- `name::String`: 参数名.

返回:
- `Any`: 对应参数值; 参数不存在时抛出错误.
"""
function get_twist_emery_argument(args::AbstractDict, name::String)
    haskey(args, name) && return args[name]
    symbol_name = Symbol(name)
    haskey(args, symbol_name) && return args[symbol_name]
    error("Missing twist Emery argument $(name).")
end

"""
用途: 当显式 mean-field 初值为 `NaN` 时使用物理 one-body 比值作为默认值.

参数:
- `value::Real`: CLI 给定值.
- `default_value::Real`: 由物理参数计算的默认值.

返回:
- `Float64`: 若 `value` 为 `NaN` 则返回 `default_value`, 否则返回 `value`.
"""
function twist_emery_value_or_default(value::Real, default_value::Real)::Float64
    numeric_value = Float64(value)
    return isnan(numeric_value) ? Float64(default_value) : numeric_value
end

"""
用途: 按 AFM 或 Stripe ansatz 生成有限 mean-field 参数名、初值和 stripe 几何.

数学公式:
- 固定 gauge 为 `chi1_dp_x=1`, 未显式指定时
  `chi1_dp_y=tpd_y/tpd_x`, `chi1_pp=tpp/tpd_x`,
  `mu_px=ep_x/tpd_x`, `mu_py=ep_y/tpd_x`.
- Stripe 使用 `Q=2π/lambda`; `site` center 对应 `x0=0`,
  `bond` center 对应 `x0=1/2`.

参数:
- `args::AbstractDict`: 至少包含 lattice、physical one-body、mean-field 初值、
  `ansatz`, `lambda` 和 `stripe_center`.

返回:
- `NamedTuple`: `param_names`, `param_values`, `stripe_wavevector`,
  `stripe_center_offset`.
"""
function build_twist_emery_mean_field_parameter_setup(args::AbstractDict)
    tpd_x = Float64(get_twist_emery_argument(args, "tpd_x"))
    iszero(tpd_x) && error("tpd_x must be nonzero because chi1_dp_x=1 fixes the mean-field gauge.")

    common_names = [:chi1_dp_y, :chi1_pp, :chi1_dd, :mu_px, :mu_py]
    common_values = [
        twist_emery_value_or_default(
            get_twist_emery_argument(args, "chi1_dp_y"),
            Float64(get_twist_emery_argument(args, "tpd_y")) / tpd_x,
        ),
        twist_emery_value_or_default(
            get_twist_emery_argument(args, "chi1_pp"),
            Float64(get_twist_emery_argument(args, "tpp")) / tpd_x,
        ),
        Float64(get_twist_emery_argument(args, "chi1_dd")),
        twist_emery_value_or_default(
            get_twist_emery_argument(args, "mu_px"),
            Float64(get_twist_emery_argument(args, "ep_x")) / tpd_x,
        ),
        twist_emery_value_or_default(
            get_twist_emery_argument(args, "mu_py"),
            Float64(get_twist_emery_argument(args, "ep_y")) / tpd_x,
        ),
    ]

    ansatz = lowercase(String(get_twist_emery_argument(args, "ansatz")))
    if ansatz == "afm"
        return (
            param_names=vcat(common_names, [:Delta_AF_d]),
            param_values=vcat(
                common_values,
                [Float64(get_twist_emery_argument(args, "Delta_AF_d"))],
            ),
            stripe_wavevector=0.0,
            stripe_center_offset=0.0,
        )
    elseif ansatz == "stripe"
        stripe_period = Int(get_twist_emery_argument(args, "lambda"))
        stripe_period > 0 || error("lambda must be positive, got $(stripe_period).")
        stripe_center = lowercase(String(get_twist_emery_argument(args, "stripe_center")))
        center_offset = if stripe_center == "site"
            0.0
        elseif stripe_center == "bond"
            0.5
        else
            error("stripe_center must be site or bond, got $(stripe_center).")
        end
        return (
            param_names=vcat(
                common_names,
                [:Delta_c_d, :Delta_c_px, :Delta_c_py, :Delta_s_d],
            ),
            param_values=vcat(
                common_values,
                [
                    Float64(get_twist_emery_argument(args, "Delta_c_d")),
                    Float64(get_twist_emery_argument(args, "Delta_c_px")),
                    Float64(get_twist_emery_argument(args, "Delta_c_py")),
                    Float64(get_twist_emery_argument(args, "Delta_s_d")),
                ],
            ),
            stripe_wavevector=2 * pi / stripe_period,
            stripe_center_offset=center_offset,
        )
    end
    error("ansatz must be AFM or Stripe, got $(get_twist_emery_argument(args, "ansatz")).")
end

"""
用途: 构造方向和作用项分组的 PBC Emery physical Hamiltonian terms.

参数:
- `lx, ly::Int`: Cu unit cell 在 x/y 方向的数量.
- `tpd_x, tpd_y::Real`: 水平 Cu-p_x 和垂直 Cu-p_y hopping.
- `tpp::Real`: 各向同性 O-O hopping.
- `ep_x, ep_y::Real`: p_x/p_y orbital onsite energy.
- `Udd, Up::Real`: Cu d 和 oxygen p orbital onsite Hubbard interaction.
- `Vpd_x, Vpd_y::Real`: 水平/垂直 Cu-O density interaction.
- `Vpp::Real`: O-O density interaction.

返回:
- `NamedTuple`: 包含十个物理 term groups 和按固定顺序合并的 `all_terms`.

公式:
- `H_ep = ep_x * sum_{i in px} n_i + ep_y * sum_{i in py} n_i`.
- `H_U = Udd * sum_{i in d} n_{i,up}n_{i,dn}
  + Up * sum_{i in p} n_{i,up}n_{i,dn}`.
- `H_V = Vpd_x * sum_{<d,px>} n_d n_px
  + Vpd_y * sum_{<d,py>} n_d n_py
  + Vpp * sum_{<p,p>} n_p n_p`.
"""
function build_twist_emery_physical_term_groups(
    lx::Int,
    ly::Int;
    tpd_x::Real,
    tpd_y::Real,
    tpp::Real,
    ep_x::Real,
    ep_y::Real,
    Udd::Real,
    Up::Real,
    Vpd_x::Real,
    Vpd_y::Real,
    Vpp::Real,
)
    twist_emery_n_sites(lx, ly)
    tpd_x_terms = OperatorTerm[]
    tpd_y_terms = OperatorTerm[]
    tpp_terms = OperatorTerm[]
    ep_x_terms = OperatorTerm[]
    ep_y_terms = OperatorTerm[]
    udd_terms = OperatorTerm[]
    up_terms = OperatorTerm[]
    vpd_x_terms = OperatorTerm[]
    vpd_y_terms = OperatorTerm[]
    vpp_terms = OperatorTerm[]

    pd_groups = build_twist_emery_pd_bond_groups(
        lx,
        ly;
        amplitude_x=tpd_x,
        amplitude_y=tpd_y,
    )
    pp_bonds = build_twist_emery_pp_bonds(lx, ly; amplitude=tpp)
    for bond in pd_groups.x_bonds
        add_emery_general_model_hopping_terms!(
            tpd_x_terms,
            bond.i,
            bond.j,
            bond.coef,
        )
        push!(
            vpd_x_terms,
            OperatorTerm([:n, :n], [bond.i, bond.j], Float64(Vpd_x)),
        )
    end
    for bond in pd_groups.y_bonds
        add_emery_general_model_hopping_terms!(
            tpd_y_terms,
            bond.i,
            bond.j,
            bond.coef,
        )
        push!(
            vpd_y_terms,
            OperatorTerm([:n, :n], [bond.i, bond.j], Float64(Vpd_y)),
        )
    end
    for bond in pp_bonds
        add_emery_general_model_hopping_terms!(
            tpp_terms,
            bond.i,
            bond.j,
            bond.coef,
        )
        push!(
            vpp_terms,
            OperatorTerm([:n, :n], [bond.i, bond.j], Float64(Vpp)),
        )
    end

    for x in 1:lx, y in 1:ly
        d_site = twist_emery_xyo_to_site_index(x, y, EMERY_ORB_D, lx, ly)
        py_site = twist_emery_xyo_to_site_index(x, y, EMERY_ORB_PY, lx, ly)
        px_site = twist_emery_xyo_to_site_index(x, y, EMERY_ORB_PX, lx, ly)
        push!(ep_x_terms, OperatorTerm([:n], [px_site], Float64(ep_x)))
        push!(ep_y_terms, OperatorTerm([:n], [py_site], Float64(ep_y)))
        push!(
            udd_terms,
            OperatorTerm([:n_up, :n_dn], [d_site, d_site], Float64(Udd)),
        )
        push!(
            up_terms,
            OperatorTerm([:n_up, :n_dn], [px_site, px_site], Float64(Up)),
        )
        push!(
            up_terms,
            OperatorTerm([:n_up, :n_dn], [py_site, py_site], Float64(Up)),
        )
    end

    all_terms = vcat(
        tpd_x_terms,
        tpd_y_terms,
        tpp_terms,
        ep_x_terms,
        ep_y_terms,
        udd_terms,
        up_terms,
        vpd_x_terms,
        vpd_y_terms,
        vpp_terms,
    )
    return (;
        tpd_x_terms=tpd_x_terms,
        tpd_y_terms=tpd_y_terms,
        tpp_terms=tpp_terms,
        ep_x_terms=ep_x_terms,
        ep_y_terms=ep_y_terms,
        udd_terms=udd_terms,
        up_terms=up_terms,
        vpd_x_terms=vpd_x_terms,
        vpd_y_terms=vpd_y_terms,
        vpp_terms=vpp_terms,
        all_terms=all_terms,
    )
end

"""
用途: 用已分组的 physical terms 构造 PBC Emery `GeneralModel`.

参数:
- `lx, ly::Int`: Cu unit cell 在 x/y 方向的数量.
- `term_groups`: `build_twist_emery_physical_term_groups` 返回的 NamedTuple.

返回:
- `GeneralModel`: site 数为 `3 * lx * ly`, terms 为 `term_groups.all_terms`.
"""
function build_twist_emery_general_model(
    lx::Int,
    ly::Int,
    term_groups,
)::GeneralModel
    return GeneralModel(twist_emery_n_sites(lx, ly), term_groups.all_terms)
end

"""
用途: 计算一组 PBC Emery physical terms 在当前 VMC 构型上的 local energy 之和.

参数:
- `terms::Vector{OperatorTerm}`: 同一 physical 分量的 operator terms.
- `model`: 当前 `GeneralModel`.
- `vwf`: 当前 determinant 波函数.

返回:
- `Float64`: `sum_a Re[E_local(term_a)]`.
"""
function measure_twist_emery_term_energy_sum(
    terms::Vector{OperatorTerm},
    model,
    vwf,
)::Float64
    energy = 0.0
    for term in terms
        energy += real(mfVMC.Model.compute_term_energy(term, vwf))
    end
    return energy
end

"""
用途: 返回 PBC Emery 中所有 Cu d sites 及其整数 unit-cell 坐标.

参数:
- `lx, ly::Int`: Cu unit cell 在 x/y 方向的数量.

返回:
- `Vector{Tuple{Int, Int, Int}}`: 每项为 `(site, x, y)`.
"""
function build_twist_emery_cu_site_coordinates(
    lx::Int,
    ly::Int,
)::Vector{Tuple{Int,Int,Int}}
    coordinates = Tuple{Int,Int,Int}[]
    sizehint!(coordinates, lx * ly)
    for x in 1:lx, y in 1:ly
        push!(
            coordinates,
            (
                twist_emery_xyo_to_site_index(x, y, EMERY_ORB_D, lx, ly),
                x,
                y,
            ),
        )
    end
    return coordinates
end

"""
用途: 构造 PBC Emery measure 使用的能量、局域密度/自旋和 Cu `Szz(q)` observables.

参数:
- `lx, ly::Int`: Cu unit cell 在 x/y 方向的数量.
- `term_groups`: `build_twist_emery_physical_term_groups` 返回的 physical term groups.

返回:
- `Dict{Symbol, Function}`: 固定包含总能量、十个能量分项、所有 orbital 的
  `n/Sz` 和全部 Cu momentum index 的 `Szzq`.
"""
function build_twist_emery_observables(
    lx::Int,
    ly::Int,
    term_groups,
)::Dict{Symbol,Function}
    observables = Dict{Symbol,Function}(:E => local_energy)
    energy_group_specs = (
        (:E_tpd_x, term_groups.tpd_x_terms),
        (:E_tpd_y, term_groups.tpd_y_terms),
        (:E_tpp, term_groups.tpp_terms),
        (:E_ep_x, term_groups.ep_x_terms),
        (:E_ep_y, term_groups.ep_y_terms),
        (:E_Udd, term_groups.udd_terms),
        (:E_Up, term_groups.up_terms),
        (:E_Vpd_x, term_groups.vpd_x_terms),
        (:E_Vpd_y, term_groups.vpd_y_terms),
        (:E_Vpp, term_groups.vpp_terms),
    )
    for (observable_name, terms) in energy_group_specs
        terms_local = copy(terms)
        observables[observable_name] = (model, vwf) ->
            measure_twist_emery_term_energy_sum(terms_local, model, vwf)
    end

    cu_site_coordinates = build_twist_emery_cu_site_coordinates(lx, ly)
    for momentum_x in 0:(lx - 1), momentum_y in 0:(ly - 1)
        momentum_x_local = momentum_x
        momentum_y_local = momentum_y
        observable_name = Symbol("Szzq_$(momentum_x_local)_$(momentum_y_local)")
        observables[observable_name] = (model, vwf) ->
            measure_emery_cu_szz_structure_factor(
                vwf,
                cu_site_coordinates,
                lx,
                ly,
                momentum_x_local,
                momentum_y_local,
            )
    end

    for x in 1:lx, y in 1:ly
        d_site = twist_emery_xyo_to_site_index(x, y, EMERY_ORB_D, lx, ly)
        py_site = twist_emery_xyo_to_site_index(x, y, EMERY_ORB_PY, lx, ly)
        px_site = twist_emery_xyo_to_site_index(x, y, EMERY_ORB_PX, lx, ly)
        add_emery_site_observables!(observables, "d_$(x)_$(y)", d_site)
        add_emery_site_observables!(observables, "py_$(x)_$(y)", py_site)
        add_emery_site_observables!(observables, "px_$(x)_$(y)", px_site)
    end
    return observables
end

"""
用途: 返回 PBC Emery blocking-binning 使用的固定能量 observable 顺序.

参数:
- 无.

返回:
- `Vector{Symbol}`: 总能量后依次为十个方向/作用项分辨的能量字段.
"""
function build_twist_emery_history_observables()::Vector{Symbol}
    return [
        :E,
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
    ]
end
