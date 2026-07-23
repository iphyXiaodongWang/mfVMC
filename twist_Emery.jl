# This file is used for PBC Emery model VMC for AFM and stripe states.

include(joinpath(@__DIR__, "Emery.jl"))

const ACTIVE_TWIST_EMERY_PROJECTOR_DERIVATIVE_PARAM_NAMES =
    Ref{Union{Nothing,Vector{Symbol}}}(nothing)

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
用途: 将完整 mean-field 参数名和值转换为 `TwistEmeryNonPHParams`.

参数:
- `wf_param_names::Vector{Symbol}`: AFM 或 Stripe mean-field 参数名.
- `wf_param_values::Vector{Float64}`: 与名称对齐的参数值.
- `lx, ly::Int`: Cu unit cell 尺寸.
- `bcx, bcy::Float64`: mean-field 边界条件.
- `stripe_wavevector, stripe_center_offset::Float64`: stripe 波矢和中心.

返回:
- `TwistEmeryNonPHParams`: 固定 `chi1_dp_x=1` 的参数对象.
"""
function build_twist_emery_nonph_params_from_wf_params(
    wf_param_names::Vector{Symbol},
    wf_param_values::Vector{Float64},
    lx::Int,
    ly::Int,
    bcx::Float64,
    bcy::Float64;
    stripe_wavevector::Real=0.0,
    stripe_center_offset::Real=0.0,
)::TwistEmeryNonPHParams
    length(wf_param_names) == length(wf_param_values) ||
        error("wf_param_names and wf_param_values length mismatch.")
    supported_names = Set([
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
    ])
    for param_name in wf_param_names
        param_name in supported_names ||
            error("Unsupported twist Emery mean-field parameter $(param_name).")
    end
    length(unique(wf_param_names)) == length(wf_param_names) ||
        error("Duplicate twist Emery mean-field parameter names.")
    param_map = Dict{Symbol,Float64}(zip(wf_param_names, wf_param_values))
    return TwistEmeryNonPHParams(
        lx=lx,
        ly=ly,
        bcx=bcx,
        bcy=bcy,
        chi1_dp_x=1.0,
        chi1_dp_y=get(param_map, :chi1_dp_y, 0.0),
        chi1_pp=get(param_map, :chi1_pp, 0.0),
        chi1_dd=get(param_map, :chi1_dd, 0.0),
        mu_px=get(param_map, :mu_px, 0.0),
        mu_py=get(param_map, :mu_py, 0.0),
        delta_af_d=get(param_map, :Delta_AF_d, 0.0),
        delta_c_d=get(param_map, :Delta_c_d, 0.0),
        delta_c_px=get(param_map, :Delta_c_px, 0.0),
        delta_c_py=get(param_map, :Delta_c_py, 0.0),
        delta_s_d=get(param_map, :Delta_s_d, 0.0),
        stripe_wavevector=Float64(stripe_wavevector),
        stripe_center_offset=Float64(stripe_center_offset),
    )
end

"""
用途: 用完整参数更新 twist Emery determinant、mean-field 导数和 projector.

参数:
- `vwf`: 无 backflow determinant 波函数.
- `param_names::Vector{Symbol}`, `params::Vector{Float64}`: 完整 mean-field 后接 projector 参数.
- `lx, ly::Int`: Cu unit cell 尺寸.
- `bcx, bcy::Float64`: mean-field 边界条件.
- `n_occupied_orbitals::Int`: determinant 占据轨道数.
- `nparams_proj::Int`: 完整向量末尾的 projector 参数数量.
- `stripe_wavevector, stripe_center_offset::Float64`: Stripe 几何.
- `active_wf_param_names`: 需要构造导数的 mean-field 子集, `nothing` 表示全部.
- `dense_derivative_workspace`: MPI dense derivative 共享 workspace 或 `nothing`.

返回:
- `nothing`: 原地更新 `vwf` 并重新初始化 determinant.
"""
function update_twist_emery_ansatz!(
    vwf,
    param_names::Vector{Symbol},
    params::Vector{Float64},
    lx::Int,
    ly::Int,
    bcx::Float64,
    bcy::Float64,
    n_occupied_orbitals::Int;
    nparams_proj::Int=0,
    stripe_wavevector::Real=0.0,
    stripe_center_offset::Real=0.0,
    active_wf_param_names::Union{Nothing,Vector{Symbol}}=nothing,
    dense_derivative_workspace=nothing,
)::Nothing
    length(param_names) == length(params) ||
        error("param_names and params length mismatch.")
    0 <= nparams_proj <= length(param_names) ||
        error("nparams_proj is outside valid range.")
    mfVMC.Backflow.uses_backflow(vwf.backflow) &&
        error("twist_Emery.jl first version does not support backflow.")

    nparams_wf = length(param_names) - nparams_proj
    wf_param_names = param_names[1:nparams_wf]
    wf_param_values = params[1:nparams_wf]
    projector_param_names =
        nparams_proj == 0 ? Symbol[] : param_names[(nparams_wf+1):end]
    projector_param_values =
        nparams_proj == 0 ? Float64[] : params[(nparams_wf+1):end]
    derivative_wf_param_names =
        active_wf_param_names === nothing ? wf_param_names : active_wf_param_names
    wf_param_name_set = Set(wf_param_names)
    for param_name in derivative_wf_param_names
        param_name in wf_param_name_set ||
            error("Active mean-field parameter $(param_name) is not in the full list.")
    end

    nonph_params = build_twist_emery_nonph_params_from_wf_params(
        wf_param_names,
        wf_param_values,
        lx,
        ly,
        bcx,
        bcy;
        stripe_wavevector=stripe_wavevector,
        stripe_center_offset=stripe_center_offset,
    )
    n_spinful = 2 * twist_emery_n_sites(lx, ly)
    n_occupied_orbitals <= n_spinful ||
        error("n_occupied_orbitals exceeds the spinful one-body dimension.")

    dense_ansatz = if isempty(derivative_wf_param_names)
        dense_derivative_workspace === nothing ||
            error("Empty active mean-field list requires dense_derivative_workspace=nothing.")
        eigenvectors = real.(eigen(build_twist_emery_nonph_hamiltonian(nonph_params)).vectors)
        (
            U_occ=copy(eigenvectors[:, 1:n_occupied_orbitals]),
            dUt_tensor=zeros(
                Float64,
                n_occupied_orbitals,
                n_spinful,
                0,
            ),
        )
    else
        make_column_emery_dense_tensor_shared(
            nonph_params;
            param_names=derivative_wf_param_names,
            n_occupied_orbitals=n_occupied_orbitals,
            workspace=dense_derivative_workspace,
            nspin=n_spinful,
            build_hamiltonian=params_ref ->
                Matrix(build_twist_emery_nonph_hamiltonian(params_ref)),
            build_dh_dparam=(params_ref, param_name) ->
                Matrix(build_twist_emery_nonph_dh_dparam(params_ref, param_name)),
        )
    end

    copyto!(vwf.base_gs_U, dense_ansatz.U_occ)
    copyto!(vwf.gs_U, dense_ansatz.U_occ)
    copyto!(vwf.gs_U_t, permutedims(dense_ansatz.U_occ))
    update_vwf_params!(vwf, derivative_wf_param_names, dense_ansatz.dUt_tensor)
    if nparams_proj > 0
        update_vwf_projector_params!(
            vwf,
            projector_param_names,
            projector_param_values,
        )
    end
    init_gswf!(vwf)
    return nothing
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
用途: 解析 twist Emery 固定参数字符串.

参数:
- `fixed_params_string::AbstractString`: 形如 `"mu_px=2.5,g_d=0.7"` 的逗号分隔字符串.

返回:
- `Dict{Symbol, Float64}`: 参数名到固定值的映射.
"""
function parse_twist_emery_fixed_param_string(
    fixed_params_string::AbstractString,
)::Dict{Symbol,Float64}
    fixed_param_values = Dict{Symbol,Float64}()
    isempty(strip(fixed_params_string)) && return fixed_param_values
    for raw_assignment in split(fixed_params_string, ",")
        assignment = strip(raw_assignment)
        parts = split(assignment, "=")
        length(parts) == 2 ||
            error("Invalid fixed parameter assignment: $(assignment). Expected name=value.")
        param_name = Symbol(strip(parts[1]))
        isempty(String(param_name)) && error("Fixed parameter name cannot be empty.")
        haskey(fixed_param_values, param_name) &&
            error("Duplicate fixed parameter: $(param_name).")
        fixed_param_values[param_name] = parse(Float64, strip(parts[2]))
    end
    return fixed_param_values
end

"""
用途: 解析 twist Emery active 参数名字符串.

参数:
- `param_names_string::AbstractString`: 形如 `"Delta_c_d,vj_oo"` 的逗号分隔字符串.

返回:
- `Vector{Symbol}`: active 参数名列表, 空字符串返回空列表.
"""
function parse_twist_emery_param_name_list(
    param_names_string::AbstractString,
)::Vector{Symbol}
    isempty(strip(param_names_string)) && return Symbol[]
    param_names = [Symbol(strip(raw_name)) for raw_name in split(param_names_string, ",")]
    any(name -> isempty(String(name)), param_names) &&
        error("Active parameter list contains an empty parameter name.")
    length(unique(param_names)) == length(param_names) ||
        error("Active parameter list contains duplicate names: $(param_names).")
    return param_names
end

"""
用途: 从 JSON 文件读取 twist Emery 初始参数, 缺失字段保留当前默认值.

参数:
- `json_path::AbstractString`: JSON 文件路径.
- `param_names::Vector{Symbol}`: 完整参数顺序.
- `default_params::Vector{Float64}`: 对应默认参数值.

返回:
- `Vector{Float64}`: 按 `param_names` 排列的初始值.
"""
function build_twist_emery_init_params_from_json_with_defaults(
    json_path::AbstractString,
    param_names::Vector{Symbol},
    default_params::Vector{Float64},
)::Vector{Float64}
    isfile(json_path) || error("JSON file not found: $(json_path)")
    length(param_names) == length(default_params) ||
        error("param_names and default_params length mismatch.")
    raw_dict = JSON.parsefile(json_path)
    return [
        haskey(raw_dict, String(param_name)) ?
        Float64(raw_dict[String(param_name)]) :
        default_params[index]
        for (index, param_name) in enumerate(param_names)
    ]
end

"""
用途: 将 fixed 参数值应用到完整初始参数向量.

参数:
- `param_names::Vector{Symbol}`: 完整参数名列表.
- `init_params::Vector{Float64}`: 当前初值.
- `fixed_param_values::Dict{Symbol, Float64}`: fixed 参数映射.

返回:
- `Vector{Float64}`: 写入 fixed 值后的副本.
"""
function apply_twist_emery_fixed_params_to_values(
    param_names::Vector{Symbol},
    init_params::Vector{Float64},
    fixed_param_values::Dict{Symbol,Float64},
)::Vector{Float64}
    length(param_names) == length(init_params) ||
        error("param_names and init_params length mismatch.")
    param_index_map = Dict(name => index for (index, name) in enumerate(param_names))
    updated_params = copy(init_params)
    for (param_name, param_value) in fixed_param_values
        haskey(param_index_map, param_name) ||
            error("Unknown fixed parameter $(param_name).")
        updated_params[param_index_map[param_name]] = param_value
    end
    return updated_params
end

"""
用途: 根据 fixed/active 设置确定 SR 实际优化参数下标.

参数:
- `param_names::Vector{Symbol}`: 完整参数名列表.
- `fixed_param_values::Dict{Symbol, Float64}`: fixed 参数映射.
- `requested_active_param_names::Vector{Symbol}`: 显式 active 列表; 空列表表示除 fixed 外全部.

返回:
- `Vector{Int}`: active 参数在完整向量中的 1-based 下标.
"""
function build_twist_emery_active_param_indices(
    param_names::Vector{Symbol},
    fixed_param_values::Dict{Symbol,Float64},
    requested_active_param_names::Vector{Symbol}=Symbol[],
)::Vector{Int}
    param_index_map = Dict(name => index for (index, name) in enumerate(param_names))
    for param_name in keys(fixed_param_values)
        haskey(param_index_map, param_name) ||
            error("Unknown fixed parameter $(param_name).")
    end
    if isempty(requested_active_param_names)
        return [
            index
            for (index, param_name) in enumerate(param_names)
            if !haskey(fixed_param_values, param_name)
        ]
    end

    active_indices = Int[]
    for param_name in requested_active_param_names
        haskey(param_index_map, param_name) ||
            error("Unknown active parameter $(param_name).")
        haskey(fixed_param_values, param_name) &&
            error("Parameter $(param_name) cannot be both fixed and active.")
        push!(active_indices, param_index_map[param_name])
    end
    return active_indices
end

"""
用途: 将 SR active 参数值合并回完整参数模板.

参数:
- `full_param_template::Vector{Float64}`: 完整参数模板.
- `active_param_indices::Vector{Int}`: active 下标.
- `active_param_values::Vector{Float64}`: 当前 active 值.

返回:
- `Vector{Float64}`: 合并后的完整参数向量.
"""
function merge_twist_emery_active_params_into_full(
    full_param_template::Vector{Float64},
    active_param_indices::Vector{Int},
    active_param_values::Vector{Float64},
)::Vector{Float64}
    length(active_param_indices) == length(active_param_values) ||
        error("active_param_indices and active_param_values length mismatch.")
    full_param_values = copy(full_param_template)
    for (active_offset, param_index) in enumerate(active_param_indices)
        full_param_values[param_index] = active_param_values[active_offset]
    end
    return full_param_values
end

"""
用途: 将未参与 SR 的 fixed/inactive 参数补写到最优参数 JSON.

参数:
- `json_path::AbstractString`: `extract_min_energy` 生成的 JSON 文件路径.
- `full_param_names::Vector{Symbol}`: 完整参数名.
- `full_param_values::Vector{Float64}`: 完整参数模板值.
- `active_param_indices::Vector{Int}`: active 参数下标.

返回:
- `nothing`: 文件不存在时不执行写入.
"""
function append_twist_emery_inactive_params_to_json!(
    json_path::AbstractString,
    full_param_names::Vector{Symbol},
    full_param_values::Vector{Float64},
    active_param_indices::Vector{Int},
)::Nothing
    isfile(json_path) || return nothing
    param_dict = JSON.parsefile(json_path)
    active_index_set = Set(active_param_indices)
    for (param_index, param_name) in enumerate(full_param_names)
        if !(param_index in active_index_set)
            param_dict[String(param_name)] = full_param_values[param_index]
        end
    end
    open(json_path, "w") do io
        JSON.print(io, param_dict)
        println(io)
    end
    return nothing
end

"""
用途: 设置 SR 中参与导数计算的 twist Emery projector 参数.

参数:
- `projector_param_names::Vector{Symbol}`: projector 完整参数名.
- `active_projector_param_names::Union{Nothing, Vector{Symbol}}`: active 子集,
  `nothing` 表示全部.

返回:
- `nothing`: 名称重复或不存在时抛出错误.
"""
function set_active_twist_emery_projector_derivative_param_names!(
    projector_param_names::Vector{Symbol};
    active_projector_param_names::Union{Nothing,Vector{Symbol}}=nothing,
)::Nothing
    if active_projector_param_names !== nothing
        length(unique(active_projector_param_names)) == length(active_projector_param_names) ||
            error("Duplicate active projector derivative parameters.")
        available_names = Set(projector_param_names)
        for param_name in active_projector_param_names
            param_name in available_names ||
                error("Unknown active projector derivative parameter $(param_name).")
        end
    end
    ACTIVE_TWIST_EMERY_PROJECTOR_DERIVATIVE_PARAM_NAMES[] =
        active_projector_param_names === nothing ? nothing : copy(active_projector_param_names)
    return nothing
end

"""
用途: 覆盖 twist Emery 无 backflow SR 的 log-derivative, 只保留 active projector 参数.

数学公式:
- determinant 参数使用 `O_p=Tr(A^{-1}*dA/dp)`.
- projector 参数使用 `O_p=d log(P)/dp`.

参数:
- `vwf::mfVMC.VMC.vwf_det{T}`: determinant 波函数.

返回:
- `Vector{T}`: mean-field active 参数后接 projector active 参数的导数向量.
"""
function mfVMC.VMC.compute_grad_log_psi!(vwf::mfVMC.VMC.vwf_det{T}) where T
    mfVMC.Backflow.uses_backflow(vwf.backflow) &&
        error("twist_Emery.jl first version does not support backflow.")
    ws = mfVMC.VMC.ensure_ws!(vwf)
    sampler = vwf.sampler
    wf_param_count = length(vwf.param_keys)
    projector_param_names_all = mfVMC.Projector.projector_param_names(vwf.projector)
    active_projector_param_names =
        ACTIVE_TWIST_EMERY_PROJECTOR_DERIVATIVE_PARAM_NAMES[] === nothing ?
        projector_param_names_all :
        ACTIVE_TWIST_EMERY_PROJECTOR_DERIVATIVE_PARAM_NAMES[]

    resize!(ws.grad_buffer, wf_param_count + length(active_projector_param_names))
    derivative_vector = ws.grad_buffer
    fill!(derivative_vector, zero(T))
    if wf_param_count > 0
        mfVMC.VMC._compute_dense_tensor_gradient!(
            @view(derivative_vector[1:wf_param_count]),
            vwf,
        )
    end

    if !isempty(active_projector_param_names)
        full_derivatives = mfVMC.Projector.projector_log_derivative(vwf.projector, sampler)
        derivative_map = Dict(
            param_name => T(derivative_value)
            for (param_name, derivative_value) in
                zip(projector_param_names_all, full_derivatives)
        )
        for (active_offset, param_name) in enumerate(active_projector_param_names)
            derivative_vector[wf_param_count+active_offset] = derivative_map[param_name]
        end
    end
    return derivative_vector
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
用途: 构造 PBC Emery orbital-resolved Gutzwiller 的 site group 向量.

参数:
- `lx, ly::Int`: Cu unit cell 尺寸.

返回:
- `Vector{Int}`: Cu d orbital 属于 group 1, `p_x/p_y` orbital 属于 group 2.
"""
function twist_emery_orbital_gutzwiller_group_vector(
    lx::Int,
    ly::Int,
)::Vector{Int}
    n_sites = twist_emery_n_sites(lx, ly)
    site_groups = Vector{Int}(undef, n_sites)
    for x in 1:lx, y in 1:ly
        site_groups[twist_emery_xyo_to_site_index(x, y, EMERY_ORB_D, lx, ly)] = 1
        site_groups[twist_emery_xyo_to_site_index(x, y, EMERY_ORB_PX, lx, ly)] = 2
        site_groups[twist_emery_xyo_to_site_index(x, y, EMERY_ORB_PY, lx, ly)] = 2
    end
    return site_groups
end

"""
用途: 将 PBC hopping 代表 bonds 转换为去重后的无向 density pairs.

参数:
- `bonds::Vector{EmeryBond}`: 允许包含相反方向或重复 periodic image 的 bonds.

返回:
- `Vector{Tuple{Int, Int}}`: 按 `(min(i,j),max(i,j))` 规范化、去重且移除 self-loop 的 pairs.
"""
function twist_emery_unique_density_pairs(
    bonds::Vector{EmeryBond},
)::Vector{Tuple{Int,Int}}
    pair_set = Set{Tuple{Int,Int}}()
    for bond in bonds
        bond.i == bond.j && continue
        push!(pair_set, minmax(bond.i, bond.j))
    end
    return sort!(collect(pair_set))
end

"""
用途: 构造 PBC Emery onsite Gutzwiller 与三类最近邻 density Jastrow projector.

数学公式:
- `P_G=exp[-g_d*D_d-g_p*D_p]`.
- `P_J=exp[-vj_oo*sum_<p,p>n_i*n_j-vj_cuo*sum_<d,p>n_i*n_j
  -vj_cucu*sum_<d,d>n_i*n_j]`.

参数:
- `lx, ly::Int`: Cu unit cell 尺寸.
- `g_d, g_p::Real`: Cu d 与共享 oxygen p onsite Gutzwiller 参数.
- `vj_oo, vj_cuo, vj_cucu::Real`: O-O、Cu-O、Cu-Cu 最近邻 Jastrow 参数.

返回:
- `CompositeProjector`: 参数固定顺序为
  `g_d, g_p, vj_oo, vj_cuo, vj_cucu`.
"""
function build_twist_emery_density_jastrow_projector(
    lx::Int,
    ly::Int;
    g_d::Real,
    g_p::Real,
    vj_oo::Real,
    vj_cuo::Real,
    vj_cucu::Real,
)::CompositeProjector
    n_sites = twist_emery_n_sites(lx, ly)
    pd_groups = build_twist_emery_pd_bond_groups(
        lx,
        ly;
        amplitude_x=1.0,
        amplitude_y=1.0,
    )
    dd_groups = build_twist_emery_dd_bond_groups(lx, ly; amplitude=1.0)
    pp_bonds = build_twist_emery_pp_bonds(lx, ly; amplitude=1.0)

    pd_pairs = twist_emery_unique_density_pairs(
        vcat(pd_groups.x_bonds, pd_groups.y_bonds),
    )
    dd_pairs = twist_emery_unique_density_pairs(
        vcat(dd_groups.x_bonds, dd_groups.y_bonds),
    )
    pp_pairs = twist_emery_unique_density_pairs(pp_bonds)

    return CompositeProjector(AbstractProjectorTerm[
        SiteGroupGutzwillerProjectorTerm(
            param_names=[:g_d, :g_p],
            g_values=[Float64(g_d), Float64(g_p)],
            site_groups=twist_emery_orbital_gutzwiller_group_vector(lx, ly),
        ),
        JastrowProjectorTerm(
            param_name=:vj_oo,
            v=Float64(vj_oo),
            site_to_neighbor_sites=build_emery_jastrow_neighbor_table(n_sites, pp_pairs),
        ),
        JastrowProjectorTerm(
            param_name=:vj_cuo,
            v=Float64(vj_cuo),
            site_to_neighbor_sites=build_emery_jastrow_neighbor_table(n_sites, pd_pairs),
        ),
        JastrowProjectorTerm(
            param_name=:vj_cucu,
            v=Float64(vj_cucu),
            site_to_neighbor_sites=build_emery_jastrow_neighbor_table(n_sites, dd_pairs),
        ),
    ])
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

"""
用途: 解析 PBC twist Emery 无 backflow 计算的命令行参数.

参数:
- 无, 直接读取全局 `ARGS`.

返回:
- `Dict{String, Any}`: 包含 lattice、各向异性 physical 参数、AFM/Stripe ansatz、
  projector、SR/measure、fixed/active 与输出目录设置.
"""
function parse_twist_emery_commandline()
    settings = ArgParseSettings()
    @add_arg_table settings begin
        "--Lx"
        arg_type = Int
        default = 8
        "--Ly"
        arg_type = Int
        default = 4
        "--tpd_x"
        arg_type = Float64
        default = 1.0
        "--tpd_y"
        arg_type = Float64
        default = 1.0
        "--tpp"
        arg_type = Float64
        default = 0.0
        "--ep_x"
        arg_type = Float64
        default = 3.0
        "--ep_y"
        arg_type = Float64
        default = 3.0
        "--Udd"
        arg_type = Float64
        default = 8.0
        "--Up"
        arg_type = Float64
        default = 0.0
        "--Vpd_x"
        arg_type = Float64
        default = 0.0
        "--Vpd_y"
        arg_type = Float64
        default = 0.0
        "--Vpp"
        arg_type = Float64
        default = 0.0
        "--bcx"
        arg_type = Float64
        default = 1.0
        "--bcy"
        arg_type = Float64
        default = 1.0
        "--ansatz"
        arg_type = String
        default = "Stripe"
        "--lambda"
        arg_type = Int
        default = 4
        "--stripe_center"
        arg_type = String
        default = "site"
        "--chi1_dp_y"
        arg_type = Float64
        default = NaN
        "--chi1_pp"
        arg_type = Float64
        default = NaN
        "--chi1_dd"
        arg_type = Float64
        default = 0.0
        "--mu_px"
        arg_type = Float64
        default = NaN
        "--mu_py"
        arg_type = Float64
        default = NaN
        "--Delta_AF_d"
        arg_type = Float64
        default = 3.0
        "--Delta_c_d"
        arg_type = Float64
        default = 0.0
        "--Delta_c_px"
        arg_type = Float64
        default = 0.0
        "--Delta_c_py"
        arg_type = Float64
        default = 0.0
        "--Delta_s_d"
        arg_type = Float64
        default = 3.0
        "--g_d"
        arg_type = Float64
        default = 1.0
        "--g_p"
        arg_type = Float64
        default = 1.0
        "--vj_oo"
        arg_type = Float64
        default = 0.0
        "--vj_cuo"
        arg_type = Float64
        default = 0.0
        "--vj_cucu"
        arg_type = Float64
        default = 0.0
        "--target_sz"
        arg_type = Int
        default = 0
        "--doping"
        arg_type = String
        default = "0.125"
        "--nMC"
        arg_type = Int
        default = 10000
        "--wMC"
        arg_type = Int
        default = 100
        "--rMC"
        arg_type = Int
        default = 100
        "--dMC"
        arg_type = Int
        default = 1
        "--seed"
        arg_type = Int
        default = 5423
        "--nSR"
        arg_type = Int
        default = 50
        "--lr"
        arg_type = Float64
        default = 0.04
        "--lr_end"
        arg_type = Float64
        default = NaN
        "--eigen_cutoff"
        arg_type = Float64
        default = 0.0
        "--init_params_json"
        arg_type = String
        default = ""
        "--fixed_params"
        arg_type = String
        default = ""
        "--active_params"
        arg_type = String
        default = ""
        "--job"
        arg_type = String
        default = "SR"
        "--enable_timing"
        arg_type = String
        default = "false"
        "--output_dir"
        arg_type = String
        default = "logs"
    end
    return parse_args(settings)
end

"""
用途: 运行 PBC twist Emery 无 backflow 的 AFM/Stripe SR 或 measure 主流程.

实现约定:
- physical Hamiltonian 在 x/y 方向均为严格 PBC, 不乘 `bcx/bcy`.
- mean-field 跨边界 hopping 分别乘实数 `bcx/bcy`.
- mean-field gauge 固定 `chi1_dp_x=1`, projector 固定包含
  `g_d,g_p,vj_oo,vj_cuo,vj_cucu`.
- `active_params` 只控制 SR 导数和更新维度, `fixed_params` 的值始终保留在完整 ansatz.

参数:
- 无, 所有配置由 `parse_twist_emery_commandline()` 从 `ARGS` 读取.

返回:
- `nothing`: 输出写入 `output_dir`; SR 生成 history/min_params, measure 生成
  分项能量与局域 observable 的 JSON/blocking 文件.
"""
function main_twist_emery()::Nothing
    args = parse_twist_emery_commandline()
    enable_timing = parse_column_bool_flag(
        args["enable_timing"],
        "--enable_timing",
    )
    ENABLE_TIMING[] = enable_timing
    enable_timing && timing_reset!()

    session = init_mpi_session()
    rank = session.rank
    is_root = rank == session.root
    lx = args["Lx"]
    ly = args["Ly"]
    n_sites = twist_emery_n_sites(lx, ly)
    doping = parse_column_doping_value(args["doping"], "--doping")
    electron_count = compute_emery_electron_count(lx, ly, doping)
    target_sz = args["target_sz"]
    (target_sz + electron_count) % 2 == 0 ||
        error("target_sz and electron count must have the same parity.")
    nup = (electron_count + target_sz) ÷ 2
    ndn = electron_count - nup
    if nup < 0 || ndn < 0 || nup > n_sites || ndn > n_sites
        error(
            "Invalid particle numbers: N_up=$(nup), N_down=$(ndn), N_sites=$(n_sites).",
        )
    end

    mean_field_setup = build_twist_emery_mean_field_parameter_setup(args)
    wf_param_names = mean_field_setup.param_names
    wf_init_params = mean_field_setup.param_values
    projector = build_twist_emery_density_jastrow_projector(
        lx,
        ly;
        g_d=args["g_d"],
        g_p=args["g_p"],
        vj_oo=args["vj_oo"],
        vj_cuo=args["vj_cuo"],
        vj_cucu=args["vj_cucu"],
    )
    projector_param_name_list = projector_param_names(projector)
    projector_init_params = projector_param_values(projector)
    nparams_wf = length(wf_param_names)
    nparams_proj = length(projector_param_name_list)
    param_names = vcat(wf_param_names, projector_param_name_list)
    init_params = vcat(wf_init_params, projector_init_params)

    if !isempty(args["init_params_json"])
        init_params = build_twist_emery_init_params_from_json_with_defaults(
            args["init_params_json"],
            param_names,
            init_params,
        )
    end
    fixed_param_values = parse_twist_emery_fixed_param_string(
        args["fixed_params"],
    )
    requested_active_param_names = parse_twist_emery_param_name_list(
        args["active_params"],
    )
    init_params = apply_twist_emery_fixed_params_to_values(
        param_names,
        init_params,
        fixed_param_values,
    )
    active_param_indices = build_twist_emery_active_param_indices(
        param_names,
        fixed_param_values,
        requested_active_param_names,
    )
    uses_param_subset = active_param_indices != collect(eachindex(param_names))
    active_param_names = param_names[active_param_indices]
    active_init_params = init_params[active_param_indices]
    active_wf_param_names = [
        param_names[index]
        for index in active_param_indices
        if index <= nparams_wf
    ]
    active_projector_param_names = [
        param_names[index]
        for index in active_param_indices
        if index > nparams_wf
    ]
    set_active_twist_emery_projector_derivative_param_names!(
        projector_param_name_list;
        active_projector_param_names=uses_param_subset ?
            active_projector_param_names :
            nothing,
    )

    physical_term_groups = build_twist_emery_physical_term_groups(
        lx,
        ly;
        tpd_x=args["tpd_x"],
        tpd_y=args["tpd_y"],
        tpp=args["tpp"],
        ep_x=args["ep_x"],
        ep_y=args["ep_y"],
        Udd=args["Udd"],
        Up=args["Up"],
        Vpd_x=args["Vpd_x"],
        Vpd_y=args["Vpd_y"],
        Vpp=args["Vpp"],
    )
    hamiltonian = build_twist_emery_general_model(
        lx,
        ly,
        physical_term_groups,
    )
    sampler = config_Hubbard(n_sites, nup, ndn; ifPH=false)
    init_config_Hubbard!(sampler)
    vwf = vwf_det(
        zeros(Float64, 2 * n_sites, electron_count),
        sampler,
    )
    set_projector!(vwf, projector)
    kernel = HubbardKernel(conserve_sz=true)

    derivative_wf_param_names = uses_param_subset ?
                                active_wf_param_names :
                                wf_param_names
    dense_derivative_workspace = isempty(derivative_wf_param_names) ?
                                 nothing :
                                 setup_emery_dense_derivative_workspace(
        session,
        electron_count,
        2 * n_sites,
        length(derivative_wf_param_names),
    )
    update_twist_emery_ansatz!(
        vwf,
        param_names,
        init_params,
        lx,
        ly,
        args["bcx"],
        args["bcy"],
        electron_count;
        nparams_proj=nparams_proj,
        stripe_wavevector=mean_field_setup.stripe_wavevector,
        stripe_center_offset=mean_field_setup.stripe_center_offset,
        active_wf_param_names=derivative_wf_param_names,
        dense_derivative_workspace=dense_derivative_workspace,
    )

    output_dir = args["output_dir"]
    mkpath(output_dir)
    measurement_params = VMCParams(
        total_samples=args["nMC"],
        warmup_steps=args["wMC"],
        rebuild_every=args["rMC"],
        decorr_steps=args["dMC"],
        seed=args["seed"] + rank,
    )
    job = lowercase(args["job"])
    if is_root
        println(
            "twist Emery: ansatz=$(args["ansatz"]), Lx=$(lx), Ly=$(ly), " *
            "N_up=$(nup), N_down=$(ndn), output_dir=$(output_dir)",
        )
        println("Full initial parameters: $(Dict(zip(param_names, init_params)))")
        println("Active parameters: $(active_param_names)")
    end

    if job == "sr"
        isempty(active_param_indices) &&
            error("SR requires at least one active parameter.")
        learning_rate = args["lr"]
        final_learning_rate = isnan(args["lr_end"]) ?
                              learning_rate :
                              args["lr_end"]
        sr_params = SRParams(
            vmc_params=measurement_params,
            n_steps=args["nSR"],
            lr=learning_rate,
            eigen_cutoff=args["eigen_cutoff"],
        )
        learning_rate_function = build_exponential_lr_func(
            learning_rate,
            final_learning_rate,
            args["nSR"],
        )
        update_vwf_function! = (wavefunction, active_values) -> begin
            full_values = uses_param_subset ?
                          merge_twist_emery_active_params_into_full(
                init_params,
                active_param_indices,
                active_values,
            ) :
                          active_values
            update_twist_emery_ansatz!(
                wavefunction,
                param_names,
                full_values,
                lx,
                ly,
                args["bcx"],
                args["bcy"],
                electron_count;
                nparams_proj=nparams_proj,
                stripe_wavevector=mean_field_setup.stripe_wavevector,
                stripe_center_offset=mean_field_setup.stripe_center_offset,
                active_wf_param_names=derivative_wf_param_names,
                dense_derivative_workspace=dense_derivative_workspace,
            )
        end
        run_sr_optimization(
            hamiltonian,
            vwf,
            kernel,
            active_init_params,
            update_vwf_function!,
            sr_params;
            log_file=joinpath(output_dir, "sr_history.txt"),
            param_names=active_param_names,
            lr_func=learning_rate_function,
        )
        if is_root
            extract_min_energy(joinpath(output_dir, "sr_history.txt"))
            append_twist_emery_inactive_params_to_json!(
                joinpath(output_dir, "min_params.json"),
                param_names,
                init_params,
                active_param_indices,
            )
        end
    elseif job == "measure"
        results = run_simulation(
            hamiltonian,
            vwf,
            kernel,
            build_twist_emery_observables(lx, ly, physical_term_groups),
            measurement_params;
            history_observables=build_twist_emery_history_observables(),
        )
        if is_root && results !== nothing
            write_column_measure_outputs(output_dir, results)
        end
    else
        error("job must be SR or measure, got $(args["job"]).")
    end

    if enable_timing && is_root
        timing_report()
        open(joinpath(output_dir, "timing_report.txt"), "w") do io
            timing_report(io)
        end
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_twist_emery()
end
