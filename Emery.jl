#This file is used for OBC Emery model VMC for stripe
using Random
using Printf
using DelimitedFiles
using LinearAlgebra
using Statistics
using ArgParse
using JSON

push!(LOAD_PATH, joinpath(@__DIR__, "src"))
push!(LOAD_PATH, @__DIR__)

using mfVMC
import mfVMC.Timing: @timed, timing_reset!, timing_report
using Utils: add_term_ij_nonPH, compute_eig_and_dU_reg1

const EMERY_ORB_D = 1
const EMERY_ORB_PY = 2
const EMERY_ORB_PX = 3

"""
用途: 保存 Emery 三带模型中一条有向 hopping bond。

字段:
- `i::Int`: 起点 site index, 1-based。
- `j::Int`: 终点 site index, 1-based。
- `coef::Float64`: Hamiltonian 矩阵元 `H[i,j]` 的系数。
"""
struct EmeryBond
    i::Int
    j::Int
    coef::Float64
end

struct ColumnEmeryNonPHParams
    lx::Int
    ly::Int
    bcx::Float64
    bcy::Float64
    chi1_dd::Float64
    chi1_dp::Float64
    chi1_pp::Float64
    mud::Dict{Symbol,Float64}
    mupx::Dict{Symbol,Float64}
    mupy::Dict{Symbol,Float64}
    mzd::Dict{Symbol,Float64}
    mzpx::Dict{Symbol,Float64}
    mzpy::Dict{Symbol,Float64}
end

"""
用途: 构造 column-resolved nonPH Emery mean-field 参数对象。

参数:
- `lx, ly::Int`: Cu 晶胞在 x/y 方向的数量。
- `bcx, bcy::Float64`: 边界条件因子, 当前 x 方向固定 OBC, `bcx` 仅保留为兼容字段。
- `chi1_dd, chi1_dp, chi1_pp::Float64`: Cu-Cu, Cu-O, O-O hopping 参数。
- `mud, mzd::Dict{Symbol, Float64}`: Cu d 轨道按列保存的 `mud_x`, `mzd_x`。
- `mupx, mzpx::Dict{Symbol, Float64}`: O p_x 轨道按列保存的 `mupx_x`, `mzpx_x`, 其中 x 可为 `0:lx`。
- `mupy, mzpy::Dict{Symbol, Float64}`: O p_y 轨道按列保存的 `mupy_x`, `mzpy_x`, 其中 x 为 `1:lx`。

返回:
- `ColumnEmeryNonPHParams`: column-resolved nonPH Emery 参数。
"""
function ColumnEmeryNonPHParams(;
    lx::Int,
    ly::Int,
    bcx::Float64=1.0,
    bcy::Float64=1.0,
    chi1_dd::Float64=0.0,
    chi1_dp::Float64=0.0,
    chi1_pp::Float64=0.0,
    mud::Dict{Symbol,Float64}=Dict{Symbol,Float64}(),
    mupx::Dict{Symbol,Float64}=Dict{Symbol,Float64}(),
    mupy::Dict{Symbol,Float64}=Dict{Symbol,Float64}(),
    mzd::Dict{Symbol,Float64}=Dict{Symbol,Float64}(),
    mzpx::Dict{Symbol,Float64}=Dict{Symbol,Float64}(),
    mzpy::Dict{Symbol,Float64}=Dict{Symbol,Float64}(),
)::ColumnEmeryNonPHParams
    lx > 0 || error("lx must be positive, got $(lx).")
    ly > 0 || error("ly must be positive, got $(ly).")
    return ColumnEmeryNonPHParams(
        lx,
        ly,
        bcx,
        bcy,
        chi1_dd,
        chi1_dp,
        chi1_pp,
        mud,
        mupx,
        mupy,
        mzd,
        mzpx,
        mzpy,
    )
end

"""
用途: 返回 x 方向 OBC, y 方向 PBC 的 Emery 三带模型总 site 数。

参数:
- `lx::Int`: Cu 晶胞 x 方向数量。
- `ly::Int`: Cu 晶胞 y 方向数量。

返回:
- `Int`: 总 site 数, 公式为 `3 * lx * ly + ly`, 额外 `ly` 来自左边界 `p_x(0,y)`。
"""
function emery_n_sites(lx::Int, ly::Int)::Int
    lx > 0 || error("lx must be positive, got $(lx).")
    ly > 0 || error("ly must be positive, got $(ly).")
    return 3 * lx * ly + ly
end

"""
用途: 将空间 site index 和 spin block 编号映射到 Hamiltonian 中的 spinful 行列编号。

参数:
- `site::Int`: 空间 site index, 1-based。
- `spin_offset::Int`: spin block 编号, `1=up`, `2=down`。

返回:
- `Int`: spinful Hamiltonian 中的 1-based 行列编号。
"""
function emery_spin_index(site::Int, spin_offset::Int)::Int
    spin_offset == 1 || spin_offset == 2 || error("spin_offset must be 1 or 2.")
    return 2 * (site - 1) + spin_offset
end

"""
用途: 返回 Emery site 对应的 orbital 编号。

参数:
- `site::Int`: 1-based site index。
- `lx, ly::Int`: Cu 晶胞尺寸。

返回:
- `Int`: orbital 编号, `1=d`, `2=p_y`, `3=p_x`。
"""
function emery_orbital_of_site(site::Int, lx::Int, ly::Int)::Int
    1 <= site <= emery_n_sites(lx, ly) || error("site=$(site) is outside 1:$(emery_n_sites(lx, ly)).")
    if site <= ly
        return EMERY_ORB_PX
    end
    return mod(site - ly - 1, 3) + 1
end

"""
用途: 将 Emery 三带坐标 `(x,y,o)` 映射到一维 site index。

参数:
- `x::Int`: 轨道所在 x 列。`d/p_y` 必须在 `1:lx`, `p_x` 可在 `0:lx`。
- `y::Int`: y 坐标, 按周期边界映射到 `1:ly`。
- `o::Int`: orbital 编号, `1=d`, `2=p_y`, `3=p_x`。
- `lx, ly::Int`: Cu 晶胞尺寸。

返回:
- `Int`: 1-based site index。`p_x(0,y)` 占据 `1:ly`, 其余标准 cell 整体后移 `ly`。
"""
function Emery_xyo_to_site_index(x::Int, y::Int, o::Int, lx::Int, ly::Int)::Int
    lx > 0 || error("lx must be positive, got $(lx).")
    ly > 0 || error("ly must be positive, got $(ly).")
    y_periodic = mod1(y, ly)
    if o == 3 && x == 0
        return y_periodic
    elseif o == EMERY_ORB_D || o == EMERY_ORB_PY || o == EMERY_ORB_PX
        1 <= x <= lx || error("x=$(x) is outside 1:$(lx) for orbital $(o).")
        site_index = y_periodic + (x - 1) * ly
        return (site_index - 1) * 3 + o + ly
    end
    error("Unknown Emery orbital $(o). Expected 1=d, 2=p_y, 3=p_x.")
end

"""
用途: 向 spinful Hamiltonian 显式加入 Hermitian hopping 项。

参数:
- `hamiltonian::AbstractMatrix`: 维度为 `2N_sites x 2N_sites` 的 spinful Hamiltonian。
- `site_i, site_j::Int`: 空间 site index。
- `coef::Float64`: hopping 矩阵元。

返回:
- `AbstractMatrix`: 原地修改后的 Hamiltonian。
"""
function add_emery_spinful_hopping!(
    hamiltonian::AbstractMatrix,
    site_i::Int,
    site_j::Int,
    coef::Float64,
)
    site_i != site_j || error("Emery hopping cannot connect a site to itself.")
    for spin_offset in (1, 2)
        row = emery_spin_index(site_i, spin_offset)
        col = emery_spin_index(site_j, spin_offset)
        hamiltonian[row, col] += coef
        hamiltonian[col, row] += coef
    end
    return hamiltonian
end

"""
用途: 构造 Emery 模型的 Cu-O hopping bond 列表。

参数:
- `lx, ly::Int`: Cu 晶胞尺寸。
- `amplitude::Real`: Cu-O hopping 振幅。
- `bcy::Real`: y 方向周期边界因子。

返回:
- `Vector{EmeryBond}`: 有向代表 bond, 系数采用三带 Hubbard 标准符号。
"""
function build_emery_pd_bonds(lx::Int, ly::Int; amplitude::Real, bcy::Real=1.0)::Vector{EmeryBond}
    t = Float64(amplitude)
    bonds = EmeryBond[]
    sizehint!(bonds, 4 * lx * ly)
    for x in 1:lx, y in 1:ly
        d_site = Emery_xyo_to_site_index(x, y, EMERY_ORB_D, lx, ly)
        py_site = Emery_xyo_to_site_index(x, y, EMERY_ORB_PY, lx, ly)
        px_site = Emery_xyo_to_site_index(x, y, EMERY_ORB_PX, lx, ly)
        py_boundary = y == ly ? Float64(bcy) : 1.0

        push!(bonds, EmeryBond(d_site, py_site, t))
        push!(bonds, EmeryBond(d_site, px_site, -t))
        push!(bonds, EmeryBond(py_site, Emery_xyo_to_site_index(x, y + 1, EMERY_ORB_D, lx, ly), -t * py_boundary))
        if x < lx
            push!(bonds, EmeryBond(px_site, Emery_xyo_to_site_index(x + 1, y, EMERY_ORB_D, lx, ly), t))
        end
        if x == 1
            left_px_site = Emery_xyo_to_site_index(0, y, EMERY_ORB_PX, lx, ly)
            push!(bonds, EmeryBond(left_px_site, d_site, t))
        end
    end
    return bonds
end

"""
用途: 构造 Emery 模型的 O-O hopping bond 列表。

参数:
- `lx, ly::Int`: Cu 晶胞尺寸。
- `amplitude::Real`: O-O hopping 振幅。
- `bcy::Real`: y 方向周期边界因子。

返回:
- `Vector{EmeryBond}`: 有向代表 bond, 系数采用三带 Hubbard 标准符号。
"""
function build_emery_pp_bonds(lx::Int, ly::Int; amplitude::Real, bcy::Real=1.0)::Vector{EmeryBond}
    t = Float64(amplitude)
    bonds = EmeryBond[]
    sizehint!(bonds, 4 * lx * ly)
    for x in 1:lx, y in 1:ly
        py_site = Emery_xyo_to_site_index(x, y, EMERY_ORB_PY, lx, ly)
        px_site = Emery_xyo_to_site_index(x, y, EMERY_ORB_PX, lx, ly)
        py_to_px_y_boundary = y == ly ? Float64(bcy) : 1.0
        px_to_py_y_minus_boundary = y == 1 ? Float64(bcy) : 1.0

        push!(bonds, EmeryBond(py_site, px_site, t))
        push!(bonds, EmeryBond(py_site, Emery_xyo_to_site_index(x, y + 1, EMERY_ORB_PX, lx, ly), -t * py_to_px_y_boundary))
        if x < lx
            push!(bonds, EmeryBond(px_site, Emery_xyo_to_site_index(x + 1, y, EMERY_ORB_PY, lx, ly), -t))
            push!(bonds, EmeryBond(px_site, Emery_xyo_to_site_index(x + 1, y - 1, EMERY_ORB_PY, lx, ly), t * px_to_py_y_minus_boundary))
        end
        if x == 1
            left_px_site = Emery_xyo_to_site_index(0, y, EMERY_ORB_PX, lx, ly)
            push!(bonds, EmeryBond(left_px_site, py_site, -t))
            push!(bonds, EmeryBond(left_px_site, Emery_xyo_to_site_index(1, y - 1, EMERY_ORB_PY, lx, ly), t * px_to_py_y_minus_boundary))
        end
    end
    return bonds
end

"""
用途: 构造 Emery mean-field 中可选的 Cu-Cu hopping bond 列表。

参数:
- `lx, ly::Int`: Cu 晶胞尺寸。
- `amplitude::Real`: Cu-Cu hopping 振幅。
- `bcy::Real`: y 方向周期边界因子。

返回:
- `Vector{EmeryBond}`: 有向代表 bond, x 方向 OBC, y 方向 PBC。
"""
function build_emery_dd_bonds(lx::Int, ly::Int; amplitude::Real, bcy::Real=1.0)::Vector{EmeryBond}
    t = Float64(amplitude)
    bonds = EmeryBond[]
    sizehint!(bonds, (2 * lx - 1) * ly)
    for x in 1:lx, y in 1:ly
        d_site = Emery_xyo_to_site_index(x, y, EMERY_ORB_D, lx, ly)
        y_boundary = y == ly ? Float64(bcy) : 1.0
        push!(bonds, EmeryBond(d_site, Emery_xyo_to_site_index(x, y + 1, EMERY_ORB_D, lx, ly), t * y_boundary))
        if x < lx
            push!(bonds, EmeryBond(d_site, Emery_xyo_to_site_index(x + 1, y, EMERY_ORB_D, lx, ly), t))
        end
    end
    return bonds
end

"""
用途: 向 `GeneralModel` 项列表加入 spin up/down 的 hopping 及其 Hermitian conjugate。

参数:
- `terms::Vector{OperatorTerm}`: 待写入的 Hamiltonian 项。
- `site_i, site_j::Int`: hopping 两端 site。
- `coef::Float64`: `c_i^dagger c_j + h.c.` 的系数。

返回:
- `nothing`。
"""
function add_emery_general_model_hopping_terms!(
    terms::Vector{OperatorTerm},
    site_i::Int,
    site_j::Int,
    coef::Float64,
)::Nothing
    for (source_site, target_site) in ((site_i, site_j), (site_j, site_i))
        push!(terms, OperatorTerm([:cdag_up, :c_up], [source_site, target_site], coef))
        push!(terms, OperatorTerm([:cdag_dn, :c_dn], [source_site, target_site], coef))
    end
    return nothing
end

"""
用途: 构造 Emery 模型 Cu-O density-density interaction 使用的 bond 列表。

参数:
- `lx, ly::Int`: Cu 晶胞尺寸。

返回:
- `Vector{Tuple{Int, Int}}`: 去重后的 Cu-O site pair, 每条 bond 只出现一次。
"""
function build_emery_pd_density_bonds(lx::Int, ly::Int)::Vector{Tuple{Int,Int}}
    bond_set = Set{Tuple{Int,Int}}()
    for bond in build_emery_pd_bonds(lx, ly; amplitude=1.0)
        sorted_pair = bond.i < bond.j ? (bond.i, bond.j) : (bond.j, bond.i)
        push!(bond_set, sorted_pair)
    end
    return sort(collect(bond_set))
end

"""
用途: 构造 Emery 模型 O-O density-density interaction 使用的 bond 列表。

参数:
- `lx, ly::Int`: Cu 晶胞尺寸。

返回:
- `Vector{Tuple{Int, Int}}`: 去重后的 O-O site pair, 每条 bond 只出现一次。
"""
function build_emery_pp_density_bonds(lx::Int, ly::Int)::Vector{Tuple{Int,Int}}
    bond_set = Set{Tuple{Int,Int}}()
    for bond in build_emery_pp_bonds(lx, ly; amplitude=1.0)
        sorted_pair = bond.i < bond.j ? (bond.i, bond.j) : (bond.j, bond.i)
        push!(bond_set, sorted_pair)
    end
    return sort(collect(bond_set))
end

"""
用途: 构造 x 方向 OBC, y 方向 PBC 的 Emery 三带物理 Hamiltonian。

参数:
- `lx, ly::Int`: Cu 晶胞尺寸。
- `tpd, tpp::Real`: Cu-O 和 O-O hopping。
- `Delta_pd::Real`: O 轨道 onsite energy。
- `Udd, Up::Real`: Cu 和 O 轨道 onsite Hubbard U。
- `Vpd::Real`: Cu-O density-density interaction。
- `Vpp::Real`: O-O density-density interaction。

返回:
- `GeneralModel`: 主 VMC 使用的物理 Hamiltonian。
"""
function build_emery_general_model(
    lx::Int,
    ly::Int;
    tpd::Real,
    tpp::Real,
    Delta_pd::Real,
    Udd::Real,
    Up::Real,
    Vpd::Real,
    Vpp::Real=0.0,
)::GeneralModel
    terms = OperatorTerm[]
    n_sites = emery_n_sites(lx, ly)

    for site in 1:n_sites
        orbital = emery_orbital_of_site(site, lx, ly)
        if orbital == EMERY_ORB_D
            push!(terms, OperatorTerm([:n_up, :n_dn], [site, site], Float64(Udd)))
        else
            push!(terms, OperatorTerm([:n], [site], Float64(Delta_pd)))
            Up == 0 || push!(terms, OperatorTerm([:n_up, :n_dn], [site, site], Float64(Up)))
        end
    end

    for bond in build_emery_pd_bonds(lx, ly; amplitude=Float64(tpd))
        add_emery_general_model_hopping_terms!(terms, bond.i, bond.j, -bond.coef)
    end
    for bond in build_emery_pp_bonds(lx, ly; amplitude=Float64(tpp))
        tpp == 0 || add_emery_general_model_hopping_terms!(terms, bond.i, bond.j, -bond.coef)
    end
    for (site_i, site_j) in build_emery_pd_density_bonds(lx, ly)
        Vpd == 0 || push!(terms, OperatorTerm([:n, :n], [site_i, site_j], Float64(Vpd)))
    end
    for (site_i, site_j) in build_emery_pp_density_bonds(lx, ly)
        Vpp == 0 || push!(terms, OperatorTerm([:n, :n], [site_i, site_j], Float64(Vpp)))
    end

    return GeneralModel(n_sites, terms)
end

"""
用途: 按 Cu 晶胞数量和 hole doping 计算 nonPH Emery 模型电子/空穴数。

参数:
- `lx, ly::Int`: Cu 晶胞尺寸。
- `doping::Float64`: 相对 Cu 数定义的 doping。

返回:
- `Int`: `round(Int, lx * ly * (1 + doping))`, 若不是整数则报错。
"""
function compute_emery_electron_count(lx::Int, ly::Int, doping::Float64)::Int
    electron_count_float = lx * ly * (1.0 + doping)
    electron_count = round(Int, electron_count_float)
    if !isapprox(electron_count_float, electron_count; atol=1e-8, rtol=0.0)
        error("Lx * Ly * (1 + doping) must be an integer, got $(electron_count_float).")
    end
    return electron_count
end

"""
用途: 计算单个 sampler site 的总占据数。

参数:
- `site_state::Int8`: sampler 中单个 site 的局域状态。

返回:
- `Float64`: `n_up + n_down`, 取值为 `0.0, 1.0, 2.0`。
"""
function emery_site_occupation(site_state::Int8)::Float64
    return Float64(has_up(site_state)) + Float64(has_dn(site_state))
end

"""
用途: 为单个 Emery site 构造密度和自旋观测量。

参数:
- `observables::Dict{Symbol, Function}`: 待写入的观测量字典。
- `name_prefix::AbstractString`: key 前缀, 如 `d_1_1`, `px_0_1`。
- `site::Int`: 对应的 site index。

返回:
- `nothing`。
"""
function add_emery_site_observables!(
    observables::Dict{Symbol,Function},
    name_prefix::AbstractString,
    site::Int,
)::Nothing
    density_key = Symbol("n_$(name_prefix)")
    spin_key = Symbol("Sz_$(name_prefix)")
    observables[density_key] = (model, vwf) -> emery_site_occupation(vwf.sampler.state[site])
    observables[spin_key] = (model, vwf) -> get_Sz(vwf.sampler.state[site])
    return nothing
end

"""
用途: 构造 Emery stripe measure 使用的 orbital-resolved observables。

参数:
- `lx, ly::Int`: Cu 晶胞尺寸。

返回:
- `Dict{Symbol, Function}`: 包含 `:E`, 以及所有 `d/py/px` 轨道的 `n` 和 `Sz`。
"""
function build_emery_observables(lx::Int, ly::Int)::Dict{Symbol,Function}
    observables = Dict{Symbol,Function}()
    observables[:E] = local_energy

    for y in 1:ly
        site = Emery_xyo_to_site_index(0, y, EMERY_ORB_PX, lx, ly)
        add_emery_site_observables!(observables, "px_0_$(y)", site)
    end
    for x in 1:lx, y in 1:ly
        d_site = Emery_xyo_to_site_index(x, y, EMERY_ORB_D, lx, ly)
        py_site = Emery_xyo_to_site_index(x, y, EMERY_ORB_PY, lx, ly)
        px_site = Emery_xyo_to_site_index(x, y, EMERY_ORB_PX, lx, ly)
        add_emery_site_observables!(observables, "d_$(x)_$(y)", d_site)
        add_emery_site_observables!(observables, "py_$(x)_$(y)", py_site)
        add_emery_site_observables!(observables, "px_$(x)_$(y)", px_site)
    end

    return observables
end

"""
用途: 构造 Emery orbital-resolved Gutzwiller 的 site group 向量。

参数:
- `lx, ly::Int`: Cu 晶胞尺寸。

返回:
- `Vector{Int}`: `site_groups[site] = 1` 表示 Cu d 轨道, `= 2` 表示 O p_x/p_y 轨道。
"""
function emery_orbital_gutzwiller_group_vector(lx::Int, ly::Int)::Vector{Int}
    site_groups = Vector{Int}(undef, emery_n_sites(lx, ly))
    for site in eachindex(site_groups)
        orbital = emery_orbital_of_site(site, lx, ly)
        site_groups[site] = orbital == EMERY_ORB_D ? 1 : 2
    end
    return site_groups
end

"""
用途: 构造 Emery 最小 orbital-resolved onsite Gutzwiller projector。

参数:
- `lx, ly::Int`: Cu 晶胞尺寸。
- `g_d::Real`: Cu d 轨道 doublon penalty 初值。
- `g_p::Real`: O p_x/p_y 轨道共享 doublon penalty 初值。

返回:
- `CompositeProjector`: 包含一个 `SiteGroupGutzwillerProjectorTerm`。
"""
function build_emery_orbital_gutzwiller_projector(
    lx::Int,
    ly::Int;
    g_d::Real,
    g_p::Real,
)::CompositeProjector
    return CompositeProjector([
        SiteGroupGutzwillerProjectorTerm(
            param_names=Symbol[:g_d, :g_p],
            g_values=Float64[g_d, g_p],
            site_groups=emery_orbital_gutzwiller_group_vector(lx, ly),
        ),
    ])
end

"""
用途: 给单个 Emery site 添加 onsite chemical potential 和 staggered magnetic field。

参数:
- `hamiltonian::AbstractMatrix`: 维度为 `2N_sites x 2N_sites` 的 spinful Hamiltonian。
- `site::Int`: 空间 site index。
- `mu_value::Float64`: chemical potential onsite 项。
- `mz_value::Float64`: staggered magnetic field 项。
- `staggered_sign::Float64`: staggered sign, 通常为 `(-1)^(x+y)`。

返回:
- `AbstractMatrix`: 原地修改后的 Hamiltonian。
"""
function add_emery_onsite!(
    hamiltonian::AbstractMatrix,
    site::Int,
    mu_value::Float64,
    mz_value::Float64,
    staggered_sign::Float64,
)
    hamiltonian[emery_spin_index(site, 1), emery_spin_index(site, 1)] += mu_value + staggered_sign * mz_value
    hamiltonian[emery_spin_index(site, 2), emery_spin_index(site, 2)] += mu_value - staggered_sign * mz_value
    return hamiltonian
end

"""
用途: 构造 column-resolved nonPH Emery number-conserving mean-field Hamiltonian。

数学公式:
- hopping: 对每条 `EmeryBond(i,j,coef)`, 加入 `H_{ij} += -coef` 和 `H_{ji} += -coef`,
  使 `chi1_dp=tpd`, `chi1_pp=tpp` 时 mean-field one-body 符号与物理 Hamiltonian 对齐。
- onsite field:
  `H_{i up,i up} += mu_i + (-1)^(x+y) mz_i`,
  `H_{i down,i down} += mu_i - (-1)^(x+y) mz_i`。

参数:
- `params::ColumnEmeryNonPHParams`: column-resolved nonPH Emery mean-field 参数。

返回:
- `Hermitian{Float64, Matrix{Float64}}`: 维度为 `2N_sites x 2N_sites` 的 Hamiltonian。
"""
function build_column_emery_nonph_hamiltonian(
    params::ColumnEmeryNonPHParams,
)
    lx = params.lx
    ly = params.ly
    n_sites = emery_n_sites(lx, ly)
    hamiltonian = zeros(Float64, 2 * n_sites, 2 * n_sites)

    for bond in build_emery_dd_bonds(lx, ly; amplitude=params.chi1_dd, bcy=params.bcy)
        add_emery_spinful_hopping!(hamiltonian, bond.i, bond.j, -bond.coef)
    end
    for bond in build_emery_pd_bonds(lx, ly; amplitude=params.chi1_dp, bcy=params.bcy)
        add_emery_spinful_hopping!(hamiltonian, bond.i, bond.j, -bond.coef)
    end
    for bond in build_emery_pp_bonds(lx, ly; amplitude=params.chi1_pp, bcy=params.bcy)
        add_emery_spinful_hopping!(hamiltonian, bond.i, bond.j, -bond.coef)
    end

    for x in 1:lx, y in 1:ly
        staggered_sign = Float64((-1)^(x + y))
        d_site = Emery_xyo_to_site_index(x, y, EMERY_ORB_D, lx, ly)
        py_site = Emery_xyo_to_site_index(x, y, EMERY_ORB_PY, lx, ly)
        add_emery_onsite!(
            hamiltonian,
            d_site,
            get(params.mud, Symbol("mud_$(x)"), 0.0),
            get(params.mzd, Symbol("mzd_$(x)"), 0.0),
            staggered_sign,
        )
        add_emery_onsite!(
            hamiltonian,
            py_site,
            get(params.mupy, Symbol("mupy_$(x)"), 0.0),
            get(params.mzpy, Symbol("mzpy_$(x)"), 0.0),
            staggered_sign,
        )
    end
    for x in 0:lx, y in 1:ly
        staggered_sign = Float64((-1)^(x + y))
        px_site = Emery_xyo_to_site_index(x, y, EMERY_ORB_PX, lx, ly)
        add_emery_onsite!(
            hamiltonian,
            px_site,
            get(params.mupx, Symbol("mupx_$(x)"), 0.0),
            get(params.mzpx, Symbol("mzpx_$(x)"), 0.0),
            staggered_sign,
        )
    end

    return Hermitian(hamiltonian)
end

"""
用途: 构造 column-resolved nonPH Emery Hamiltonian 对单个参数的导数矩阵。

参数:
- `params::ColumnEmeryNonPHParams`: 当前 mean-field 参数。
- `param_name::Symbol`: 参数名, 支持 hopping 参数和按列 onsite 参数。

返回:
- `Matrix{Float64}`: `dH/dp` 矩阵。
"""
function build_column_emery_nonph_dh_dparam(
    params::ColumnEmeryNonPHParams,
    param_name::Symbol,
)::Matrix{Float64}
    name_string = String(param_name)
    derivative_params_kwargs = Dict{Symbol,Any}(
        :lx => params.lx,
        :ly => params.ly,
        :bcx => params.bcx,
        :bcy => params.bcy,
    )

    if param_name in (:chi1_dd, :chi1_dp, :chi1_pp)
        derivative_params_kwargs[param_name] = 1.0
    elseif startswith(name_string, "mud_")
        derivative_params_kwargs[:mud] = Dict(param_name => 1.0)
    elseif startswith(name_string, "mzd_")
        derivative_params_kwargs[:mzd] = Dict(param_name => 1.0)
    elseif startswith(name_string, "mupx_")
        derivative_params_kwargs[:mupx] = Dict(param_name => 1.0)
    elseif startswith(name_string, "mzpx_")
        derivative_params_kwargs[:mzpx] = Dict(param_name => 1.0)
    elseif startswith(name_string, "mupy_")
        derivative_params_kwargs[:mupy] = Dict(param_name => 1.0)
    elseif startswith(name_string, "mzpy_")
        derivative_params_kwargs[:mzpy] = Dict(param_name => 1.0)
    else
        error("Unknown column Emery mean-field parameter: $(param_name).")
    end

    derivative_params = ColumnEmeryNonPHParams(; derivative_params_kwargs...)
    return Matrix(build_column_emery_nonph_hamiltonian(derivative_params))
end

"""
用途: 生成 column-resolved nonPH Emery determinant 的占据轨道和参数导数。

数学公式:
- 对 number-conserving Hamiltonian `H U = U epsilon` 对角化。
- nonPH determinant 取最低 `N_e` 个单粒子轨道。

参数:
- `params::ColumnEmeryNonPHParams`: mean-field 参数。
- `param_names::Vector{Symbol}`: 需要求导的 mean-field 参数名。
- `n_occupied_orbitals::Int`: 占据轨道数, 等于真实电子数。

返回:
- `(epsilon, occupied_orbitals, d_ut_params)`。
"""
function make_column_emery_nonph_ansatz_and_derivs(
    params::ColumnEmeryNonPHParams;
    param_names::Vector{Symbol}=Symbol[],
    n_occupied_orbitals::Int,
)
    hamiltonian = Matrix(build_column_emery_nonph_hamiltonian(params))
    hamiltonian_derivatives = Dict{Symbol,Matrix{Float64}}()
    for param_name in param_names
        hamiltonian_derivatives[param_name] = build_column_emery_nonph_dh_dparam(params, param_name)
    end

    epsilon, full_orbitals, _, orbital_derivatives = compute_eig_and_dU_reg1(
        hamiltonian,
        hamiltonian_derivatives,
    )
    if n_occupied_orbitals < 0 || n_occupied_orbitals > size(full_orbitals, 2)
        error("n_occupied_orbitals=$(n_occupied_orbitals) is outside 0:$(size(full_orbitals, 2)).")
    end

    occupied_orbitals = real.(full_orbitals[:, 1:n_occupied_orbitals])
    d_ut_params = Dict{Symbol,Matrix{Float64}}()
    for param_name in param_names
        d_ut_params[param_name] = permutedims(real.(orbital_derivatives[param_name][:, 1:n_occupied_orbitals]))
    end
    return epsilon, occupied_orbitals, d_ut_params
end

"""
用途: 从 mean-field 参数名和值构造 `ColumnEmeryNonPHParams`。

参数:
- `wf_param_names::Vector{Symbol}`: mean-field 参数名, 支持 `chi1_dd/chi1_dp/chi1_pp`
  以及 `mud_x/mzd_x/mupx_x/mzpx_x/mupy_x/mzpy_x`。
- `wf_param_values::Vector{Float64}`: 与参数名一一对应的参数值。
- `lx, ly::Int`: Cu 晶胞尺寸。
- `bcx, bcy::Float64`: 边界条件因子, 当前 x 方向 OBC, `bcx` 保留为兼容字段。
- `fixed_chi1_dp::Float64`: 当 `:chi1_dp` 不在 `wf_param_names` 中时使用的固定 Cu-O hopping。

返回:
- `ColumnEmeryNonPHParams`: 可直接用于构造 Emery mean-field Hamiltonian 的参数对象。
"""
function build_column_emery_nonph_params_from_wf_params(
    wf_param_names::Vector{Symbol},
    wf_param_values::Vector{Float64},
    lx::Int,
    ly::Int,
    bcx::Float64,
    bcy::Float64,
    ;
    fixed_chi1_dp::Float64=0.0,
)::ColumnEmeryNonPHParams
    length(wf_param_names) == length(wf_param_values) ||
        error("wf_param_names and wf_param_values must have the same length.")

    param_map = Dict{Symbol,Float64}(zip(wf_param_names, wf_param_values))
    mud = Dict{Symbol,Float64}()
    mzd = Dict{Symbol,Float64}()
    mupx = Dict{Symbol,Float64}()
    mzpx = Dict{Symbol,Float64}()
    mupy = Dict{Symbol,Float64}()
    mzpy = Dict{Symbol,Float64}()

    for (param_name, param_value) in param_map
        name_string = String(param_name)
        if startswith(name_string, "mud_")
            mud[param_name] = param_value
        elseif startswith(name_string, "mzd_")
            mzd[param_name] = param_value
        elseif startswith(name_string, "mupx_")
            mupx[param_name] = param_value
        elseif startswith(name_string, "mzpx_")
            mzpx[param_name] = param_value
        elseif startswith(name_string, "mupy_")
            mupy[param_name] = param_value
        elseif startswith(name_string, "mzpy_")
            mzpy[param_name] = param_value
        elseif param_name in (:chi1_dd, :chi1_dp, :chi1_pp)
            continue
        else
            error("Unknown column Emery mean-field parameter: $(param_name).")
        end
    end

    return ColumnEmeryNonPHParams(
        lx=lx,
        ly=ly,
        bcx=bcx,
        bcy=bcy,
        chi1_dd=get(param_map, :chi1_dd, 0.0),
        chi1_dp=get(param_map, :chi1_dp, fixed_chi1_dp),
        chi1_pp=get(param_map, :chi1_pp, 0.0),
        mud=mud,
        mupx=mupx,
        mupy=mupy,
        mzd=mzd,
        mzpx=mzpx,
        mzpy=mzpy,
    )
end

"""
用途: 将 stripe 中心类型映射为 Emery 初始参数公式中的 `x0` 偏移量。

参数:
- `stripe_center::AbstractString`: stripe 中心类型, 支持 `site` 或 `bond`。

返回:
- `Float64`: `site -> 0.0`, `bond -> 0.5`。
"""
function get_emery_stripe_center_offset(stripe_center::AbstractString)::Float64
    stripe_center_lowercase = lowercase(strip(stripe_center))
    if stripe_center_lowercase == "site"
        return 0.0
    elseif stripe_center_lowercase == "bond"
        return 0.5
    end
    error("Unknown stripe_center: $(stripe_center). Expected 'site' or 'bond'.")
end

"""
用途: 计算 Emery stripe 初始态在指定 x 坐标处的 chemical potential。

参数:
- `x_coordinate::Float64`: orbital 的实际 x 坐标, Cu 和 `p_y` 使用整数 x, `p_x(x)` 使用 `x + 0.5`。
- `lambda::Int`: stripe 电荷调制周期 `λ`。
- `stripe_center::AbstractString`: stripe 中心类型, 支持 `site` 或 `bond`。
- `mu_uniform::Float64`: 平均 chemical potential `μ`。
- `stripe_mu_amp::Float64`: 电荷调制振幅 `Δc`。
- `stripe_spin_peak_x::Float64`: spin envelope 峰值位置, `NaN` 时使用 `stripe_center` 相位。

返回:
- `Float64`: `mu_uniform + stripe_mu_amp * cos(Q * (x_coordinate - x0))`, 其中 `Q = 2π / λ`。
"""
function compute_emery_stripe_mu_value(
    x_coordinate::Float64,
    lambda::Int,
    stripe_center::AbstractString,
    mu_uniform::Float64,
    stripe_mu_amp::Float64,
    stripe_spin_peak_x::Float64,
)::Float64
    lambda > 0 || error("lambda must be positive.")
    stripe_center_offset = if isnan(stripe_spin_peak_x)
        get_emery_stripe_center_offset(stripe_center)
    else
        stripe_spin_peak_x - lambda / 2.0
    end
    stripe_wave_vector = 2.0 * pi / lambda
    return mu_uniform + stripe_mu_amp * cos(stripe_wave_vector * (x_coordinate - stripe_center_offset))
end

"""
用途: 计算 Emery stripe 初始态在指定 Cu x 坐标处的 d 轨道 staggered magnetic field。

参数:
- `x_coordinate::Float64`: Cu 的实际 x 坐标。
- `lambda::Int`: stripe 电荷调制周期 `λ`。
- `stripe_center::AbstractString`: stripe 中心类型, 支持 `site` 或 `bond`。
- `mz_amp::Float64`: 自旋调制振幅 `Δs`。
- `stripe_spin_peak_x::Float64`: spin envelope 峰值位置, `NaN` 时使用 `stripe_center` 相位。

返回:
- `Float64`: `mz_amp * sin(Q / 2 * (x_coordinate - x0))`, 其中 `Q = 2π / λ`。
"""
function compute_emery_stripe_mzd_value(
    x_coordinate::Float64,
    lambda::Int,
    stripe_center::AbstractString,
    mz_amp::Float64,
    stripe_spin_peak_x::Float64,
)::Float64
    lambda > 0 || error("lambda must be positive.")
    stripe_center_offset = if isnan(stripe_spin_peak_x)
        get_emery_stripe_center_offset(stripe_center)
    else
        stripe_spin_peak_x - lambda / 2.0
    end
    stripe_wave_vector = 2.0 * pi / lambda
    return mz_amp * sin(stripe_wave_vector / 2.0 * (x_coordinate - stripe_center_offset))
end

"""
用途: 生成 column-resolved nonPH Emery mean-field 参数名和初值。

参数:
- `ansatz::AbstractString`: `AFM` 或 `Stripe`。
- `lx, lambda::Int`: Cu 晶胞 x 方向长度和 stripe 周期。
- `stripe_center::AbstractString`: `site` 或 `bond`。
- `mu_uniform, stripe_mu_amp, mz_amp::Float64`: 初态 onsite 参数。
- `chi1_dd, chi1_dp, chi1_pp::Float64`: mean-field hopping 初值, 其中 `chi1_dp` 固定为 gauge。
- `stripe_spin_peak_x::Float64`: spin envelope 峰值位置, `NaN` 时使用 `stripe_center`。

返回:
- `NamedTuple`: 包含 `wf_param_names`, `wf_init_params`, `fixed_chi1_dp`。
"""
function build_column_emery_nonph_mean_field_parameter_setup(
    ansatz::AbstractString,
    lx::Int,
    lambda::Int,
    stripe_center::AbstractString,
    mu_uniform::Float64,
    stripe_mu_amp::Float64,
    mz_amp::Float64,
    chi1_dd::Float64,
    chi1_dp::Float64,
    chi1_pp::Float64,
    stripe_spin_peak_x::Float64,
)
    wf_param_names = Symbol[:chi1_dd, :chi1_pp]
    wf_init_params = Float64[chi1_dd, chi1_pp]
    if ansatz == "Stripe"
        for x in 1:lx
            x_coordinate = Float64(x)
            push!(wf_param_names, Symbol("mud_$(x)"))
            push!(wf_param_names, Symbol("mzd_$(x)"))
            push!(wf_param_names, Symbol("mupy_$(x)"))
            push!(wf_param_names, Symbol("mzpy_$(x)"))
            push!(wf_init_params, compute_emery_stripe_mu_value(x_coordinate, lambda, stripe_center, mu_uniform, stripe_mu_amp, stripe_spin_peak_x))
            push!(wf_init_params, compute_emery_stripe_mzd_value(x_coordinate, lambda, stripe_center, mz_amp, stripe_spin_peak_x))
            push!(wf_init_params, compute_emery_stripe_mu_value(x_coordinate, lambda, stripe_center, mu_uniform, stripe_mu_amp, stripe_spin_peak_x))
            push!(wf_init_params, 0.0)
        end
        for x in 0:lx
            x_coordinate = Float64(x) + 0.5
            push!(wf_param_names, Symbol("mupx_$(x)"))
            push!(wf_param_names, Symbol("mzpx_$(x)"))
            push!(wf_init_params, compute_emery_stripe_mu_value(x_coordinate, lambda, stripe_center, mu_uniform, stripe_mu_amp, stripe_spin_peak_x))
            push!(wf_init_params, 0.0)
        end
    elseif ansatz == "AFM"
        for x in 1:lx
            push!(wf_param_names, Symbol("mud_$(x)"))
            push!(wf_param_names, Symbol("mzd_$(x)"))
            push!(wf_param_names, Symbol("mupy_$(x)"))
            push!(wf_param_names, Symbol("mzpy_$(x)"))
            push!(wf_init_params, mu_uniform)
            push!(wf_init_params, mz_amp)
            push!(wf_init_params, mu_uniform)
            push!(wf_init_params, 0.0)
        end
        for x in 0:lx
            push!(wf_param_names, Symbol("mupx_$(x)"))
            push!(wf_param_names, Symbol("mzpx_$(x)"))
            push!(wf_init_params, mu_uniform)
            push!(wf_init_params, 0.0)
        end
    else
        error("Unknown ansatz type: $(ansatz).")
    end
    return (; wf_param_names=wf_param_names, wf_init_params=wf_init_params, fixed_chi1_dp=chi1_dp)
end

"""
用途: 根据 NN/NNN bond 生成 backflow 使用的有向 source 数据。

参数:
- `bonds1, bonds2::Vector{Tuple{Int, Int}}`: 无向 bond 的代表方向。
- `t1, t2::Float64`: 对应 hopping 振幅。

返回:
- `Tuple{Vector{Tuple{Int, Int}}, Vector{Float64}}`: 有向 source bonds 和振幅。
"""
function build_column_backflow_source_data(
    bonds1::Vector{Tuple{Int,Int}},
    bonds2::Vector{Tuple{Int,Int}},
    t1::Float64,
    t2::Float64,
)
    source_bonds = Tuple{Int,Int}[]
    source_amplitudes = Float64[]
    for (site_i, site_j) in bonds1
        push!(source_bonds, (site_i, site_j))
        push!(source_amplitudes, t1)
        push!(source_bonds, (site_j, site_i))
        push!(source_amplitudes, t1)
    end
    for (site_i, site_j) in bonds2
        push!(source_bonds, (site_i, site_j))
        push!(source_amplitudes, t2)
        push!(source_bonds, (site_j, site_i))
        push!(source_amplitudes, t2)
    end
    return source_bonds, source_amplitudes
end

"""
用途: 构造 column nonPH 使用的 Eq.(5) composite backflow。

参数:
- `source_bonds, source_amplitudes`: backflow source 数据。
- `bf_epsilon, bf_eta1, bf_eta2, bf_eta3::Float64`: Eq.(5) 参数。

返回:
- `CompositeBackflowTerm`: 按 `epsilon, eta1, eta2, eta3` 排列。
"""
function build_column_composite_backflow(
    source_bonds::Vector{Tuple{Int,Int}},
    source_amplitudes::Vector{Float64},
    bf_epsilon::Float64,
    bf_eta1::Float64,
    bf_eta2::Float64,
    bf_eta3::Float64,
)
    return CompositeBackflowTerm([
        BackflowEpsilonTerm(
            param_name=:bf_epsilon,
            epsilon_bf=bf_epsilon,
            epsilon_mask_terms=Symbol[:eta1, :eta2, :eta3],
            source_bonds=source_bonds,
            source_amplitudes=source_amplitudes,
        ),
        BackflowEta1DoublonHoleTerm(
            param_name=:bf_eta1,
            eta1_bf=bf_eta1,
            source_bonds=source_bonds,
            source_amplitudes=source_amplitudes,
        ),
        BackflowEta2SpinExchangeTerm(
            param_name=:bf_eta2,
            eta2_bf=bf_eta2,
            source_bonds=source_bonds,
            source_amplitudes=source_amplitudes,
        ),
        BackflowEta3MixedVirtualHopTerm(
            param_name=:bf_eta3,
            eta3_bf=bf_eta3,
            source_bonds=source_bonds,
            source_amplitudes=source_amplitudes,
        ),
    ])
end

"""
用途: 根据开关构造 column nonPH backflow 对象。

参数:
- `enable_backflow::Bool`: 是否启用 backflow。
- 其余参数同 `build_column_composite_backflow`。

返回:
- `AbstractBackflowTerm`: 启用时为 composite backflow, 禁用时为 `NoBackflowTerm()`。
"""
function build_column_optional_backflow(
    enable_backflow::Bool,
    source_bonds::Vector{Tuple{Int,Int}},
    source_amplitudes::Vector{Float64},
    bf_epsilon::Float64,
    bf_eta1::Float64,
    bf_eta2::Float64,
    bf_eta3::Float64,
)
    if !enable_backflow
        return NoBackflowTerm()
    end
    return build_column_composite_backflow(
        source_bonds,
        source_amplitudes,
        bf_epsilon,
        bf_eta1,
        bf_eta2,
        bf_eta3,
    )
end

"""
用途: 解析布尔命令行字符串。

参数:
- `raw_value::AbstractString`: 支持 `true/false`, `1/0`, `yes/no`, `on/off`。
- `option_name::AbstractString`: 选项名, 用于错误信息。

返回:
- `Bool`: 解析结果。
"""
function parse_column_bool_flag(raw_value::AbstractString, option_name::AbstractString)::Bool
    normalized_value = lowercase(strip(raw_value))
    if normalized_value in ("true", "t", "1", "yes", "y", "on")
        return true
    elseif normalized_value in ("false", "f", "0", "no", "n", "off")
        return false
    end
    error("Invalid value for $(option_name): $(raw_value).")
end

"""
用途: 构造指数衰减学习率函数。

参数:
- `lr_start::Float64`: 初始学习率。
- `lr_end::Float64`: 最终学习率。
- `n_steps::Int`: SR 总步数。

返回:
- `Function`: `(lr0, step) -> lr_value`。
"""
function build_exponential_lr_func(
    lr_start::Float64,
    lr_end::Float64,
    n_steps::Int,
)::Function
    if n_steps <= 1
        return (lr0, step) -> lr_end
    end
    if lr_start == 0.0
        return (lr0, step) -> 0.0
    end
    if lr_start < 0.0 || lr_end < 0.0
        error("lr and lr_end must be non-negative.")
    end
    lr_decay_gamma = (lr_end / lr_start)^(1.0 / (n_steps - 1))
    return (lr0, step) -> lr0 * (lr_decay_gamma^(step - 1))
end

#=
"""
用途: 更新 column Emery nonPH determinant 波函数和参数导数。

参数:
- `vwf`: determinant 波函数。
- `param_names, params`: 完整参数名和值, 顺序为 mean-field, projector, backflow。
- `lx, ly, bcx, bcy`: 晶格和边界参数, 当前 Emery x 方向固定 OBC。
- `n_occupied_orbitals::Int`: nonPH 占据轨道数。
- `nparams_proj, nparams_backflow::Int`: projector/backflow 参数数量。
- `fixed_chi1_dp::Float64`: 固定的 Cu-O mean-field hopping, 用于固定 hopping gauge。

返回:
- `nothing`。
"""
const COLUMN_NONPH_ANSATZ_DOC_PLACEHOLDER = nothing
=#

"""
用途: 将 `measure` 任务的结果写入 Hubbard_bf.jl 使用的输出文件。

参数:
- `output_dir::AbstractString`: 输出目录, 函数内部会确保目录存在。
- `results`: `run_simulation` 返回的结果, 需要包含 `:means`, 可选包含 `:histories`。

返回:
- `nothing`。

公式:
- 对 `histories` 调用 `blocking_binning`, 得到每个观测量的平均值 `Mean`, 标准误差 `SE`,
  有效样本数 `N_eff`, 以及 integrated autocorrelation time `Tau_int`。
"""
function write_column_measure_outputs(output_dir::AbstractString, results)::Nothing
    mkpath(output_dir)

    means = results[:means]
    mean_dict_str = Dict{String,Any}()
    for (key, value) in means
        mean_dict_str[String(key)] = value isa Number ? real(value) : value
    end
    open(joinpath(output_dir, "block_binning_mean.json"), "w") do io
        JSON.print(io, mean_dict_str)
    end

    histories = get(results, :histories, Dict{Symbol,Any}())
    if !isempty(histories)
        mean_hist, se_dict, n_eff_dict, tau_int_dict, _ = blocking_binning(histories)
        open(joinpath(output_dir, "block_binning.txt"), "w") do io
            println(io, "# Observable\tMean\tSE\tN_eff\tTau_int")
            for name in sort(collect(keys(mean_hist)))
                mean_value = mean_hist[name]
                se_value = se_dict[name]
                n_eff_value = n_eff_dict[name]
                tau_int_value = tau_int_dict[name]

                if mean_value isa Number && se_value isa Number && n_eff_value isa Number && tau_int_value isa Number
                    @printf(
                        io,
                        "%s\t%.10f\t%.10f\t%.6f\t%.6f\n",
                        String(name),
                        real(mean_value),
                        real(se_value),
                        real(n_eff_value),
                        real(tau_int_value),
                    )
                else
                    println(io, "$(String(name))\t$(mean_value)\t$(se_value)\t$(n_eff_value)\t$(tau_int_value)")
                end
            end
        end
    end
    return nothing
end

"""
用途: 更新 column nonPH determinant 波函数和参数导数。

参数:
- `vwf`: determinant 波函数。
- `param_names, params`: 完整参数名和值, 顺序为 mean-field, projector, backflow。
- `lx, ly, bcx, bcy`: 晶格和边界参数, 当前 Emery x 方向固定 OBC。
- `n_occupied_orbitals::Int`: nonPH 占据轨道数。
- `nparams_proj, nparams_backflow::Int`: projector/backflow 参数数量。

返回:
- `nothing`。
"""
function update_column_nonph_ansatz!(
    vwf,
    param_names::Vector{Symbol},
    params::Vector{Float64},
    lx::Int,
    ly::Int,
    bcx::Float64,
    bcy::Float64,
    n_occupied_orbitals::Int;
    nparams_proj::Int=0,
    nparams_backflow::Int=0,
    fixed_chi1_dp::Float64=0.0,
)::Nothing
    total_param_count = length(param_names)
    nparams_wf = total_param_count - nparams_proj - nparams_backflow
    wf_param_names = param_names[1:nparams_wf]
    wf_param_values = params[1:nparams_wf]
    projector_param_names = param_names[(nparams_wf+1):(nparams_wf+nparams_proj)]
    projector_param_values = params[(nparams_wf+1):(nparams_wf+nparams_proj)]
    backflow_param_names = param_names[(nparams_wf+nparams_proj+1):end]
    backflow_param_values = params[(nparams_wf+nparams_proj+1):end]

    nonph_params = build_column_emery_nonph_params_from_wf_params(
        wf_param_names,
        wf_param_values,
        lx,
        ly,
        bcx,
        bcy;
        fixed_chi1_dp=fixed_chi1_dp,
    )
    _, gs_u, d_ut_params = make_column_emery_nonph_ansatz_and_derivs(
        nonph_params;
        param_names=wf_param_names,
        n_occupied_orbitals=n_occupied_orbitals,
    )

    copyto!(vwf.base_gs_U, gs_u)
    copyto!(vwf.gs_U, gs_u)
    copyto!(vwf.backflow_u, gs_u)
    copyto!(vwf.gs_U_t, permutedims(gs_u))

    d_ut_matrix = zeros(Float64, size(gs_u, 2), size(gs_u, 1), length(wf_param_names))
    for (param_index, param_name) in enumerate(wf_param_names)
        d_ut_matrix[:, :, param_index] = d_ut_params[param_name]
    end
    update_vwf_params!(vwf, wf_param_names, d_ut_matrix)

    if !isempty(projector_param_names)
        update_vwf_projector_params!(vwf, projector_param_names, projector_param_values)
    end
    if !isempty(backflow_param_names)
        update_vwf_backflow_params!(vwf, backflow_param_names, backflow_param_values)
    end
    init_gswf!(vwf)
    return nothing
end

"""
用途: 解析 `Hubbard_bf.jl` 命令行参数。

参数:
- 无。读取 `ARGS`。

返回:
- `Dict{String, Any}`: ArgParse 结果。
"""
function parse_column_bf_commandline()
    settings = ArgParseSettings()
    @add_arg_table settings begin
        "--Lx"
        arg_type = Int
        default = 8
        "--Ly"
        arg_type = Int
        default = 3
        "--tpd"
        arg_type = Float64
        default = 1.0
        "--tpp"
        arg_type = Float64
        default = 0.0
        "--Delta_pd"
        arg_type = Float64
        default = 3.0
        "--Udd"
        arg_type = Float64
        default = 8.0
        "--Up"
        arg_type = Float64
        default = 0.0
        "--Vpd"
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
        "--edge_pinning"
        arg_type = Float64
        default = 0.0
        "--chi1_dd"
        arg_type = Float64
        default = 0.0
        "--chi1_dp"
        arg_type = Float64
        default = 1.0
        "--chi1_pp"
        arg_type = Float64
        default = 0.0
        "--mz"
        arg_type = Float64
        default = 3.0
        "--mu"
        arg_type = Float64
        default = -3.0
        "--stripe_mu_amp"
        arg_type = Float64
        default = 0.0
        "--stripe_spin_peak_x"
        arg_type = Float64
        default = NaN
        "--target_sz"
        arg_type = Int
        default = 0
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
        "--init_params_json"
        arg_type = String
        default = ""
        "--job"
        arg_type = String
        default = "SR"
        "--doping"
        arg_type = Float64
        default = 0.125
        "--ansatz"
        arg_type = String
        default = "Stripe"
        "--lambda"
        arg_type = Int
        default = 4
        "--stripe_center"
        arg_type = String
        default = "site"
        "--enable_orbital_gutzwiller"
        arg_type = String
        default = "true"
        "--g_d"
        arg_type = Float64
        default = 1.0
        "--g_p"
        arg_type = Float64
        default = 1.0
        "--vj1"
        arg_type = Float64
        default = 0.0
        "--vj2"
        arg_type = Float64
        default = 0.0
        "--enable_backflow"
        arg_type = String
        default = "false"
        "--bf_epsilon"
        arg_type = Float64
        default = 1.0
        "--bf_eta1"
        arg_type = Float64
        default = 0.0
        "--bf_eta2"
        arg_type = Float64
        default = 0.0
        "--bf_eta3"
        arg_type = Float64
        default = 0.0
    end
    return parse_args(settings)
end

"""
用途: 运行 column-resolved nonPH + backflow Hubbard VMC/SR 主流程。

参数:
- 无。所有参数来自命令行。

返回:
- `nothing`。
"""
function main_column_nonph_backflow()::Nothing
    args = parse_column_bf_commandline()
    session = init_mpi_session()
    rank = session.rank
    is_root = rank == session.root

    lx = args["Lx"]
    ly = args["Ly"]
    bcx = args["bcx"]
    bcy = args["bcy"]
    n_sites = emery_n_sites(lx, ly)
    target_sz = args["target_sz"]
    doping = args["doping"]
    lr = args["lr"]
    lr_end = isnan(args["lr_end"]) ? lr : args["lr_end"]
    job = args["job"]

    # Reset timing at start (for measure mode, one-shot; for SR, we accumulate per step)
    timing_reset!()

    mean_field_setup = build_column_emery_nonph_mean_field_parameter_setup(
        args["ansatz"],
        lx,
        args["lambda"],
        args["stripe_center"],
        args["mu"],
        args["stripe_mu_amp"],
        args["mz"],
        args["chi1_dd"],
        args["chi1_dp"],
        args["chi1_pp"],
        args["stripe_spin_peak_x"],
    )
    wf_param_names = mean_field_setup.wf_param_names
    wf_init_params = mean_field_setup.wf_init_params
    fixed_chi1_dp = mean_field_setup.fixed_chi1_dp

    meas_params = VMCParams(
        total_samples=args["nMC"],
        warmup_steps=args["wMC"],
        rebuild_every=args["rMC"],
        decorr_steps=args["dMC"],
        seed=args["seed"] + rank,
    )

    if parse_column_bool_flag(args["enable_backflow"], "--enable_backflow")
        error("Emery backflow source bonds are not migrated yet. Use --enable_backflow false for this step.")
    end
    backflow = NoBackflowTerm()
    projector = if parse_column_bool_flag(args["enable_orbital_gutzwiller"], "--enable_orbital_gutzwiller")
        build_emery_orbital_gutzwiller_projector(lx, ly; g_d=args["g_d"], g_p=args["g_p"])
    else
        CompositeProjector([NoProjectorTerm()])
    end
    proj_param_names = projector_param_names(projector)
    proj_init_params = projector_param_values(projector)
    nparams_proj = length(proj_param_names)
    nparams_backflow = 0
    init_params = vcat(wf_init_params, proj_init_params)
    param_names = vcat(wf_param_names, proj_param_names)

    if !isempty(args["init_params_json"])
        init_params = build_init_params_from_json(args["init_params_json"], param_names)
        if is_root
            println("Loaded initial parameters from json: $(args["init_params_json"])")
        end
    end

    ham = @timed "build_emery_general_model" build_emery_general_model(
        lx,
        ly;
        tpd=args["tpd"],
        tpp=args["tpp"],
        Delta_pd=args["Delta_pd"],
        Udd=args["Udd"],
        Up=args["Up"],
        Vpd=args["Vpd"],
        Vpp=args["Vpp"],
    )
    if args["edge_pinning"] != 0.0
        for y in 1:ly
            left_site = Emery_xyo_to_site_index(1, y, EMERY_ORB_D, lx, ly)
            push!(ham.terms, OperatorTerm([:Sz], [left_site], args["edge_pinning"] * (-1)^(y + 1)))
        end
    end

    nelec = compute_emery_electron_count(lx, ly, doping)
    @assert (target_sz + nelec) % 2 == 0 "Wrong parity between target_sz and electron count."
    nup = (nelec + target_sz) ÷ 2
    ndn = nelec - nup
    if nup < 0 || ndn < 0 || nup > n_sites || ndn > n_sites
        error("Invalid nonPH particle numbers: nup=$(nup), ndn=$(ndn), N_sites=$(n_sites).")
    end

    sampler = config_Hubbard(n_sites, nup, ndn; ifPH=false)
    init_config_Hubbard!(sampler)
    vwf = vwf_det(zeros(Float64, 2 * n_sites, nelec), sampler; backflow=backflow)
    set_projector!(vwf, projector)
    kernel = HubbardKernel(conserve_sz=true)

    if is_root
        println("Initial parameters: $(init_params)")
        println("Fixed mean-field parameters: chi1_dp=$(fixed_chi1_dp)")
        println("column Emery nonPH particle numbers: N_up=$(nup), N_down=$(ndn), N_e=$(nelec), N_sites=$(n_sites)")
    end

    update_column_nonph_ansatz!(
        vwf,
        param_names,
        init_params,
        lx,
        ly,
        bcx,
        bcy,
        nelec;
        nparams_proj=nparams_proj,
        nparams_backflow=nparams_backflow,
        fixed_chi1_dp=fixed_chi1_dp,
    )

    folder = "logs"
    mkpath(folder)
    if job == "SR"
        sr_params = SRParams(vmc_params=meas_params, n_steps=args["nSR"], lr=lr)
        exp_lr_func = build_exponential_lr_func(lr, lr_end, args["nSR"])
        update_vwf_func! = (vwf, params) -> @timed "update_column_nonph_ansatz!" update_column_nonph_ansatz!(
            vwf,
            param_names,
            params,
            lx,
            ly,
            bcx,
            bcy,
            nelec;
            nparams_proj=nparams_proj,
            nparams_backflow=nparams_backflow,
            fixed_chi1_dp=fixed_chi1_dp,
        )
        run_sr_optimization(
            ham,
            vwf,
            kernel,
            init_params,
            update_vwf_func!,
            sr_params;
            log_file=joinpath(folder, "sr_history.txt"),
            param_names=param_names,
            lr_func=exp_lr_func,
        )
        if is_root
            extract_min_energy(joinpath(folder, "sr_history.txt"))
            println("[Timing] SR optimization complete. Printing timing report...")
            timing_report()
            open(joinpath(folder, "timing_report.txt"), "w") do io
                timing_report(io)
            end
            println("[Timing] Timing report saved to $(joinpath(folder, "timing_report.txt")).")
        end
    elseif job == "measure"
        results = run_simulation(
            ham,
            vwf,
            kernel,
            build_emery_observables(lx, ly),
            meas_params;
            history_observables=[:E],
        )
        if is_root && results !== nothing
            write_column_measure_outputs(folder, results)
            println("[Timing] Measurement complete. Printing timing report...")
            timing_report()
            open(joinpath(folder, "timing_report.txt"), "w") do io
                timing_report(io)
            end
            println("[Timing] Timing report saved to $(joinpath(folder, "timing_report.txt")).")
        end
    else
        error("Unknown job: $(job)")
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_column_nonph_backflow()
end
