using MPI
using Random
using Printf
using LinearAlgebra
using ArgParse
using JSON

include(joinpath(@__DIR__, "src", "mfVMC.jl"))
using .mfVMC

"""
    DefectThreeBandRunState

用途: 保存 defect three-band Hubbard 主程序构造出的运行状态。

字段:
- `geom`: `ThreeBandGeometry`, 三带模型几何.
- `metadata`: `DefectThreeBandMetadata`, defect 锚点及相关站点信息.
- `model`: `DefectThreeBandHubbardModel`, 物理 Hamiltonian.
- `nelec::Int`: 固定电子数.
- `nup::Int`: 固定 spin-up 电子数.
- `ndn::Int`: 固定 spin-down 电子数.
- `param_names::Vector{Symbol}`: 变分参数名.
- `init_params::Vector{Float64}`: 与 `param_names` 对齐的初值.
- `projector`: 主 `src/Projector.jl` 中的 Gutzwiller/Jastrow projector; 若未启用则为 `nothing`.
- `uniform_nondefect_mu::Bool`: 是否用共享的非 defect 化学势参数.
- `site_resolved_oxygen_mz::Bool`: oxygen 的 `mz` 是否也逐 site 参数化.
- `jastrow_shells::Int`: clean orbital-pair Jastrow 的 shell 数, 用于参数拆分.
"""
struct DefectThreeBandRunState
    geom
    metadata
    model
    nelec::Int
    nup::Int
    ndn::Int
    param_names::Vector{Symbol}
    init_params::Vector{Float64}
    projector
    uniform_nondefect_mu::Bool
    site_resolved_oxygen_mz::Bool
    jastrow_shells::Int
end

"""
    parse_defect_anchors(value)

用途: 解析命令行中的 defect anchor 字符串。

参数:
- `value::AbstractString`: 形如 `"x1,y1;x2,y2"` 的字符串, 坐标采用 1-based 编号.

返回:
- `Vector{Tuple{Int,Int}}`: defect anchor 坐标列表.
"""
function parse_defect_anchors(value::AbstractString)
    stripped = strip(value)
    isempty(stripped) && return Tuple{Int,Int}[]

    anchors = Tuple{Int,Int}[]
    seen = Set{Tuple{Int,Int}}()
    for raw_token in split(stripped, ";"; keepempty=true)
        token = strip(raw_token)
        fields = split(token, ","; keepempty=true)
        length(fields) == 2 ||
            throw(ArgumentError("invalid defect anchor '$token'; expected x,y"))

        anchor = try
            (parse(Int, strip(fields[1])), parse(Int, strip(fields[2])))
        catch err
            err isa ArgumentError || rethrow()
            throw(ArgumentError("invalid integer in defect anchor '$token'"))
        end
        anchor[1] > 0 && anchor[2] > 0 ||
            throw(ArgumentError("defect anchor coordinates must be positive integers: $anchor"))
        anchor in seen && throw(ArgumentError("duplicate defect anchor: $anchor"))

        push!(anchors, anchor)
        push!(seen, anchor)
    end
    return anchors
end

"""
    defect_threeband_argparse_settings()

用途: 构造 defect three-band Hubbard 主程序的命令行参数表。

参数:
- 无.

返回:
- `ArgParseSettings`: 可传给 `parse_args` 的参数配置.
"""
function defect_threeband_argparse_settings()
    settings = ArgParseSettings()

    @add_arg_table settings begin
        "--Lx"
        help = "Lattice size in x unit cells"
        arg_type = Int
        default = 4
        "--Ly"
        help = "Lattice size in y unit cells"
        arg_type = Int
        default = 4
        "--bcx"
        help = "Boundary condition phase in x"
        arg_type = Float64
        default = 1.0
        "--bcy"
        help = "Boundary condition phase in y"
        arg_type = Float64
        default = 1.0
        "--tpd"
        help = "Cu-O hopping amplitude"
        arg_type = Float64
        default = 1.0
        "--tpp"
        help = "O-O hopping amplitude"
        arg_type = Float64
        default = 0.0
        "--Delta_pd"
        help = "Oxygen orbital energy offset"
        arg_type = Float64
        default = 0.0
        "--Udd"
        help = "Cu onsite interaction"
        arg_type = Float64
        default = 8.0
        "--Up"
        help = "O onsite interaction"
        arg_type = Float64
        default = 0.0
        "--Vpd"
        help = "Cu-O density interaction"
        arg_type = Float64
        default = 0.0
        "--defect_Epp"
        help = "Defect oxygen onsite energy shift"
        arg_type = Float64
        default = 0.0
        "--nelec"
        help = "Electron count over the 3*Lx*Ly sites; used when positive"
        arg_type = Int
        default = 0
        "--Nhole"
        help = "Hole count relative to one electron per unit cell when --nelec <= 0"
        arg_type = Int
        default = 0
        "--target_sz"
        help = "Target N_up - N_dn"
        arg_type = Int
        default = 0
        "--defect_anchors"
        help = "Semicolon-separated defect oxygen anchor cells as x,y;x,y"
        arg_type = String
        default = ""
        "--chi1_00"
        help = "Cu-Cu mean-field hopping"
        arg_type = Float64
        default = 0.0
        "--chi1_01"
        help = "Cu-O mean-field hopping"
        arg_type = Float64
        default = 0.0
        "--chi1_11"
        help = "O-O mean-field hopping"
        arg_type = Float64
        default = 0.0
        "--mu0"
        help = "Cu mean-field chemical potential"
        arg_type = Float64
        default = 0.0
        "--mu1"
        help = "O mean-field chemical potential"
        arg_type = Float64
        default = 0.0
        "--mu0_d0"
        help = "Defect Cu patch mean-field chemical potential; NaN follows --mu0"
        arg_type = Float64
        default = NaN
        "--mu1_d0"
        help = "Defect oxygen mean-field chemical potential; NaN follows --mu1"
        arg_type = Float64
        default = NaN
        "--mz_00"
        help = "Cu mean-field spin polarization"
        arg_type = Float64
        default = 0.0
        "--mz_11"
        help = "O mean-field spin polarization"
        arg_type = Float64
        default = 0.0
        "--mz_00_d0"
        help = "Defect Cu patch mean-field spin polarization; NaN follows --mz_00"
        arg_type = Float64
        default = NaN
        "--mz_11_d0"
        help = "Defect oxygen mean-field spin polarization; NaN follows --mz_11"
        arg_type = Float64
        default = NaN
        "--uniform_nondefect_mu"
        help = "Use shared mu0/mu1 parameters for non-defect Cu/O sites"
        arg_type = Bool
        default = false
        "--site_resolved_oxygen_mz"
        help = "Use independent mz_site_* parameters on oxygen sites"
        arg_type = Bool
        default = false
        "--chi_def_pd_init"
        help = "Initial defect Cu-O hopping parameter; NaN follows clean background"
        arg_type = Float64
        default = NaN
        "--chi_def_pp_init"
        help = "Initial defect O-O hopping parameter; NaN follows clean background"
        arg_type = Float64
        default = NaN
        "--chi_def_dd_init"
        help = "Initial defect Cu-Cu hopping parameter; NaN follows clean background"
        arg_type = Float64
        default = NaN
        "--gutzwiller_orbital"
        help = "Enable orbital-resolved onsite Gutzwiller projector"
        arg_type = Bool
        default = false
        "--g_d"
        help = "Initial Cu orbital Gutzwiller parameter; NaN follows --g_site_init"
        arg_type = Float64
        default = NaN
        "--g_py"
        help = "Initial py orbital Gutzwiller parameter; NaN follows --g_site_init"
        arg_type = Float64
        default = NaN
        "--g_px"
        help = "Initial px orbital Gutzwiller parameter; NaN follows --g_site_init"
        arg_type = Float64
        default = NaN
        "--site_gutzwiller"
        help = "Enable site-resolved onsite Gutzwiller projector"
        arg_type = Bool
        default = false
        "--g_site_init"
        help = "Initial value for site-resolved Gutzwiller parameters"
        arg_type = Float64
        default = 0.0
        "--jastrow_shells"
        help = "Number of orbital-pair density Jastrow cell-distance shells"
        arg_type = Int
        default = 0
        "--jastrow_init"
        help = "Initial value for each clean orbital-pair Jastrow parameter"
        arg_type = Float64
        default = 0.0
        "--jastrow_init_file"
        help = "Optional text file with two columns: Jastrow parameter name and initial value"
        arg_type = String
        default = ""
        "--defect_jastrow"
        help = "Enable defect-pair density Jastrow projector"
        arg_type = Bool
        default = false
        "--defect_jastrow_init"
        help = "Initial value for defect-pair Jastrow parameters"
        arg_type = Float64
        default = 0.0
        "--nMC"
        help = "Number of Monte Carlo samples"
        arg_type = Int
        default = 10000
        "--wMC"
        help = "Number of warmup sweeps"
        arg_type = Int
        default = 100
        "--rMC"
        help = "Accepted moves between determinant rebuilds"
        arg_type = Int
        default = 100
        "--dMC"
        help = "Decorrelating sweeps between samples"
        arg_type = Int
        default = 1
        "--seed"
        help = "Random seed"
        arg_type = Int
        default = 5423
        "--nSR"
        help = "Number of SR steps"
        arg_type = Int
        default = 50
        "--lr"
        help = "SR learning rate"
        arg_type = Float64
        default = 0.04
        "--lr_end"
        help = "Final learning rate; default follows --lr"
        arg_type = Float64
        default = NaN
        "--diag_shift"
        help = "Relative diagonal shift for stabilized SR"
        arg_type = Float64
        default = 1e-3
        "--eps_wf"
        help = "SR eigenmode truncation threshold"
        arg_type = Float64
        default = 1e-4
        "--max_step_size"
        help = "Maximum absolute parameter update per SR step"
        arg_type = Float64
        default = 0.1
        "--numa_tensor_replica"
        help = "Use one shared dense derivative tensor replica per NUMA domain on single-node MPI runs"
        arg_type = Bool
        default = false
        "--not_opt_params"
        help = "Comma-separated initial parameters to keep fixed during SR"
        arg_type = String
        default = ""
        "--job"
        help = "Job to run: SR or measure"
        arg_type = String
        default = "SR"
        "--init_params_json"
        help = "Path to JSON file with initial parameters"
        arg_type = String
        default = ""
        "--init_params_txt"
        help = "Path to a two-column txt file with initial parameters; may contain a subset"
        arg_type = String
        default = ""
    end

    return settings
end

parse_commandline(argv=ARGS) = parse_args(argv, defect_threeband_argparse_settings())

"""
    _arg(args, name, default=nothing)

用途: 同时兼容 `Dict{String,Any}` 和 `Dict{Symbol,Any}` 风格的参数读取。

参数:
- `args`: 参数字典.
- `name::Symbol`: 参数名.
- `default`: 缺省值.

返回:
- 参数值或 `default`.
"""
_arg(args, name::Symbol, default=nothing) =
    haskey(args, String(name)) ? args[String(name)] :
    haskey(args, name) ? args[name] :
    default

"""
    _nan_to_nothing(value)

用途: 把命令行中用作占位的 `NaN` 转为 `nothing`, 便于调用 core 中的 fallback 逻辑。

参数:
- `value`: 任意值.

返回:
- 若 `value` 是 `NaN` 浮点数, 返回 `nothing`; 否则返回原值.
"""
_nan_to_nothing(value) = value isa AbstractFloat && isnan(value) ? nothing : value

const ORB_D = 1
const ORB_PY = 2
const ORB_PX = 3

"""
    ThreeBandGeometry

用途: 保存三带 Hubbard 模型的晶胞几何信息。

字段:
- `Lx::Int`: x 方向晶胞数.
- `Ly::Int`: y 方向晶胞数.
- `bcx::Float64`: x 方向边界相位.
- `bcy::Float64`: y 方向边界相位.
"""
struct ThreeBandGeometry
    Lx::Int
    Ly::Int
    bcx::Float64
    bcy::Float64
end

function ThreeBandGeometry(Lx::Integer, Ly::Integer; bcx::Real=1.0, bcy::Real=1.0)
    Lx > 0 && Ly > 0 || throw(ArgumentError("Lx and Ly must be positive"))
    return ThreeBandGeometry(Int(Lx), Int(Ly), Float64(bcx), Float64(bcy))
end

"""
    n_cells(geom), n_sites(geom)

用途: 返回三带几何中的晶胞数和 site 数。

参数:
- `geom::ThreeBandGeometry`: 三带几何.

返回:
- `Int`: `n_cells = Lx * Ly`, `n_sites = 3 * Lx * Ly`.
"""
n_cells(geom::ThreeBandGeometry) = geom.Lx * geom.Ly
n_sites(geom::ThreeBandGeometry) = 3 * n_cells(geom)

"""
    cell_index(geom, x, y)

用途: 将 1-based 晶胞坐标映射到晶胞编号。

参数:
- `geom::ThreeBandGeometry`: 三带几何.
- `x`, `y`: 1-based 晶胞坐标, 会按周期边界归一化.

返回:
- `Int`: 1-based 晶胞编号.
"""
cell_index(geom::ThreeBandGeometry, x::Integer, y::Integer) =
    mod1(Int(x), geom.Lx) + (mod1(Int(y), geom.Ly) - 1) * geom.Lx

"""
    site_index(geom, cell, orbital)

用途: 将晶胞编号和 orbital 编号映射到 site 编号。

参数:
- `geom::ThreeBandGeometry`: 三带几何.
- `cell::Integer`: 1-based 晶胞编号.
- `orbital::Integer`: orbital 编号, `1=d`, `2=py`, `3=px`.

返回:
- `Int`: 1-based site 编号.
"""
function site_index(geom::ThreeBandGeometry, cell::Integer, orbital::Integer)
    1 <= Int(cell) <= n_cells(geom) || throw(ArgumentError("cell index out of range: $cell"))
    1 <= Int(orbital) <= 3 || throw(ArgumentError("orbital index must be 1, 2, or 3"))
    return 3 * (Int(cell) - 1) + Int(orbital)
end

"""
    cell_of_site(geom, site), orbital_of_site(geom, site)

用途: 将 site 编号反解为晶胞编号和 orbital 编号。

参数:
- `geom::ThreeBandGeometry`: 三带几何.
- `site::Integer`: 1-based site 编号.

返回:
- `Int`: `cell_of_site` 返回晶胞编号, `orbital_of_site` 返回 orbital 编号.
"""
function cell_of_site(geom::ThreeBandGeometry, site::Integer)
    1 <= Int(site) <= n_sites(geom) || throw(ArgumentError("site index out of range: $site"))
    return div(Int(site) - 1, 3) + 1
end

function orbital_of_site(geom::ThreeBandGeometry, site::Integer)
    1 <= Int(site) <= n_sites(geom) || throw(ArgumentError("site index out of range: $site"))
    return mod(Int(site) - 1, 3) + 1
end

"""
    orbital_group_vector(geom)

用途: 返回每个 site 所属 orbital group, 用于 orbital-resolved Gutzwiller。

参数:
- `geom::ThreeBandGeometry`: 三带几何.

返回:
- `Vector{Int}`: 第 `site` 个元素为 `1=d`, `2=py`, `3=px`.
"""
orbital_group_vector(geom::ThreeBandGeometry) = [orbital_of_site(geom, site) for site in 1:n_sites(geom)]

"""
    threeband_jastrow_names(n_shells)

用途: 生成 clean orbital-pair Jastrow 的参数名。

参数:
- `n_shells::Integer`: cell-distance shell 数.

返回:
- `Vector{Symbol}`: 顺序为 `v_dd_s, v_dp_s, v_pp_s`.
"""
function threeband_jastrow_names(n_shells::Integer)
    n = Int(n_shells)
    n >= 0 || throw(ArgumentError("n_shells must be nonnegative"))
    names = Symbol[]
    for shell in 1:n
        append!(names, (Symbol("v_dd_$shell"), Symbol("v_dp_$shell"), Symbol("v_pp_$shell")))
    end
    return names
end

"""
    build_orbital_pair_jastrow_index(geom, n_shells; pair_mode=:dp_pp)

用途: 构造 clean orbital-pair Jastrow 的 pair 参数编号矩阵。

参数:
- `geom::ThreeBandGeometry`: 三带几何.
- `n_shells::Integer`: cell-distance shell 数.
- `pair_mode::Symbol`: 当前仅支持 `:dp_pp`.

返回:
- `Matrix{Int}`: `pair_index[i,j] = 0` 表示不耦合, 否则为 `threeband_jastrow_names(n_shells)` 中的参数编号.

公式:
- `shell(i,j) = min(|dx|, Lx-|dx|) + min(|dy|, Ly-|dy|)`, 即 periodic Manhattan distance.
"""
function build_orbital_pair_jastrow_index(
    geom::ThreeBandGeometry,
    n_shells::Integer;
    pair_mode::Symbol=:dp_pp,
)
    pair_mode === :dp_pp || throw(ArgumentError("unknown orbital pair jastrow pair_mode: $pair_mode"))
    n_shells_int = Int(n_shells)
    n_shells_int >= 0 || throw(ArgumentError("n_shells must be nonnegative"))

    pair_index = zeros(Int, n_sites(geom), n_sites(geom))
    for site_i in 1:n_sites(geom)
        cell_i = cell_of_site(geom, site_i)
        x_i = mod(cell_i - 1, geom.Lx) + 1
        y_i = div(cell_i - 1, geom.Lx) + 1
        orb_i = orbital_of_site(geom, site_i)
        for site_j in (site_i + 1):n_sites(geom)
            cell_j = cell_of_site(geom, site_j)
            x_j = mod(cell_j - 1, geom.Lx) + 1
            y_j = div(cell_j - 1, geom.Lx) + 1
            dx = abs(x_i - x_j)
            dy = abs(y_i - y_j)
            dx = min(dx, geom.Lx - dx)
            dy = min(dy, geom.Ly - dy)
            shell = dx + dy
            shell == 0 && continue
            shell <= n_shells_int || continue

            orb_j = orbital_of_site(geom, site_j)
            family_offset = if orb_i == ORB_D && orb_j == ORB_D
                1
            elseif orb_i == ORB_D || orb_j == ORB_D
                2
            elseif orb_i != ORB_D && orb_j != ORB_D
                3
            else
                0
            end
            family_offset == 0 && continue
            param_id = 3 * (shell - 1) + family_offset
            pair_index[site_i, site_j] = param_id
            pair_index[site_j, site_i] = param_id
        end
    end
    return pair_index
end

"""
    DefectThreeBandMetadata

用途: 保存 defect anchor 派生出的 defect oxygen 和附近 Cu patch 信息。

字段:
- `geom::ThreeBandGeometry`: 三带几何.
- `anchors::Vector{Tuple{Int,Int}}`: defect anchor 晶胞坐标.
- `anchor_cells::Vector{Int}`: defect anchor 晶胞编号.
- `defect_oxygen_sites::Vector{Int}`: defect 对应的 oxygen site.
- `defect_cu_patch_sites::Vector{Int}`: defect 附近的 Cu site.
"""
struct DefectThreeBandMetadata
    geom::ThreeBandGeometry
    anchors::Vector{Tuple{Int,Int}}
    anchor_cells::Vector{Int}
    defect_oxygen_sites::Vector{Int}
    defect_cu_patch_sites::Vector{Int}
end

"""
    _site_at(geom, x, y, orbital)

用途: 用 1-based 晶胞坐标和 orbital 编号得到 site 编号。

参数:
- `geom::ThreeBandGeometry`: 三带几何.
- `x`, `y`: 1-based 晶胞坐标.
- `orbital::Integer`: orbital 编号, `1=d`, `2=py`, `3=px`.

返回:
- `Int`: 1-based site 编号.
"""
_site_at(geom::ThreeBandGeometry, x::Integer, y::Integer, orbital::Integer) =
    site_index(geom, cell_index(geom, x, y), orbital)

function DefectThreeBandMetadata(geom::ThreeBandGeometry, anchors)
    normalized_anchors = [(mod1(Int(x), geom.Lx), mod1(Int(y), geom.Ly)) for (x, y) in anchors]
    anchor_cells = [cell_index(geom, x, y) for (x, y) in normalized_anchors]
    defect_oxygen_sites = Int[]
    defect_cu_patch_sites = Int[]
    for (x, y) in normalized_anchors
        append!(defect_oxygen_sites, (
            _site_at(geom, x, y, ORB_PY),
            _site_at(geom, x, y, ORB_PX),
            _site_at(geom, x + 1, y, ORB_PY),
            _site_at(geom, x, y + 1, ORB_PX),
        ))
        append!(defect_cu_patch_sites, (
            _site_at(geom, x, y, ORB_D),
            _site_at(geom, x + 1, y, ORB_D),
            _site_at(geom, x, y + 1, ORB_D),
            _site_at(geom, x + 1, y + 1, ORB_D),
        ))
    end
    return DefectThreeBandMetadata(
        geom,
        collect(normalized_anchors),
        unique(anchor_cells),
        sort(unique(defect_oxygen_sites)),
        sort(unique(defect_cu_patch_sites)),
    )
end

"""
    _add_hopping_terms!(terms, site_i, site_j, coef)

用途: 向 `GeneralModel` 项列表加入 spin up/down 的 hopping 及其 Hermitian conjugate。

参数:
- `terms::Vector{OperatorTerm}`: 待写入的 Hamiltonian 项.
- `site_i`, `site_j`: hopping 两端 site.
- `coef::Real`: `c_i^dagger c_j + h.c.` 的系数.

返回:
- `nothing`.
"""
function _add_hopping_terms!(terms::Vector{OperatorTerm}, site_i::Integer, site_j::Integer, coef::Real)
    for (cdag, c) in ((Int(site_i), Int(site_j)), (Int(site_j), Int(site_i)))
        push!(terms, OperatorTerm(Symbol[:cdag_up, :c_up], Int[cdag, c], Float64(coef)))
        push!(terms, OperatorTerm(Symbol[:cdag_dn, :c_dn], Int[cdag, c], Float64(coef)))
    end
    return nothing
end

"""
    build_pd_bonds(geom), build_pp_bonds(geom), build_dd_bonds(geom)

用途: 构造三带模型中 Cu-O, O-O, Cu-Cu 的简化周期边界 bond 列表。

参数:
- `geom::ThreeBandGeometry`: 三带几何.

返回:
- `Vector{Tuple{Int,Int}}`: site pair 列表, 每条 bond 只出现一次.
"""
function build_pd_bonds(geom::ThreeBandGeometry)
    bonds = Tuple{Int,Int}[]
    for y in 1:geom.Ly, x in 1:geom.Lx
        d_site = site_index(geom, cell_index(geom, x, y), ORB_D)
        for oxygen_site in (
            site_index(geom, cell_index(geom, x, y), ORB_PX),
            site_index(geom, cell_index(geom, x - 1, y), ORB_PX),
            site_index(geom, cell_index(geom, x, y), ORB_PY),
            site_index(geom, cell_index(geom, x, y - 1), ORB_PY),
        )
            push!(bonds, extrema((d_site, oxygen_site)))
        end
    end
    return sort(unique(bonds))
end

function build_pp_bonds(geom::ThreeBandGeometry)
    bonds = Tuple{Int,Int}[]
    for y in 1:geom.Ly, x in 1:geom.Lx
        py_site = site_index(geom, cell_index(geom, x, y), ORB_PY)
        for px_site in (
            site_index(geom, cell_index(geom, x, y), ORB_PX),
            site_index(geom, cell_index(geom, x - 1, y), ORB_PX),
            site_index(geom, cell_index(geom, x, y + 1), ORB_PX),
            site_index(geom, cell_index(geom, x - 1, y + 1), ORB_PX),
        )
            push!(bonds, extrema((py_site, px_site)))
        end
    end
    return sort(unique(bonds))
end

function build_dd_bonds(geom::ThreeBandGeometry)
    bonds = Tuple{Int,Int}[]
    for y in 1:geom.Ly, x in 1:geom.Lx
        d_site = site_index(geom, cell_index(geom, x, y), ORB_D)
        push!(bonds, extrema((d_site, site_index(geom, cell_index(geom, x + 1, y), ORB_D))))
        push!(bonds, extrema((d_site, site_index(geom, cell_index(geom, x, y + 1), ORB_D))))
    end
    return sort(unique(bonds))
end

"""
    DefectHoppingBond

用途: 描述一个可独立优化的 defect hopping bond。

字段:
- `name::Symbol`: 参数名, 如 `:chi_def_pd_1`.
- `kind::Symbol`: bond 类型, 可为 `:pd`, `:pp`, `:dd`.
- `i::Int`, `j::Int`: 两端 site 编号.
- `base_coef::Float64`: clean mean-field 中该 bond 的参考系数符号和幅度.
- `direction_label::Symbol`: 几何方向标签.
- `source_anchor_ids::Vector{Int}`: 该 bond 来源于哪些 defect anchor.
"""
struct DefectHoppingBond
    name::Symbol
    kind::Symbol
    i::Int
    j::Int
    base_coef::Float64
    direction_label::Symbol
    source_anchor_ids::Vector{Int}
end

"""
    DefectHoppingBonds

用途: 分类型保存 defect hopping bond 列表。
"""
struct DefectHoppingBonds
    pd::Vector{DefectHoppingBond}
    pp::Vector{DefectHoppingBond}
    dd::Vector{DefectHoppingBond}
end

_bond_key(i::Int, j::Int, kind::Symbol) = (kind, min(i, j), max(i, j))

function _cell_coords_of_site(geom::ThreeBandGeometry, site::Int)
    cell = cell_of_site(geom, site)
    return mod(cell - 1, geom.Lx) + 1, div(cell - 1, geom.Lx) + 1
end

function _direction_label(geom::ThreeBandGeometry, site_i::Int, site_j::Int)
    x_i, y_i = _cell_coords_of_site(geom, site_i)
    x_j, y_j = _cell_coords_of_site(geom, site_j)
    dx = mod(x_j - x_i + geom.Lx, geom.Lx)
    dy = mod(y_j - y_i + geom.Ly, geom.Ly)
    return Symbol("dx$(dx)_dy$(dy)")
end

function _anchor_oxygen_site_set(geom::ThreeBandGeometry, anchor::Tuple{Int,Int})
    x, y = anchor
    return Set((
        _site_at(geom, x, y, ORB_PY),
        _site_at(geom, x, y, ORB_PX),
        _site_at(geom, x + 1, y, ORB_PY),
        _site_at(geom, x, y + 1, ORB_PX),
    ))
end

function _anchor_cu_patch_site_set(geom::ThreeBandGeometry, anchor::Tuple{Int,Int})
    x, y = anchor
    return Set((
        _site_at(geom, x, y, ORB_D),
        _site_at(geom, x + 1, y, ORB_D),
        _site_at(geom, x, y + 1, ORB_D),
        _site_at(geom, x + 1, y + 1, ORB_D),
    ))
end

_anchor_oxygen_site_sets(metadata::DefectThreeBandMetadata) =
    [_anchor_oxygen_site_set(metadata.geom, anchor) for anchor in metadata.anchors]

_anchor_cu_patch_site_sets(metadata::DefectThreeBandMetadata) =
    [_anchor_cu_patch_site_set(metadata.geom, anchor) for anchor in metadata.anchors]

_oxygen_source_anchor_ids(anchor_oxygen_sites, site_i::Int, site_j::Int) =
    [anchor_id for (anchor_id, sites) in pairs(anchor_oxygen_sites) if site_i in sites || site_j in sites]

_cu_patch_source_anchor_ids(anchor_cu_patch_sites, site_i::Int, site_j::Int) =
    [anchor_id for (anchor_id, sites) in pairs(anchor_cu_patch_sites) if site_i in sites || site_j in sites]

function _candidate_sort_key(candidate)
    key = _bond_key(candidate.i, candidate.j, candidate.kind)
    return (String(key[1]), key[2], key[3], candidate.i, candidate.j, String(candidate.direction_label))
end

function _finalize_defect_bonds(candidates, prefix::AbstractString)
    sorted_candidates = sort(collect(candidates); by=_candidate_sort_key)
    isempty(sorted_candidates) && return DefectHoppingBond[]
    representatives = Dict{Tuple{Symbol,Int,Int},eltype(sorted_candidates)}()
    summed_base_coef = Dict{Tuple{Symbol,Int,Int},Float64}()
    source_anchor_ids = Dict{Tuple{Symbol,Int,Int},Set{Int}}()

    for candidate in sorted_candidates
        key = _bond_key(candidate.i, candidate.j, candidate.kind)
        get!(representatives, key, candidate)
        summed_base_coef[key] = get(summed_base_coef, key, 0.0) + candidate.base_coef
        union!(get!(source_anchor_ids, key, Set{Int}()), candidate.source_anchor_ids)
    end

    sorted_keys = sort(collect(keys(summed_base_coef)); by=key -> (String(key[1]), key[2], key[3]))
    return [
        DefectHoppingBond(
            Symbol("$(prefix)_$(id)"),
            representatives[key].kind,
            representatives[key].i,
            representatives[key].j,
            summed_base_coef[key],
            representatives[key].direction_label,
            sort(collect(source_anchor_ids[key])),
        ) for (id, key) in pairs(sorted_keys)
    ]
end

function _clean_bond_candidates(geom::ThreeBandGeometry, kind::Symbol, bonds, anchor_oxygen_sites)
    candidates = NamedTuple[]
    for (site_i, site_j) in bonds
        source_anchor_ids = _oxygen_source_anchor_ids(anchor_oxygen_sites, site_i, site_j)
        isempty(source_anchor_ids) && continue
        push!(
            candidates,
            (
                kind=kind,
                i=site_i,
                j=site_j,
                base_coef=-1.0,
                direction_label=_direction_label(geom, site_i, site_j),
                source_anchor_ids=source_anchor_ids,
            ),
        )
    end
    return candidates
end

function _dd_bond_candidates(geom::ThreeBandGeometry, defect_cu_patch_sites, anchor_cu_patch_sites)
    candidates = NamedTuple[]
    defect_cu_set = Set(defect_cu_patch_sites)
    for d_site in sort(collect(defect_cu_set))
        x, y = _cell_coords_of_site(geom, d_site)
        for neighbor in (_site_at(geom, x + 1, y, ORB_D), _site_at(geom, x, y + 1, ORB_D))
            source_anchor_ids = _cu_patch_source_anchor_ids(anchor_cu_patch_sites, d_site, neighbor)
            if neighbor in defect_cu_set && !isempty(source_anchor_ids)
                push!(
                    candidates,
                    (
                        kind=:dd,
                        i=d_site,
                        j=neighbor,
                        base_coef=-1.0,
                        direction_label=_direction_label(geom, d_site, neighbor),
                        source_anchor_ids=source_anchor_ids,
                    ),
                )
            end
        end
    end
    return candidates
end

"""
    build_defect_hopping_bonds(metadata)

用途: 按同事代码的规则构造 defect hopping 参数对应的 bond 列表。

参数:
- `metadata::DefectThreeBandMetadata`: defect anchor 派生信息.

返回:
- `DefectHoppingBonds`: `pd`, `pp`, `dd` 三类 defect bond。
"""
function build_defect_hopping_bonds(metadata::DefectThreeBandMetadata)
    geom = metadata.geom
    anchor_oxygen_sites = _anchor_oxygen_site_sets(metadata)
    anchor_cu_patch_sites = _anchor_cu_patch_site_sets(metadata)
    defect_cu_patch_sites = Set(metadata.defect_cu_patch_sites)

    pd_candidates = _clean_bond_candidates(geom, :pd, build_pd_bonds(geom), anchor_oxygen_sites)
    pp_candidates = _clean_bond_candidates(geom, :pp, build_pp_bonds(geom), anchor_oxygen_sites)
    dd_candidates = _dd_bond_candidates(geom, defect_cu_patch_sites, anchor_cu_patch_sites)

    return DefectHoppingBonds(
        _finalize_defect_bonds(pd_candidates, "chi_def_pd"),
        _finalize_defect_bonds(pp_candidates, "chi_def_pp"),
        _finalize_defect_bonds(dd_candidates, "chi_def_dd"),
    )
end

_defect_all_hopping_bonds(bonds::DefectHoppingBonds) =
    Iterators.flatten((bonds.pd, bonds.pp, bonds.dd))

_defect_hopping_reference_sign(bond::DefectHoppingBond) =
    iszero(bond.base_coef) ? 1.0 : sign(bond.base_coef)

"""
    DefectThreeBandHubbardModel(geom, metadata; ...)

用途: 用主 `GeneralModel` 构造 defect three-band Hubbard 物理 Hamiltonian。

参数:
- `geom::ThreeBandGeometry`: 三带几何.
- `metadata::DefectThreeBandMetadata`: defect 信息.
- `tpd`, `tpp`: Cu-O 和 O-O hopping.
- `Delta_pd`: oxygen onsite energy.
- `Udd`, `Up`: Cu 和 oxygen onsite Hubbard U.
- `Vpd`: Cu-O density-density 相互作用.
- `defect_Epp`: defect oxygen onsite energy shift.

返回:
- `GeneralModel`: 主 `src/Model.jl` 的通用 Hamiltonian.
"""
function DefectThreeBandHubbardModel(
    geom::ThreeBandGeometry,
    metadata::DefectThreeBandMetadata;
    tpd::Real=1.0,
    tpp::Real=0.0,
    Delta_pd::Real=0.0,
    Udd::Real=8.0,
    Up::Real=0.0,
    Vpd::Real=0.0,
    defect_Epp::Real=0.0,
)
    terms = OperatorTerm[]
    defect_oxygen_set = Set(metadata.defect_oxygen_sites)
    for site in 1:n_sites(geom)
        orbital = orbital_of_site(geom, site)
        if orbital == ORB_D
            push!(terms, OperatorTerm(Symbol[:n_up, :n_dn], Int[site, site], Float64(Udd)))
        else
            push!(terms, OperatorTerm(Symbol[:n], Int[site], Float64(Delta_pd)))
            Up == 0 || push!(terms, OperatorTerm(Symbol[:n_up, :n_dn], Int[site, site], Float64(Up)))
            if site in defect_oxygen_set
                push!(terms, OperatorTerm(Symbol[:n], Int[site], Float64(defect_Epp)))
            end
        end
    end
    for (site_i, site_j) in build_pd_bonds(geom)
        _add_hopping_terms!(terms, site_i, site_j, -Float64(tpd))
        Vpd == 0 || push!(terms, OperatorTerm(Symbol[:n, :n], Int[site_i, site_j], Float64(Vpd)))
    end
    for (site_i, site_j) in build_pp_bonds(geom)
        tpp == 0 || _add_hopping_terms!(terms, site_i, site_j, -Float64(tpp))
    end
    return GeneralModel(n_sites(geom), terms)
end

"""
    _value_or_fallback(value, fallback)

用途: 将可选参数转为 `Float64`, 若为 `nothing` 则使用 fallback。

参数:
- `value`: 参数值或 `nothing`.
- `fallback`: 缺省值.

返回:
- `Float64`: 解析后的数值.
"""
_value_or_fallback(value, fallback) = value === nothing ? Float64(fallback) : Float64(value)

"""
    _defect_background_gutzwiller_args(args)

用途: 读取 orbital-resolved Gutzwiller 的背景初值。

参数:
- `args`: 命令行参数字典.

返回:
- `(g_d, g_py, g_px)`: 每个元素为 `Float64` 或 `nothing`, 其中 `nothing` 表示沿用 `g_site_init`.
"""
function _defect_background_gutzwiller_args(args)
    return (
        _nan_to_nothing(Float64(_arg(args, :g_d, NaN))),
        _nan_to_nothing(Float64(_arg(args, :g_py, NaN))),
        _nan_to_nothing(Float64(_arg(args, :g_px, NaN))),
    )
end

"""
    _defect_orbital_gutzwiller_values(args)

用途: 构造 orbital-resolved onsite Gutzwiller 的 `[g_d, g_py, g_px]` 初值。

参数:
- `args`: 命令行参数字典.

返回:
- `Vector{Float64}`: 与 `[:g_d, :g_py, :g_px]` 对齐的初值.
"""
function _defect_orbital_gutzwiller_values(args)
    g_site_init = Float64(_arg(args, :g_site_init, 0.0))
    g_d, g_py, g_px = _defect_background_gutzwiller_args(args)
    return Float64[
        _value_or_fallback(g_d, g_site_init),
        _value_or_fallback(g_py, g_site_init),
        _value_or_fallback(g_px, g_site_init),
    ]
end

"""
    _load_defect_clean_jastrow_init_file(path, expected_names)

用途: 从两列文本文件读取 clean orbital-pair Jastrow 初值。

参数:
- `path::AbstractString`: 文本路径, 每行格式为 `param_name value`.
- `expected_names`: 允许出现的 Jastrow 参数名.

返回:
- `Vector{Float64}`: 按 `expected_names` 顺序排列的初值.
"""
function _load_defect_clean_jastrow_init_file(path::AbstractString, expected_names)
    names = Symbol.(expected_names)
    values = Dict{Symbol,Float64}()
    isfile(path) || throw(ArgumentError("Jastrow initialization file does not exist: $path"))

    open(path, "r") do io
        for (line_no, raw_line) in enumerate(eachline(io))
            line = strip(split(raw_line, "#"; limit=2)[1])
            isempty(line) && continue
            fields = split(line)
            length(fields) == 2 ||
                throw(ArgumentError("invalid Jastrow initialization line $line_no in $path; expected: param_name value"))
            name = Symbol(fields[1])
            name in names ||
                throw(ArgumentError("unknown clean Jastrow parameter $name in $path line $line_no"))
            haskey(values, name) &&
                throw(ArgumentError("duplicate clean Jastrow parameter $name in $path line $line_no"))
            values[name] = parse(Float64, fields[2])
        end
    end

    missing_names = [name for name in names if !haskey(values, name)]
    isempty(missing_names) ||
        throw(ArgumentError("Jastrow initialization file $path misses parameters: $missing_names"))
    return [values[name] for name in names]
end

"""
    _defect_clean_jastrow_names_values(args)

用途: 根据 `--jastrow_shells` 和初值设置生成 clean orbital-pair Jastrow 参数。

参数:
- `args`: 命令行参数字典.

返回:
- `(n_shells, names, values)`: shell 数, 参数名, 参数初值.
"""
function _defect_clean_jastrow_names_values(args)
    n_shells = Int(_arg(args, :jastrow_shells, 0))
    n_shells >= 0 || throw(ArgumentError("jastrow_shells must be non-negative"))
    names = threeband_jastrow_names(n_shells)
    init_file = String(_arg(args, :jastrow_init_file, ""))
    values = if n_shells == 0
        Float64[]
    elseif isempty(init_file)
        fill(Float64(_arg(args, :jastrow_init, 0.0)), length(names))
    else
        _load_defect_clean_jastrow_init_file(init_file, names)
    end
    return n_shells, names, values
end

"""
    _build_jastrow_terms_from_pair_index(names, values, pair_index)

用途: 将参数编号矩阵转换为主 `JastrowProjectorTerm` 使用的邻接表。

参数:
- `names`: Jastrow 参数名列表.
- `values`: Jastrow 参数值列表.
- `pair_index::AbstractMatrix{<:Integer}`: `pair_index[i,j]` 给出 pair `(i,j)` 属于哪个参数, `0` 表示不耦合.

返回:
- `Vector{AbstractProjectorTerm}`: 每个参数一个 `JastrowProjectorTerm`.
"""
function _build_jastrow_terms_from_pair_index(names, values, pair_index)
    name_vector = Symbol.(names)
    value_vector = Float64.(values)
    length(name_vector) == length(value_vector) ||
        throw(ArgumentError("Jastrow names and values must have the same length"))
    size(pair_index, 1) == size(pair_index, 2) ||
        throw(ArgumentError("Jastrow pair_index must be square"))

    n_sites_total = size(pair_index, 1)
    terms = AbstractProjectorTerm[]
    for param_id in eachindex(name_vector)
        site_to_neighbor_sites = [Int[] for _ in 1:n_sites_total]
        for site_i in 1:n_sites_total
            for site_j in (site_i + 1):n_sites_total
                if Int(pair_index[site_i, site_j]) == param_id || Int(pair_index[site_j, site_i]) == param_id
                    push!(site_to_neighbor_sites[site_i], site_j)
                    push!(site_to_neighbor_sites[site_j], site_i)
                end
            end
        end
        push!(
            terms,
            JastrowProjectorTerm(
                param_name=name_vector[param_id],
                v=value_vector[param_id],
                site_to_neighbor_sites=site_to_neighbor_sites,
            ),
        )
    end
    return terms
end

"""
    _build_clean_shell_jastrow_terms(geom, n_shells, names, values)

用途: 构造不区分 defect/clean 的 clean orbital-pair density Jastrow projector 项。

参数:
- `geom`: `ThreeBandGeometry`.
- `n_shells::Integer`: Jastrow cell-distance shell 数.
- `names`: Jastrow 参数名.
- `values`: Jastrow 参数值.

返回:
- `Vector{AbstractProjectorTerm}`: 每个 Jastrow 参数对应一个 `JastrowProjectorTerm`.

公式:
- projector 形式为 `exp(-sum_a v_a sum_{(i,j) in E_a} n_i n_j)`,
  其中 `E_a` 由 orbital pair 类型和 cell-distance shell 决定。
"""
function _build_clean_shell_jastrow_terms(geom, n_shells::Integer, names, values)
    n_shells_int = Int(n_shells)
    n_shells_int > 0 || return AbstractProjectorTerm[]

    pair_index = build_orbital_pair_jastrow_index(geom, n_shells_int; pair_mode=:dp_pp)
    active_indices = Set(vec(pair_index))
    delete!(active_indices, 0)
    !isempty(active_indices) ||
        throw(ArgumentError("jastrow_shells produces no nonzero orbital-pair Jastrow pairs for this geometry"))
    inactive = setdiff(collect(1:length(names)), active_indices)
    isempty(inactive) ||
        throw(ArgumentError("jastrow_shells creates inactive orbital-pair Jastrow parameters $(names[inactive]); reduce jastrow_shells or use a larger geometry"))

    return _build_jastrow_terms_from_pair_index(names, values, pair_index)
end

"""
    _compact_jastrow_pair_index(names, values, pair_index)

用途: 删除已经没有任何 pair 的 Jastrow 参数, 并把 `pair_index` 重新压缩到连续编号。

参数:
- `names`: 原始 Jastrow 参数名。
- `values`: 原始 Jastrow 参数值。
- `pair_index::AbstractMatrix{<:Integer}`: pair 到参数编号的映射, `0` 表示不耦合。

返回:
- `(names, values, pair_index)`: 只保留 active 参数后的压缩结果。
"""
function _compact_jastrow_pair_index(names, values, pair_index)
    name_vector = Symbol.(names)
    value_vector = Float64.(values)
    length(name_vector) == length(value_vector) ||
        throw(ArgumentError("Jastrow names and values must have the same length"))

    active_ids = sort(unique([Int(param_id) for param_id in pair_index if Int(param_id) != 0]))
    isempty(active_ids) && return (
        names=Symbol[],
        values=Float64[],
        pair_index=zeros(Int, size(pair_index)...),
    )

    remap = Dict(old_id => new_id for (new_id, old_id) in pairs(active_ids))
    compact_pair_index = zeros(Int, size(pair_index))
    for index in CartesianIndices(pair_index)
        old_id = Int(pair_index[index])
        old_id == 0 && continue
        compact_pair_index[index] = remap[old_id]
    end

    return (
        names=name_vector[active_ids],
        values=value_vector[active_ids],
        pair_index=compact_pair_index,
    )
end

"""
    build_shell_split_defect_jastrow_projector_data(geom, metadata, n_shells; ...)

用途: 按同事代码的 cost 口径拆分 clean Jastrow 和 defect Jastrow。只有原本处在
clean shell 内, 且至少一端是 defect oxygen site 的 pair 会从 clean 参数中移出,
并变成独立的 `v_defpair_*` 参数。

参数:
- `geom`: `ThreeBandGeometry`。
- `metadata`: `DefectThreeBandMetadata`。
- `n_shells::Integer`: clean Jastrow shell 数。
- `clean_names`: 原 clean Jastrow 参数名。
- `clean_values`: 原 clean Jastrow 参数值。
- `defect_init::Real`: defect-pair Jastrow 参数初值。

返回:
- named tuple, 字段为 `clean_names`, `clean_values`, `clean_pair_index`,
  `defect_names`, `defect_values`, `defect_pair_index`。

公式:
- clean 部分: `P_clean = exp(-sum_a v_a sum_{(i,j) in E_a clean} n_i n_j)`。
- defect 部分: `P_def = exp(-sum_b v_b n_{i_b} n_{j_b})`, 其中 `(i_b,j_b)`
  是从 clean shell 中被 defect oxygen 替换出来的 pair。
"""
function build_shell_split_defect_jastrow_projector_data(
    geom,
    metadata,
    n_shells::Integer;
    clean_names,
    clean_values,
    defect_init::Real=0.0,
)
    n_shells_int = Int(n_shells)
    n_shells_int > 0 || return (
        clean_names=Symbol[],
        clean_values=Float64[],
        clean_pair_index=zeros(Int, n_sites(geom), n_sites(geom)),
        defect_names=Symbol[],
        defect_values=Float64[],
        defect_pair_index=zeros(Int, n_sites(geom), n_sites(geom)),
    )

    expected_clean_names = threeband_jastrow_names(n_shells_int)
    clean_name_vector = Symbol.(clean_names)
    clean_value_vector = Float64.(clean_values)
    clean_name_vector == expected_clean_names ||
        throw(ArgumentError("clean_names must match threeband_jastrow_names(n_shells)"))
    length(clean_value_vector) == length(clean_name_vector) ||
        throw(ArgumentError("clean_values length must match clean_names"))

    clean_pair_index = build_orbital_pair_jastrow_index(geom, n_shells_int; pair_mode=:dp_pp)
    active_indices = Set(vec(clean_pair_index))
    delete!(active_indices, 0)
    inactive = setdiff(collect(1:length(clean_name_vector)), active_indices)
    isempty(inactive) ||
        throw(ArgumentError("jastrow_shells creates inactive orbital-pair Jastrow parameters $(clean_name_vector[inactive]); reduce jastrow_shells or use a larger geometry"))

    remaining_clean_pair_index = copy(clean_pair_index)
    defect_pair_index = zeros(Int, n_sites(geom), n_sites(geom))
    defect_sites = Set(Int.(metadata.defect_oxygen_sites))
    defect_pair_count = 0
    for site_i in 1:(n_sites(geom) - 1)
        for site_j in (site_i + 1):n_sites(geom)
            clean_pair_index[site_i, site_j] == 0 && continue
            (site_i in defect_sites || site_j in defect_sites) || continue

            remaining_clean_pair_index[site_i, site_j] = 0
            remaining_clean_pair_index[site_j, site_i] = 0
            defect_pair_count += 1
            defect_pair_index[site_i, site_j] = defect_pair_count
            defect_pair_index[site_j, site_i] = defect_pair_count
        end
    end

    compact_clean = _compact_jastrow_pair_index(
        clean_name_vector,
        clean_value_vector,
        remaining_clean_pair_index,
    )
    defect_names = [Symbol("v_defpair_$idx") for idx in 1:defect_pair_count]
    defect_values = fill(Float64(defect_init), defect_pair_count)

    return (
        clean_names=compact_clean.names,
        clean_values=compact_clean.values,
        clean_pair_index=compact_clean.pair_index,
        defect_names=defect_names,
        defect_values=defect_values,
        defect_pair_index=defect_pair_index,
    )
end

"""
    defect_pair_jastrow_names(metadata, geom)

用途: 生成 defect-pair Jastrow 参数名。每个 defect oxygen site 与一个非 defect oxygen site 的 pair 对应一个参数。

参数:
- `metadata`: `DefectThreeBandMetadata`, 需要提供 `defect_oxygen_sites`.
- `geom`: `ThreeBandGeometry`.

返回:
- `Vector{Symbol}`: `:v_defpair_1, :v_defpair_2, ...`.
"""
function defect_pair_jastrow_names(metadata, geom)
    defect_sites = Set(Int.(metadata.defect_oxygen_sites))
    count = 0
    for defect_site in sort(collect(defect_sites))
        for other_site in 1:n_sites(geom)
            other_site == defect_site && continue
            other_site in defect_sites && continue
            count += 1
        end
    end
    return [Symbol("v_defpair_$idx") for idx in 1:count]
end

"""
    build_defect_pair_jastrow_projector_terms(metadata, geom; defect_init=0.0)

用途: 构造 defect-pair density Jastrow projector 项。

参数:
- `metadata`: `DefectThreeBandMetadata`, 需要提供 `defect_oxygen_sites`.
- `geom`: `ThreeBandGeometry`.
- `defect_init::Real`: 每个 defect-pair Jastrow 参数初值.

返回:
- `Vector{AbstractProjectorTerm}`: 每个 defect pair 一个 `JastrowProjectorTerm`.

公式:
- `P_def(C) = exp(-sum_a v_defpair_a n_{i_a} n_{j_a})`。
"""
function build_defect_pair_jastrow_projector_terms(metadata, geom; defect_init::Real=0.0)
    defect_sites = Set(Int.(metadata.defect_oxygen_sites))
    terms = AbstractProjectorTerm[]
    param_index = 0
    for defect_site in sort(collect(defect_sites))
        for other_site in 1:n_sites(geom)
            other_site == defect_site && continue
            other_site in defect_sites && continue
            param_index += 1
            site_to_neighbor_sites = [Int[] for _ in 1:n_sites(geom)]
            push!(site_to_neighbor_sites[defect_site], other_site)
            push!(site_to_neighbor_sites[other_site], defect_site)
            push!(
                terms,
                JastrowProjectorTerm(
                    param_name=Symbol("v_defpair_$param_index"),
                    v=Float64(defect_init),
                    site_to_neighbor_sites=site_to_neighbor_sites,
                ),
            )
        end
    end
    return terms
end

"""
    build_shell_split_defect_jastrow_projector_terms(geom, metadata, n_shells; ...)

用途: 同时构造 clean orbital-pair Jastrow 和 defect-pair Jastrow 的 projector 项。

参数:
- `geom`: `ThreeBandGeometry`.
- `metadata`: `DefectThreeBandMetadata`.
- `n_shells::Integer`: clean Jastrow shell 数.
- `clean_names`: clean Jastrow 参数名.
- `clean_values`: clean Jastrow 参数值.
- `defect_init::Real`: defect-pair Jastrow 初值.

返回:
- `(clean_terms, defect_terms)`: 两组 `Vector{AbstractProjectorTerm}`.
"""
function build_shell_split_defect_jastrow_projector_terms(
    geom,
    metadata,
    n_shells::Integer;
    clean_names,
    clean_values,
    defect_init::Real=0.0,
)
    split_data = build_shell_split_defect_jastrow_projector_data(
        geom,
        metadata,
        n_shells;
        clean_names=clean_names,
        clean_values=clean_values,
        defect_init=defect_init,
    )
    clean_terms = _build_jastrow_terms_from_pair_index(
        split_data.clean_names,
        split_data.clean_values,
        split_data.clean_pair_index,
    )
    defect_terms = _build_jastrow_terms_from_pair_index(
        split_data.defect_names,
        split_data.defect_values,
        split_data.defect_pair_index,
    )
    return clean_terms, defect_terms
end

"""
    _defect_threeband_geometry_from_args(args)

用途: 从命令行参数构造三带晶格几何。

参数:
- `args`: 命令行参数字典.

返回:
- `ThreeBandGeometry`.
"""
function _defect_threeband_geometry_from_args(args)
    return ThreeBandGeometry(
        Int(_arg(args, :Lx)),
        Int(_arg(args, :Ly));
        bcx=Float64(_arg(args, :bcx, 1.0)),
        bcy=Float64(_arg(args, :bcy, 1.0)),
    )
end

"""
    _validate_wrapped_defect_anchors(geom, anchors)

用途: 检查 defect anchor 在周期边界归一化后是否重复。

参数:
- `geom::ThreeBandGeometry`: 三带几何.
- `anchors`: 1-based anchor 坐标列表.

返回:
- `nothing`.
"""
function _validate_wrapped_defect_anchors(geom, anchors)
    seen = Set{Tuple{Int,Int}}()
    for anchor in anchors
        wrapped = (mod1(Int(anchor[1]), geom.Lx), mod1(Int(anchor[2]), geom.Ly))
        wrapped in seen && throw(ArgumentError("duplicate geometry-normalized defect anchor: $wrapped"))
        push!(seen, wrapped)
    end
    return nothing
end

"""
    canonical_defect_threeband_nelec(args)

用途: 根据 `--nelec` 或 `--Nhole` 确定固定电子数。

参数:
- `args`: 命令行参数字典.

返回:
- `Int`: 固定电子数 `N_e`.

公式:
- 若 `nelec > 0`, 使用 `nelec`.
- 否则使用 `N_e = Lx * Ly - Nhole`, 即相对每个 unit cell 一个电子的 hole 计数约定.
"""
function canonical_defect_threeband_nelec(args)
    geom = _defect_threeband_geometry_from_args(args)
    nsite = n_sites(geom)
    nelec_arg = Int(_arg(args, :nelec, 0))
    nhole = Int(_arg(args, :Nhole, 0))
    nhole >= 0 || throw(ArgumentError("Nhole must be non-negative"))

    nelec = nelec_arg > 0 ? nelec_arg : Int(_arg(args, :Lx)) * Int(_arg(args, :Ly)) - nhole
    0 < nelec <= 2 * nsite ||
        throw(ArgumentError("nelec must satisfy 0 < nelec <= $(2 * nsite) for this geometry"))
    return nelec
end

"""
    defect_threeband_spin_counts(nelec, target_sz, max_per_spin=nothing)

用途: 在固定电子数和固定 `N_up - N_down` 下计算两个自旋粒子数。

参数:
- `nelec::Integer`: 总电子数.
- `target_sz::Integer`: `N_up - N_down`.
- `max_per_spin`: 每个 spin sector 的最大容量, 可为 `nothing`.

返回:
- `(nup, ndn)`: 两个 `Int`.

公式:
- `N_up = (N_e + target_sz) / 2`
- `N_down = N_e - N_up`
"""
function defect_threeband_spin_counts(nelec::Integer, target_sz::Integer, max_per_spin::Union{Nothing,Integer}=nothing)
    (Int(nelec) + Int(target_sz)) % 2 == 0 ||
        throw(ArgumentError("nelec + target_sz must be even"))
    nup = div(Int(nelec) + Int(target_sz), 2)
    ndn = Int(nelec) - nup
    nup >= 0 && ndn >= 0 ||
        throw(ArgumentError("target_sz is incompatible with nelec"))
    if max_per_spin !== nothing
        nmax = Int(max_per_spin)
        nup <= nmax && ndn <= nmax ||
            throw(ArgumentError("per-spin occupancy must satisfy nup, ndn <= $nmax"))
    end
    return nup, ndn
end

"""
    build_defect_threeband_projector(geom, metadata, args)

用途: 使用主 `src/Projector.jl` 的 API 构造 site/orbital Gutzwiller 与 Jastrow projector。

参数:
- `geom`: `ThreeBandGeometry`.
- `metadata`: `DefectThreeBandMetadata`.
- `args`: 命令行参数字典.

返回:
- `(projector, names, values)`: projector 对象或 `nothing`, 参数名, 参数值.

公式:
- onsite Gutzwiller: `exp(sum_i g_i n_{i up} n_{i down})`.
- density Jastrow: `exp(-sum_{i<j} v_{ij} n_i n_j)`.
"""
function build_defect_threeband_projector(geom, metadata, args)
    n_shells, clean_names, clean_values = _defect_clean_jastrow_names_values(args)
    terms = AbstractProjectorTerm[]

    if Bool(_arg(args, :site_gutzwiller, false))
        push!(
            terms,
            SiteGroupGutzwillerProjectorTerm(
                param_names=[Symbol("g_site_$site") for site in 1:n_sites(geom)],
                g_values=fill(Float64(_arg(args, :g_site_init, 0.0)), n_sites(geom)),
                site_groups=collect(1:n_sites(geom)),
            ),
        )
    elseif Bool(_arg(args, :gutzwiller_orbital, false))
        push!(
            terms,
            SiteGroupGutzwillerProjectorTerm(
                param_names=Symbol[:g_d, :g_py, :g_px],
                g_values=_defect_orbital_gutzwiller_values(args),
                site_groups=orbital_group_vector(geom),
            ),
        )
    end

    if n_shells > 0
        if Bool(_arg(args, :defect_jastrow, false))
            clean_terms, defect_terms = build_shell_split_defect_jastrow_projector_terms(
                geom,
                metadata,
                n_shells;
                clean_names=clean_names,
                clean_values=clean_values,
                defect_init=Float64(_arg(args, :defect_jastrow_init, 0.0)),
            )
            append!(terms, clean_terms)
            append!(terms, defect_terms)
        else
            clean_terms = _build_clean_shell_jastrow_terms(
                geom,
                n_shells,
                clean_names,
                clean_values,
            )
            append!(terms, clean_terms)
        end
    end

    if isempty(terms)
        return nothing, Symbol[], Float64[]
    end
    projector = CompositeProjector(terms)
    check_projector_consistency(projector)
    return projector, projector_param_names(projector), projector_param_values(projector)
end

"""
    build_defect_initial_params(geom, metadata; ...)

用途: 构造 defect three-band 主程序的初始参数名和值。

参数:
- `geom::ThreeBandGeometry`: 三带几何.
- `metadata::DefectThreeBandMetadata`: defect 信息.
- keyword 参数: mean-field, Gutzwiller, Jastrow 初值和开关.

返回:
- `(param_names, init_params)`: 参数名和值, 顺序与后续 SR 优化一致.
"""
function build_defect_initial_params(
    geom::ThreeBandGeometry,
    metadata::DefectThreeBandMetadata;
    mu0::Real=0.0,
    mu1::Real=0.0,
    mu0_d0=nothing,
    mu1_d0=nothing,
    mz_00::Real=0.0,
    mz_11::Real=0.0,
    mz_00_d0=nothing,
    mz_11_d0=nothing,
    chi1_00::Real=0.0,
    chi1_01::Real=0.0,
    chi1_11::Real=0.0,
    chi_def_pd_init=nothing,
    chi_def_pp_init=nothing,
    chi_def_dd_init=nothing,
    g_site_init::Real=0.0,
    g_d=nothing,
    g_py=nothing,
    g_px=nothing,
    jastrow_shells::Integer=0,
    jastrow_init::Real=0.0,
    clean_jastrow_names=nothing,
    clean_jastrow_values=nothing,
    defect_jastrow_init::Real=0.0,
    use_site_gutzwiller::Bool=false,
    use_orbital_gutzwiller::Bool=false,
    use_defect_jastrow::Bool=false,
    uniform_nondefect_mu::Bool=false,
    site_resolved_oxygen_mz::Bool=false,
)
    names = Symbol[]
    values = Float64[]
    defect_oxygen_set = Set(metadata.defect_oxygen_sites)
    defect_cu_set = Set(metadata.defect_cu_patch_sites)

    if uniform_nondefect_mu
        append!(names, Symbol[:mu0, :mu1, :mu0_d0, :mu1_d0])
        append!(values, Float64[mu0, mu1, _value_or_fallback(mu0_d0, mu0), _value_or_fallback(mu1_d0, mu1)])
    else
        for site in 1:n_sites(geom)
            push!(names, Symbol("mu_site_$site"))
            if orbital_of_site(geom, site) == ORB_D
                push!(values, site in defect_cu_set ? _value_or_fallback(mu0_d0, mu0) : Float64(mu0))
            else
                push!(values, site in defect_oxygen_set ? _value_or_fallback(mu1_d0, mu1) : Float64(mu1))
            end
        end
    end

    for site in 1:n_sites(geom)
        if orbital_of_site(geom, site) == ORB_D
            push!(names, Symbol("mz_site_$site"))
            push!(values, site in defect_cu_set ? _value_or_fallback(mz_00_d0, mz_00) : Float64(mz_00))
        elseif site_resolved_oxygen_mz
            push!(names, Symbol("mz_site_$site"))
            push!(values, site in defect_oxygen_set ? _value_or_fallback(mz_11_d0, mz_11) : Float64(mz_11))
        end
    end
    if !site_resolved_oxygen_mz
        append!(names, Symbol[:mz_11, :mz_11_d0])
        append!(values, Float64[mz_11, _value_or_fallback(mz_11_d0, mz_11)])
    end

    append!(names, Symbol[:chi1_00, :chi1_01, :chi1_11])
    append!(values, Float64[chi1_00, chi1_01, chi1_11])

    defect_hopping_bonds = build_defect_hopping_bonds(metadata)
    for (bonds, init_value, fallback) in (
        (defect_hopping_bonds.pd, chi_def_pd_init, chi1_01),
        (defect_hopping_bonds.pp, chi_def_pp_init, chi1_11),
        (defect_hopping_bonds.dd, chi_def_dd_init, chi1_00),
    )
        for bond in bonds
            push!(names, bond.name)
            default_value = abs(bond.base_coef) * Float64(fallback)
            push!(values, _value_or_fallback(init_value, default_value))
        end
    end

    if use_site_gutzwiller
        for site in 1:n_sites(geom)
            push!(names, Symbol("g_site_$site"))
            push!(values, Float64(g_site_init))
        end
    elseif use_orbital_gutzwiller
        append!(names, Symbol[:g_d, :g_py, :g_px])
        append!(
            values,
            Float64[
                _value_or_fallback(g_d, g_site_init),
                _value_or_fallback(g_py, g_site_init),
                _value_or_fallback(g_px, g_site_init),
            ],
        )
    end

    n_shells = Int(jastrow_shells)
    if n_shells > 0
        jastrow_names = clean_jastrow_names === nothing ? threeband_jastrow_names(n_shells) : Symbol.(clean_jastrow_names)
        jastrow_values = clean_jastrow_values === nothing ?
            fill(Float64(jastrow_init), length(jastrow_names)) :
            Float64.(clean_jastrow_values)
        if use_defect_jastrow
            split_data = build_shell_split_defect_jastrow_projector_data(
                geom,
                metadata,
                n_shells;
                clean_names=jastrow_names,
                clean_values=jastrow_values,
                defect_init=Float64(defect_jastrow_init),
            )
            append!(names, split_data.clean_names)
            append!(values, split_data.clean_values)
            append!(names, split_data.defect_names)
            append!(values, split_data.defect_values)
        else
            append!(names, jastrow_names)
            append!(values, jastrow_values)
        end
    end
    return names, values
end

"""
    _param_value_map(param_names, params)

用途: 将参数名和值转换为查表字典。

参数:
- `param_names`: 参数名.
- `params`: 参数值.

返回:
- `Dict{Symbol,Float64}`.
"""
function _param_value_map(param_names, params)
    names = Symbol.(param_names)
    length(names) == length(params) || throw(ArgumentError("param_names and params length mismatch"))
    return Dict{Symbol,Float64}(name => Float64(value) for (name, value) in zip(names, params))
end

function _get_param(param_map::Dict{Symbol,Float64}, name::Symbol, default::Real=0.0)
    return get(param_map, name, Float64(default))
end

function _add_spatial_hopping!(matrix::AbstractMatrix, site_i::Integer, site_j::Integer, value)
    matrix[Int(site_i), Int(site_j)] += value
    matrix[Int(site_j), Int(site_i)] += value
    return nothing
end

"""
    _zero_spatial_hopping!(matrix, site_i, site_j)

用途: 将某个 spatial hopping bond 的矩阵元清零。

参数:
- `matrix::AbstractMatrix`: mean-field 矩阵.
- `site_i`, `site_j`: bond 两端 site.

返回:
- `nothing`.
"""
function _zero_spatial_hopping!(matrix::AbstractMatrix, site_i::Integer, site_j::Integer)
    matrix[Int(site_i), Int(site_j)] = 0.0
    matrix[Int(site_j), Int(site_i)] = 0.0
    return nothing
end

"""
    build_defect_meanfield_blocks(geom, metadata, param_names, params)

用途: 根据参数构造 spin up/down 的 mean-field 矩阵。

参数:
- `geom::ThreeBandGeometry`: 三带几何.
- `metadata::DefectThreeBandMetadata`: defect 信息.
- `param_names`, `params`: 完整参数名和值.

返回:
- `(ham_up, ham_down)`: 两个 `n_sites x n_sites` 实矩阵.
"""
function build_defect_meanfield_blocks(geom, metadata, param_names, params)
    param_map = _param_value_map(param_names, params)
    n_total = n_sites(geom)
    ham_up = zeros(Float64, n_total, n_total)
    ham_down = zeros(Float64, n_total, n_total)
    defect_oxygen_set = Set(metadata.defect_oxygen_sites)
    defect_cu_set = Set(metadata.defect_cu_patch_sites)

    for site in 1:n_total
        orbital = orbital_of_site(geom, site)
        mu = if haskey(param_map, Symbol("mu_site_$site"))
            param_map[Symbol("mu_site_$site")]
        elseif orbital == ORB_D
            site in defect_cu_set ? _get_param(param_map, :mu0_d0, _get_param(param_map, :mu0)) : _get_param(param_map, :mu0)
        else
            site in defect_oxygen_set ? _get_param(param_map, :mu1_d0, _get_param(param_map, :mu1)) : _get_param(param_map, :mu1)
        end
        mz = if haskey(param_map, Symbol("mz_site_$site"))
            param_map[Symbol("mz_site_$site")]
        elseif orbital == ORB_D
            0.0
        else
            site in defect_oxygen_set ? _get_param(param_map, :mz_11_d0, _get_param(param_map, :mz_11)) : _get_param(param_map, :mz_11)
        end
        ham_up[site, site] += mu + mz
        ham_down[site, site] += mu - mz
    end

    for (site_i, site_j) in build_dd_bonds(geom)
        _add_spatial_hopping!(ham_up, site_i, site_j, -_get_param(param_map, :chi1_00))
        _add_spatial_hopping!(ham_down, site_i, site_j, -_get_param(param_map, :chi1_00))
    end
    for (site_i, site_j) in build_pd_bonds(geom)
        _add_spatial_hopping!(ham_up, site_i, site_j, -_get_param(param_map, :chi1_01))
        _add_spatial_hopping!(ham_down, site_i, site_j, -_get_param(param_map, :chi1_01))
    end
    for (site_i, site_j) in build_pp_bonds(geom)
        _add_spatial_hopping!(ham_up, site_i, site_j, -_get_param(param_map, :chi1_11))
        _add_spatial_hopping!(ham_down, site_i, site_j, -_get_param(param_map, :chi1_11))
    end

    defect_hopping_bonds = build_defect_hopping_bonds(metadata)
    for bond in _defect_all_hopping_bonds(defect_hopping_bonds)
        _zero_spatial_hopping!(ham_up, bond.i, bond.j)
        _zero_spatial_hopping!(ham_down, bond.i, bond.j)
    end
    for bond in defect_hopping_bonds.pd
        value = _defect_hopping_reference_sign(bond) * _get_param(param_map, bond.name)
        _add_spatial_hopping!(ham_up, bond.i, bond.j, value)
        _add_spatial_hopping!(ham_down, bond.i, bond.j, value)
    end
    for bond in defect_hopping_bonds.pp
        value = _defect_hopping_reference_sign(bond) * _get_param(param_map, bond.name)
        _add_spatial_hopping!(ham_up, bond.i, bond.j, value)
        _add_spatial_hopping!(ham_down, bond.i, bond.j, value)
    end
    for bond in defect_hopping_bonds.dd
        value = _defect_hopping_reference_sign(bond) * _get_param(param_map, bond.name)
        _add_spatial_hopping!(ham_up, bond.i, bond.j, value)
        _add_spatial_hopping!(ham_down, bond.i, bond.j, value)
    end
    return ham_up, ham_down
end

"""
    build_defect_meanfield_derivative_blocks(geom, metadata, active_names)

用途: 构造 active mean-field 参数对应的 spin up/down mean-field 导数矩阵。

参数:
- `geom::ThreeBandGeometry`: 三带几何.
- `metadata::DefectThreeBandMetadata`: defect 信息.
- `active_names`: active mean-field 参数名.

返回:
- `(deriv_up, deriv_down)`: 两个 `Dict{Symbol,Matrix{Float64}}`.
"""
function build_defect_meanfield_derivative_blocks(geom, metadata, active_names)
    n_total = n_sites(geom)
    deriv_up = Dict{Symbol,Matrix{Float64}}()
    deriv_down = Dict{Symbol,Matrix{Float64}}()
    defect_hopping_bonds = build_defect_hopping_bonds(metadata)
    defect_bond_by_name = Dict{Symbol,DefectHoppingBond}(
        bond.name => bond for bond in _defect_all_hopping_bonds(defect_hopping_bonds)
    )
    for raw_name in active_names
        name = Symbol(raw_name)
        up_block = zeros(Float64, n_total, n_total)
        down_block = zeros(Float64, n_total, n_total)
        text = String(name)
        if startswith(text, "mu_site_")
            site = parse(Int, replace(text, "mu_site_" => ""))
            up_block[site, site] = 1.0
            down_block[site, site] = 1.0
        elseif startswith(text, "mz_site_")
            site = parse(Int, replace(text, "mz_site_" => ""))
            up_block[site, site] = 1.0
            down_block[site, site] = -1.0
        elseif name in (:mu0, :mu1, :mu0_d0, :mu1_d0, :mz_11, :mz_11_d0)
            for site in 1:n_total
                orbital = orbital_of_site(geom, site)
                is_target = if name == :mu0
                    orbital == ORB_D && !(site in Set(metadata.defect_cu_patch_sites))
                elseif name == :mu1
                    orbital != ORB_D && !(site in Set(metadata.defect_oxygen_sites))
                elseif name == :mu0_d0
                    site in Set(metadata.defect_cu_patch_sites)
                elseif name == :mu1_d0
                    site in Set(metadata.defect_oxygen_sites)
                elseif name == :mz_11
                    orbital != ORB_D && !(site in Set(metadata.defect_oxygen_sites))
                else
                    site in Set(metadata.defect_oxygen_sites)
                end
                is_target || continue
                if name in (:mz_11, :mz_11_d0)
                    up_block[site, site] = 1.0
                    down_block[site, site] = -1.0
                else
                    up_block[site, site] = 1.0
                    down_block[site, site] = 1.0
                end
            end
        elseif name == :chi1_00
            for (site_i, site_j) in build_dd_bonds(geom)
                _add_spatial_hopping!(up_block, site_i, site_j, -1.0)
                _add_spatial_hopping!(down_block, site_i, site_j, -1.0)
            end
            for bond in defect_hopping_bonds.dd
                _zero_spatial_hopping!(up_block, bond.i, bond.j)
                _zero_spatial_hopping!(down_block, bond.i, bond.j)
            end
        elseif name == :chi1_01
            for (site_i, site_j) in build_pd_bonds(geom)
                _add_spatial_hopping!(up_block, site_i, site_j, -1.0)
                _add_spatial_hopping!(down_block, site_i, site_j, -1.0)
            end
            for bond in defect_hopping_bonds.pd
                _zero_spatial_hopping!(up_block, bond.i, bond.j)
                _zero_spatial_hopping!(down_block, bond.i, bond.j)
            end
        elseif name == :chi1_11
            for (site_i, site_j) in build_pp_bonds(geom)
                _add_spatial_hopping!(up_block, site_i, site_j, -1.0)
                _add_spatial_hopping!(down_block, site_i, site_j, -1.0)
            end
            for bond in defect_hopping_bonds.pp
                _zero_spatial_hopping!(up_block, bond.i, bond.j)
                _zero_spatial_hopping!(down_block, bond.i, bond.j)
            end
        elseif haskey(defect_bond_by_name, name)
            bond = defect_bond_by_name[name]
            value = _defect_hopping_reference_sign(bond)
            _add_spatial_hopping!(up_block, bond.i, bond.j, value)
            _add_spatial_hopping!(down_block, bond.i, bond.j, value)
        else
            throw(ArgumentError("unknown defect three-band mean-field parameter: $name"))
        end
        deriv_up[name] = up_block
        deriv_down[name] = down_block
    end
    return deriv_up, deriv_down
end

"""
    _session_size(session)

用途: 读取 MPI session 的进程数, 用于判断是否需要 shared derivative tensor。

参数:
- `session`: `MPISession`。

返回:
- `Int`: MPI communicator 中的 rank 数。
"""
_session_size(session) = Int(session.size)

"""
    _current_cpu_id()

用途: 尝试读取当前进程所在 CPU 编号, 用于 NUMA replica 分组。

参数:
- 无。

返回:
- `Union{Nothing,Int}`: Linux 上通常返回 CPU id; 若系统不支持则返回 `nothing`。
"""
function _current_cpu_id()
    try
        cpu = ccall(:sched_getcpu, Cint, ())
        return cpu < 0 ? nothing : Int(cpu)
    catch
        return nothing
    end
end

"""
    _numa_node_id_for_cpu(cpu)

用途: 根据 Linux `/sys/devices/system/cpu` 查询 CPU 所属 NUMA node。

参数:
- `cpu::Integer`: CPU 编号。

返回:
- `Int`: NUMA node 编号; 查询不到时返回 `0`。
"""
function _numa_node_id_for_cpu(cpu::Integer)
    cpu_dir = "/sys/devices/system/cpu/cpu$(Int(cpu))"
    isdir(cpu_dir) || return 0
    for entry in readdir(cpu_dir)
        startswith(entry, "node") || continue
        node_id = tryparse(Int, entry[5:end])
        node_id === nothing || return node_id
    end
    return 0
end

"""
    current_numa_node_id()

用途: 返回当前进程所在 NUMA node id, 用于 shared tensor replica 选择。

参数:
- 无。

返回:
- `Int`: NUMA node id。
"""
function current_numa_node_id()
    cpu = _current_cpu_id()
    cpu === nothing && return 0
    return _numa_node_id_for_cpu(cpu)
end

"""
    _replica_metadata_from_node_ids(node_ids, local_rank)

用途: 根据同一节点内所有 rank 的 NUMA node id, 计算当前 rank 应使用的 tensor replica。

参数:
- `node_ids::Vector{Int}`: node-local ranks 对应的 NUMA node id。
- `local_rank::Integer`: 当前 rank 在 shared-memory communicator 中的编号。

返回:
- named tuple, 包含 replica 标签, 当前 replica index, 当前 replica 内 rank/size。
"""
function _replica_metadata_from_node_ids(node_ids::Vector{Int}, local_rank::Integer)
    isempty(node_ids) && return (labels=[0], index=1, rank=0, size=1, node_id=0)
    idx = Int(local_rank) + 1
    1 <= idx <= length(node_ids) || throw(BoundsError(node_ids, idx))
    labels = sort(unique(node_ids))
    node_id = node_ids[idx]
    replica_index = findfirst(==(node_id), labels)
    replica_index === nothing && throw(ArgumentError("local NUMA node id is missing from gathered node ids"))
    replica_members = findall(==(node_id), node_ids)
    replica_rank = findfirst(==(idx), replica_members)
    replica_rank === nothing && throw(ArgumentError("local rank is missing from its NUMA replica group"))
    return (
        labels=labels,
        index=Int(replica_index),
        rank=Int(replica_rank) - 1,
        size=length(replica_members),
        node_id=node_id,
    )
end

"""
    setup_defect_dense_derivative_workspace(session, nocc, nspin, nparams; numa_tensor_replica=false)

用途: 为 dense determinant derivative tensor 创建 MPI shared-memory storage。

参数:
- `session`: 全局 MPI session。
- `nocc::Integer`: 占据轨道数, 即电子数。
- `nspin::Integer`: spin-orbital 行数, 等于 `2 * n_sites`。
- `nparams::Integer`: active mean-field 参数数。
- `numa_tensor_replica::Bool`: 若为 `true`, 每个 NUMA node 使用一份 replica。

返回:
- `nothing`: 单 rank 或无 mean-field 参数时不需要 shared storage。
- named tuple: 包含 `shared_tensor`, shared-memory communicator 和 replica 信息。

公式:
- tensor 形状为 `(N_occ, N_spin, N_param)`。
- 多 NUMA replica 时实际底层 storage 为 `(N_occ, N_spin, N_param, N_replica)`。
"""
function setup_defect_dense_derivative_workspace(
    session,
    nocc::Integer,
    nspin::Integer,
    nparams::Integer;
    numa_tensor_replica::Bool=false,
)
    Int(nparams) <= 0 && return nothing
    _session_size(session) <= 1 && return nothing

    session_shm = init_node_mpi_session(session)
    local_numa_node = numa_tensor_replica ? current_numa_node_id() : 0
    gathered_node_ids = MPI.Gather(local_numa_node, session_shm.root, session_shm.comm)
    node_ids = session_shm.rank == session_shm.root ? Int.(gathered_node_ids) : Int[]
    node_ids = MPI.bcast(node_ids, session_shm.root, session_shm.comm)
    replica = numa_tensor_replica ?
        _replica_metadata_from_node_ids(node_ids, session_shm.rank) :
        (labels=[0], index=1, rank=session_shm.rank, size=session_shm.size, node_id=0)

    nreplicas = length(replica.labels)
    local_length = session_shm.rank == session_shm.root ? Int(nocc) * Int(nspin) * Int(nparams) * nreplicas : 0
    win, _ = MPI.Win_allocate_shared(Ptr{Float64}, local_length, session_shm.comm)
    shared_tensor_storage = MPI.Win_shared_query(
        Array{Float64},
        (Int(nocc), Int(nspin), Int(nparams), nreplicas),
        win;
        rank=0,
    )
    shared_tensor = @view shared_tensor_storage[:, :, :, replica.index]

    return (
        session=session,
        session_shm=session_shm,
        win=win,
        shared_tensor_storage=shared_tensor_storage,
        shared_tensor=shared_tensor,
        numa_tensor_replica=Bool(numa_tensor_replica),
        numa_node_id=replica.node_id,
        replica_labels=replica.labels,
        replica_index=replica.index,
        replica_rank=replica.rank,
        replica_size=replica.size,
    )
end

"""
    _defect_response_F(evals; eta=1e-8)

用途: 构造 mean-field eigenvector 一阶微扰响应矩阵。

参数:
- `evals`: mean-field 本征值。
- `eta::Real`: 小的虚部正则化, 用于处理近简并能级。

返回:
- `Matrix{Float64}`: 响应矩阵 `F`。

公式:
- `F_ab = -real(1 / (epsilon_a - epsilon_b + i eta))`, 且 `F_aa = 0`。
"""
function _defect_response_F(evals; eta::Real=1e-8)
    diff_mat = evals .- evals' .+ im * eta
    F = -real.(1.0 ./ diff_mat)
    F[diagind(F)] .= 0.0
    return F
end

"""
    _ordered_fixed_sz_eigensystem(ham_up, ham_down, n_sites_total, nelec, nup, ndn)

用途: 构造 fixed-Sz determinant 使用的 full eigensystem, 列顺序保证前 `nelec`
列正好是 occupied orbitals。

参数:
- `ham_up`, `ham_down`: spin-up/down 空间 mean-field 矩阵。
- `n_sites_total::Integer`: 空间 site 数。
- `nelec::Integer`: 总电子数。
- `nup`, `ndn`: 两个 spin sector 的电子数。

返回:
- named tuple: `evals`, `u_occ`, `u_full`, `response_F`。
"""
function _ordered_fixed_sz_eigensystem(
    ham_up,
    ham_down,
    n_sites_total::Integer,
    nelec::Integer,
    nup::Integer,
    ndn::Integer,
)
    n_total = Int(n_sites_total)
    n_occ = Int(nelec)
    n_up = Int(nup)
    n_down = Int(ndn)
    eval_up, eigvec_up = eigen(Hermitian(ham_up))
    eval_down, eigvec_down = eigen(Hermitian(ham_down))

    evals = Vector{Float64}(undef, 2 * n_total)
    u_full = zeros(Float64, 2 * n_total, 2 * n_total)
    col = 1
    for sector_col in 1:n_up
        evals[col] = real(eval_up[sector_col])
        u_full[1:2:end, col] .= real.(eigvec_up[:, sector_col])
        col += 1
    end
    for sector_col in 1:n_down
        evals[col] = real(eval_down[sector_col])
        u_full[2:2:end, col] .= real.(eigvec_down[:, sector_col])
        col += 1
    end
    for sector_col in (n_up + 1):n_total
        evals[col] = real(eval_up[sector_col])
        u_full[1:2:end, col] .= real.(eigvec_up[:, sector_col])
        col += 1
    end
    for sector_col in (n_down + 1):n_total
        evals[col] = real(eval_down[sector_col])
        u_full[2:2:end, col] .= real.(eigvec_down[:, sector_col])
        col += 1
    end

    return (
        evals=evals,
        u_occ=copy(u_full[:, 1:n_occ]),
        u_full=u_full,
        response_F=_defect_response_F(evals),
    )
end

"""
    _fill_defect_dense_derivative_slice!(dest, geom, metadata, name, u_full, response_F, nocc)

用途: 计算单个 mean-field 参数的 determinant orbital derivative, 并写入预分配切片。

参数:
- `dest::AbstractMatrix{Float64}`: 形状为 `(N_occ, N_spin)` 的输出切片。
- `geom`, `metadata`: defect three-band 几何与 defect 信息。
- `name::Symbol`: active mean-field 参数名。
- `u_full::AbstractMatrix`: full eigensystem 的轨道矩阵。
- `response_F::AbstractMatrix`: 一阶微扰响应矩阵。
- `nocc::Integer`: occupied orbital 数。

返回:
- `dest`。

公式:
- `dH_MO = U' dH U`
- `dU = U * (F .* dH_MO)`
- `dest = transpose(dU[:, 1:N_occ])`
"""
function _fill_defect_dense_derivative_slice!(
    dest::AbstractMatrix{Float64},
    geom,
    metadata,
    name::Symbol,
    u_full::AbstractMatrix,
    response_F::AbstractMatrix,
    nocc::Integer,
)
    n_total = n_sites(geom)
    size(dest, 1) == Int(nocc) ||
        throw(DimensionMismatch("dense derivative slice has inconsistent occupied dimension"))
    size(dest, 2) == 2 * n_total ||
        throw(DimensionMismatch("dense derivative slice has inconsistent spin-orbital dimension"))

    deriv_up, deriv_down = build_defect_meanfield_derivative_blocks(geom, metadata, [name])
    dH = zeros(Float64, 2 * n_total, 2 * n_total)
    dH[1:2:end, 1:2:end] .= deriv_up[name]
    dH[2:2:end, 2:2:end] .= deriv_down[name]
    dH_MO = u_full' * dH * u_full
    dU = u_full * (response_F .* dH_MO)
    dest .= permutedims(real.(dU[:, 1:Int(nocc)]))
    return dest
end

"""
    make_defect_threeband_ansatz_response(geom, metadata, param_names, params; ...)

用途: 构造 determinant 使用的占据轨道矩阵和 active mean-field 参数导数。

参数:
- `geom::ThreeBandGeometry`: 三带几何.
- `metadata::DefectThreeBandMetadata`: defect 信息.
- `param_names`, `params`: 完整参数名和值.
- `active_param_names`: active mean-field 参数名.
- `nelec::Integer`: 总电子数.
- `nup`, `ndn`: 固定 spin-up/down 粒子数.

返回:
- `(U_occ, derivatives)`: `U_occ` 为 `2Nsite x Nelec`, `derivatives` 为 `Nelec x 2Nsite x Nparam`.
"""
function make_defect_threeband_ansatz_response(
    geom,
    metadata,
    param_names,
    params;
    active_param_names,
    nelec::Integer,
    nup::Union{Nothing,Integer}=nothing,
    ndn::Union{Nothing,Integer}=nothing,
    derivative_storage=nothing,
    derivative_fill_rank::Integer=0,
    derivative_fill_size::Integer=1,
)
    n_up = nup === nothing ? div(Int(nelec), 2) : Int(nup)
    n_down = ndn === nothing ? Int(nelec) - n_up : Int(ndn)
    n_up + n_down == Int(nelec) || throw(ArgumentError("nup + ndn must equal nelec"))

    ham_up, ham_down = build_defect_meanfield_blocks(geom, metadata, param_names, params)
    n_total = n_sites(geom)
    active_names = Symbol.(active_param_names)

    if derivative_storage !== nothing
        expected_size = (Int(nelec), 2 * n_total, length(active_names))
        size(derivative_storage) == expected_size ||
            throw(DimensionMismatch("derivative_storage size $(size(derivative_storage)) != expected $expected_size"))
        fill_rank = Int(derivative_fill_rank)
        fill_size = Int(derivative_fill_size)
        0 <= fill_rank < fill_size ||
            throw(ArgumentError("derivative_fill_rank must satisfy 0 <= rank < size"))
        eig = _ordered_fixed_sz_eigensystem(ham_up, ham_down, n_total, nelec, n_up, n_down)
        for param_index in (fill_rank + 1):fill_size:length(active_names)
            _fill_defect_dense_derivative_slice!(
                @view(derivative_storage[:, :, param_index]),
                geom,
                metadata,
                active_names[param_index],
                eig.u_full,
                eig.response_F,
                nelec,
            )
        end
        return (U_occ=eig.u_occ, derivatives=derivative_storage)
    end

    deriv_up, deriv_down = build_defect_meanfield_derivative_blocks(geom, metadata, active_names)
    _, eigvec_up, _, du_up = compute_eig_and_dU_reg1(ham_up, deriv_up)
    _, eigvec_down, _, du_down = compute_eig_and_dU_reg1(ham_down, deriv_down)

    u_occ = zeros(Float64, 2 * n_total, Int(nelec))
    if n_up > 0
        u_occ[1:2:end, 1:n_up] .= real.(eigvec_up[:, 1:n_up])
    end
    if n_down > 0
        u_occ[2:2:end, (n_up + 1):Int(nelec)] .= real.(eigvec_down[:, 1:n_down])
    end

    derivatives = zeros(Float64, Int(nelec), 2 * n_total, length(active_names))
    for (param_index, name) in enumerate(active_names)
        du_occ = zeros(Float64, 2 * n_total, Int(nelec))
        if n_up > 0
            du_occ[1:2:end, 1:n_up] .= real.(du_up[name][:, 1:n_up])
        end
        if n_down > 0
            du_occ[2:2:end, (n_up + 1):Int(nelec)] .= real.(du_down[name][:, 1:n_down])
        end
        derivatives[:, :, param_index] .= permutedims(du_occ)
    end
    return (U_occ=u_occ, derivatives=derivatives)
end

"""
    build_defect_threeband_state(args)

用途: 构造第一版 defect three-band Hubbard 运行状态。

参数:
- `args`: 命令行参数字典.

返回:
- `DefectThreeBandRunState`.
"""
function build_defect_threeband_state(args)
    geom = _defect_threeband_geometry_from_args(args)
    anchors = parse_defect_anchors(String(_arg(args, :defect_anchors, "")))
    _validate_wrapped_defect_anchors(geom, anchors)
    metadata = DefectThreeBandMetadata(geom, anchors)
    model = DefectThreeBandHubbardModel(
        geom,
        metadata;
        tpd=Float64(_arg(args, :tpd)),
        tpp=Float64(_arg(args, :tpp)),
        Delta_pd=Float64(_arg(args, :Delta_pd)),
        Udd=Float64(_arg(args, :Udd)),
        Up=Float64(_arg(args, :Up)),
        Vpd=Float64(_arg(args, :Vpd)),
        defect_Epp=Float64(_arg(args, :defect_Epp, 0.0)),
    )
    nelec = canonical_defect_threeband_nelec(args)
    nup, ndn = defect_threeband_spin_counts(nelec, Int(_arg(args, :target_sz, 0)), n_sites(geom))
    uniform_nondefect_mu = Bool(_arg(args, :uniform_nondefect_mu, false))
    site_resolved_oxygen_mz = Bool(_arg(args, :site_resolved_oxygen_mz, false))
    use_site_gutzwiller = Bool(_arg(args, :site_gutzwiller, false))
    use_orbital_gutzwiller = Bool(_arg(args, :gutzwiller_orbital, false))
    use_defect_jastrow = Bool(_arg(args, :defect_jastrow, false))
    n_jastrow_shells, clean_jastrow_names, clean_jastrow_values =
        _defect_clean_jastrow_names_values(args)
    g_d, g_py, g_px = _defect_background_gutzwiller_args(args)

    param_names, init_params = build_defect_initial_params(
        geom,
        metadata;
        mu0=Float64(_arg(args, :mu0, 0.0)),
        mu1=Float64(_arg(args, :mu1, 0.0)),
        mu0_d0=_nan_to_nothing(Float64(_arg(args, :mu0_d0, NaN))),
        mu1_d0=_nan_to_nothing(Float64(_arg(args, :mu1_d0, NaN))),
        mz_00=Float64(_arg(args, :mz_00, 0.0)),
        mz_11=Float64(_arg(args, :mz_11, 0.0)),
        mz_00_d0=_nan_to_nothing(Float64(_arg(args, :mz_00_d0, NaN))),
        mz_11_d0=_nan_to_nothing(Float64(_arg(args, :mz_11_d0, NaN))),
        chi1_00=Float64(_arg(args, :chi1_00, 0.0)),
        chi1_01=Float64(_arg(args, :chi1_01, 0.0)),
        chi1_11=Float64(_arg(args, :chi1_11, 0.0)),
        chi_def_pd_init=_nan_to_nothing(Float64(_arg(args, :chi_def_pd_init, NaN))),
        chi_def_pp_init=_nan_to_nothing(Float64(_arg(args, :chi_def_pp_init, NaN))),
        chi_def_dd_init=_nan_to_nothing(Float64(_arg(args, :chi_def_dd_init, NaN))),
        g_site_init=Float64(_arg(args, :g_site_init, 0.0)),
        g_d=g_d,
        g_py=g_py,
        g_px=g_px,
        jastrow_shells=n_jastrow_shells,
        jastrow_init=Float64(_arg(args, :jastrow_init, 0.0)),
        clean_jastrow_names=clean_jastrow_names,
        clean_jastrow_values=clean_jastrow_values,
        defect_jastrow_init=Float64(_arg(args, :defect_jastrow_init, 0.0)),
        use_site_gutzwiller=use_site_gutzwiller,
        use_orbital_gutzwiller=use_orbital_gutzwiller,
        use_defect_jastrow=use_defect_jastrow,
        uniform_nondefect_mu=uniform_nondefect_mu,
        site_resolved_oxygen_mz=site_resolved_oxygen_mz,
    )
    projector, projector_names, _ = build_defect_threeband_projector(geom, metadata, args)
    if projector !== nothing
        initial_values = Dict(zip(Symbol.(param_names), Float64.(init_params)))
        update_projector_params!(projector, [initial_values[name] for name in projector_names])
    end

    return DefectThreeBandRunState(
        geom,
        metadata,
        model,
        nelec,
        nup,
        ndn,
        Symbol.(param_names),
        Float64.(init_params),
        projector,
        uniform_nondefect_mu,
        site_resolved_oxygen_mz,
        n_jastrow_shells,
    )
end

"""
    parse_defect_not_opt_params(value)

用途: 解析 `--not_opt_params` 中固定不优化的参数名。

参数:
- `value::AbstractString`: 逗号分隔的参数名字符串.

返回:
- `Vector{Symbol}`.
"""
function parse_defect_not_opt_params(value::AbstractString)
    stripped = strip(value)
    isempty(stripped) && return Symbol[]

    names = Symbol[]
    seen = Set{Symbol}()
    for raw in split(stripped, ","; keepempty=true)
        name_string = strip(raw)
        isempty(name_string) && throw(ArgumentError("empty entry in --not_opt_params: $value"))
        name = Symbol(name_string)
        name in seen && throw(ArgumentError("duplicate entry in --not_opt_params: $name"))
        push!(names, name)
        push!(seen, name)
    end
    return names
end

"""
    _checked_defect_name_vector(param_names)

用途: 将参数名转为 `Symbol` 并检查是否重复。

参数:
- `param_names`: 参数名列表.

返回:
- `Vector{Symbol}`.
"""
function _checked_defect_name_vector(param_names)
    names = Symbol.(param_names)
    seen = Set{Symbol}()
    for name in names
        name in seen && throw(ArgumentError("duplicate defect three-band parameter name: $name"))
        push!(seen, name)
    end
    return names
end

"""
    select_defect_threeband_opt_params(param_names, params, not_opt_params)

用途: 根据 `--not_opt_params` 从完整参数中选出参与 SR 的子集。

参数:
- `param_names`: 完整参数名列表.
- `params`: 完整参数值列表.
- `not_opt_params::AbstractString`: 逗号分隔的固定参数.

返回:
- `(opt_names, opt_values)`.
"""
function select_defect_threeband_opt_params(param_names, params, not_opt_params::AbstractString)
    all_names = _checked_defect_name_vector(param_names)
    length(all_names) == length(params) ||
        throw(ArgumentError("param_names and params must have the same length"))

    fixed_names = parse_defect_not_opt_params(not_opt_params)
    available = Set(all_names)
    for name in fixed_names
        name in available ||
            throw(ArgumentError("unknown fixed defect three-band parameter in --not_opt_params: $name"))
    end

    fixed_set = Set(fixed_names)
    opt_names = Symbol[]
    opt_values = Float64[]
    for (name, value) in zip(all_names, params)
        if !(name in fixed_set)
            push!(opt_names, name)
            push!(opt_values, Float64(value))
        end
    end
    return opt_names, opt_values
end

"""
    merge_defect_threeband_opt_params(all_param_names, all_params, opt_param_names, opt_params)

用途: 把 SR 更新后的优化参数合并回完整参数向量。

参数:
- `all_param_names`: 完整参数名.
- `all_params`: 完整初始参数值.
- `opt_param_names`: 优化参数名.
- `opt_params`: 优化参数当前值.

返回:
- `Vector{Float64}`: 按 `all_param_names` 顺序排列的完整参数.
"""
function merge_defect_threeband_opt_params(all_param_names, all_params, opt_param_names, opt_params)
    all_names = _checked_defect_name_vector(all_param_names)
    opt_names = _checked_defect_name_vector(opt_param_names)
    length(all_names) == length(all_params) ||
        throw(ArgumentError("all_param_names and all_params must have the same length"))
    length(opt_names) == length(opt_params) ||
        throw(ArgumentError("opt_param_names and opt_params must have the same length"))

    values = Dict{Symbol,Float64}(name => Float64(value) for (name, value) in zip(all_names, all_params))
    for (name, value) in zip(opt_names, opt_params)
        haskey(values, name) ||
            throw(ArgumentError("optimized parameter $name is not present in full defect three-band parameter list"))
        values[name] = Float64(value)
    end
    return [values[name] for name in all_names]
end

"""
    build_init_params_from_txt(path, param_names, params)

用途: 从两列文本文件覆盖部分初始参数。

参数:
- `path::AbstractString`: 文本文件路径, 每行格式为 `param_name value`.
- `param_names`: 完整参数名列表.
- `params`: 默认参数值.

返回:
- `Vector{Float64}`: 覆盖后的参数值.
"""
function build_init_params_from_txt(path::AbstractString, param_names, params)
    names = _checked_defect_name_vector(param_names)
    values = Dict{Symbol,Float64}(name => Float64(value) for (name, value) in zip(names, params))
    isfile(path) || throw(ArgumentError("initialization txt file does not exist: $path"))

    open(path, "r") do io
        for (line_no, raw_line) in enumerate(eachline(io))
            line = strip(split(raw_line, "#"; limit=2)[1])
            isempty(line) && continue
            fields = split(line)
            length(fields) == 2 ||
                throw(ArgumentError("invalid initialization line $line_no in $path; expected: param_name value"))
            name = Symbol(fields[1])
            haskey(values, name) ||
                throw(ArgumentError("unknown defect three-band initialization parameter $name in $path line $line_no"))
            values[name] = parse(Float64, fields[2])
        end
    end

    return [values[name] for name in names]
end

"""
    _is_defect_projector_param(name)

用途: 判断参数名是否属于 Gutzwiller/Jastrow projector。

参数:
- `name::Symbol`: 参数名.

返回:
- `Bool`: `true` 表示该参数应由 projector 更新, `false` 表示属于 mean-field.
"""
function _is_defect_projector_param(name::Symbol)
    text = String(name)
    return name in Symbol[:g_d, :g_py, :g_px] ||
        startswith(text, "g_site_") ||
        startswith(text, "v_dd_") ||
        startswith(text, "v_dp_") ||
        startswith(text, "v_pp_") ||
        startswith(text, "v_defpair_")
end

"""
    update_defect_threeband_ansatz!(vwf, param_names, params, geom, metadata, nelec; ...)

用途: 根据当前变分参数更新 determinant 波函数矩阵和参数导数。

参数:
- `vwf`: `vwf_det` 波函数.
- `param_names`: 完整参数名列表.
- `params`: 完整参数值列表.
- `geom`: `ThreeBandGeometry`.
- `metadata`: `DefectThreeBandMetadata`.
- `nelec`: 固定电子数.
- `initialize::Bool`: 是否调用 `init_gswf!`.
- `active_param_names`: 本轮需要计算导数的参数名.
- `jastrow_shells::Integer`: clean Jastrow shell 数, 用于 projector 参数名校验.
- `nup`, `ndn`: 固定 Sz 的两个自旋粒子数.
- `uniform_nondefect_mu`: 是否使用共享非 defect 化学势.
- `site_resolved_oxygen_mz`: oxygen `mz` 是否逐 site 参数化.
- `dense_derivative_workspace`: 可选 MPI shared-memory derivative tensor workspace; 为 `nothing` 时使用普通 per-rank dense tensor.

返回:
- `vwf`.
"""
function update_defect_threeband_ansatz!(
    vwf,
    param_names,
    params,
    geom,
    metadata,
    nelec;
    initialize::Bool=true,
    active_param_names=nothing,
    jastrow_shells::Integer=0,
    nup::Union{Nothing,Integer}=nothing,
    ndn::Union{Nothing,Integer}=nothing,
    uniform_nondefect_mu::Bool=false,
    site_resolved_oxygen_mz::Bool=false,
    dense_derivative_workspace=nothing,
)
    full_names = _checked_defect_name_vector(param_names)
    length(full_names) == length(params) ||
        throw(ArgumentError("param_names and params must have the same length"))
    active_names = active_param_names === nothing ? copy(full_names) : _checked_defect_name_vector(active_param_names)
    full_name_set = Set(full_names)
    for name in active_names
        name in full_name_set ||
            throw(ArgumentError("active defect three-band parameter $name is not present in the supplied parameter list"))
    end
    projector_names = [name for name in full_names if _is_defect_projector_param(name)]
    projector_values = [
        Float64(params[idx])
        for idx in eachindex(full_names)
        if _is_defect_projector_param(full_names[idx])
    ]
    active_mf_names = Symbol[]
    active_projector_names = Symbol[]
    for name in active_names
        if _is_defect_projector_param(name)
            push!(active_projector_names, name)
        else
            push!(active_mf_names, name)
        end
    end

    if !isempty(projector_values)
        attached_projector_names = get_vwf_projector_param_names(vwf)
        isempty(attached_projector_names) &&
            throw(ArgumentError("defect three-band projector parameters were supplied, but no projector is attached"))
        projector_names == attached_projector_names ||
            throw(ArgumentError("defect three-band projector parameter names/order $(projector_names) do not match attached projector $(attached_projector_names)"))
        active_projector_names == projector_names ||
            throw(ArgumentError("current main-src projector path requires all projector parameters to be active together; fixed projector subsets are not supported yet"))
    end
    ansatz = if dense_derivative_workspace !== nothing && !isempty(active_mf_names)
        response = make_defect_threeband_ansatz_response(
            geom,
            metadata,
            full_names,
            params;
            active_param_names=active_mf_names,
            nelec=nelec,
            nup=nup,
            ndn=ndn,
            derivative_storage=dense_derivative_workspace.shared_tensor,
            derivative_fill_rank=dense_derivative_workspace.replica_rank,
            derivative_fill_size=dense_derivative_workspace.replica_size,
        )
        MPI.Barrier(dense_derivative_workspace.session_shm.comm)
        response
    else
        make_defect_threeband_ansatz_response(
            geom,
            metadata,
            full_names,
            params;
            active_param_names=active_mf_names,
            nelec=nelec,
            nup=nup,
            ndn=ndn,
        )
    end

    copyto!(vwf.gs_U, ansatz.U_occ)
    copyto!(vwf.gs_U_t, permutedims(ansatz.U_occ))
    copyto!(vwf.base_gs_U, ansatz.U_occ)
    update_vwf_params!(vwf, active_mf_names, ansatz.derivatives)
    if !isempty(projector_values)
        update_vwf_projector_params!(vwf, projector_names, projector_values)
    end
    if initialize
        init_gswf!(vwf)
    end
    return vwf
end

"""
    build_defect_exponential_lr_func(lr_start, lr_end, n_steps)

用途: 构造指数学习率衰减函数。

参数:
- `lr_start::Float64`: 初始学习率.
- `lr_end::Float64`: 末步学习率.
- `n_steps::Int`: SR 步数.

返回:
- `Function`: 形如 `(lr0, step) -> lr_step`.

公式:
- `gamma = (lr_end / lr_start)^(1 / (n_steps - 1))`
- `lr_step = lr_start * gamma^(step - 1)`
"""
function build_defect_exponential_lr_func(lr_start::Float64, lr_end::Float64, n_steps::Int)::Function
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

"""
    defect_threeband_observables()

用途: 构造第一版测量所需的 observable 字典。

参数:
- 无.

返回:
- `Dict{Symbol,Function}`: 当前只包含局域能量 `:E`.
"""
defect_threeband_observables() = Dict{Symbol,Function}(:E => (model, vwf) -> local_energy(model, vwf))

"""
    main()

用途: defect three-band Hubbard 主程序入口。支持 `SR` 和 `measure` 两种任务。

参数:
- 无, 直接读取命令行 `ARGS`.

返回:
- `nothing`.
"""
function main()
    args = parse_commandline()
    session = init_mpi_session()
    rank = session.rank
    is_root = rank == session.root

    built = build_defect_threeband_state(args)
    sampler = config_Hubbard(n_sites(built.geom), built.nup, built.ndn; ifPH=false)
    init_config_Hubbard!(sampler)
    vwf = vwf_det(zeros(Float64, 2 * n_sites(built.geom), built.nelec), sampler)
    if built.projector !== nothing
        set_projector!(vwf, built.projector)
    end
    kernel = HubbardKernel(conserve_sz=true)

    all_param_names = built.param_names
    all_init_params = built.init_params
    init_params_json = String(_arg(args, :init_params_json, ""))
    init_params_txt = String(_arg(args, :init_params_txt, ""))
    if !isempty(init_params_json) && !isempty(init_params_txt)
        throw(ArgumentError("--init_params_json cannot be combined with --init_params_txt"))
    end
    if !isempty(init_params_json)
        all_init_params = build_init_params_from_json(init_params_json, all_param_names)
        is_root && println("Loaded initial parameters from json: $(init_params_json)")
    elseif !isempty(init_params_txt)
        all_init_params = build_init_params_from_txt(init_params_txt, all_param_names, all_init_params)
        is_root && println("Loaded initial parameters from txt: $(init_params_txt)")
    end

    opt_param_names, opt_init_params = select_defect_threeband_opt_params(
        all_param_names,
        all_init_params,
        String(_arg(args, :not_opt_params, "")),
    )
    job = lowercase(String(_arg(args, :job)))
    opt_mf_param_names = [name for name in opt_param_names if !_is_defect_projector_param(name)]
    dense_derivative_workspace = job == "sr" ?
        setup_defect_dense_derivative_workspace(
            session,
            built.nelec,
            2 * n_sites(built.geom),
            length(opt_mf_param_names);
            numa_tensor_replica=Bool(_arg(args, :numa_tensor_replica, false)),
        ) :
        nothing

    if is_root
        println("Initial parameters: $all_init_params")
        println("Optimized parameter names: $opt_param_names")
        println("Defect anchors: $(built.metadata.anchors)")
        println("Three-band particle numbers: N_up=$(built.nup), N_down=$(built.ndn), N_e=$(built.nelec)")
        if dense_derivative_workspace !== nothing
            println(
                "Dense derivative shared tensor: shape=$(size(dense_derivative_workspace.shared_tensor)), " *
                "NUMA replica=$(dense_derivative_workspace.numa_tensor_replica), " *
                "replicas=$(length(dense_derivative_workspace.replica_labels))",
            )
        end
    end

    update_defect_threeband_ansatz!(
        vwf,
        all_param_names,
        all_init_params,
        built.geom,
        built.metadata,
        built.nelec;
        initialize=false,
        active_param_names=opt_param_names,
        jastrow_shells=built.jastrow_shells,
        nup=built.nup,
        ndn=built.ndn,
        uniform_nondefect_mu=built.uniform_nondefect_mu,
        site_resolved_oxygen_mz=built.site_resolved_oxygen_mz,
        dense_derivative_workspace=dense_derivative_workspace,
    )

    vmc_params = VMCParams(
        total_samples=Int(_arg(args, :nMC)),
        warmup_steps=Int(_arg(args, :wMC)),
        rebuild_every=Int(_arg(args, :rMC)),
        decorr_steps=Int(_arg(args, :dMC)),
        seed=Int(_arg(args, :seed)) + rank,
    )

    lr = Float64(_arg(args, :lr))
    lr_end = Float64(_arg(args, :lr_end))
    if isnan(lr_end)
        lr_end = lr
    end

    folder = "logs"
    mkpath(folder)
    if job == "sr"
        isempty(opt_param_names) &&
            throw(ArgumentError("--not_opt_params leaves no active defect three-band parameters to optimize"))
        sr_params = SRParams(
            vmc_params=vmc_params,
            n_steps=Int(_arg(args, :nSR)),
            lr=lr,
            diag_shift=Float64(_arg(args, :diag_shift, 1e-3)),
            eigen_cutoff=Float64(_arg(args, :eps_wf, 1e-4)),
            max_step_size=Float64(_arg(args, :max_step_size, 0.1)),
        )
        exp_lr_func = build_defect_exponential_lr_func(lr, lr_end, Int(_arg(args, :nSR)))
        update_vwf_func! = (vwf, params) -> begin
            merged_params = merge_defect_threeband_opt_params(
                all_param_names,
                all_init_params,
                opt_param_names,
                params,
            )
            update_defect_threeband_ansatz!(
                vwf,
                all_param_names,
                merged_params,
                built.geom,
                built.metadata,
                built.nelec;
                initialize=false,
                active_param_names=opt_param_names,
                jastrow_shells=built.jastrow_shells,
                nup=built.nup,
                ndn=built.ndn,
                uniform_nondefect_mu=built.uniform_nondefect_mu,
                site_resolved_oxygen_mz=built.site_resolved_oxygen_mz,
                dense_derivative_workspace=dense_derivative_workspace,
            )
        end
        run_sr_optimization(
            built.model,
            vwf,
            kernel,
            opt_init_params,
            update_vwf_func!,
            sr_params;
            log_file=joinpath(folder, "defect_threeband_sr_history.txt"),
            param_names=opt_param_names,
            lr_func=exp_lr_func,
        )
    elseif job == "measure"
        run_simulation(
            built.model,
            vwf,
            kernel,
            defect_threeband_observables(),
            vmc_params;
            history_observables=[:E],
        )
    else
        throw(ArgumentError("unknown job: $job"))
    end
    return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
