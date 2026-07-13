using MPI
using Random
using Printf
using DelimitedFiles
using LinearAlgebra
using Statistics
using ArgParse
using JSON
# using FFWT

# === 1. 环境设置 ===
push!(LOAD_PATH, joinpath(@__DIR__, "src"))
push!(LOAD_PATH, @__DIR__)


using mfVMC
include("PartonSquare.jl")
using .PartonSquare


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--Lx"
        help = "Lattice size in X direction"
        arg_type = Int
        default = 8
        "--Ly"
        help = "Lattice size in Y direction"
        arg_type = Int
        default = 3
        "--t1"
        help = "Hopping amplitude"
        arg_type = Float64
        default = 1.0
        "--t2"
        help = "Next-nearest neighbor hopping amplitude"
        arg_type = Float64
        default = -0.2
        "--U"
        help = "On-site interaction strength"
        arg_type = Float64
        default = 8.0
        "--bcx"
        help = "Boundary condition phase in X (1.0 or -1.0)"
        arg_type = Float64
        default = 1.001
        "--bcy"
        help = "Boundary condition phase in Y (1.0 or -1.0)"
        arg_type = Float64
        default = 0.999
        "--x_boundary"
        help = "X boundary type: pbc or obc"
        arg_type = String
        default = "pbc"
        "--edge_pinning"
        help = "Staggered Sz pinning strength on x edges. Used only when --x_boundary obc"
        arg_type = Float64
        default = 0.1
        "--etax"
        help = "Initial x-bond pairing amplitude in the mean-field ansatz"
        arg_type = Float64
        default = 0.01
        "--etay"
        help = "Initial y-bond pairing amplitude in the mean-field ansatz"
        arg_type = Float64
        default = 0.01
        "--chi2"
        help = "Next-nearest neighbor hopping in MF ansatz. Default follows --t2"
        arg_type = Float64
        default = -0.2
        "--mz"
        help = "AFM order parameters"
        arg_type = Float64
        default = 3.0
        "--mu"
        help = "chemical potential"
        arg_type = Float64
        default = -3.0
        "--target_sz"
        help = "target total sz"
        arg_type = Int
        default = 0
        "--nMC"
        help = "Number of Monte Carlo total_samples"
        arg_type = Int
        default = 10000
        "--wMC"
        help = "Number of Monte Carlo warnming up"
        arg_type = Int
        default = 100
        "--rMC"
        help = "Number of rebuild inserve"
        arg_type = Int
        default = 100
        "--dMC"
        help = "Number of Monte Carlo decorrelation sweeps"
        arg_type = Int
        default = 1
        "--seed"
        help = "random seed"
        arg_type = Int
        default = 5423
        "--nSR"
        help = "total steps for SR"
        arg_type = Int
        default = 50
        "--lr"
        help = "SR learn rate"
        arg_type = Float64
        default = 0.04
        "--lr_end"
        help = "Target learning rate at the last SR step. Default follows --lr"
        arg_type = Float64
        default = NaN
        "--init_params_json"
        help = "Path to json file that provides initial parameters"
        arg_type = String
        default = ""
        "--job"
        help = "Job to be done. Can be SR and measure"
        arg_type = String
        default = "SR"
        "--doping"
        help = "Doping level"
        arg_type = Float64
        default = 0.125
        "--ansatz"
        help = "Ansatz type, can be 'AFM' or 'Stripe'"
        arg_type = String
        default = "Stripe"
        "--lambda"
        help = "assuming length of stripe"
        arg_type = Float64
        default = 4.0
        "--stripe_center"
        help = "Stripe center type, can be 'site' or 'bond'"
        arg_type = String
        default = "site"
        "--stripe_mu_amp"
        help = "Charge modulation amplitude for Stripe initial mean-field parameters"
        arg_type = Float64
        default = 0.0
        "--stripe_spin_peak_x"
        help = "X coordinate where the Stripe spin modulation envelope reaches its peak. NaN keeps the old --stripe_center phase"
        arg_type = Float64
        default = NaN
        "--g"
        help = "Gutzwiller projector parameter"
        arg_type = Float64
        default = 1.0
        "--vj1"
        help = "Jastrow projector parameter on nearest-neighbor bonds"
        arg_type = Float64
        default = 0.0
        "--vj2"
        help = "Jastrow projector parameter on next-nearest-neighbor bonds"
        arg_type = Float64
        default = 0.0
        "--enable_backflow"
        help = "Enable PH-basis backflow using the same local-state rules as Hubbard_bf.jl"
        arg_type = String
        default = "true"
        "--bf_epsilon"
        help = "Backflow epsilon parameter"
        arg_type = Float64
        default = 1.0
        "--bf_eta1"
        help = "Backflow eta1 doublon-holon parameter"
        arg_type = Float64
        default = 0.0
        "--bf_eta2"
        help = "Backflow eta2 spin-exchange parameter"
        arg_type = Float64
        default = 0.0
        "--bf_eta3"
        help = "Backflow eta3 mixed virtual-hop parameter"
        arg_type = Float64
        default = 0.0
    end

    return parse_args(s)
end

# ==============================================================================
# 3. 辅助函数
# ==============================================================================

"""
用途: 将 stripe 中心类型映射为论文中的 `x0` 偏移量。

参数:
- `stripe_center::AbstractString`: stripe 中心类型, 支持 `site` 或 `bond`。

返回:
- `Float64`: 论文公式中的 `x0`, `site -> 0.0`, `bond -> 0.5`。
"""
function get_stripe_center_offset(stripe_center::AbstractString)::Float64
    stripe_center_lowercase = lowercase(strip(stripe_center))
    if stripe_center_lowercase == "site"
        return 0.0
    elseif stripe_center_lowercase == "bond"
        return 0.5
    end
    error("Unknown stripe_center: $(stripe_center). Expected 'site' or 'bond'.")
end

"""
用途: 根据论文 `2111.04623v4` 的 Eq.(6)(7)(10) 生成 Stripe 初始 mean-field 列参数。

参数:
- `lx::Int`: 晶格在 `x` 方向的列数。
- `lambda::Real`: stripe 电荷调制周期 `λ`, 支持非整数周期。
- `stripe_center::AbstractString`: stripe 中心类型, 支持 `site` 或 `bond`。
- `mu_uniform::Float64`: 平均 chemical potential `μ`。
- `stripe_mu_amp::Float64`: 电荷调制振幅 `Δc`。
- `mz_amp::Float64`: 自旋调制振幅 `Δs`。
- `etax_uniform::Float64`: x 方向 bond pairing 初值。
- `etay_uniform::Float64`: y 方向 bond pairing 初值。
- `stripe_spin_peak_x::Float64`: spin modulation envelope 峰值所在的 x 坐标。若为 `NaN`, 使用 `stripe_center` 给出的旧相位。

返回:
- `NamedTuple`: 包含 `etax`, `etay`, `mz`, `mu` 四个 `Dict{Symbol, Float64}`。

公式:
- `Q = 2π / λ`
- 若 `stripe_spin_peak_x` 为 `NaN`, `x0` 来自 `stripe_center`; 否则 `x0 = stripe_spin_peak_x - λ / 2`。
- `mu_x = μ + Δc * cos[Q * (x - x0)]`
- `mz_x = Δs * sin[Q / 2 * (x - x0)]`
- `Δx(x) = etax * |cos[Q / 2 * (x + 1/2 - x0)]|`
- `Δy(x) = etay * |cos[Q / 2 * (x - x0)]|`
- `etax_x = Δx(x)`
- `etay_x = Δy(x)`
"""
function build_stripe_initial_column_params(
    lx::Int,
    lambda::Real,
    stripe_center::AbstractString,
    mu_uniform::Float64,
    stripe_mu_amp::Float64,
    mz_amp::Float64,
    etax_uniform::Float64,
    etay_uniform::Float64,
    stripe_spin_peak_x::Float64=NaN,
)
    lambda_value = Float64(lambda)
    if lambda_value <= 0
        error("lambda must be positive.")
    end

    stripe_center_offset = if isnan(stripe_spin_peak_x)
        get_stripe_center_offset(stripe_center)
    else
        stripe_spin_peak_x - lambda_value / 2.0
    end
    stripe_wave_vector = 2.0 * pi / lambda_value
    x_bond_pairing_uniform = etax_uniform
    y_bond_pairing_uniform = etay_uniform

    etax_by_x = Dict{Symbol,Float64}()
    etay_by_x = Dict{Symbol,Float64}()
    mz_by_x = Dict{Symbol,Float64}()
    mu_by_x = Dict{Symbol,Float64}()

    for x in 1:lx
        x_coordinate = Float64(x)
        mu_by_x[Symbol("mu_$(x)")] = mu_uniform + stripe_mu_amp * cos(stripe_wave_vector * (x_coordinate - stripe_center_offset))
        mz_by_x[Symbol("mz_$(x)")] = mz_amp * sin(stripe_wave_vector / 2.0 * (x_coordinate - stripe_center_offset))

        x_bond_pairing = x_bond_pairing_uniform * abs(cos(stripe_wave_vector / 2.0 * (x_coordinate + 0.5 - stripe_center_offset)))
        y_bond_pairing = y_bond_pairing_uniform * abs(cos(stripe_wave_vector / 2.0 * (x_coordinate - stripe_center_offset)))

        etax_by_x[Symbol("etax_$(x)")] = x_bond_pairing
        etay_by_x[Symbol("etay_$(x)")] = y_bond_pairing
    end

    return (; etax=etax_by_x, etay=etay_by_x, mz=mz_by_x, mu=mu_by_x)
end

"""
用途: 将命令行输入的 x 方向边界条件名称规范化为内部 Symbol.

参数:
- `x_boundary::AbstractString`: 命令行输入, 允许 `pbc` 或 `obc`, 大小写不敏感.

返回:
- `Symbol`: `:pbc` 表示 x 方向周期边界, `:obc` 表示 x 方向开边界.
"""
function normalize_x_boundary_name(x_boundary::AbstractString)::Symbol
    normalized_name = Symbol(lowercase(strip(x_boundary)))
    if normalized_name == :pbc || normalized_name == :obc
        return normalized_name
    end
    error("Unknown x_boundary=$(x_boundary). Expected pbc or obc.")
end

"""
用途: 将二维格点坐标 `(x, y)` 映射到一维 site index.

参数:
- `x::Int`: x 方向列坐标, 必须在 `1:lx` 内.
- `y::Int`: y 方向坐标, 按周期边界取模.
- `lx::Int`: x 方向长度.
- `ly::Int`: y 方向长度.

返回:
- `Int`: 一维 site index, 取值范围为 `1:(lx * ly)`.
"""
function hubbard_column_site_index(x::Int, y::Int, lx::Int, ly::Int)::Int
    if x < 1 || x > lx
        error("x=$(x) is outside 1:$(lx).")
    end
    return (x - 1) * ly + mod(y - 1, ly) + 1
end

"""
用途: 为 x-column Hubbard 模型构造最近邻和次近邻 bond 列表.

参数:
- `lx::Int`: x 方向长度.
- `ly::Int`: y 方向长度.
- `x_boundary::Symbol`: `:pbc` 表示 x 方向周期, `:obc` 表示 x 方向开边界.

返回:
- `Tuple{Vector{Tuple{Int, Int}}, Vector{Tuple{Int, Int}}}`:
  `(bonds1, bonds2)`, 分别为无向 bond 的一个方向代表. 后续构造 Hamiltonian 时会显式加入 Hermitian conjugate.

公式:
- 最近邻包含 `+x` 与 `+y` 两个方向.
- 次近邻包含 `(+x, +y)` 与 `(+x, -y)` 两个方向.
- `:obc` 时不包含任何跨越 `x = lx -> 1` 的 bond.
"""
function build_hubbard_column_bonds(lx::Int, ly::Int, x_boundary::Symbol)
    if lx <= 0 || ly <= 0
        error("lx and ly must be positive, got lx=$(lx), ly=$(ly).")
    end
    if x_boundary != :pbc && x_boundary != :obc
        error("Unknown x_boundary=$(x_boundary). Expected :pbc or :obc.")
    end

    bonds1 = Tuple{Int,Int}[]
    bonds2 = Tuple{Int,Int}[]
    for y in 1:ly, x in 1:lx
        site = hubbard_column_site_index(x, y, lx, ly)
        has_forward_x_bond = x < lx || x_boundary == :pbc
        if has_forward_x_bond
            x_forward = x == lx ? 1 : x + 1
            push!(bonds1, (site, hubbard_column_site_index(x_forward, y, lx, ly)))
            push!(bonds2, (site, hubbard_column_site_index(x_forward, y + 1, lx, ly)))
            push!(bonds2, (site, hubbard_column_site_index(x_forward, y - 1, lx, ly)))
        end
        push!(bonds1, (site, hubbard_column_site_index(x, y + 1, lx, ly)))
    end
    return bonds1, bonds2
end

"""
用途: 构造截断 displacement Jastrow 的位移标签列表。

参数:
- `lx::Int`: 晶格 x 方向长度。
- `ly::Int`: 晶格 y 方向长度。
- `dx_max::Int`: x 方向最大位移, 默认 `div(lx, 2)`。
- `dy_max::Int`: y 方向最大位移, 默认 `div(ly, 2)`。

返回:
- `Vector{Tuple{Int, Int}}`: 按 `(dx, dy)` 排序的位移标签, 不包含 `(0, 0)`。
"""
function build_truncated_jastrow_displacement_labels(
    lx::Int,
    ly::Int;
    dx_max::Int=div(lx, 2),
    dy_max::Int=div(ly, 2),
)::Vector{Tuple{Int,Int}}
    if lx <= 0 || ly <= 0
        error("lx and ly must be positive, got lx=$(lx), ly=$(ly).")
    end
    if dx_max < 0 || dy_max < 0
        error("dx_max and dy_max must be non-negative, got dx_max=$(dx_max), dy_max=$(dy_max).")
    end

    labels = Tuple{Int,Int}[]
    for dx in 0:min(dx_max, lx - 1)
        for dy in 0:min(dy_max, div(ly, 2))
            if dx == 0 && dy == 0
                continue
            end
            push!(labels, (dx, dy))
        end
    end
    return labels
end

"""
用途: 为截断 displacement Jastrow 构造参数名。

参数:
- `dx::Int`: x 方向位移。
- `dy::Int`: y 方向最短周期位移。

返回:
- `Symbol`: 形如 `:vj_dx_dy` 的参数名。
"""
function build_truncated_jastrow_param_name(dx::Int, dy::Int)::Symbol
    return Symbol("vj_$(dx)_$(dy)")
end

"""
用途: 为给定 displacement 生成唯一无序 pair 集合。

数学公式:
- 对每个格点 `i = (x, y)`, 连接 `j = (x + dx, y +/- dy)`。
- x 方向在 `:obc` 下不 wrap, 在 `:pbc` 下 wrap。
- y 方向始终按周期边界 wrap。

参数:
- `lx::Int`: 晶格 x 方向长度。
- `ly::Int`: 晶格 y 方向长度。
- `dx::Int`: x 方向位移。
- `dy::Int`: y 方向最短周期位移。
- `x_boundary::Symbol`: `:pbc` 或 `:obc`。

返回:
- `Vector{Tuple{Int, Int}}`: 经过 `i < j` 规范化的唯一 pair 列表。
"""
function build_truncated_jastrow_pair_set_for_displacement(
    lx::Int,
    ly::Int,
    dx::Int,
    dy::Int,
    x_boundary::Symbol,
)::Vector{Tuple{Int,Int}}
    if x_boundary != :pbc && x_boundary != :obc
        error("Unknown x_boundary=$(x_boundary). Expected :pbc or :obc.")
    end
    if dx < 0 || dy < 0
        error("dx and dy must be non-negative, got dx=$(dx), dy=$(dy).")
    end

    y_offsets = dy == 0 ? (0,) : (dy, -dy)
    unique_pairs = Set{Tuple{Int,Int}}()
    for x in 1:lx
        neighbor_x = x + dx
        if x_boundary == :pbc
            neighbor_x = mod(neighbor_x - 1, lx) + 1
        elseif neighbor_x > lx
            continue
        end

        for y in 1:ly
            site_index = hubbard_column_site_index(x, y, lx, ly)
            for offset_y in y_offsets
                neighbor_y = mod(y - 1 + offset_y, ly) + 1
                neighbor_index = hubbard_column_site_index(neighbor_x, neighbor_y, lx, ly)
                if neighbor_index == site_index
                    continue
                end
                push!(unique_pairs, (min(site_index, neighbor_index), max(site_index, neighbor_index)))
            end
        end
    end
    return sort!(collect(unique_pairs))
end

"""
用途: 为给定截断 displacement 构造 Jastrow 对称邻接表。

参数:
- `lx::Int`: 晶格 x 方向长度。
- `ly::Int`: 晶格 y 方向长度。
- `dx::Int`: x 方向位移。
- `dy::Int`: y 方向最短周期位移。
- `x_boundary::Symbol`: `:pbc` 或 `:obc`。

返回:
- `Vector{Vector{Int}}`: 每个 site 的 Jastrow 邻居列表, 无自环且对称。
"""
function build_truncated_jastrow_neighbor_table_for_displacement(
    lx::Int,
    ly::Int,
    dx::Int,
    dy::Int,
    x_boundary::Symbol,
)::Vector{Vector{Int}}
    unique_pairs = build_truncated_jastrow_pair_set_for_displacement(lx, ly, dx, dy, x_boundary)
    neighbor_table = [Int[] for _ in 1:(lx * ly)]
    for (site_i, site_j) in unique_pairs
        push!(neighbor_table[site_i], site_j)
        push!(neighbor_table[site_j], site_i)
    end
    for neighbors in neighbor_table
        sort!(neighbors)
    end
    return neighbor_table
end

"""
用途: 给截断 displacement Jastrow 参数设置兼容旧命令行的初值。

参数:
- `dx::Int`: x 方向位移。
- `dy::Int`: y 方向最短周期位移。
- `vj1::Float64`: 最近邻 Jastrow 初值。
- `vj2::Float64`: 对角次近邻 Jastrow 初值。

返回:
- `Float64`: 当前 displacement 的 Jastrow 初值。
"""
function truncated_jastrow_initial_value(dx::Int, dy::Int, vj1::Float64, vj2::Float64)::Float64
    if (dx == 1 && dy == 0) || (dx == 0 && dy == 1)
        return vj1
    elseif dx == 1 && dy == 1
        return vj2
    end
    return 0.0
end

"""
用途: 构造截断 displacement Jastrow terms、参数名和初值。

参数:
- `lx::Int`: 晶格 x 方向长度。
- `ly::Int`: 晶格 y 方向长度。
- `x_boundary::Symbol`: `:pbc` 或 `:obc`。
- `dx_max::Int`: x 方向截断, 默认 `div(lx, 2)`。
- `dy_max::Int`: y 方向截断, 默认 `div(ly, 2)`。
- `vj1::Float64`: `(1, 0)` 与 `(0, 1)` 的初值。
- `vj2::Float64`: `(1, 1)` 的初值。

返回:
- `Tuple{Vector{JastrowProjectorTerm{Float64}}, Vector{Symbol}, Vector{Float64}}`:
  `(jastrow_terms, param_names, init_params)`。
"""
function build_truncated_displacement_jastrow_terms(
    lx::Int,
    ly::Int,
    x_boundary::Symbol;
    dx_max::Int=div(lx, 2),
    dy_max::Int=div(ly, 2),
    vj1::Float64=0.0,
    vj2::Float64=0.0,
)::Tuple{Vector{JastrowProjectorTerm{Float64}},Vector{Symbol},Vector{Float64}}
    jastrow_terms = JastrowProjectorTerm{Float64}[]
    param_names = Symbol[]
    init_params = Float64[]

    for (dx, dy) in build_truncated_jastrow_displacement_labels(lx, ly; dx_max=dx_max, dy_max=dy_max)
        param_name = build_truncated_jastrow_param_name(dx, dy)
        init_value = truncated_jastrow_initial_value(dx, dy, vj1, vj2)
        neighbor_table = build_truncated_jastrow_neighbor_table_for_displacement(lx, ly, dx, dy, x_boundary)
        push!(
            jastrow_terms,
            JastrowProjectorTerm(
                param_name=param_name,
                v=init_value,
                site_to_neighbor_sites=neighbor_table,
            ),
        )
        push!(param_names, param_name)
        push!(init_params, init_value)
    end

    return jastrow_terms, param_names, init_params
end

"""
用途: 构造 Hubbard.jl 使用的截断 displacement projector。

参数:
- `lx::Int`: 晶格 x 方向长度。
- `ly::Int`: 晶格 y 方向长度。
- `g::Float64`: Gutzwiller projector 初值。
- `x_boundary::Symbol`: `:pbc` 或 `:obc`。
- `dx_max::Int`: x 方向 Jastrow 截断, 默认 `div(lx, 2)`。
- `dy_max::Int`: y 方向 Jastrow 截断, 默认 `div(ly, 2)`。
- `vj1::Float64`: 最近邻 Jastrow 初值。
- `vj2::Float64`: 对角次近邻 Jastrow 初值。

返回:
- `CompositeProjector`: 包含 `g` 与多个 `vj_dx_dy` 的 projector。
"""
function build_hubbard_truncated_projector(
    lx::Int,
    ly::Int,
    g::Float64,
    x_boundary::Symbol;
    dx_max::Int=div(lx, 2),
    dy_max::Int=div(ly, 2),
    vj1::Float64=0.0,
    vj2::Float64=0.0,
)::CompositeProjector
    jastrow_terms, _, _ = build_truncated_displacement_jastrow_terms(
        lx,
        ly,
        x_boundary;
        dx_max=dx_max,
        dy_max=dy_max,
        vj1=vj1,
        vj2=vj2,
    )
    projector_terms = AbstractProjectorTerm[
        GutzwillerProjectorTerm(param_name=:g, g=g),
    ]
    append!(projector_terms, jastrow_terms)
    return CompositeProjector(projector_terms)
end

"""
用途: 根据 Hubbard Hamiltonian 的 NN/NNN bond 生成 backflow 使用的有向 source 数据。

参数:
- `bonds1, bonds2::Vector{Tuple{Int, Int}}`: 无向 NN/NNN bond 的代表方向。
- `t1, t2::Float64`: 对应 hopping 振幅。

返回:
- `Tuple{Vector{Tuple{Int, Int}}, Vector{Float64}}`: 有向 source bonds 和振幅。
"""
function build_hubbard_backflow_source_data(
    bonds1::Vector{Tuple{Int,Int}},
    bonds2::Vector{Tuple{Int,Int}},
    t1::Float64,
    t2::Float64,
)::Tuple{Vector{Tuple{Int,Int}},Vector{Float64}}
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
用途: 构造 PH Hubbard 使用的 composite backflow。这里故意沿用 `Hubbard_bf.jl`
中 nonPH 的 local-state 规则, 用于比较非标准 PH backflow 对结果的影响。

参数:
- `source_bonds, source_amplitudes`: backflow source 数据。
- `bf_epsilon, bf_eta1, bf_eta2, bf_eta3::Float64`: backflow 参数。

返回:
- `CompositeBackflowTerm`: 顺序为 `epsilon, eta1, eta2, eta3` 的 backflow。
"""
function build_hubbard_composite_backflow(
    source_bonds::Vector{Tuple{Int,Int}},
    source_amplitudes::Vector{Float64},
    bf_epsilon::Float64,
    bf_eta1::Float64,
    bf_eta2::Float64,
    bf_eta3::Float64,
)::CompositeBackflowTerm
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
用途: 根据命令行开关构造 PH Hubbard backflow 对象。

参数:
- `enable_backflow::Bool`: 是否启用 backflow。
- 其余参数同 `build_hubbard_composite_backflow`。

返回:
- `AbstractBackflowTerm`: 启用时为 `CompositeBackflowTerm`, 禁用时为 `NoBackflowTerm()`。
"""
function build_hubbard_optional_backflow(
    enable_backflow::Bool,
    source_bonds::Vector{Tuple{Int,Int}},
    source_amplitudes::Vector{Float64},
    bf_epsilon::Float64,
    bf_eta1::Float64,
    bf_eta2::Float64,
    bf_eta3::Float64,
)::AbstractBackflowTerm
    if !enable_backflow
        return NoBackflowTerm()
    end
    return build_hubbard_composite_backflow(
        source_bonds,
        source_amplitudes,
        bf_epsilon,
        bf_eta1,
        bf_eta2,
        bf_eta3,
    )
end

"""
用途: 解析命令行中的布尔字符串。

参数:
- `raw_value::AbstractString`: 支持 `true/false`, `1/0`, `yes/no`, `on/off`。
- `option_name::AbstractString`: 命令行选项名, 用于错误信息。

返回:
- `Bool`: 解析后的布尔值。
"""
function parse_hubbard_bool_flag(raw_value::AbstractString, option_name::AbstractString)::Bool
    normalized_value = lowercase(strip(raw_value))
    if normalized_value in ("true", "t", "1", "yes", "y", "on")
        return true
    elseif normalized_value in ("false", "f", "0", "no", "n", "off")
        return false
    end
    error("Invalid value for $(option_name): $(raw_value).")
end

function update_ansatz!(
    vwf,
    param_names::Vector{Symbol},
    params::Vector{Float64},
    lx,
    ly,
    bcx,
    bcy,
    x_boundary::Symbol,
    target_sz::Int;
    nparams_proj::Int=0,
    nparams_backflow::Int=0,
)
    # 支持输入为 wf 参数 + projector 参数 + backflow 参数的拼接向量
    nparms = length(param_names)
    nparams_wf = nparms - nparams_proj - nparams_backflow
    wf_param_names = param_names[1:nparams_wf]
    wf_param_values = params[1:nparams_wf]
    projector_param_names = param_names[(nparams_wf+1):(nparams_wf+nparams_proj)]
    projector_param_values = params[(nparams_wf+1):(nparams_wf+nparams_proj)]
    backflow_param_names = param_names[(nparams_wf+nparams_proj+1):end]
    backflow_param_values = params[(nparams_wf+nparams_proj+1):end]
    # 这里也可以把 bcx, bcy 提出来作为参数
    param_map = Dict{Symbol,Float64}(zip(wf_param_names, wf_param_values))

    chi2 = get(param_map, :chi2, 0.0)

    etax = Dict{Symbol,Float64}()
    etay = Dict{Symbol,Float64}()
    mz = Dict{Symbol,Float64}()
    mu = Dict{Symbol,Float64}()

    for (name, value) in param_map
        name_str = String(name)
        if startswith(name_str, "etax_")
            etax[name] = value
        elseif startswith(name_str, "etay_")
            etay[name] = value
        elseif startswith(name_str, "mz_")
            mz[name] = value
        elseif startswith(name_str, "mu_")
            mu[name] = value
        elseif name == :etax || name == :etay || name == :chi2
            continue
        else
            error("Unknown parameter name: $name")
        end
    end

    if haskey(param_map, :etax)
        if !isempty(etax)
            error("Found both uniform etax and x-dependent etax_* parameters.")
        end
        for x in 1:lx
            etax[Symbol("etax_$(x)")] = param_map[:etax]
        end
    end
    if haskey(param_map, :etay)
        if !isempty(etay)
            error("Found both uniform etay and x-dependent etay_* parameters.")
        end
        for x in 1:lx
            etay[Symbol("etay_$(x)")] = param_map[:etay]
        end
    end

    hubbard_params = PartonSquare.HubbardParams(
        Lx=lx,
        Ly=ly,
        bcx=bcx,
        bcy=bcy,
        x_boundary=x_boundary,
        chi1=1.0,
        etax=etax,
        etay=etay,
        chi2=chi2,
        mu=mu,
        mz=mz
    )

    _, gs_U, dUt_params = PartonSquare.make_ansatz_and_derivs(hubbard_params; param_names=wf_param_names, target_sz=target_sz)

    copyto!(vwf.base_gs_U, gs_U)
    copyto!(vwf.gs_U, gs_U)
    copyto!(vwf.backflow_u, gs_U)
    copyto!(vwf.gs_U_t, permutedims(gs_U))
    dUt_matrix = zeros(Float64, size(gs_U, 2), size(gs_U, 1), length(wf_param_names))
    for (idx, name) in enumerate(wf_param_names)
        dUt_matrix[:, :, idx] = dUt_params[name]
    end
    update_vwf_params!(vwf, wf_param_names, dUt_matrix)
    if !isempty(projector_param_names)
        update_vwf_projector_params!(vwf, projector_param_names, projector_param_values)
    end
    if !isempty(backflow_param_names)
        update_vwf_backflow_params!(vwf, backflow_param_names, backflow_param_values)
    end
    init_gswf!(vwf)
end

function build_exponential_lr_func(
    lr_start::Float64,
    lr_end::Float64,
    n_steps::Int
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

function defination_observabels(lx::Int, ly::Int)::Dict{Symbol,Function}
    observables = Dict{Symbol,Function}()
    observables[:E] = local_energy
    for x in 1:lx, y in 1:ly
        i = idx(x, y, lx, ly)
        key = Symbol("Sz_$(x)_$(y)")
        observables[key] = (model, vwf) -> begin
            val = get_Sz(vwf.sampler.state[i])
            return val
        end
        key = Symbol("n_$(x)_$(y)")
        observables[key] = (model, vwf) -> begin
            st = vwf.sampler.state[i]
            n_up = (st & UP) != 0 ? 1.0 : 0.0
            n_dn = (st & DN) != 0 ? 1.0 : 0.0
            return n_up + n_dn
        end
    end
    return observables
end
function idx(x::Int, y::Int, lx::Int, ly::Int)
    return mod(x - 1, lx) * ly + mod(y - 1, ly) + 1
end

# ==============================================================================
# 4. 主程序
# ==============================================================================

function main()
    args = parse_commandline()

    session = init_mpi_session()
    rank = session.rank
    is_root = (rank == session.root)

    # ---------------------------------------------------------
    # A. 参数设定 (全部集中在这里)
    # ---------------------------------------------------------
    lx = args["Lx"]
    ly = args["Ly"]
    BCX = args["bcx"]
    BCY = args["bcy"]
    x_boundary = normalize_x_boundary_name(args["x_boundary"])
    edge_pinning = args["edge_pinning"]
    target_sz = args["target_sz"]
    doping = args["doping"]
    lambda = args["lambda"]
    stripe_center = args["stripe_center"]
    stripe_mu_amp = args["stripe_mu_amp"]
    stripe_spin_peak_x = args["stripe_spin_peak_x"]
    # if mod(lx, 4) == 0
    #     BCX = -1
    # end
    # if mod(ly, 4) == 0
    #     BCY = -1
    # end
    nMC = args["nMC"]
    wMC = args["wMC"]
    rMC = args["rMC"]
    dMC = args["dMC"]
    seed = args["seed"]
    n_steps = args["nSR"]
    lr = args["lr"]
    lr_end = args["lr_end"]
    if isnan(lr_end)
        lr_end = lr
    end

    t1 = args["t1"]
    t2 = args["t2"]
    U = args["U"]
    job = args["job"]
    ansatz = args["ansatz"]
    g = args["g"]
    vj1 = args["vj1"]
    vj2 = args["vj2"]
    init_params_json = args["init_params_json"]
    N_sites = lx * ly
    #要优化的参数
    wf_param_names = [:chi2]
    wf_init_params = [args["chi2"]]
    #对每一列的mz，构建mean field参数mz_i,i为第几列
    if ansatz == "Stripe"
        stripe_column_params = build_stripe_initial_column_params(
            lx,
            lambda,
            stripe_center,
            args["mu"],
            stripe_mu_amp,
            args["mz"],
            args["etax"],
            args["etay"],
            stripe_spin_peak_x,
        )
        for i in 1:lx
            push!(wf_param_names, Symbol("etax_$i"))
            push!(wf_param_names, Symbol("etay_$i"))
            push!(wf_param_names, Symbol("mz_$i"))
            push!(wf_param_names, Symbol("mu_$i"))
            push!(wf_init_params, stripe_column_params.etax[Symbol("etax_$i")])
            push!(wf_init_params, stripe_column_params.etay[Symbol("etay_$i")])
            push!(wf_init_params, stripe_column_params.mz[Symbol("mz_$i")])
            push!(wf_init_params, stripe_column_params.mu[Symbol("mu_$i")])
        end
    elseif ansatz == "AFM"
        for i in 1:lx
            push!(wf_param_names, Symbol("etax_$i"))
            push!(wf_param_names, Symbol("etay_$i"))
            push!(wf_param_names, Symbol("mz_$i"))
            push!(wf_param_names, Symbol("mu_$i"))
            push!(wf_init_params, args["etax"])
            push!(wf_init_params, args["etay"])
            push!(wf_init_params, args["mz"])
            push!(wf_init_params, args["mu"])
        end
    else
        error("Unknown ansatz type: $ansatz")
    end
    # VMC 采样参数
    meas_params = VMCParams(
        total_samples=nMC,
        warmup_steps=wMC,
        rebuild_every=rMC,
        decorr_steps=dMC,
        seed=args["seed"] + rank
    )
    # ---------------------------------------------------------

    # B. 模型与波函数初始化
    #GeneralModel定义
    bonds1, bonds2 = build_hubbard_column_bonds(lx, ly, x_boundary)

    # Projector 定义
    projector = build_hubbard_truncated_projector(lx, ly, g, x_boundary; vj1=vj1, vj2=vj2)
    source_bonds, source_amplitudes = build_hubbard_backflow_source_data(bonds1, bonds2, t1, t2)
    backflow = build_hubbard_optional_backflow(
        parse_hubbard_bool_flag(args["enable_backflow"], "--enable_backflow"),
        source_bonds,
        source_amplitudes,
        args["bf_epsilon"],
        args["bf_eta1"],
        args["bf_eta2"],
        args["bf_eta3"],
    )
    proj_param_names = projector_param_names(projector)
    proj_init_params = projector_param_values(projector)
    nparams_proj = length(proj_param_names)
    backflow_param_name_list = backflow_param_names(backflow)
    backflow_init_params = backflow_param_values(backflow)
    nparams_backflow = length(backflow_param_name_list)
    # 把波函数参数, 投影算符参数和 backflow 参数拼接成一个向量, 供优化器使用
    init_params = vcat(wf_init_params, proj_init_params, backflow_init_params)
    param_names = vcat(wf_param_names, proj_param_names, backflow_param_name_list)

    if !isempty(init_params_json)
        init_params = build_init_params_from_json(init_params_json, param_names)
        if is_root
            println("Loaded initial parameters from json: $(init_params_json)")
        end
    end

    terms = OperatorTerm[]
    for (i, j) in bonds1
        push!(terms, OperatorTerm([:cdag_up, :c_up], [i, j], -t1))
        push!(terms, OperatorTerm([:cdag_up, :c_up], [j, i], -t1))
        push!(terms, OperatorTerm([:cdag_dn, :c_dn], [i, j], -t1))
        push!(terms, OperatorTerm([:cdag_dn, :c_dn], [j, i], -t1))
    end
    for (i, j) in bonds2
        push!(terms, OperatorTerm([:cdag_up, :c_up], [i, j], -t2))
        push!(terms, OperatorTerm([:cdag_up, :c_up], [j, i], -t2))
        push!(terms, OperatorTerm([:cdag_dn, :c_dn], [i, j], -t2))
        push!(terms, OperatorTerm([:cdag_dn, :c_dn], [j, i], -t2))
    end
    for i in 1:N_sites
        push!(terms, OperatorTerm([:n_up, :n_dn], [i, i], U))
    end
    if x_boundary == :obc && edge_pinning != 0.0
        for y in 1:ly
            left_site = hubbard_column_site_index(1, y, lx, ly)
            right_site = hubbard_column_site_index(lx, y, lx, ly)
            push!(terms, OperatorTerm([:Sz], [left_site], edge_pinning * (-1)^(y + 1)))
            #push!(terms, OperatorTerm([:Sz], [right_site], edge_pinning * (-1)^(y + lx)))
        end
    end
    ham = GeneralModel(N_sites, terms)

    nelec = Int(N_sites * (1 + doping))
    #检查target_sz的parity
    @assert (target_sz + nelec) % 2 == 0 "Wrong parity!"
    nup = (nelec + target_sz) ÷ 2
    ndn = nelec - nup
    sampler = config_Hubbard(N_sites, nup, ndn; ifPH=true)
    init_config_Hubbard!(sampler)

    vwf = vwf_det(zeros(Float64, 2 * N_sites, N_sites + target_sz), sampler; backflow=backflow)
    set_projector!(vwf, projector)
    kernel = HubbardKernel(conserve_sz=true)

    # C. 更新波函数参数
    if rank == 0
        println("Initial parameters: $init_params")
    end
    update_ansatz!(
        vwf,
        param_names,
        init_params,
        lx,
        ly,
        BCX,
        BCY,
        x_boundary,
        target_sz;
        nparams_proj=nparams_proj,
        nparams_backflow=nparams_backflow,
    )


    # D. 运行模拟
    folder = "logs"
    mkpath(folder)

    if job == "SR"
        sr_params = SRParams(vmc_params=meas_params, n_steps=n_steps, lr=lr)
        exp_lr_func = build_exponential_lr_func(lr, lr_end, n_steps)

        update_vwf_func! = (vwf, params) -> update_ansatz!(
            vwf,
            param_names,
            params,
            lx,
            ly,
            BCX,
            BCY,
            x_boundary,
            target_sz;
            nparams_proj=nparams_proj,
            nparams_backflow=nparams_backflow,
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
            lr_func=exp_lr_func
        )
        if is_root
            min_energy = extract_min_energy(joinpath(folder, "sr_history.txt"))
        end
    elseif job == "measure"
        observables = defination_observabels(lx, ly)
        # 默认不保留历史, 如需阻塞法(Binning)请在此列出观测量名称
        history_observables = [:E]
        results = run_simulation(
            ham,
            vwf,
            kernel,
            observables,
            meas_params;
            history_observables=history_observables
        )
        if is_root && results !== nothing
            means = results[:means]
            mean_dict = Dict{Symbol,Any}()
            for (key, value) in means
                if value isa Number
                    mean_dict[key] = real(value)
                else
                    mean_dict[key] = value
                end
            end

            histories = results[:histories]
            if !isempty(histories)
                mean_hist, se_dict, n_eff_dict, tau_int_dict, _ = blocking_binning(histories)

                txt_file = joinpath(folder, "block_binning.txt")
                open(txt_file, "w") do io
                    println(io, "# Observable\tMean\tSE\tN_eff\tTau_int")
                    for name in sort(collect(keys(mean_hist)))
                        mean_val = mean_hist[name]
                        se_val = se_dict[name]
                        n_eff_val = n_eff_dict[name]
                        tau_val = tau_int_dict[name]

                        if mean_val isa Number && se_val isa Number && n_eff_val isa Number && tau_val isa Number
                            @printf(io, "%s\t%.10f\t%.10f\t%.6f\t%.6f\n",
                                String(name), mean_val, se_val, n_eff_val, tau_val)
                        else
                            println(io, "$(String(name))\t$(mean_val)\t$(se_val)\t$(n_eff_val)\t$(tau_val)")
                        end
                    end
                end
            end

            json_file = joinpath(folder, "block_binning_mean.json")
            mean_dict_str = Dict{String,Any}()
            for (key, value) in mean_dict
                mean_dict_str[String(key)] = value
            end
            open(json_file, "w") do io
                JSON.print(io, mean_dict_str)
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
