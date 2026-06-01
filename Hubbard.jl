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
        arg_type = Int
        default = 4
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
- `lambda::Int`: stripe 电荷调制周期 `λ`。
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
    lambda::Int,
    stripe_center::AbstractString,
    mu_uniform::Float64,
    stripe_mu_amp::Float64,
    mz_amp::Float64,
    etax_uniform::Float64,
    etay_uniform::Float64,
    stripe_spin_peak_x::Float64=NaN,
)
    if lambda <= 0
        error("lambda must be positive.")
    end

    stripe_center_offset = if isnan(stripe_spin_peak_x)
        get_stripe_center_offset(stripe_center)
    else
        stripe_spin_peak_x - lambda / 2.0
    end
    stripe_wave_vector = 2.0 * pi / lambda
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
)
    # 支持输入为 wf 参数 + projector 参数的拼接向量
    nparms = length(param_names)
    nparams_wf = nparms - nparams_proj
    wf_param_names = param_names[1:nparams_wf]
    wf_param_values = params[1:nparams_wf]
    projector_param_names = param_names[(nparams_wf+1):end]
    projector_param_values = params[(nparams_wf+1):end]
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

    site_to_neighbor_sites_j1 = [Int[] for _ in 1:N_sites]
    for (site_i, site_j) in bonds1
        if !(site_j in site_to_neighbor_sites_j1[site_i])
            push!(site_to_neighbor_sites_j1[site_i], site_j)
        end
        if !(site_i in site_to_neighbor_sites_j1[site_j])
            push!(site_to_neighbor_sites_j1[site_j], site_i)
        end
    end

    site_to_neighbor_sites_j2 = [Int[] for _ in 1:N_sites]
    for (site_i, site_j) in bonds2
        if !(site_j in site_to_neighbor_sites_j2[site_i])
            push!(site_to_neighbor_sites_j2[site_i], site_j)
        end
        if !(site_i in site_to_neighbor_sites_j2[site_j])
            push!(site_to_neighbor_sites_j2[site_j], site_i)
        end
    end

    # Projector 定义
    projector = CompositeProjector([
        GutzwillerProjectorTerm(param_name=:g, g=g),
        JastrowProjectorTerm(param_name=:vj1, v=vj1, site_to_neighbor_sites=site_to_neighbor_sites_j1),
        JastrowProjectorTerm(param_name=:vj2, v=vj2, site_to_neighbor_sites=site_to_neighbor_sites_j2)
    ])
    proj_param_names = projector_param_names(projector)
    proj_init_params = projector_param_values(projector)
    nparams_proj = length(proj_param_names)
    # 把波函数参数和投影算符参数拼接成一个向量, 供优化器使用
    init_params = vcat(wf_init_params, proj_init_params)
    param_names = vcat(wf_param_names, proj_param_names)

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

    vwf = vwf_det(zeros(Float64, 2 * N_sites, N_sites + target_sz), sampler)
    set_projector!(vwf, projector)
    kernel = HubbardKernel(conserve_sz=true)

    # C. 更新波函数参数
    if rank == 0
        println("Initial parameters: $init_params")
    end
    update_ansatz!(vwf, param_names, init_params, lx, ly, BCX, BCY, x_boundary, target_sz; nparams_proj=nparams_proj)


    # D. 运行模拟
    folder = "logs"
    mkpath(folder)

    if job == "SR"
        sr_params = SRParams(vmc_params=meas_params, n_steps=n_steps, lr=lr)
        exp_lr_func = build_exponential_lr_func(lr, lr_end, n_steps)

        update_vwf_func! = (vwf, params) -> update_ansatz!(vwf, param_names, params, lx, ly, BCX, BCY, x_boundary, target_sz; nparams_proj=nparams_proj)

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
