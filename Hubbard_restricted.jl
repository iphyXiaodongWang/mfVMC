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
        "--etax"
        help = "MF parameters"
        arg_type = Float64
        default = 0.01
        "--etay"
        help = "MF parameters"
        arg_type = Float64
        default = 0.01
        "--chi2"
        help = "Next-nearest neighbor hopping in MF ansatz. Default follows --t2"
        arg_type = Float64
        default = -0.2
        "--Delta_AF"
        help = "AFM order parameters"
        arg_type = Float64
        default = 3.0
        "--Delta_c"
        help = "charge stripe order parameters"
        arg_type = Float64
        default = 3.0
        "--Delta_s"
        help = "spin stripe order parameters"
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
        "--g"
        help = "Gutzwiller projector parameter"
        arg_type = Float64
        default = 1.0
        "--bf_epsilon"
        help = "Eq.(4) backflow epsilon parameter"
        arg_type = Float64
        default = 1.0
        "--bf_eta"
        help = "Eq.(4) backflow eta parameter"
        arg_type = Float64
        default = 0.0
    end

    return parse_args(s)
end

# ==============================================================================
# 3. 辅助函数
# ==============================================================================

"""
用途: 枚举全位移 Jastrow 参数的合法最短镜像位移标签。

参数:
- `lx::Int`: 晶格在 `x` 方向的长度。
- `ly::Int`: 晶格在 `y` 方向的长度。

返回:
- `Vector{Tuple{Int, Int}}`: 按稳定顺序排列的 `(dx, dy)` 标签列表, 其中
  `0 <= dx <= floor(lx / 2)`, `0 <= dy <= floor(ly / 2)`, 且排除 `(0, 0)`。

说明:
- 为了去除 `g` 与全位移 Jastrow 参数共同构成的一维冗余自由度, 这里统一移除
  稳定排序后的最后一个位移标签。
"""
function build_jastrow_displacement_labels(
    lx::Int,
    ly::Int,
)::Vector{Tuple{Int,Int}}
    if lx <= 0 || ly <= 0
        error("lx and ly must be positive, got lx=$(lx), ly=$(ly).")
    end

    displacement_labels = Tuple{Int,Int}[]
    max_dx = fld(lx, 2)
    max_dy = fld(ly, 2)
    for dx in 0:max_dx
        for dy in 0:max_dy
            if dx == 0 && dy == 0
                continue
            end
            push!(displacement_labels, (dx, dy))
        end
    end
    if !isempty(displacement_labels)
        pop!(displacement_labels)
    end
    return displacement_labels
end

"""
用途: 根据 Jastrow 位移标签构造参数名。

参数:
- `dx::Int`: 最短镜像后的 `x` 方向绝对位移。
- `dy::Int`: 最短镜像后的 `y` 方向绝对位移。

返回:
- `Symbol`: 形如 `:vj_dx_dy` 的参数名, 例如 `:vj_1_2`。
"""
function build_jastrow_param_name(
    dx::Int,
    dy::Int,
)::Symbol
    if dx < 0 || dy < 0
        error("dx and dy must be non-negative, got dx=$(dx), dy=$(dy).")
    end
    return Symbol("vj_$(dx)_$(dy)")
end

"""
用途: 为给定的 Jastrow 位移类生成去重后的有向 PBC offset 列表。

参数:
- `lx::Int`: 晶格在 `x` 方向的长度。
- `ly::Int`: 晶格在 `y` 方向的长度。
- `dx::Int`: `x` 方向绝对位移。
- `dy::Int`: `y` 方向绝对位移。

返回:
- `Vector{Tuple{Int, Int}}`: 在模 `lx`、`ly` 意义下去重后的 offset 列表。

说明:
- 当 `dx = 0` 或 `dy = 0` 时, 对应方向不会重复枚举正负号。
- 当 `dx = lx / 2` 或 `dy = ly / 2` 时, 通过模运算后会自动去重。
"""
function build_jastrow_wrapped_offsets_for_displacement(
    lx::Int,
    ly::Int,
    dx::Int,
    dy::Int,
)::Vector{Tuple{Int,Int}}
    if lx <= 0 || ly <= 0
        error("lx and ly must be positive, got lx=$(lx), ly=$(ly).")
    end
    if dx < 0 || dy < 0
        error("dx and dy must be non-negative, got dx=$(dx), dy=$(dy).")
    end
    if dx == 0 && dy == 0
        error("Displacement (0, 0) is not allowed for Jastrow terms.")
    end
    if dx > fld(lx, 2) || dy > fld(ly, 2)
        error("Displacement exceeds shortest-image range: dx=$(dx), dy=$(dy), lx=$(lx), ly=$(ly).")
    end

    sign_choices_x = dx == 0 ? [1] : [1, -1]
    sign_choices_y = dy == 0 ? [1] : [1, -1]
    wrapped_offset_set = Set{Tuple{Int,Int}}()
    for sign_x in sign_choices_x
        for sign_y in sign_choices_y
            push!(wrapped_offset_set, (mod(sign_x * dx, lx), mod(sign_y * dy, ly)))
        end
    end
    return sort!(collect(wrapped_offset_set))
end

"""
用途: 为给定的 Jastrow 位移类构造唯一无序 pair 集合。

参数:
- `lx::Int`: 晶格在 `x` 方向的长度。
- `ly::Int`: 晶格在 `y` 方向的长度。
- `dx::Int`: `x` 方向绝对位移。
- `dy::Int`: `y` 方向绝对位移。

返回:
- `Vector{Tuple{Int, Int}}`: 经 `i < j` 规范化并按字典序排序后的唯一 pair 列表。
"""
function build_jastrow_pair_set_for_displacement(
    lx::Int,
    ly::Int,
    dx::Int,
    dy::Int,
)::Vector{Tuple{Int,Int}}
    wrapped_offsets = build_jastrow_wrapped_offsets_for_displacement(lx, ly, dx, dy)
    unique_pairs = Set{Tuple{Int,Int}}()
    for x in 1:lx
        for y in 1:ly
            site_index = (x - 1) * ly + y
            for (offset_x, offset_y) in wrapped_offsets
                neighbor_x = mod(x - 1 + offset_x, lx) + 1
                neighbor_y = mod(y - 1 + offset_y, ly) + 1
                neighbor_index = (neighbor_x - 1) * ly + neighbor_y
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
用途: 为给定的 Jastrow 位移类构造对称邻接表。

参数:
- `lx::Int`: 晶格在 `x` 方向的长度。
- `ly::Int`: 晶格在 `y` 方向的长度。
- `dx::Int`: `x` 方向绝对位移。
- `dy::Int`: `y` 方向绝对位移。

返回:
- `Vector{Vector{Int}}`: 满足无自环、无重复、对称的邻接表。
"""
function build_jastrow_neighbor_table_for_displacement(
    lx::Int,
    ly::Int,
    dx::Int,
    dy::Int,
)::Vector{Vector{Int}}
    unique_pairs = build_jastrow_pair_set_for_displacement(lx, ly, dx, dy)
    neighbor_table = [Int[] for _ in 1:(lx*ly)]
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
用途: 根据全位移分类批量构造 Jastrow terms、参数名与默认初值。

参数:
- `lx::Int`: 晶格在 `x` 方向的长度。
- `ly::Int`: 晶格在 `y` 方向的长度。
- `default_value::Float64`: 每个 `vj_dx_dy` 的默认初值。

返回:
- `Tuple{Vector{JastrowProjectorTerm{Float64}}, Vector{Symbol}, Vector{Float64}}`:
  `(jastrow_terms, param_names, init_params)`。
"""
function build_full_displacement_jastrow_terms(
    lx::Int,
    ly::Int;
    default_value::Float64=0.0,
)::Tuple{Vector{JastrowProjectorTerm{Float64}},Vector{Symbol},Vector{Float64}}
    displacement_labels = build_jastrow_displacement_labels(lx, ly)
    jastrow_terms = JastrowProjectorTerm{Float64}[]
    param_names = Symbol[]
    init_params = Float64[]

    for (dx, dy) in displacement_labels
        param_name = build_jastrow_param_name(dx, dy)
        neighbor_table = build_jastrow_neighbor_table_for_displacement(lx, ly, dx, dy)
        push!(
            jastrow_terms,
            JastrowProjectorTerm(
                param_name=param_name,
                v=default_value,
                site_to_neighbor_sites=neighbor_table,
            ),
        )
        push!(param_names, param_name)
        push!(init_params, default_value)
    end

    return jastrow_terms, param_names, init_params
end

"""
用途: 构造受约束 Hubbard 主程序使用的 projector。

参数:
- `lx::Int`: 晶格在 `x` 方向的长度。
- `ly::Int`: 晶格在 `y` 方向的长度。
- `g::Float64`: Gutzwiller projector 参数。
- `jastrow_default_value::Float64`: 全位移 Jastrow 参数的默认初值。

返回:
- `CompositeProjector`: 由一个 Gutzwiller term 和全部 `vj_dx_dy` Jastrow terms 组成的 projector。
"""
function build_restricted_projector(
    lx::Int,
    ly::Int,
    g::Float64;
    jastrow_default_value::Float64=0.0,
)::CompositeProjector
    jastrow_terms, _, _ = build_full_displacement_jastrow_terms(
        lx,
        ly;
        default_value=jastrow_default_value,
    )
    projector_terms = AbstractProjectorTerm[
        GutzwillerProjectorTerm(param_name=:g, g=g),
    ]
    append!(projector_terms, jastrow_terms)
    return CompositeProjector(projector_terms)
end

function update_ansatz!(
    vwf,
    param_names::Vector{Symbol},
    params::Vector{Float64},
    lx,
    ly,
    bcx,
    bcy,
    target_sz::Int;
    nparams_proj::Int=0,
    nparams_backflow::Int=0,
    Q::Float64=0.0,
    x0::Float64=0.0
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
    etax = get(param_map, :etax, 0.0)
    etay = get(param_map, :etay, 0.0)
    mu = get(param_map, :mu, 0.0)
    Delta_AF = get(param_map, :Delta_AF, 0.0)
    Delta_c = get(param_map, :Delta_c, 0.0)
    Delta_s = get(param_map, :Delta_s, 0.0)

    hubbard_params = PartonSquare.RestrictedHubbardParams(
        Lx=lx,
        Ly=ly,
        bcx=bcx,
        bcy=bcy,
        chi1=1.0,
        chi2=chi2,
        etax=etax,
        etay=etay,
        mu=mu,
        Delta_AF=Delta_AF,
        Delta_c=Delta_c,
        Delta_s=Delta_s,
        Q=Q,
        x0=x0
    )

    _, gs_U, dUt_params = PartonSquare.make_ansatz_and_derivs(hubbard_params; param_names=wf_param_names, target_sz=target_sz, Q=Q, x0=x0)

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
    target_sz = args["target_sz"]
    doping = args["doping"]
    lambda = args["lambda"]
    stripe_center = args["stripe_center"]
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
    bf_epsilon = args["bf_epsilon"]
    bf_eta = args["bf_eta"]
    init_params_json = args["init_params_json"]
    N_sites = lx * ly
    #要优化的参数
    if ansatz == "AFM"
        wf_param_names = [:chi2, :etax, :etay, :Delta_AF, :mu]
        wf_init_params = [args["chi2"], args["etax"], args["etay"], args["Delta_AF"], args["mu"]]
        Q = 0.0
        x0 = 0.0
    elseif ansatz == "Stripe"
        wf_param_names = [:chi2, :etax, :etay, :Delta_c, :Delta_s, :mu]
        wf_init_params = [args["chi2"], args["etax"], args["etay"], args["Delta_c"], args["Delta_s"], args["mu"]]
        Q = 2π / lambda
        if stripe_center == "site"
            x0 = 0.0
        elseif stripe_center == "bond"
            x0 = 0.5
        else
            error("Unknown stripe_center type: $stripe_center")
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
    bonds1 = Tuple{Int,Int}[]
    bonds2 = Tuple{Int,Int}[]
    idx(x, y) = mod(x - 1, lx) * ly + mod(y - 1, ly) + 1
    for y in 1:ly, x in 1:lx
        u = idx(x, y)
        push!(bonds1, (u, idx(x + 1, y)))
        push!(bonds1, (u, idx(x, y + 1)))
        push!(bonds2, (u, idx(x + 1, y + 1)))
        push!(bonds2, (u, idx(x - 1, y + 1)))
    end

    backflow_source_bonds = Tuple{Int,Int}[]
    backflow_source_amplitudes = Float64[]
    for (site_i, site_j) in bonds1
        push!(backflow_source_bonds, (site_i, site_j))
        push!(backflow_source_amplitudes, t1)
        push!(backflow_source_bonds, (site_j, site_i))
        push!(backflow_source_amplitudes, t1)
    end
    for (site_i, site_j) in bonds2
        push!(backflow_source_bonds, (site_i, site_j))
        push!(backflow_source_amplitudes, t2)
        push!(backflow_source_bonds, (site_j, site_i))
        push!(backflow_source_amplitudes, t2)
    end

    # Projector 定义
    projector = build_restricted_projector(lx, ly, g)
    backflow = Eq4BackflowTerm(
        param_name_epsilon=:bf_epsilon,
        param_name_eta=:bf_eta,
        epsilon_bf=bf_epsilon,
        eta_bf=bf_eta,
        source_bonds=backflow_source_bonds,
        source_amplitudes=backflow_source_amplitudes,
    )
    #backflow = NoBackflowTerm()
    proj_param_names = projector_param_names(projector)
    proj_init_params = projector_param_values(projector)
    nparams_proj = length(proj_param_names)
    backflow_param_name_list = backflow_param_names(backflow)
    backflow_init_params = backflow_param_values(backflow)
    nparams_backflow = length(backflow_param_name_list)
    # 把波函数参数和投影算符参数拼接成一个向量, 供优化器使用
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
    update_ansatz!(vwf, param_names, init_params, lx, ly, BCX, BCY, target_sz; nparams_proj=nparams_proj, nparams_backflow=nparams_backflow, Q=Q, x0=x0)


    # D. 运行模拟
    folder = "logs"
    mkpath(folder)

    if job == "SR"
        sr_params = SRParams(vmc_params=meas_params, n_steps=n_steps, lr=lr)
        exp_lr_func = build_exponential_lr_func(lr, lr_end, n_steps)

        update_vwf_func! = (vwf, params) -> update_ansatz!(vwf, param_names, params, lx, ly, BCX, BCY, target_sz; nparams_proj=nparams_proj, nparams_backflow=nparams_backflow, Q=Q, x0=x0)

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
