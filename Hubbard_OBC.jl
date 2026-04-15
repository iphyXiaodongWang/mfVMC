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
        help = "Boundary condition phase in X. Use 0.0 for OBC in X"
        arg_type = Float64
        default = 0.0
        "--bcy"
        help = "Boundary condition phase in Y (1.0 or -1.0) for periodic Y"
        arg_type = Float64
        default = 0.999
        "--etad1"
        help = "MF parameters"
        arg_type = Float64
        default = 0.01
        "--etas1"
        help = "MF parameters"
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

function update_ansatz!(vwf, param_names::Vector{Symbol}, params::Vector{Float64}, lx, ly, bcx, bcy, target_sz::Int; nparams_proj::Int=0)
    # 支持输入为 wf 参数 + projector 参数的拼接向量
    nparms = length(param_names)
    wf_param_names = param_names[1:(nparms-nparams_proj)]
    wf_param_values = params[1:(nparms-nparams_proj)]
    projector_param_names = param_names[(nparms-nparams_proj+1):end]
    projector_param_values = params[(nparms-nparams_proj+1):end]
    # 这里也可以把 bcx, bcy 提出来作为参数
    param_map = Dict{Symbol,Float64}(zip(wf_param_names, wf_param_values))

    etad1 = get(param_map, :etad1, 0.0)
    etas1 = get(param_map, :etas1, 0.0)
    chi2 = get(param_map, :chi2, 0.0)

    mz = Dict{Symbol,Float64}()
    mu = Dict{Symbol,Float64}()

    for (name, value) in param_map
        name_str = String(name)
        if startswith(name_str, "mz_")
            mz[name] = value
        elseif startswith(name_str, "mu_")
            mu[name] = value
        elseif name == :etad1 || name == :etas1 || name == :chi2
            continue
        else
            error("Unknown parameter name: $name")
        end
    end

    hubbard_params = PartonSquare.HubbardParams(
        Lx=lx,
        Ly=ly,
        bcx=bcx,
        bcy=bcy,
        chi1=1.0,
        etad1=etad1,
        etas1=etas1,
        chi2=chi2,
        mu=mu,
        mz=mz
    )

    _, gs_U, dUt_params = PartonSquare.make_ansatz_and_derivs(hubbard_params; param_names=wf_param_names, target_sz=target_sz)

    copyto!(vwf.gs_U, gs_U)
    copyto!(vwf.gs_U_t, permutedims(gs_U))
    dUt_matrix = zeros(Float64, size(gs_U, 2), size(gs_U, 1), length(wf_param_names))
    for (idx, name) in enumerate(wf_param_names)
        dUt_matrix[:, :, idx] = dUt_params[name]
    end
    update_vwf_params!(vwf, wf_param_names, dUt_matrix)
    init_gswf!(vwf)
    if !isempty(projector_param_names)
        update_vwf_projector_params!(vwf, projector_param_names, projector_param_values)
    end
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
    if abs(BCX) > 1e-12
        if is_root
            println("[Hubbard_OBC] Force x-open boundary: override --bcx=$BCX to 0.0")
        end
        BCX = 0.0
    end
    if abs(BCY) <= 1e-12
        if is_root
            println("[Hubbard_OBC] Force y-periodic boundary: override --bcy=$BCY to 1.0")
        end
        BCY = 1.0
    end
    target_sz = args["target_sz"]
    doping = args["doping"]
    lambda = args["lambda"]
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
    wf_param_names = [:etad1, :etas1, :chi2]
    wf_init_params = [args["etad1"], args["etas1"], args["chi2"]]
    #对每一列的mz，构建mean field参数mz_i,i为第几列
    offset = lambda ÷ 2
    if ansatz == "Stripe"
        for i in 1:lx
            i_offset = mod(i - offset - 1, lx) + 1
            istripe = div(i_offset - 1, lambda)
            Q = (-1)^istripe
            push!(wf_param_names, Symbol("mz_$i"))
            push!(wf_param_names, Symbol("mu_$i"))
            if i_offset % lambda == 1 || i_offset % lambda == 0
                #stripe的边界mz设成0,内部mz每隔一个stripe反号一次
                push!(wf_init_params, 0.0)
                push!(wf_init_params, args["mu"])
            else
                push!(wf_init_params, args["mz"] * Q)
                push!(wf_init_params, args["mu"])
            end
        end
    elseif ansatz == "AFM"
        for i in 1:lx
            push!(wf_param_names, Symbol("mz_$i"))
            push!(wf_param_names, Symbol("mu_$i"))
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
    # GeneralModel定义
    # OBC的情况下, 只有x方向开边界, y方向周期边界
    bonds1 = Tuple{Int,Int}[]
    bonds2 = Tuple{Int,Int}[]

    # 线性索引: x方向不开边界绕回, y方向周期
    idx(x, y) = (x - 1) * ly + mod(y - 1, ly) + 1

    for y in 1:ly, x in 1:lx
        u = idx(x, y)
        # x方向最近邻: 仅在内部列连接, 不做 x=Lx -> 1 绕回
        if x < lx
            push!(bonds1, (u, idx(x + 1, y)))
        end

        # y方向最近邻: 始终周期连接
        push!(bonds1, (u, idx(x, y + 1)))

        # 对角t2项: y方向周期, x方向不开边界绕回
        if x < lx
            push!(bonds2, (u, idx(x + 1, y + 1)))
        end
        if x > 1
            push!(bonds2, (u, idx(x - 1, y + 1)))
        end
    end

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
    #pinning potential
    for i in 1:ly
        push!(terms, OperatorTerm([:Sz], [idx(1, i)], 0.1 * (-1)^(i + 1)))
        push!(terms, OperatorTerm([:Sz], [idx(lx, i)], 0.1 * (-1)^(i + lx)))
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
    update_ansatz!(vwf, param_names, init_params, lx, ly, BCX, BCY, target_sz; nparams_proj=nparams_proj)


    # D. 运行模拟
    folder = "logs"
    mkpath(folder)

    if job == "SR"
        sr_params = SRParams(vmc_params=meas_params, n_steps=n_steps, lr=lr)
        exp_lr_func = build_exponential_lr_func(lr, lr_end, n_steps)

        update_vwf_func! = (vwf, params) -> update_ansatz!(vwf, param_names, params, lx, ly, BCX, BCY, target_sz; nparams_proj=nparams_proj)

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

main()
