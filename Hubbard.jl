using MPI
using Random
using Printf
using DelimitedFiles
using LinearAlgebra
using Statistics
using ArgParse
# using FFWT

# === 1. 环境设置 ===
push!(LOAD_PATH, "./src")
push!(LOAD_PATH, ".")

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
        "--etad1"
        help = "MF parameters"
        arg_type = Float64
        default = 0.01
        "--etas1"
        help = "MF parameters"
        arg_type = Float64
        default = 0.01
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
    end

    return parse_args(s)
end

# ==============================================================================
# 3. 辅助函数
# ==============================================================================

function update_ansatz!(vwf, param_names::Vector{Symbol}, params::Vector{Float64}, lx, ly, bcx, bcy, target_sz::Int)
    # 这里也可以把 bcx, bcy 提出来作为参数
    param_map = Dict{Symbol,Float64}(zip(param_names, params))

    mu = get(param_map, :mu, 0.0)
    etad1 = get(param_map, :etad1, 0.0)
    etas1 = get(param_map, :etas1, 0.0)

    mz = Dict{Symbol,Float64}()

    for (name, value) in param_map
        name_str = String(name)
        if startswith(name_str, "mz_")
            mz[name] = value
        elseif name == :mu || name == :etad1 || name == :etas1
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
        mu=mu,
        chi1=1.0,
        etad1=etad1,
        etas1=etas1,
        mz=mz
    )

    _, gs_U, dUt_params = PartonSquare.make_ansatz_and_derivs(hubbard_params; param_names=param_names, target_sz=target_sz)

    copyto!(vwf.gs_U, gs_U)
    copyto!(vwf.gs_U_t, permutedims(gs_U))
    update_vwf_params!(vwf, dUt_params)
    init_gswf!(vwf)
end

# ==============================================================================
# 4. 主程序
# ==============================================================================

function main()
    args = parse_commandline()

    session = init_mpi_session()
    rank = session.rank

    # ---------------------------------------------------------
    # A. 参数设定 (全部集中在这里)
    # ---------------------------------------------------------
    LX = args["Lx"]
    LY = args["Ly"]
    BCX = args["bcx"]
    BCY = args["bcy"]
    target_sz = args["target_sz"]
    doping = args["doping"]
    # if mod(LX, 4) == 0
    #     BCX = -1
    # end
    # if mod(LY, 4) == 0
    #     BCY = -1
    # end
    nMC = args["nMC"]
    wMC = args["wMC"]
    rMC = args["rMC"]
    dMC = args["dMC"]
    seed = args["seed"]
    nSR = args["nSR"]
    lr = args["lr"]
    lr_end = args["lr_end"]

    t1 = args["t1"]
    t2 = args["t2"]
    U = args["U"]

    N_sites = LX * LY
    #要优化的参数
    para_names = [:etad1, :etas1, :mu]
    init_params = [args[String(alpha)] for alpha in para_names]
    #对每一列的mz，构建mean field参数mz_i,i为第几列
    for i in 1:LY
        push!(para_names, Symbol("mz_$i"))
        push!(init_params, args["mz"])
    end


    # VMC 采样参数
    meas_params = VMCParams(
        total_samples=nMC,
        warmup_steps=wMC,
        rebuild_every=rMC,
        decorr_steps=dMC,
        seed=args["seed"] + rank
    )
    sr_params = SRParams(vmc_params=meas_params, n_steps=nSR)
    # ---------------------------------------------------------

    # B. 模型与波函数初始化
    #GeneralModel定义
    bonds1 = Tuple{Int,Int}[]
    bonds2 = Tuple{Int,Int}[]
    idx(x, y) = mod(x - 1, LX) * LY + mod(y - 1, LY) + 1
    for y in 1:LY, x in 1:LX
        u = idx(x, y)
        push!(bonds1, (u, idx(x + 1, y)))
        push!(bonds1, (u, idx(x, y + 1)))
        push!(bonds2, (u, idx(x + 1, y + 1)))
        push!(bonds2, (u, idx(x - 1, y + 1)))
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

    vwf = vwf_det(zeros(Float64, 2 * N_sites, N_sites + target_sz), sampler)
    kernel = HubbardKernel(conserve_sz=true)

    # C. 更新波函数参数
    if rank == 0
        println("Initial parameters: $init_params")
    end
    update_ansatz!(vwf, para_names, init_params, LX, LY, BCX, BCY, target_sz)
    update_vwf_func! = (vwf, params) -> update_ansatz!(vwf, para_names, params, LX, LY, BCX, BCY, target_sz)


    # D. 运行模拟
    results = run_sr_optimization(ham, vwf, kernel, init_params, update_vwf_func!, sr_params; log_file="logs/sr_history.txt", param_names=para_names)

end

main()
