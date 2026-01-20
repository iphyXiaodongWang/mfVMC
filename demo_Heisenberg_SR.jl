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
        default = 8
        "--bcx"
        help = "Boundary condition phase in X (1.0 or -1.0)"
        arg_type = Float64
        default = 1.0
        "--bcy"
        help = "Boundary condition phase in Y (1.0 or -1.0)"
        arg_type = Float64
        default = 1.0
        "--etad1"
        help = "MF parameters"
        arg_type = Float64
        default = 0.5
        "--etas1"
        help = "MF parameters"
        arg_type = Float64
        default = 0.01
        "--mz"
        help = "AFM order parameters"
        arg_type = Float64
        default = 0.1
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
        "--seed"
        help = "random seed"
        arg_type = Int
        default = 5423
    end

    return parse_args(s)
end

# ==============================================================================
# 3. 辅助函数
# ==============================================================================

function update_ansatz!(vwf, para_names::Vector{Symbol}, params::Vector{Float64}, LX, LY, BCX, BCY; target_sz::Int=0)
    # 这里也可以把 bcx, bcy 提出来作为参数
    kwargs = NamedTuple{Tuple(para_names)}(params)
    heisenberg_params = PartonSquare.HeisenbergParams(; Lx=LX, Ly=LY, bcx=BCX, bcy=BCY, chi1=1.0, kwargs...)
    _, gs_U, dUt_params = PartonSquare.make_ansatz_and_derivs(heisenberg_params; para_names=para_names, target_sz=target_sz)

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
    # if mod(LX, 4) == 0
    #     BCX = -1
    # end
    # if mod(LY, 4) == 0
    #     BCY = -1
    # end
    nMC = args["nMC"]
    wMC = args["wMC"]
    rMC = args["rMC"]
    dMC = 1
    seed = args["seed"]

    N_sites = LX * LY
    #要优化的参数
    para_names = [:etad1, :etas1, :mz]
    init_params = [args[String(alpha)] for alpha in para_names]

    # VMC 采样参数
    meas_params = VMCParams(
        total_samples=nMC,
        warmup_steps=wMC,
        rebuild_every=rMC,
        decorr_steps=dMC,
        seed=args["seed"] + rank
    )
    sr_params = SRParams(vmc_params=meas_params)
    # ---------------------------------------------------------

    # B. 模型与波函数初始化
    model_params = Dict(:lx => LX, :ly => LY, :J1 => 1.0, :J2 => 0.0)
    ham = HeisenbergModel(N_sites; model_params=model_params)
    #检查target_sz的parity
    @assert (target_sz + N_sites) % 2 == 0 "Wrong parity!"
    sampler = config_Heisenberg(N_sites, (N_sites + target_sz) ÷ 2; ifPH=true)
    init_config_Heisenberg!(sampler)

    vwf = vwf_det(zeros(Float64, 2 * N_sites, N_sites + target_sz), sampler)
    kernel = HeisenbergKernel(conserve_sz=true)

    # C. 更新波函数参数
    if rank == 0
        println("Initial parameters: $init_params")
    end
    update_ansatz!(vwf, para_names, init_params, LX, LY, BCX, BCY; target_sz=target_sz)
    update_vwf_func! = (vwf, params) -> update_ansatz!(vwf, para_names, params, LX, LY, BCX, BCY; target_sz=target_sz)


    # D. 运行模拟
    results = run_sr_optimization(ham, vwf, kernel, init_params, update_vwf_func!, sr_params; log_file="logs/sr_history.txt", param_names=para_names)

end

main()
