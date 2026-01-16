using MPI
using Random
using Printf
using DelimitedFiles
using LinearAlgebra
using Statistics 
using ArgParse 

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
            default = 6
        "--Ly"
            help = "Lattice size in Y direction"
            arg_type = Int
            default = 6
        "--bcx"
            help = "Boundary condition phase in X (1.0 or -1.0)"
            arg_type = Float64
            default = 1.0
        "--bcy"
            help = "Boundary condition phase in Y (1.0 or -1.0)"
            arg_type = Float64
            default = 1.0
        "--phi"
            help = "U1 parameters"
            arg_type = Float64
            default = 0.22
        "--qx"
            help = "target qx to be computed"
            arg_type = Float64
            default = 1.0
        "--qy"
            help = "target qy to be computed"
            arg_type = Float64
            default = 1.0
        "--nMC"
            help = "Number of Monte Carlo total_samples"
            arg_type = Int
            default = 100000
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
# 2. 核心测量器: 固定 i 的 S^xx(q)
# ==============================================================================

struct SxSq_Estimator
    N::Int
    Lx::Int
    Ly::Int
    phases::Vector{ComplexF64} # 存储 exp(-i * q * r_j)

    function SxSq_Estimator(Lx, Ly, qx, qy)
        N = Lx * Ly
        phases = Vector{ComplexF64}(undef, N)
        
        for x in 1:Lx
            for y in 1:Ly
                # 计算格点索引 (需与 Model.jl 一致)
                # idx = (x-1)*Ly + (y-1) + 1
                site_idx = mod(x-1, Lx)*Ly + mod(y-1, Ly) + 1
                
                # 物理坐标
                rx = x - 1
                ry = y - 1
                
                # S(q) = 1/N sum_{ij} ...
                # 这里相位取 exp(-i q r)
                dot_prod = qx * rx + qy * ry
                phases[site_idx] = cis(-dot_prod) 
            end
        end
        new(N, Lx, Ly, phases)
    end
end

# Functor 接口
function (est::SxSq_Estimator)(model, vwf)
    acc = 0.0 + 0.0im
    
    # 利用平移对称性，固定 site_i = 1
    site_i = div(est.N, 2)
    
    # 遍历所有格点 j
    for site_j in 1:est.N
        # 1. 获取相位因子
        phase = est.phases[site_j]
        
        # 2. 计算实空间关联 <Sx_i Sx_j>
        # 注意：这里显式传入 conserve_sz=true，确保只计算 spin exchange 部分
        val_sxsx = measure_SxSx(vwf, site_i, site_j; conserve_sz=true)
        # val_sxsx = measure_SzSz(vwf, site_i, site_j)
        
        # 3. 累加
        acc += val_sxsx * phase
    end
    
    # 4. [修复] 归一化: 除以 N
    # 原始公式 sum_{j} <S_0 S_j> 随尺寸线性增长
    # 除以 N 后得到的是 intensive quantity (类似 magnetization squared)
    acc /= est.N
    
    return acc
end

function total_Sz_est(model, vwf)
    return measure_total_Sz(vwf)
end

# ==============================================================================
# 3. 辅助函数
# ==============================================================================

function update_ansatz_for_phi!(vwf, params::Vector{Float64}, LX, LY, BCX, BCY)
    phi_val = params[1]
    # 这里也可以把 bcx, bcy 提出来作为参数
    flux_params = PartonSquare.U1SFluxParams(Lx=LX, Ly=LY, phi=phi_val, bcy=BCX, bcx=BCY)
    _, gs_U, dUt_params = PartonSquare.make_ansatz_and_derivs(flux_params)
    
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
    PHI = args["phi"]
    BCX = args["bcx"]
    BCY = args["bcy"]
    # if mod(LX, 4) == 0
    #     BCX = -1
    # end
    # if mod(LY, 4) == 0
    #     BCY = -1
    # end
    target_qx = args["qx"]*π
    target_qy = args["qy"]*π
    nMC = args["nMC"]
    wMC = args["wMC"]
    rMC = args["rMC"]
    dMC = 1
    seed = args["seed"]

    N_sites = LX * LY
    
    opt_params = [PHI] 

    # VMC 采样参数
    meas_params = VMCParams(
        total_samples = nMC, 
        warmup_steps  = wMC,
        rebuild_every = rMC,
        decorr_steps  = dMC,
        seed          = args["seed"] + rank
    )
    # ---------------------------------------------------------

    # B. 模型与波函数初始化
    model_params = Dict(:lx=>LX, :ly=>LY, :J1=>1.0, :J2=>0.0)
    ham = HeisenbergModel(N_sites; model_params=model_params)
    
    sampler = config_Heisenberg(N_sites, N_sites÷2)
    init_config_Heisenberg!(sampler)
    
    vwf = vwf_det(zeros(ComplexF64, 2*N_sites, N_sites), sampler)
    kernel = HeisenbergKernel(conserve_sz=true) 

    # C. 更新波函数参数
    if rank == 0
        println("Using parameters: $opt_params")
    end
    update_ansatz_for_phi!(vwf, opt_params, LX, LY, BCX, BCY)


    # D. 配置测量任务
    sx_sq_op = SxSq_Estimator(LX, LY, target_qx, target_qy)
    
    # [关键修复]：这里使用匿名函数 (m, v) -> sx_sq_op(m, v)
    # 这样类型就是 Function，解决了 MethodError
    observables = Dict{Symbol, Function}(
        :SxSq_PiPi => (m, v) -> sx_sq_op(m, v)
        # :Total_Sz  => (m, v) -> total_Sz_est(m, v)
    )
    
    if rank == 0
        println("Starting Measurement for SxSq at q=($target_qx, $target_qy)...")
    end
    
    # E. 运行模拟
    results = run_simulation(ham, vwf, kernel, observables, meas_params)

    
    # F. 输出结果
    if rank == 0
        site_i = div(N_sites, 2)
        for site_j in 1:N_sites
            x = div(site_j - 1, LY) 
            y = mod(site_j - 1, LY)    
            corr = measure_SxSx(vwf, site_i, site_j; conserve_sz=true)
            @printf "%4d   | %3d  %3d  |  %.8f\n" site_j x y corr
        end
        val_Sq = results[:means][:SxSq_PiPi]
        
        # 计算误差
        history_Sq = results[:histories][:SxSq_PiPi]
        err_Sq = std(history_Sq) / sqrt(length(history_Sq))
        
        println("\n" * "="^40)
        println("Measurement Results")
        println("="^40)
        println(@sprintf("Params          : %s", opt_params))
        println(@sprintf("SxSq(pi, pi)    : %.6f +/- %.6f", abs(val_Sq), err_Sq))
        println("="^40)
        
        # 简单写入文件
        outfile = "logs/Square_SxSxq_Lx$(LX)_Ly$(LY)_Phi$(PHI)_BCXY$(BCX)_$(BCY).txt"
        open(outfile, "a") do io
            # println(io, "# qx qy SxSq Error")
            qx = args["qx"]; qy = args["qy"]
            nSMPL = length(history_Sq)
            println(io, "$(qx)$pi $qy$pi $val_Sq $err_Sq $nSMPL")
        end
    end
end

main()
