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
        default = 10
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

struct SxSq_Estimator
    N::Int
    Lx::Int
    Ly::Int
    qx::Float64
    qy::Float64
    # 预先计算坐标，避免在循环中反复计算
    coords::Vector{Tuple{Int,Int}}

    function SxSq_Estimator(Lx, Ly, qx, qy)
        N = Lx * Ly
        coords = Vector{Tuple{Int,Int}}(undef, N)
        for idx in 1:N
            # 这里的索引转换逻辑需确保与你的 Model/Sampler 一致
            # 假设 idx 1-based, 先x后y: x = (idx-1) // Ly, y = (idx-1) % Ly
            x = div(idx - 1, Ly)
            y = mod(idx - 1, Ly)
            coords[idx] = (x, y)
        end
        new(N, Lx, Ly, qx, qy, coords)
    end
end

function (est::SxSq_Estimator)(model, vwf)
    acc = 0.0 + 0.0im

    # --- 策略选择 ---
    # 如果 N 很大 (>100)，全双重求和 O(N^2) 会非常慢，因为 measure_SxSx 本身涉及矩阵更新。
    # 我们可以采用 "随机批次参考点" (Stochastic Sampling of Reference Sites)
    # 或者如果 N 较小 (6x6=36)，直接全求和即可。

    # 这里演示全双重求和 (最稳健，适用于 Lx=6, Ly=6)

    for site_i in 1:est.N
        ri_x, ri_y = est.coords[site_i]

        for site_j in 1:est.N
            rj_x, rj_y = est.coords[site_j]

            # 1. 计算位移向量 r = rj - ri
            dx = rj_x - ri_x
            dy = rj_y - ri_y

            # 2. 计算相位 exp(-i * q * (rj - ri))
            dot_prod = est.qx * dx + est.qy * dy
            phase = cis(-dot_prod)

            # 3. 测量 <Sx_i Sx_j>
            # 注意：measure_SxSx 内部通常已经处理了 Hermitian 对称性，
            # 即它返回 real(<Si Sj>)。如果它只计算 S+ S-，则需要小心。
            val_sxsx = measure_SxSx(vwf, site_i, site_j; conserve_sz=true)

            acc += val_sxsx * phase
        end
    end

    # 4. 归一化
    # S(q) = 1/N * sum_{ij} e^{...} <Si Sj>
    return acc / est.N / est.N
end

struct SzSq_Estimator
    N::Int
    phases::Vector{ComplexF64} # 存储预计算好的 exp(-i * q * r_j)

    function SzSq_Estimator(Lx, Ly, qx, qy)
        N = Lx * Ly
        phases = Vector{ComplexF64}(undef, N)

        # 预先计算所有格点的相位因子
        # 遍历顺序必须与 Sampler/Wavefunction 中的格点索引顺序一致
        # 假设索引逻辑是: idx = (x-1)*Ly + y  (Column-major, Julia默认)
        for x in 1:Lx
            for y in 1:Ly
                idx = (x - 1) * Ly + y

                # 物理坐标 (0-based)
                rx = x - 1
                ry = y - 1

                # 计算相位: exp(-i * q * r)
                dot_prod = qx * rx + qy * ry
                phases[idx] = cis(-dot_prod)
            end
        end
        new(N, phases)
    end
end

function (est::SzSq_Estimator)(model, vwf)
    # magnetisation_q = sum_j Sz_j * exp(-i q r_j)
    # 这是一个复数序参数
    m_q = sum(get_Sz(s) * p for (s, p) in zip(vwf.sampler.state, est.phases))
    return abs2(m_q) / est.N / est.N
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
    target_qx = args["qx"] * π
    target_qy = args["qy"] * π
    nMC = args["nMC"]
    wMC = args["wMC"]
    rMC = args["rMC"]
    dMC = 1
    seed = args["seed"]

    N_sites = LX * LY

    opt_params = [PHI]

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
    model_params = Dict(:lx => LX, :ly => LY, :J1 => 1.0, :J2 => 0.0)
    ham = HeisenbergModel(N_sites; model_params=model_params)

    sampler = config_Heisenberg(N_sites, N_sites ÷ 2)
    init_config_Heisenberg!(sampler)

    vwf = vwf_det(zeros(ComplexF64, 2 * N_sites, N_sites), sampler)
    kernel = HeisenbergKernel(conserve_sz=true)

    # C. 更新波函数参数
    if rank == 0
        println("Using parameters: $opt_params")
    end
    update_ansatz_for_phi!(vwf, opt_params, LX, LY, BCX, BCY)


    # D. 配置测量任务
    sx_sq_op = SxSq_Estimator(LX, LY, target_qx, target_qy)
    sz_sq_op = SzSq_Estimator(LX, LY, target_qx, target_qy)

    # [关键修复]：这里使用匿名函数 (m, v) -> sx_sq_op(m, v)
    # 这样类型就是 Function，解决了 MethodError
    observables = Dict{Symbol,Function}(
        # :SxSq_PiPi => (m, v) -> sx_sq_op(m, v),
        :SzSq_PiPi => (m, v) -> sz_sq_op(m, v)
        # :Total_Sz  => (m, v) -> total_Sz_est(m, v)
    )

    if rank == 0
        println("Starting Measurement for SxSq at q=($target_qx, $target_qy)...")
    end

    # E. 运行模拟
    results = run_simulation(ham, vwf, kernel, observables, meas_params)


    # F. 输出结果
    if rank == 0
        # site_i = div(N_sites, 2)
        # for site_j in 1:N_sites
        #     x = div(site_j - 1, LY) 
        #     y = mod(site_j - 1, LY)    
        #     corr = measure_SxSx(vwf, site_i, site_j; conserve_sz=true)
        #     @printf "%4d   | %3d  %3d  |  %.8f\n" site_j x y corr
        # end
        # val_Sq = results[:means][:SxSq_PiPi]
        # history_Sq = results[:histories][:SxSq_PiPi]
        # err_Sq = std(history_Sq) / sqrt(length(history_Sq))

        val_Sqz = results[:means][:SzSq_PiPi]
        history_Sqz = results[:histories][:SzSq_PiPi]
        err_Sqz = std(history_Sqz) / sqrt(length(history_Sqz))

        println("\n" * "="^40)
        println("Measurement Results")
        println("="^40)
        println(@sprintf("Params          : %s", opt_params))
        # println(@sprintf("SxSq(pi, pi)    : %.6f +/- %.6f", abs(val_Sq), err_Sq))
        println(@sprintf("SzSq(pi, pi)    : %.6f +/- %.6f", abs(val_Sqz), err_Sqz))
        println("="^40)

        # 简单写入文件
        outfile = "logs/Square_SzSzq_Lx$(LX)_Ly$(LY)_Phi$(PHI)_BCXY$(BCX)_$(BCY).txt"
        mkpath(dirname(outfile))
        open(outfile, "a") do io
            # println(io, "# qx qy SxSq Error")
            qx = args["qx"]
            qy = args["qy"]
            nSMPL = length(history_Sqz)
            println(io, "$(qx)$pi $qy$pi $val_Sqz $err_Sqz $nSMPL")
        end
    end
end

main()
