module Driver

using MPI
using Random
using Dates
using Statistics
using LinearAlgebra   # 新增：用于 SR 矩阵运算
using Printf
using DelimitedFiles

using ..Sampler       # 使用上层模块定义的子模块
using ..VMC
using ..Model
using ..MPI_VMC_Utils

export VMCParams, SRParams
export run_simulation, run_sr_optimization


# ==============================================================================
# 1. 参数配置
# ==============================================================================
struct VMCParams
    total_samples::Int      # 总采样数 (所有核之和)
    warmup_steps::Int       # 热身步数 (Sweep)
    decorr_steps::Int       # 间隔步数 (Sweep)
    rebuild_every::Int      # 重建逆矩阵频率
    seed::Int               # 随机数种子

    function VMCParams(;
        total_samples=10000,
        warmup_steps=1000,
        decorr_steps=4,
        rebuild_every=100,
        seed=1234
    )
        new(total_samples, warmup_steps, decorr_steps, rebuild_every, seed)
    end
end

# ==============================================================================
# 2. SR 优化参数
# ==============================================================================
struct SRParams
    n_steps::Int            # 优化总步数
    lr::Float64             # 学习率
    diag_shift::Float64     # S矩阵对角偏移 (Ridge)
    max_step_size::Float64  # 单次更新最大幅度 (Clipping)
    vmc_params::VMCParams

    function SRParams(;
        n_steps=50,
        lr=0.05,
        diag_shift=1e-3,
        max_step_size=0.1,
        vmc_params::VMCParams=VMCParams()
    )
        new(n_steps, lr, diag_shift, max_step_size, vmc_params)
    end
end

# ==============================================================================
# 2. 通用驱动函数
# ==============================================================================
"""
    run_simulation(model, vwf, kernel, observables, params)

通用 VMC 采样驱动器。

# 参数
- `observables`: Dict{Symbol, Function}。Key是名字, Value是函数 `f(model, vwf) -> Number`。
                 注意：代码默认只对 :E (能量) 进行全历史记录(Binning)，其他量只记录均值。
"""
function run_simulation(model, vwf, kernel, observables::Dict{Symbol,Function}, params::VMCParams)

    # --- 1. MPI 初始化 ---
    session = init_mpi_session()
    rank = session.rank
    is_root = (rank == session.root)

    # 计算本地采样数
    n_samples_local = div(params.total_samples, session.size)
    Nlat = vwf.sampler.N_sites

    # [关键] 获取排序后的键名列表，确保所有 MPI 进程按相同顺序处理
    obs_names = sort(collect(keys(observables)))

    if is_root
        println("="^60)
        println(" VMC Simulation Driver")
        println(" MPI Size: $(session.size)")
        println(" Total Samples: $(params.total_samples) (Local: $n_samples_local)")
        println(" Observables: $obs_names")
        println("="^60)
    end

    # --- 2. 初始化 Runner & RNG ---
    runner = VMCRunner(model, vwf; kernel=kernel, auto_fix=true)

    rng = Random.default_rng()
    Random.seed!(rng, hash(rank, UInt(params.seed)))

    # --- 3. Warmup (热身) ---
    if is_root
        println("[Driver] Starting Warmup ($(params.warmup_steps) sweeps)...")
    end

    steps_since_rebuild = 0
    for _ in 1:params.warmup_steps
        for _ in 1:Nlat
            mcmc_step!(runner, rng)
            steps_since_rebuild += 1
            if steps_since_rebuild >= params.rebuild_every
                rebuild_inverse!(vwf)
                steps_since_rebuild = 0
            end
        end
    end
    rebuild_inverse!(vwf)
    steps_since_rebuild = 0

    # --- 4. 准备测量 Buffer ---
    obs_buf = ObservableBuffer(ComplexF64)

    for name in obs_names
        register_scalar!(obs_buf, name)
    end

    MPI.Barrier(session.comm)
    t_start = time()

    # --- 5. 采样循环 ---
    if is_root
        println("[Driver] Starting Measurement Loop...")
    end

    for i in 1:n_samples_local
        # Decorrelation
        for _ in 1:params.decorr_steps
            for _ in 1:Nlat
                mcmc_step!(runner, rng)
                steps_since_rebuild += 1
                if steps_since_rebuild >= params.rebuild_every
                    rebuild_inverse!(vwf)
                    steps_since_rebuild = 0
                end
            end
        end

        # Measurement
        for name in obs_names
            func = observables[name]
            val = func(model, vwf)

            # 1. 累加到 Buffer (用于计算均值)
            accumulate_sample!(obs_buf, name, val)

            # 2. [修改] 记录全历史 (用于 Binning)
            # 现在对所有注册的标量都进行记录，不仅仅是能量
            record_scalar!(obs_buf, name, real(val))
        end

        increment_counter!(obs_buf)
    end

    t_end = time()
    elapsed = t_end - t_start

    if is_root
        println("[Driver] Sampling done in $(round(elapsed, digits=2)) s.")
    end

    # --- 6. 数据收集 ---

    # A. 收集所有量的均值
    means = mpi_reduce_all(obs_buf, session)

    # B. [修改] 收集所有量的完整历史
    # 我们将结果存在一个新的字典 histories 中
    histories = Dict{Symbol,Vector{Float64}}()

    # 必须所有 Rank 都参与这个循环，顺序必须一致 (由 obs_names 保证)
    for name in obs_names
        full_list = mpi_gather_scalar(obs_buf, session, name)

        # 只有 Root 会收到数据，其他 Rank 收到 nothing
        if is_root && full_list !== nothing
            histories[name] = full_list
        end
    end

    if is_root
        # 返回结构改为包含 histories
        return Dict(:means => means, :histories => histories)
    else
        return nothing
    end
end


# ==============================================================================
# 3. SR Optimization Helpers
# ==============================================================================

"""
    run_sr_optimization(model, vwf, kernel, initial_params, update_vwf_func!, 
                        sr_params; log_file="sr_history.txt")

SR 优化主循环。
- `sr_params`: 包含优化策略 (lr, steps) 和采样策略 (vmc_params)。
"""
function run_sr_optimization(model, vwf, kernel,
    initial_params::Vector{Float64},
    update_vwf_func!::Function,
    sr_params::SRParams; # <--- 接口变简洁了
    log_file="sr_history.txt", param_names::Union{Nothing,Vector{Symbol}}=nothing)

    session = init_mpi_session()
    rank = session.rank
    is_root = (rank == session.root)

    # 为了方便引用，提取内部的 vmc params
    vmc = sr_params.vmc_params

    # --- Setup ---
    current_params = copy(initial_params)
    n_params = length(current_params)

    # RNG Setup
    rng = Random.default_rng()
    Random.seed!(rng, hash(rank, UInt(vmc.seed)))

    # Buffer Setup
    obs_buf = ObservableBuffer(ComplexF64)
    register_scalar!(obs_buf, :E)
    register_vector!(obs_buf, :O_avg, n_params)
    register_vector!(obs_buf, :EO_avg, n_params)
    register_matrix!(obs_buf, :S_mat, n_params, n_params)

    # Local counters
    n_local = div(vmc.total_samples, session.size)
    Nlat = vwf.sampler.N_sites

    if is_root
        println("="^60)
        println(" SR Optimization Started")
        println(" Opt: Steps=$(sr_params.n_steps), LR=$(sr_params.lr)")
        println(" VMC: Samples=$(vmc.total_samples), Warmup=$(vmc.warmup_steps)")
        println(" Params: $n_params dimensions")
        println("="^60)

        # [改进 1] 动态生成文件表头
        # Step, E_mean, E_err, GradNorm, P_1, P_2, ...
        if param_names === nothing
            param_headers = ["P_$i" for i in 1:n_params]
        else
            param_headers = String.(param_names)
        end
        headers = ["Step", "E_mean", "E_err", "GradNorm", param_headers...]

        open(log_file, "w") do io
            # 使用 join 确保表头格式工整
            println(io, "# " * join(headers, "\t"))
        end
    end

    # --- Optimization Loop ---
    for step in 1:sr_params.n_steps

        # 1. Sync Parameters
        current_params = MPI.bcast(current_params, session.root, session.comm)

        # 2. Update VWF Matrix (User Callback)
        update_vwf_func!(vwf, current_params)

        # 3. [Safety] Re-create Runner
        # 确保 Runner 内部的 log_psi 缓存是基于新参数的
        runner = VMCRunner(model, vwf; kernel=kernel, auto_fix=true)

        # 4. Sampling Prep
        reset_buffers!(obs_buf)

        # Warmup (Necessary after parameter change)
        steps_since_rebuild = 0
        for _ in 1:vmc.warmup_steps
            for _ in 1:Nlat
                mcmc_step!(runner, rng)
            end
            steps_since_rebuild += Nlat
            if steps_since_rebuild >= vmc.rebuild_every
                rebuild_inverse!(vwf)
                steps_since_rebuild = 0
            end
        end
        rebuild_inverse!(vwf)

        # 5. Measurement Loop
        for _ in 1:n_local
            for _ in 1:vmc.decorr_steps
                for _ in 1:Nlat
                    mcmc_step!(runner, rng)
                end
            end

            E_loc = local_energy(model, vwf)
            accumulate_sr_stats!(obs_buf, vwf, E_loc)
            increment_counter!(obs_buf)
        end

        # 6. Collect & Solve
        means, E_history = collect_sr_data(obs_buf, session)

        if is_root
            # [Fix] 使用泛型 Solver，自动处理类型
            delta, grad_vec, E_err = solve_sr_update(means, E_history;
                diag_shift=sr_params.diag_shift,
                is_real_param=true)

            # Apply update
            step_vector = sr_params.lr .* delta
            step_vector = clamp.(step_vector, -sr_params.max_step_size, sr_params.max_step_size)
            current_params += step_vector

            # Logging
            E_mean = real(means[:E])
            grad_norm = norm(grad_vec)

            # [改进 2] 智能终端输出
            # 如果参数少于等于4个，全部显示；否则显示前3个加省略号
            param_str = ""
            if n_params <= 4
                param_str = join([@sprintf("%.4f", p) for p in current_params], ", ")
            else
                param_preview = join([@sprintf("%.4f", p) for p in current_params[1:3]], ", ")
                param_str = "$param_preview, ..."
            end

            @printf("Step %3d | E: %.6f +/- %.6f | |g|: %.4e | Params: [%s]\n",
                step, E_mean, E_err, grad_norm, param_str)

            # [改进 3] 文件写入保持完整数据，使用制表符分割
            open(log_file, "a") do io
                row = [step, E_mean, E_err, grad_norm, current_params...]
                writedlm(io, permutedims(row))
            end
            flush(stdout)
        end
    end

    return current_params
end

function accumulate_sr_stats!(obs_buf::ObservableBuffer{T}, vwf, E_loc::Number) where T
    O_vec = compute_grad_log_psi!(vwf)
    accumulate_sample!(obs_buf, :E, E_loc)
    accumulate_sample!(obs_buf, :O_avg, O_vec)
    accumulate_sample!(obs_buf, :EO_avg, E_loc .* O_vec)
    accumulate_sr_matrix!(obs_buf, :S_mat, O_vec)
    record_scalar!(obs_buf, :E, real(E_loc))
    return nothing
end

function collect_sr_data(obs_buf, session)
    means = mpi_reduce_all(obs_buf, session)
    E_history = mpi_gather_scalar(obs_buf, session, :E)
    return means, E_history
end

function solve_sr_update(means::Dict{Symbol,Any}, E_history::Vector{T};
    diag_shift::Float64=1e-3, is_real_param::Bool=true) where T<:Number
    N_total = length(E_history)
    E_err = std(E_history) / sqrt(N_total)

    E_mean = means[:E]
    O_mean = means[:O_avg]
    EO_mean = means[:EO_avg]
    S_raw = means[:S_mat]
    n_params = length(O_mean)

    grad_vec_complex = EO_mean - (E_mean .* O_mean)
    S_mat_complex = S_raw - (conj(O_mean) * transpose(O_mean))

    if is_real_param
        grad_final = 2.0 .* real(grad_vec_complex)
        S_final = real.(S_mat_complex)
    else
        grad_final = grad_vec_complex
        S_final = S_mat_complex
    end
    S_reg = S_final + diag_shift * I(n_params)
    delta = -(S_reg \ grad_final)
    return delta, grad_final, E_err
end

end # module
