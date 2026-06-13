# 用途: 详细计时对比 simplify 与 backflow_stripe 的 Emery backflow 性能。
# 系统: Lx=8, Ly=4, doping=0.125 (参考 time 测试参数)
# 单核运行。

using Random
using LinearAlgebra
using Printf
using Dates

push!(LOAD_PATH, joinpath(@__DIR__, "src"))
push!(LOAD_PATH, @__DIR__)
include(joinpath(@__DIR__, "Emery.jl"))

using mfVMC
import mfVMC.Timing: ENABLE_TIMING, timing_reset!, timing_report
import mfVMC.VMC: mcmc_step!, rebuild_slater_state!, find_stable_config!
import mfVMC.VMC: calc_ratio_rebuild
import mfVMC.Sampler: config_Hubbard, init_config_Hubbard!

ENABLE_TIMING[] = true

const N_SITES_X = 8
const N_SITES_Y = 4
const DOPING = 0.125
const TARGET_SZ = 0
const N_THERM = 2000
const N_STEPS = 5000

function main()
    lx = N_SITES_X; ly = N_SITES_Y
    n_sites = emery_n_sites(lx, ly)
    nelec = compute_emery_electron_count(lx, ly, DOPING)
    nup = (nelec + TARGET_SZ) ÷ 2; ndn = nelec - nup

    println("="^60)
    println("  Emery Backflow 详细计时对比")
    @printf("  Lx=%d Ly=%d  sites=%d  nelec=%d\n", lx, ly, n_sites, nelec)
    println("="^60)

    # --- 1. 构造 ---
    timing_reset!()
    sampler = config_Hubbard(n_sites, nup, ndn; ifPH=false)
    init_config_Hubbard!(sampler)

    dd_src, dd_amp, dp_src, dp_amp,
    pd_src, pd_amp, pp_src, pp_amp =
        build_emery_backflow_source_data_by_directed_orbital_type(
            lx, ly; tpd=1.0, tpp=0.5, bcy=1.0)
    backflow = build_column_directed_emery_backflow(
        dd_src, dd_amp, dp_src, dp_amp,
        pd_src, pd_amp, pp_src, pp_amp,
        0.91, 1.08, 0.11, -0.07, 0.05, -0.03, -0.13, 0.17,
        -0.19, 0.23, 0.29, -0.31, 0.37, -0.41, -0.43, 0.47,
        -0.53, 0.59)

    n_rows = 2 * n_sites; n_electrons = length(sampler.electron_locs)
    raw = randn(MersenneTwister(42), n_rows, n_electrons)
    base_orbitals = Matrix(qr(raw).Q)
    vwf = mfVMC.vwf_det(base_orbitals, sampler; backflow=backflow)
    kernel = mfVMC.HubbardKernel(; conserve_sz=true)

    # --- 2. 寻找稳定构型 ---
    println("\n--- 2. find_stable_config! ---")
    timing_reset!()
    find_stable_config!(vwf, kernel, MersenneTwister(123))
    timing_report()

    # --- 3. 全量 rebuild ---
    println("\n--- 3. rebuild_slater_state! (x200) ---")
    timing_reset!()
    for _ in 1:200
        rebuild_slater_state!(vwf)
    end
    timing_report()

    # --- 4. 预热 ---
    println("\n--- 4. 预热 (x$N_THERM) ---")
    timing_reset!()
    rng = MersenneTwister(456)
    n_acc = 0
    for _ in 1:N_THERM
        acc, _, _, _ = mcmc_step!(vwf, kernel, rng)
        n_acc += acc
    end
    timing_report()
    @printf("  接受: %d / %d (%.1f%%)\n", n_acc, N_THERM, 100*n_acc/N_THERM)

    # --- 5. MC 采样 ---
    println("\n--- 5. MC 采样 (x$N_STEPS) ---")
    timing_reset!()
    rng = MersenneTwister(789)
    n_acc = 0
    for _ in 1:N_STEPS
        acc, _, _, _ = mcmc_step!(vwf, kernel, rng)
        n_acc += acc
    end
    timing_report()
    @printf("  接受: %d / %d (%.1f%%)\n", n_acc, N_STEPS, 100*n_acc/N_STEPS)

    # --- 6. calc_ratio_rebuild ---
    println("\n--- 6. calc_ratio_rebuild (x200) ---")
    rng = MersenneTwister(111)
    prop, _, _ = mfVMC.Sampler.propose_move(kernel, vwf.sampler, rng)
    while prop.site1 == 0
        prop, _, _ = mfVMC.Sampler.propose_move(kernel, vwf.sampler, rng)
    end
    timing_reset!()
    for _ in 1:200
        calc_ratio_rebuild(vwf, prop)
    end
    timing_report()

    println("\n" * "="^60)
    println("  完成")
end

main()
