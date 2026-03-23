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
push!(LOAD_PATH, "./src")
push!(LOAD_PATH, ".")

using mfVMC
include("PartonSquare.jl")
include("FPS.jl")
using .FPS
using .PartonSquare



include("defect_sr_utils.jl")

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
    bcx = args["bcx"]
    bcy = args["bcy"]
    j1 = args["J1"]
    j2 = args["J2"]
    j3 = args["J3"]
    n_defect = args["Ndefect"]
    defect_ansatz = args["defectansatz"]
    target_sz = args["target_sz"]
    n_mc = args["nMC"]
    w_mc = args["wMC"]
    r_mc = args["rMC"]
    d_mc = args["dMC"]
    seed = args["seed"]
    n_steps = args["nSR"]
    lr = args["lr"]
    lr_end = args["lr_end"]
    if isnan(lr_end)
        lr_end = lr
    end
    init_params_json = args["init_params_json"]
    job = args["job"]

    n_sites_full = lx * ly
    defect_positions = if defect_ansatz == "FPS"
        FPS.generate_defect_positions_fps(lx, ly, n_defect)
    else
        error("Unsupported defectansatz: $defect_ansatz")
    end
    defect_index = [xy_to_id_1based(x, y, lx, ly) for (x, y) in defect_positions]
    n_sites = n_sites_full - n_defect

    bonds_j1, bonds_j2, bonds_j3 = build_defect_bonds_obc(lx, ly, defect_positions)
    bonds_j1 = defect_correction_bond(bonds_j1, defect_index)
    bonds_j2 = defect_correction_bond(bonds_j2, defect_index)
    bonds_j3 = defect_correction_bond(bonds_j3, defect_index)

    terms = OperatorTerm[]
    for (i, j) in bonds_j1
        push!(terms, OperatorTerm([:SS], [i, j], j1))
    end
    for (i, j) in bonds_j2
        push!(terms, OperatorTerm([:SS], [i, j], j2))
    end
    for (i, j) in bonds_j3
        push!(terms, OperatorTerm([:SS], [i, j], j3))
    end

    index_1_ly = xy_to_reduced_id_1based(1, ly, lx, ly, defect_index)
    index_lx_1 = xy_to_reduced_id_1based(lx, 1, lx, ly, defect_index)
    index_lx_ly = xy_to_reduced_id_1based(lx, ly, lx, ly, defect_index)
    push!(terms, OperatorTerm([:Sz], [index_1_ly], -0.3))
    push!(terms, OperatorTerm([:Sz], [index_lx_1], -0.3))
    push!(terms, OperatorTerm([:Sz], [index_lx_ly], 0.3))

    ham = GeneralModel(n_sites, terms)
    if target_sz != 0 && (target_sz + n_sites) % 2 != 0
        error("Wrong parity!")
    elseif target_sz == 0
        target_sz = n_sites % 2
    end
    sampler = config_Heisenberg(n_sites, (n_sites + target_sz) ÷ 2; ifPH=true)
    init_config_Heisenberg!(sampler)

    vwf = vwf_det(zeros(Float64, 2 * n_sites, n_sites + target_sz), sampler)
    kernel = HeisenbergKernel(conserve_sz=true)

    param_names, init_params = build_defect_param_names_and_init_params(lx, ly, defect_positions, args)
    if !isempty(init_params_json)
        init_params = build_init_params_from_json(init_params_json, param_names)
        if is_root
            println("Loaded initial parameters from json: $(init_params_json)")
        end
    end

    if is_root
        println("Initial parameters: $init_params")
    end

    meas_params = VMCParams(
        total_samples=n_mc,
        warmup_steps=w_mc,
        rebuild_every=r_mc,
        decorr_steps=d_mc,
        seed=seed + rank
    )
    update_defect_ansatz!(
        vwf,
        param_names,
        init_params,
        lx,
        ly,
        bcx,
        bcy,
        args["chi1"],
        defect_positions,
        defect_index,
        target_sz
    )
    folder = "logs/target_sz_$(target_sz)"
    mkpath(folder)
    if job == "SR"
        sr_params = SRParams(vmc_params=meas_params, n_steps=n_steps, lr=lr)
        exp_lr_func = build_exponential_lr_func(lr, lr_end, n_steps)

        update_vwf_func! = (vwf, params) -> update_defect_ansatz!(
            vwf,
            param_names,
            params,
            lx,
            ly,
            bcx,
            bcy,
            args["chi1"],
            defect_positions,
            defect_index,
            target_sz
        )

        run_sr_optimization(
            ham,
            vwf,
            kernel,
            init_params,
            update_vwf_func!,
            sr_params;
            log_file=joinpath(folder, "sr_defect_history.txt"),
            param_names=param_names,
            lr_func=exp_lr_func
        )
        if is_root
            min_energy = extract_min_energy(joinpath(folder, "sr_defect_history.txt"))
        end
    elseif job == "measure"
        observables = defination_observabels(lx, ly, n_sites, defect_index)
        # 默认不保留历史, 如需阻塞法(Binning)请在此列出观测量名称
        history_observables = [:E, :staggered_mz, :S_pi_pi]
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

                txt_file = joinpath(folder, "defect_block_binning.txt")
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

            json_file = joinpath(folder, "defect_block_binning_mean.json")
            mean_dict_str = Dict{String,Any}()
            for (key, value) in mean_dict
                mean_dict_str[String(key)] = value
            end
            open(json_file, "w") do io
                JSON.print(io, mean_dict_str)
            end
            save_measurement_outputs(mean_dict, lx, ly, n_sites, defect_index, folder)
        end
    end
end

main()
