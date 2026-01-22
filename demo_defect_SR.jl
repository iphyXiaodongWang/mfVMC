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
include("FPS.jl")
using .FPS
using .PartonSquare



include("defect_sr_utils.jl")

function main()
    args = parse_commandline()

    session = init_mpi_session()
    rank = session.rank

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
    d_mc = 1
    seed = args["seed"]
    n_steps = args["nSR"]
    lr = args["lr"]
    init_params_json = args["init_params_json"]

    n_sites_full = lx * ly
    defect_positions = if defect_ansatz == "FPS"
        FPS.generate_defect_positions_fps(lx, ly, n_defect)
    else
        error("Unsupported defectansatz: $defect_ansatz")
    end
    defect_index = [xy_to_id_1based(x, y, lx, ly) for (x, y) in defect_positions]
    n_sites = n_sites_full - n_defect

    bonds_j1, bonds_j2, bonds_j3 = build_defect_bonds(lx, ly, defect_positions)
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
    ham = GeneralModel(n_sites, terms)

    @assert (target_sz + n_sites) % 2 == 0 "Wrong parity!"
    sampler = config_Heisenberg(n_sites, (n_sites + target_sz) ÷ 2; ifPH=true)
    init_config_Heisenberg!(sampler)

    vwf = vwf_det(zeros(Float64, 2 * n_sites, n_sites + target_sz), sampler)
    kernel = HeisenbergKernel(conserve_sz=true)

    param_names, init_params = build_defect_param_names_and_init_params(lx, ly, defect_positions, args)
    if !isempty(init_params_json)
        init_params = build_init_params_from_json(init_params_json, param_names)
        if rank == 0
            println("Loaded initial parameters from json: $(init_params_json)")
        end
    end

    meas_params = VMCParams(
        total_samples=n_mc,
        warmup_steps=w_mc,
        rebuild_every=r_mc,
        decorr_steps=d_mc,
        seed=seed + rank
    )
    sr_params = SRParams(vmc_params=meas_params, n_steps=n_steps, lr=lr)

    if rank == 0
        println("Initial parameters: $init_params")
    end

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
        log_file="logs/sr_defect_history.txt",
        param_names=param_names
    )
end

main()
