using Random
using Printf
using DelimitedFiles
using LinearAlgebra
using Statistics
using ArgParse
using JSON

push!(LOAD_PATH, joinpath(@__DIR__, "src"))
push!(LOAD_PATH, @__DIR__)

using mfVMC

# Reuse the PH script's parameter helpers and observable definitions.
include("Hubbard_restricted.jl")
using .PartonSquare

"""
用途: 解析 restricted Hubbard Pfaffian 入口脚本的命令行参数。

参数:
- 无, 直接读取 `ARGS`。

返回:
- `Dict{String, Any}`: ArgParse 解析后的参数字典。
"""
function parse_pfa_commandline()
    settings = ArgParseSettings()

    @add_arg_table settings begin
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
        help = "Boundary condition phase in X"
        arg_type = Float64
        default = 1.001
        "--bcy"
        help = "Boundary condition phase in Y"
        arg_type = Float64
        default = 0.999
        "--etax"
        help = "x-bond pairing mean-field parameter"
        arg_type = Float64
        default = 0.01
        "--etay"
        help = "y-bond pairing mean-field parameter"
        arg_type = Float64
        default = 0.01
        "--chi2"
        help = "Next-nearest neighbor hopping in the mean-field ansatz"
        arg_type = Float64
        default = -0.2
        "--Delta_AF"
        help = "AFM order parameter"
        arg_type = Float64
        default = 3.0
        "--Delta_c"
        help = "Charge stripe order parameter"
        arg_type = Float64
        default = 3.0
        "--Delta_s"
        help = "Spin stripe order parameter"
        arg_type = Float64
        default = 3.0
        "--mu"
        help = "Chemical potential in the mean-field ansatz"
        arg_type = Float64
        default = -3.0
        "--target_sz"
        help = "Target total Sz"
        arg_type = Int
        default = 0
        "--nMC"
        help = "Number of Monte Carlo samples"
        arg_type = Int
        default = 10000
        "--wMC"
        help = "Number of warmup sweeps"
        arg_type = Int
        default = 100
        "--rMC"
        help = "Number of accepted moves before rebuilding inverse"
        arg_type = Int
        default = 100
        "--dMC"
        help = "Number of decorrelation sweeps"
        arg_type = Int
        default = 1
        "--seed"
        help = "Random seed"
        arg_type = Int
        default = 5423
        "--nSR"
        help = "Total SR steps"
        arg_type = Int
        default = 50
        "--lr"
        help = "SR learning rate"
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
        "--fixed_params"
        help = "Comma-separated fixed parameter assignments, e.g. 'mu=0.1'"
        arg_type = String
        default = ""
        "--active_params"
        help = "Comma-separated parameter names optimized by SR. Empty means all non-fixed parameters."
        arg_type = String
        default = ""
        "--job"
        help = "Job to be done. Can be SR or measure"
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
        help = "Assumed stripe wavelength"
        arg_type = Int
        default = 4
        "--stripe_center"
        help = "Stripe center type, can be 'site' or 'bond'"
        arg_type = String
        default = "site"
        "--g"
        help = "Ignored in the current bare Pfaffian version; kept for command compatibility"
        arg_type = Float64
        default = 1.0
    end

    return parse_args(settings)
end

"""
用途: 根据 ansatz 类型构造 Pfaffian mean-field 参数名, 初值, stripe 波矢与中心。

参数:
- `args::Dict{String, Any}`: 命令行参数字典。

返回:
- `Tuple{Vector{Symbol}, Vector{Float64}, Float64, Float64}`:
  mean-field 参数名, 初始值, stripe 波矢 `Q`, stripe 中心 `x0`。
"""
function build_pfa_mean_field_parameter_setup(args)
    ansatz = args["ansatz"]
    stripe_lambda = args["lambda"]
    stripe_center = args["stripe_center"]

    if ansatz == "AFM"
        wf_param_names = [:chi2, :etax, :etay, :Delta_AF, :mu]
        wf_init_params = [args["chi2"], args["etax"], args["etay"], args["Delta_AF"], args["mu"]]
        return wf_param_names, wf_init_params, 0.0, 0.0
    elseif ansatz == "Stripe"
        wf_param_names = [:chi2, :etax, :etay, :Delta_c, :Delta_s, :mu]
        wf_init_params = [args["chi2"], args["etax"], args["etay"], args["Delta_c"], args["Delta_s"], args["mu"]]
        wave_vector = 2π / stripe_lambda
        if stripe_center == "site"
            return wf_param_names, wf_init_params, wave_vector, 0.0
        elseif stripe_center == "bond"
            return wf_param_names, wf_init_params, wave_vector, 0.5
        end
        error("Unknown stripe_center type: $(stripe_center)")
    end

    error("Unknown ansatz type: $(ansatz)")
end

"""
用途: 用当前参数更新 restricted Hubbard Pfaffian trial wavefunction。

参数:
- `vwf::vwf_pfa`: Pfaffian 波函数对象。
- `param_names::Vector{Symbol}`: mean-field 参数名。
- `params::Vector{Float64}`: 与 `param_names` 对应的参数值。
- `lx, ly, bcx, bcy`: 系统尺寸和边界条件。
- `Q::Float64`: stripe 波矢。
- `x0::Float64`: stripe 中心。
- `active_wf_param_names`: 需要求导的 active mean-field 参数名; `nothing` 表示全部。

返回:
- `nothing`。
"""
function update_restricted_pfa_ansatz!(
    vwf::vwf_pfa,
    param_names::Vector{Symbol},
    params::Vector{Float64},
    lx::Int,
    ly::Int,
    bcx::Float64,
    bcy::Float64;
    Q::Float64=0.0,
    x0::Float64=0.0,
    active_wf_param_names::Union{Nothing,Vector{Symbol}}=nothing,
)
    param_map = Dict{Symbol,Float64}(zip(param_names, params))
    derivative_param_names = active_wf_param_names === nothing ? param_names : active_wf_param_names
    wf_param_name_set = Set(param_names)
    for derivative_param_name in derivative_param_names
        if !(derivative_param_name in wf_param_name_set)
            error("Active mean-field parameter $(derivative_param_name) is not in full mean-field parameter list.")
        end
    end

    restricted_params = PartonSquare.RestrictedHubbardParams(
        Lx=lx,
        Ly=ly,
        bcx=bcx,
        bcy=bcy,
        chi1=1.0,
        chi2=get(param_map, :chi2, 0.0),
        etax=get(param_map, :etax, 0.0),
        etay=get(param_map, :etay, 0.0),
        mu=get(param_map, :mu, 0.0),
        Delta_AF=get(param_map, :Delta_AF, 0.0),
        Delta_c=get(param_map, :Delta_c, 0.0),
        Delta_s=get(param_map, :Delta_s, 0.0),
        Q=Q,
        x0=x0,
    )

    _, pairing_matrix, derivative_dict = PartonSquare.make_ansatz_and_derivs_pfa(
        restricted_params;
        param_names=derivative_param_names,
    )
    copyto!(vwf.gs_F, pairing_matrix)
    update_vwf_params!(vwf, derivative_dict)
    return nothing
end

"""
用途: 构造 Hubbard 模型的一阶和二阶 hopping bond 列表。

参数:
- `lx, ly::Int`: 系统尺寸。

返回:
- `Tuple{Vector{Tuple{Int, Int}}, Vector{Tuple{Int, Int}}}`: 最近邻和次近邻 bond。
"""
function build_hubbard_bond_lists(lx::Int, ly::Int)
    bonds1 = Tuple{Int,Int}[]
    bonds2 = Tuple{Int,Int}[]
    site_index(x, y) = mod(x - 1, lx) * ly + mod(y - 1, ly) + 1
    for y in 1:ly, x in 1:lx
        site = site_index(x, y)
        push!(bonds1, (site, site_index(x + 1, y)))
        push!(bonds1, (site, site_index(x, y + 1)))
        push!(bonds2, (site, site_index(x + 1, y + 1)))
        push!(bonds2, (site, site_index(x - 1, y + 1)))
    end
    return bonds1, bonds2
end

"""
用途: 从 bond 列表构造 Hubbard `GeneralModel`。

参数:
- `n_sites::Int`: 总格点数。
- `bonds1, bonds2`: 最近邻和次近邻 bond。
- `t1, t2, interaction_u::Float64`: hopping 和 onsite interaction。

返回:
- `GeneralModel`: Hubbard Hamiltonian。
"""
function build_hubbard_general_model(
    n_sites::Int,
    bonds1::Vector{Tuple{Int,Int}},
    bonds2::Vector{Tuple{Int,Int}},
    t1::Float64,
    t2::Float64,
    interaction_u::Float64,
)
    terms = OperatorTerm[]
    for (site_i, site_j) in bonds1
        push!(terms, OperatorTerm([:cdag_up, :c_up], [site_i, site_j], -t1))
        push!(terms, OperatorTerm([:cdag_up, :c_up], [site_j, site_i], -t1))
        push!(terms, OperatorTerm([:cdag_dn, :c_dn], [site_i, site_j], -t1))
        push!(terms, OperatorTerm([:cdag_dn, :c_dn], [site_j, site_i], -t1))
    end
    for (site_i, site_j) in bonds2
        push!(terms, OperatorTerm([:cdag_up, :c_up], [site_i, site_j], -t2))
        push!(terms, OperatorTerm([:cdag_up, :c_up], [site_j, site_i], -t2))
        push!(terms, OperatorTerm([:cdag_dn, :c_dn], [site_i, site_j], -t2))
        push!(terms, OperatorTerm([:cdag_dn, :c_dn], [site_j, site_i], -t2))
    end
    for site in 1:n_sites
        push!(terms, OperatorTerm([:n_up, :n_dn], [site, site], interaction_u))
    end
    return GeneralModel(n_sites, terms)
end

"""
用途: 运行 bare Pfaffian restricted Hubbard VMC/SR 主流程。

参数:
- 无, 直接读取命令行参数。

返回:
- `nothing`。
"""
function main()
    args = parse_pfa_commandline()
    session = init_mpi_session()
    rank = session.rank
    is_root = (rank == session.root)

    lx = args["Lx"]
    ly = args["Ly"]
    bcx = args["bcx"]
    bcy = args["bcy"]
    target_sz = args["target_sz"]
    doping = args["doping"]
    n_sites = lx * ly
    n_steps = args["nSR"]
    lr = args["lr"]
    lr_end = args["lr_end"]
    if isnan(lr_end)
        lr_end = lr
    end

    wf_param_names, wf_init_params, wave_vector, stripe_center = build_pfa_mean_field_parameter_setup(args)
    init_params = copy(wf_init_params)
    param_names = copy(wf_param_names)

    if !isempty(args["init_params_json"])
        init_params = build_init_params_from_json_with_defaults(args["init_params_json"], param_names, init_params)
        if is_root
            println("Loaded initial parameters from json: $(args["init_params_json"])")
        end
    end

    fixed_param_values = parse_fixed_param_string(args["fixed_params"])
    requested_active_param_names = parse_param_name_list(args["active_params"])
    validate_fixed_mean_field_params!(wf_param_names, fixed_param_values)
    if !isempty(fixed_param_values)
        init_params = apply_fixed_params_to_values(param_names, init_params, fixed_param_values)
    end
    active_param_indices = build_active_param_indices(param_names, fixed_param_values, requested_active_param_names)
    if args["job"] == "SR" && isempty(active_param_indices)
        error("At least one parameter must remain active for SR optimization.")
    end
    uses_param_subset = length(active_param_indices) != length(param_names) || !isempty(requested_active_param_names)
    sr_param_names = uses_param_subset ? param_names[active_param_indices] : param_names
    sr_init_params = uses_param_subset ? init_params[active_param_indices] : init_params
    active_wf_param_names = uses_param_subset ? sr_param_names : nothing

    bonds1, bonds2 = build_hubbard_bond_lists(lx, ly)
    hamiltonian = build_hubbard_general_model(n_sites, bonds1, bonds2, args["t1"], args["t2"], args["U"])

    nelec = Int(n_sites * (1 + doping))
    @assert (target_sz + nelec) % 2 == 0 "Wrong parity!"
    @assert nelec % 2 == 0 "Pfaffian wavefunction requires an even total particle number."
    nup = (nelec + target_sz) ÷ 2
    ndn = nelec - nup
    sampler = config_Hubbard(n_sites, nup, ndn; ifPH=false)
    init_config_Hubbard!(sampler)

    vwf = vwf_pfa(zeros(Float64, 2 * n_sites, 2 * n_sites), sampler)
    kernel = HubbardKernel(conserve_sz=true)

    if is_root
        println("Initial parameters: $init_params")
        println("Bare Pfaffian mode: projector and backflow are disabled.")
        println("nonPH particle numbers: N_up=$(nup), N_down=$(ndn), N_e=$(nelec)")
        if !isempty(fixed_param_values)
            fixed_param_messages = [
                "$(String(param_name))=$(fixed_param_values[param_name])"
                for param_name in sort(collect(keys(fixed_param_values)); by=String)
            ]
            println("Fixed parameters: $(join(fixed_param_messages, ", "))")
        end
        if uses_param_subset
            println("Active parameters: $(join(String.(sr_param_names), ", "))")
        end
    end

    update_restricted_pfa_ansatz!(
        vwf,
        param_names,
        init_params,
        lx,
        ly,
        bcx,
        bcy;
        Q=wave_vector,
        x0=stripe_center,
    )

    measure_params = VMCParams(
        total_samples=args["nMC"],
        warmup_steps=args["wMC"],
        rebuild_every=args["rMC"],
        decorr_steps=args["dMC"],
        seed=args["seed"] + rank,
    )
    folder = "logs"
    mkpath(folder)

    if args["job"] == "SR"
        sr_params = SRParams(vmc_params=measure_params, n_steps=n_steps, lr=lr)
        exp_lr_func = build_exponential_lr_func(lr, lr_end, n_steps)
        update_vwf_func! = (vwf, params) -> begin
            full_params = uses_param_subset ?
                          merge_active_params_into_full(init_params, active_param_indices, params) :
                          params
            update_restricted_pfa_ansatz!(
                vwf,
                param_names,
                full_params,
                lx,
                ly,
                bcx,
                bcy;
                Q=wave_vector,
                x0=stripe_center,
                active_wf_param_names=active_wf_param_names,
            )
        end

        run_sr_optimization(
            hamiltonian,
            vwf,
            kernel,
            sr_init_params,
            update_vwf_func!,
            sr_params;
            log_file=joinpath(folder, "sr_history.txt"),
            param_names=sr_param_names,
            lr_func=exp_lr_func,
        )
        if is_root
            extract_min_energy(joinpath(folder, "sr_history.txt"))
            append_inactive_params_to_json!(
                joinpath(folder, "min_params.json"),
                param_names,
                init_params,
                active_param_indices,
            )
        end
    elseif args["job"] == "measure"
        observables = defination_observabels(lx, ly)
        results = run_simulation(
            hamiltonian,
            vwf,
            kernel,
            observables,
            measure_params;
            history_observables=[:E],
        )
        if is_root && results !== nothing
            means = results[:means]
            mean_dict = Dict{Symbol,Any}()
            for (key, value) in means
                mean_dict[key] = value isa Number ? real(value) : value
            end

            histories = results[:histories]
            if !isempty(histories)
                mean_hist, se_dict, n_eff_dict, tau_int_dict, _ = blocking_binning(histories)
                open(joinpath(folder, "block_binning.txt"), "w") do io
                    println(io, "# Observable\tMean\tSE\tN_eff\tTau_int")
                    for name in sort(collect(keys(mean_hist)))
                        @printf(io, "%s\t%.10f\t%.10f\t%.6f\t%.6f\n",
                            String(name), mean_hist[name], se_dict[name], n_eff_dict[name], tau_int_dict[name])
                    end
                end
            end

            mean_dict_str = Dict{String,Any}()
            for (key, value) in mean_dict
                mean_dict_str[String(key)] = value
            end
            open(joinpath(folder, "block_binning_mean.json"), "w") do io
                JSON.print(io, mean_dict_str)
            end
        end
    else
        error("Unknown job type: $(args["job"])")
    end

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
