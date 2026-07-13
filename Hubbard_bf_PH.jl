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

if !isdefined(@__MODULE__, :build_hubbard_column_bonds)
    include("Hubbard.jl")
end

"""
用途: 计算 column Hubbard PH determinant 的占据轨道数。

数学公式:
- PH basis 使用 `(up electron, down hole)` 两个 row block。
- determinant 列数固定为 `N_sites + target_sz`, 其中 `N_sites = lx * ly`。

参数:
- `lx, ly::Int`: 晶格尺寸。
- `target_sz::Int`: 目标 total Sz。

返回:
- `Int`: PH determinant 占据轨道数。
"""
function compute_column_ph_determinant_orbital_count(
    lx::Int,
    ly::Int,
    target_sz::Int,
)::Int
    n_sites = lx * ly
    n_occupied_orbitals = n_sites + target_sz
    if n_occupied_orbitals < 0 || n_occupied_orbitals > 2 * n_sites
        error("Invalid PH determinant column count N_sites + target_sz = $(n_occupied_orbitals).")
    end
    return n_occupied_orbitals
end

"""
用途: 生成 Hubbard_bf_PH.jl 的 column-resolved PH mean-field 参数名和初值。

参数:
- `ansatz::AbstractString`: `AFM` 或 `Stripe`。
- `lx::Int`: x 方向长度。
- `lambda::Real`: stripe 周期, 支持非整数周期。
- `stripe_center::AbstractString`: `site` 或 `bond`。
- `mu_uniform, stripe_mu_amp, mz_amp::Float64`: chemical potential 和 staggered field 初值。
- `chi2, etax, etay::Float64`: 次近邻 hopping 以及 x/y bond pairing 初值。
- `stripe_spin_peak_x::Float64`: spin envelope 峰值位置, `NaN` 时使用 `stripe_center`。

返回:
- `NamedTuple`: 包含 `wf_param_names` 和 `wf_init_params`。

说明:
- pairing 参数总是按列展开为 `etax_x` 和 `etay_x`。
- Stripe 初值使用 `build_stripe_initial_column_params` 中的 modulation。
"""
function build_column_ph_mean_field_parameter_setup(
    ansatz::AbstractString,
    lx::Int,
    lambda::Real,
    stripe_center::AbstractString,
    mu_uniform::Float64,
    stripe_mu_amp::Float64,
    mz_amp::Float64,
    chi2::Float64,
    etax::Float64,
    etay::Float64,
    stripe_spin_peak_x::Float64,
)
    wf_param_names = Symbol[:chi2]
    wf_init_params = Float64[chi2]

    if ansatz == "Stripe"
        stripe_column_params = build_stripe_initial_column_params(
            lx,
            lambda,
            stripe_center,
            mu_uniform,
            stripe_mu_amp,
            mz_amp,
            etax,
            etay,
            stripe_spin_peak_x,
        )
        for x in 1:lx
            push!(wf_param_names, Symbol("etax_$(x)"))
            push!(wf_param_names, Symbol("etay_$(x)"))
            push!(wf_param_names, Symbol("mz_$(x)"))
            push!(wf_param_names, Symbol("mu_$(x)"))
            push!(wf_init_params, stripe_column_params.etax[Symbol("etax_$(x)")])
            push!(wf_init_params, stripe_column_params.etay[Symbol("etay_$(x)")])
            push!(wf_init_params, stripe_column_params.mz[Symbol("mz_$(x)")])
            push!(wf_init_params, stripe_column_params.mu[Symbol("mu_$(x)")])
        end
    elseif ansatz == "AFM"
        for x in 1:lx
            push!(wf_param_names, Symbol("etax_$(x)"))
            push!(wf_param_names, Symbol("etay_$(x)"))
            push!(wf_param_names, Symbol("mz_$(x)"))
            push!(wf_param_names, Symbol("mu_$(x)"))
            push!(wf_init_params, etax)
            push!(wf_init_params, etay)
            push!(wf_init_params, mz_amp)
            push!(wf_init_params, mu_uniform)
        end
    else
        error("Unknown ansatz type: $(ansatz).")
    end
    return (; wf_param_names=wf_param_names, wf_init_params=wf_init_params)
end

"""
用途: 解析 PH split backflow 的单个初值。

参数:
- `args`: 命令行参数字典。
- `block_param_key::String`: split 参数名, 例如 `"bf_eta2_up"`。
- `fallback_param_key::String`: 共享 fallback 参数名, 例如 `"bf_eta2"`。

返回:
- `Float64`: split 参数非 `NaN` 时使用 split 值, 否则使用 fallback 值。
"""
function resolve_column_ph_split_backflow_initial_value(
    args,
    block_param_key::String,
    fallback_param_key::String,
)::Float64
    block_value = args[block_param_key]
    return isnan(block_value) ? args[fallback_param_key] : block_value
end

"""
用途: 从命令行参数中生成 PH upper/down-hole 两套 backflow 初值。

参数:
- `args`: 命令行参数字典。

返回:
- `NamedTuple`: 包含 `bf_epsilon_up`, `bf_eta1_up`, ..., `bf_eta4_dn_hole`。
"""
function build_column_ph_split_backflow_initial_values(args)
    return (
        bf_epsilon_up=resolve_column_ph_split_backflow_initial_value(args, "bf_epsilon_up", "bf_epsilon"),
        bf_eta1_up=resolve_column_ph_split_backflow_initial_value(args, "bf_eta1_up", "bf_eta1"),
        bf_eta2_up=resolve_column_ph_split_backflow_initial_value(args, "bf_eta2_up", "bf_eta2"),
        bf_eta3_up=resolve_column_ph_split_backflow_initial_value(args, "bf_eta3_up", "bf_eta3"),
        bf_eta4_up=resolve_column_ph_split_backflow_initial_value(args, "bf_eta4_up", "bf_eta4"),
        bf_epsilon_dn_hole=resolve_column_ph_split_backflow_initial_value(args, "bf_epsilon_dn_hole", "bf_epsilon"),
        bf_eta1_dn_hole=resolve_column_ph_split_backflow_initial_value(args, "bf_eta1_dn_hole", "bf_eta1"),
        bf_eta2_dn_hole=resolve_column_ph_split_backflow_initial_value(args, "bf_eta2_dn_hole", "bf_eta2"),
        bf_eta3_dn_hole=resolve_column_ph_split_backflow_initial_value(args, "bf_eta3_dn_hole", "bf_eta3"),
        bf_eta4_dn_hole=resolve_column_ph_split_backflow_initial_value(args, "bf_eta4_dn_hole", "bf_eta4"),
    )
end

"""
用途: 构造 column Hubbard PH 专用的 10 参数 split backflow。

数学公式:
- upper row `(i, up)` 使用参数 `bf_eta*_up`:
  `eta1_up D_i H_j + eta2_up n_i↑ h_i↓ n_j↓ h_j↑
   + eta3_up D_i n_j↓ h_j↑ + eta4_up n_i↑ h_i↓ H_j`。
- lower row `(i, down-hole)` 使用参数 `bf_eta*_dn_hole` 和 swapped-sites 规则:
  `eta1_dn_hole H_i D_j + eta2_dn_hole n_j↓ h_j↑ n_i↑ h_i↓
   + eta3_dn_hole D_j n_i↑ h_i↓ + eta4_dn_hole n_j↓ h_j↑ H_i`。

参数:
- `source_bonds::Vector{Tuple{Int, Int}}`: 有向 source bond 列表。
- `source_amplitudes::Vector{Float64}`: 与 source bond 对齐的 hopping 振幅。
- 后续 10 个 `bf_*::Float64`: upper 和 down-hole 两套 backflow 初值。

返回:
- `CompositeBackflowTerm`: 参数顺序为 upper 5 个后接 down-hole 5 个。
"""
function build_column_ph_split_backflow(
    source_bonds::Vector{Tuple{Int,Int}},
    source_amplitudes::Vector{Float64},
    bf_epsilon_up::Float64,
    bf_eta1_up::Float64,
    bf_eta2_up::Float64,
    bf_eta3_up::Float64,
    bf_eta4_up::Float64,
    bf_epsilon_dn_hole::Float64,
    bf_eta1_dn_hole::Float64,
    bf_eta2_dn_hole::Float64,
    bf_eta3_dn_hole::Float64,
    bf_eta4_dn_hole::Float64,
)::CompositeBackflowTerm
    upper_epsilon_terms = [
        BackflowEpsilonTerm(
            param_name=:bf_epsilon_up,
            epsilon_bf=bf_epsilon_up,
            group_names=Symbol[:hubbard],
        ),
    ]
    lower_epsilon_terms = [
        BackflowEpsilonTerm(
            param_name=:bf_epsilon_dn_hole,
            epsilon_bf=bf_epsilon_dn_hole,
            group_names=Symbol[:hubbard],
        ),
    ]
    upper_group = mfVMC.Backflow.build_directed_backflow_source_group(
        :hubbard,
        source_bonds,
        source_amplitudes,
        BackflowEta1DoublonHoleTerm(param_name=:bf_eta1_up, eta1_bf=bf_eta1_up),
        BackflowEta2SpinExchangeTerm(param_name=:bf_eta2_up, eta2_bf=bf_eta2_up),
        BackflowEta3DoublonSingleTerm(param_name=:bf_eta3_up, eta3_bf=bf_eta3_up),
        BackflowEta4SingleHoleTerm(param_name=:bf_eta4_up, eta4_bf=bf_eta4_up),
    )
    lower_group = mfVMC.Backflow.build_directed_backflow_source_group(
        :hubbard,
        source_bonds,
        source_amplitudes,
        BackflowEta1DoublonHoleTerm(param_name=:bf_eta1_dn_hole, eta1_bf=bf_eta1_dn_hole),
        BackflowEta2SpinExchangeTerm(param_name=:bf_eta2_dn_hole, eta2_bf=bf_eta2_dn_hole),
        BackflowEta3DoublonSingleTerm(param_name=:bf_eta3_dn_hole, eta3_bf=bf_eta3_dn_hole),
        BackflowEta4SingleHoleTerm(param_name=:bf_eta4_dn_hole, eta4_bf=bf_eta4_dn_hole),
    )
    return CompositeBackflowTerm(
        upper_epsilon_terms,
        [upper_group];
        particle_hole_lower_block=true,
        lower_epsilon_terms=lower_epsilon_terms,
        lower_source_groups=[lower_group],
    )
end

"""
用途: 根据开关构造 column Hubbard PH split backflow。

参数:
- `enable_backflow::Bool`: 是否启用 backflow。
- `source_bonds, source_amplitudes`: backflow source 数据。
- `backflow_values`: `build_column_ph_split_backflow_initial_values` 的返回值。

返回:
- `AbstractBackflowTerm`: 开启时为 10 参数 PH split backflow, 关闭时为 `NoBackflowTerm()`。
"""
function build_column_optional_ph_split_backflow(
    enable_backflow::Bool,
    source_bonds::Vector{Tuple{Int,Int}},
    source_amplitudes::Vector{Float64},
    backflow_values,
)::AbstractBackflowTerm
    if !enable_backflow
        return NoBackflowTerm()
    end
    return build_column_ph_split_backflow(
        source_bonds,
        source_amplitudes,
        backflow_values.bf_epsilon_up,
        backflow_values.bf_eta1_up,
        backflow_values.bf_eta2_up,
        backflow_values.bf_eta3_up,
        backflow_values.bf_eta4_up,
        backflow_values.bf_epsilon_dn_hole,
        backflow_values.bf_eta1_dn_hole,
        backflow_values.bf_eta2_dn_hole,
        backflow_values.bf_eta3_dn_hole,
        backflow_values.bf_eta4_dn_hole,
    )
end

"""
用途: 从 JSON 构造 Hubbard_bf_PH.jl 的初始参数, 缺失参数使用默认值。

参数:
- `json_path::AbstractString`: 参数 JSON 路径。
- `param_names::Vector{Symbol}`: 当前 ansatz 的参数名顺序。
- `default_params::Vector{Float64}`: 当前命令行和构造器给出的默认参数。

返回:
- `Vector{Float64}`: 按 `param_names` 顺序排列的初始参数。

说明:
- 这样可以从 no-backflow 或较旧 PH JSON 启动, 新增的 split backflow 参数使用命令行默认值。
"""
function build_column_ph_init_params_from_json_with_defaults(
    json_path::AbstractString,
    param_names::Vector{Symbol},
    default_params::Vector{Float64},
)::Vector{Float64}
    isfile(json_path) || error("JSON file not found: $(json_path)")
    length(param_names) == length(default_params) ||
        error("param_names and default_params length mismatch.")

    raw_dict = JSON.parsefile(json_path)
    init_params = Float64[]
    for (param_index, param_name) in enumerate(param_names)
        key = String(param_name)
        if haskey(raw_dict, key)
            push!(init_params, Float64(raw_dict[key]))
        else
            push!(init_params, default_params[param_index])
        end
    end
    return init_params
end

"""
用途: 将 `measure` 任务的结果写入 Hubbard_bf_PH.jl 使用的输出文件。

参数:
- `output_dir::AbstractString`: 输出目录, 函数内部会确保目录存在。
- `results`: `run_simulation` 返回的结果, 需要包含 `:means`, 可选包含 `:histories`。

返回:
- `nothing`。

公式:
- 对 `histories` 调用 `blocking_binning`, 得到每个观测量的平均值 `Mean`, 标准误差 `SE`,
  有效样本数 `N_eff`, 以及 integrated autocorrelation time `Tau_int`。
"""
function write_column_ph_measure_outputs(output_dir::AbstractString, results)::Nothing
    mkpath(output_dir)

    means = results[:means]
    mean_dict_str = Dict{String,Any}()
    for (key, value) in means
        mean_dict_str[String(key)] = value isa Number ? real(value) : value
    end
    open(joinpath(output_dir, "block_binning_mean.json"), "w") do io
        JSON.print(io, mean_dict_str)
    end

    histories = get(results, :histories, Dict{Symbol,Any}())
    if !isempty(histories)
        mean_hist, se_dict, n_eff_dict, tau_int_dict, _ = blocking_binning(histories)
        open(joinpath(output_dir, "block_binning.txt"), "w") do io
            println(io, "# Observable\tMean\tSE\tN_eff\tTau_int")
            for name in sort(collect(keys(mean_hist)))
                mean_value = mean_hist[name]
                se_value = se_dict[name]
                n_eff_value = n_eff_dict[name]
                tau_int_value = tau_int_dict[name]

                if mean_value isa Number && se_value isa Number && n_eff_value isa Number && tau_int_value isa Number
                    @printf(
                        io,
                        "%s\t%.10f\t%.10f\t%.6f\t%.6f\n",
                        String(name),
                        real(mean_value),
                        real(se_value),
                        real(n_eff_value),
                        real(tau_int_value),
                    )
                else
                    println(io, "$(String(name))\t$(mean_value)\t$(se_value)\t$(n_eff_value)\t$(tau_int_value)")
                end
            end
        end
    end
    return nothing
end

"""
用途: 构造 column Hubbard PH Hamiltonian 对单个参数的导数矩阵。

参数:
- `params::PartonSquare.HubbardParams`: 当前 PH mean-field 参数, 提供尺寸和边界。
- `param_name::Symbol`: 参数名, 支持 `:chi2`, `etax_x`, `etay_x`, `mz_x`, `mu_x`。

返回:
- `Matrix{Float64}`: `dH/dp` 矩阵。
"""
function build_column_ph_hubbard_dh_dparam(
    params::PartonSquare.HubbardParams,
    param_name::Symbol,
)::Matrix{Float64}
    name_string = String(param_name)
    derivative_kwargs = Dict{Symbol,Any}(
        :Lx => params.Lx,
        :Ly => params.Ly,
        :bcx => params.bcx,
        :bcy => params.bcy,
        :x_boundary => params.x_boundary,
    )

    if occursin(r"_\d+$", name_string)
        split_index = findfirst('_', name_string)
        param_prefix = Symbol(name_string[1:(split_index-1)])
        derivative_kwargs[param_prefix] = Dict(param_name => 1.0)
    elseif param_name == :chi2
        derivative_kwargs[:chi2] = 1.0
    else
        error("Unknown column PH mean-field parameter: $(param_name).")
    end

    derivative_params = PartonSquare.HubbardParams(; derivative_kwargs...)
    return Matrix(PartonSquare.build_ham_PH(derivative_params))
end

"""
用途: 对角化 column Hubbard PH Hamiltonian 并生成 determinant orbitals 及参数导数。

参数:
- `params::PartonSquare.HubbardParams`: PH mean-field 参数。
- `param_names::Vector{Symbol}`: 需要计算导数的 mean-field 参数名。
- `target_sz::Int`: 目标 total Sz, PH determinant 占据轨道数为 `N_sites + target_sz`。

返回:
- `Tuple`: `(epsilon, occupied_orbitals, d_ut_params)`。
  `epsilon` 是全部本征值, `occupied_orbitals` 是 `(2N_sites, N_sites+target_sz)` 矩阵,
  `d_ut_params` 是参数名到转置导数矩阵 `dU'` 的映射。
"""
function make_column_ph_hubbard_ansatz_and_derivs(
    params::PartonSquare.HubbardParams;
    param_names::Vector{Symbol}=Symbol[],
    target_sz::Int=0,
)
    hamiltonian = Matrix(PartonSquare.build_ham_PH(params))
    hamiltonian_derivatives = Dict{Symbol,Matrix{Float64}}()
    for param_name in param_names
        hamiltonian_derivatives[param_name] = build_column_ph_hubbard_dh_dparam(params, param_name)
    end

    epsilon, full_orbitals, _, orbital_derivatives = compute_eig_and_dU_reg1(
        hamiltonian,
        hamiltonian_derivatives,
    )
    n_sites = params.Lx * params.Ly
    n_occupied_orbitals = compute_column_ph_determinant_orbital_count(params.Lx, params.Ly, target_sz)

    if PartonSquare.is_root_rank()
        eig_eq_error = norm(hamiltonian * full_orbitals - full_orbitals * Diagonal(epsilon))
        left_index = max(1, n_occupied_orbitals - 4)
        right_index = min(length(epsilon), n_occupied_orbitals + 4)
        println("Eigen equation error (HU - Uepsilon): ", eig_eq_error)
        println("epsilon is ", epsilon[left_index:right_index])
    end

    occupied_orbitals = real.(full_orbitals[:, 1:n_occupied_orbitals])
    d_ut_params = Dict{Symbol,Matrix{Float64}}()
    for param_name in param_names
        d_ut_params[param_name] = permutedims(real.(orbital_derivatives[param_name][:, 1:n_occupied_orbitals]))
    end
    return epsilon, occupied_orbitals, d_ut_params
end

"""
用途: 根据参数更新 column Hubbard PH determinant 波函数和参数导数。

参数:
- `vwf`: determinant 波函数对象。
- `param_names::Vector{Symbol}`: 完整参数名列表, 顺序为 mean-field, projector, backflow。
- `params::Vector{Float64}`: 与 `param_names` 对齐的完整参数值。
- `lx, ly::Int`: 晶格尺寸。
- `bcx, bcy::Float64`: mean-field 边界条件因子。
- `x_boundary::Symbol`: `:pbc` 或 `:obc`。
- `target_sz::Int`: 目标 total Sz, PH determinant 列数为 `lx * ly + target_sz`。
- `nparams_proj, nparams_backflow::Int`: projector/backflow 参数数量。

返回:
- `nothing`。

公式:
- PH mean-field 使用 `PartonSquare.HubbardParams` 的 Nambu basis。
- determinant 占据轨道数为 `N_sites + target_sz`。
- log-derivative 中 determinant 参数对应 `dU_t[p] = (dU_occ/dp)'`。
"""
function update_column_ph_ansatz!(
    vwf,
    param_names::Vector{Symbol},
    params::Vector{Float64},
    lx::Int,
    ly::Int,
    bcx::Float64,
    bcy::Float64,
    x_boundary::Symbol,
    target_sz::Int;
    nparams_proj::Int=0,
    nparams_backflow::Int=0,
)::Nothing
    total_param_count = length(param_names)
    total_param_count == length(params) ||
        error("param_names and params length mismatch.")
    nparams_wf = total_param_count - nparams_proj - nparams_backflow
    nparams_wf >= 0 || error("Invalid parameter split.")

    wf_param_names = param_names[1:nparams_wf]
    wf_param_values = params[1:nparams_wf]
    projector_param_names = nparams_proj > 0 ? param_names[(nparams_wf+1):(nparams_wf+nparams_proj)] : Symbol[]
    projector_param_values = nparams_proj > 0 ? params[(nparams_wf+1):(nparams_wf+nparams_proj)] : Float64[]
    backflow_param_names = nparams_backflow > 0 ? param_names[(nparams_wf+nparams_proj+1):total_param_count] : Symbol[]
    backflow_param_values = nparams_backflow > 0 ? params[(nparams_wf+nparams_proj+1):total_param_count] : Float64[]

    param_map = Dict{Symbol,Float64}(zip(wf_param_names, wf_param_values))
    etax = Dict{Symbol,Float64}()
    etay = Dict{Symbol,Float64}()
    mz = Dict{Symbol,Float64}()
    mu = Dict{Symbol,Float64}()

    for (param_name, param_value) in param_map
        name_string = String(param_name)
        if startswith(name_string, "etax_")
            etax[param_name] = param_value
        elseif startswith(name_string, "etay_")
            etay[param_name] = param_value
        elseif startswith(name_string, "mz_")
            mz[param_name] = param_value
        elseif startswith(name_string, "mu_")
            mu[param_name] = param_value
        elseif param_name == :chi2
            continue
        else
            error("Unknown column PH mean-field parameter: $(param_name).")
        end
    end

    hubbard_params = PartonSquare.HubbardParams(
        Lx=lx,
        Ly=ly,
        bcx=bcx,
        bcy=bcy,
        x_boundary=x_boundary,
        chi1=1.0,
        etax=etax,
        etay=etay,
        chi2=get(param_map, :chi2, 0.0),
        mu=mu,
        mz=mz,
    )

    _, gs_u, d_ut_params = make_column_ph_hubbard_ansatz_and_derivs(
        hubbard_params;
        param_names=wf_param_names,
        target_sz=target_sz,
    )

    copyto!(vwf.base_gs_U, gs_u)
    copyto!(vwf.gs_U, gs_u)
    copyto!(vwf.gs_U_t, permutedims(gs_u))

    d_ut_matrix = zeros(Float64, size(gs_u, 2), size(gs_u, 1), length(wf_param_names))
    for (param_index, param_name) in enumerate(wf_param_names)
        d_ut_matrix[:, :, param_index] = d_ut_params[param_name]
    end
    update_vwf_params!(vwf, wf_param_names, d_ut_matrix)

    if !isempty(projector_param_names)
        update_vwf_projector_params!(vwf, projector_param_names, projector_param_values)
    end
    if !isempty(backflow_param_names)
        update_vwf_backflow_params!(vwf, backflow_param_names, backflow_param_values)
    end
    init_gswf!(vwf)
    return nothing
end

"""
用途: 解析 `Hubbard_bf_PH.jl` 命令行参数。

参数:
- 无。读取 `ARGS`。

返回:
- `Dict{String, Any}`: ArgParse 结果。
"""
function parse_column_ph_bf_commandline()
    settings = ArgParseSettings()
    @add_arg_table settings begin
        "--Lx"
        arg_type = Int
        default = 8
        "--Ly"
        arg_type = Int
        default = 3
        "--t1"
        arg_type = Float64
        default = 1.0
        "--t2"
        arg_type = Float64
        default = -0.2
        "--U"
        arg_type = Float64
        default = 8.0
        "--bcx"
        arg_type = Float64
        default = 1.0
        "--bcy"
        arg_type = Float64
        default = 1.0
        "--x_boundary"
        arg_type = String
        default = "pbc"
        "--edge_pinning"
        arg_type = Float64
        default = 0.0
        "--chi2"
        arg_type = Float64
        default = -0.2
        "--etax"
        arg_type = Float64
        default = 0.01
        "--etay"
        arg_type = Float64
        default = 0.01
        "--mz"
        arg_type = Float64
        default = 3.0
        "--mu"
        arg_type = Float64
        default = -3.0
        "--stripe_mu_amp"
        arg_type = Float64
        default = 0.0
        "--stripe_spin_peak_x"
        arg_type = Float64
        default = NaN
        "--target_sz"
        arg_type = Int
        default = 0
        "--nMC"
        arg_type = Int
        default = 10000
        "--wMC"
        arg_type = Int
        default = 100
        "--rMC"
        arg_type = Int
        default = 100
        "--dMC"
        arg_type = Int
        default = 1
        "--seed"
        arg_type = Int
        default = 5423
        "--nSR"
        arg_type = Int
        default = 50
        "--lr"
        arg_type = Float64
        default = 0.04
        "--lr_end"
        arg_type = Float64
        default = NaN
        "--init_params_json"
        arg_type = String
        default = ""
        "--job"
        arg_type = String
        default = "SR"
        "--doping"
        arg_type = Float64
        default = 0.125
        "--ansatz"
        arg_type = String
        default = "Stripe"
        "--lambda"
        arg_type = Float64
        default = 4.0
        "--stripe_center"
        arg_type = String
        default = "site"
        "--g"
        arg_type = Float64
        default = 1.0
        "--vj1"
        arg_type = Float64
        default = 0.0
        "--vj2"
        arg_type = Float64
        default = 0.0
        "--enable_backflow"
        arg_type = String
        default = "true"
        "--bf_epsilon"
        arg_type = Float64
        default = 1.0
        "--bf_eta1"
        arg_type = Float64
        default = 0.0
        "--bf_eta2"
        arg_type = Float64
        default = 0.0
        "--bf_eta3"
        arg_type = Float64
        default = 0.0
        "--bf_eta4"
        arg_type = Float64
        default = 0.0
        "--bf_epsilon_up"
        arg_type = Float64
        default = NaN
        "--bf_eta1_up"
        arg_type = Float64
        default = NaN
        "--bf_eta2_up"
        arg_type = Float64
        default = NaN
        "--bf_eta3_up"
        arg_type = Float64
        default = NaN
        "--bf_eta4_up"
        arg_type = Float64
        default = NaN
        "--bf_epsilon_dn_hole"
        arg_type = Float64
        default = NaN
        "--bf_eta1_dn_hole"
        arg_type = Float64
        default = NaN
        "--bf_eta2_dn_hole"
        arg_type = Float64
        default = NaN
        "--bf_eta3_dn_hole"
        arg_type = Float64
        default = NaN
        "--bf_eta4_dn_hole"
        arg_type = Float64
        default = NaN
    end
    return parse_args(settings)
end

"""
用途: 运行 column-resolved PH + split backflow Hubbard VMC/SR 主流程。

参数:
- 无。所有参数来自命令行。

返回:
- `nothing`。
"""
function main_column_ph_backflow()::Nothing
    args = parse_column_ph_bf_commandline()
    session = init_mpi_session()
    rank = session.rank
    is_root = rank == session.root

    lx = args["Lx"]
    ly = args["Ly"]
    bcx = args["bcx"]
    bcy = args["bcy"]
    x_boundary = normalize_x_boundary_name(args["x_boundary"])
    n_sites = lx * ly
    target_sz = args["target_sz"]
    doping = args["doping"]
    lr = args["lr"]
    lr_end = isnan(args["lr_end"]) ? lr : args["lr_end"]
    job = args["job"]
    t1 = args["t1"]
    t2 = args["t2"]
    onsite_u = args["U"]

    mean_field_setup = build_column_ph_mean_field_parameter_setup(
        args["ansatz"],
        lx,
        args["lambda"],
        args["stripe_center"],
        args["mu"],
        args["stripe_mu_amp"],
        args["mz"],
        args["chi2"],
        args["etax"],
        args["etay"],
        args["stripe_spin_peak_x"],
    )
    wf_param_names = mean_field_setup.wf_param_names
    wf_init_params = mean_field_setup.wf_init_params

    meas_params = VMCParams(
        total_samples=args["nMC"],
        warmup_steps=args["wMC"],
        rebuild_every=args["rMC"],
        decorr_steps=args["dMC"],
        seed=args["seed"] + rank,
    )

    bonds1, bonds2 = build_hubbard_column_bonds(lx, ly, x_boundary)
    source_bonds, source_amplitudes = build_hubbard_backflow_source_data(bonds1, bonds2, t1, t2)
    backflow_values = build_column_ph_split_backflow_initial_values(args)
    backflow = build_column_optional_ph_split_backflow(
        parse_hubbard_bool_flag(args["enable_backflow"], "--enable_backflow"),
        source_bonds,
        source_amplitudes,
        backflow_values,
    )
    projector = build_hubbard_truncated_projector(lx, ly, args["g"], x_boundary; vj1=args["vj1"], vj2=args["vj2"])

    proj_param_names = projector_param_names(projector)
    proj_init_params = projector_param_values(projector)
    nparams_proj = length(proj_param_names)
    backflow_param_name_list = backflow_param_names(backflow)
    backflow_init_params = backflow_param_values(backflow)
    nparams_backflow = length(backflow_param_name_list)
    init_params = vcat(wf_init_params, proj_init_params, backflow_init_params)
    param_names = vcat(wf_param_names, proj_param_names, backflow_param_name_list)

    if !isempty(args["init_params_json"])
        init_params = build_column_ph_init_params_from_json_with_defaults(args["init_params_json"], param_names, init_params)
        if is_root
            println("Loaded initial parameters from json: $(args["init_params_json"])")
        end
    end

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
    for site_i in 1:n_sites
        push!(terms, OperatorTerm([:n_up, :n_dn], [site_i, site_i], onsite_u))
    end
    if x_boundary == :obc && args["edge_pinning"] != 0.0
        for y in 1:ly
            left_site = hubbard_column_site_index(1, y, lx, ly)
            push!(terms, OperatorTerm([:Sz], [left_site], args["edge_pinning"] * (-1)^(y + 1)))
        end
    end
    ham = GeneralModel(n_sites, terms)

    electron_count_float = n_sites * (1 + doping)
    nelec = round(Int, electron_count_float)
    if !isapprox(electron_count_float, nelec; atol=1e-8, rtol=0.0)
        error("N_sites * (1 + doping) must be an integer, got $(electron_count_float).")
    end
    @assert (target_sz + nelec) % 2 == 0 "Wrong parity between target_sz and electron count."
    nup = (nelec + target_sz) ÷ 2
    ndn = nelec - nup
    if nup < 0 || ndn < 0 || nup > n_sites || ndn > n_sites
        error("Invalid PH particle numbers: nup=$(nup), ndn=$(ndn), N_sites=$(n_sites).")
    end
    n_occupied_orbitals = compute_column_ph_determinant_orbital_count(lx, ly, target_sz)

    sampler = config_Hubbard(n_sites, nup, ndn; ifPH=true)
    init_config_Hubbard!(sampler)
    vwf = vwf_det(zeros(Float64, 2 * n_sites, n_occupied_orbitals), sampler; backflow=backflow)
    set_projector!(vwf, projector)
    kernel = HubbardKernel(conserve_sz=true)

    if is_root
        println("Initial parameters: $(init_params)")
        println("column PH particle numbers: N_up=$(nup), N_down=$(ndn), N_e=$(nelec), N_occ_PH=$(n_occupied_orbitals)")
        println("Backflow enabled: $(mfVMC.Backflow.uses_backflow(backflow))")
    end

    update_column_ph_ansatz!(
        vwf,
        param_names,
        init_params,
        lx,
        ly,
        bcx,
        bcy,
        x_boundary,
        target_sz;
        nparams_proj=nparams_proj,
        nparams_backflow=nparams_backflow,
    )

    folder = "logs"
    mkpath(folder)
    if job == "SR"
        sr_params = SRParams(vmc_params=meas_params, n_steps=args["nSR"], lr=lr)
        exp_lr_func = build_exponential_lr_func(lr, lr_end, args["nSR"])
        update_vwf_func! = (vwf, params) -> update_column_ph_ansatz!(
            vwf,
            param_names,
            params,
            lx,
            ly,
            bcx,
            bcy,
            x_boundary,
            target_sz;
            nparams_proj=nparams_proj,
            nparams_backflow=nparams_backflow,
        )
        run_sr_optimization(
            ham,
            vwf,
            kernel,
            init_params,
            update_vwf_func!,
            sr_params;
            log_file=joinpath(folder, "sr_history.txt"),
            param_names=param_names,
            lr_func=exp_lr_func,
        )
        if is_root
            extract_min_energy(joinpath(folder, "sr_history.txt"))
        end
    elseif job == "measure"
        results = run_simulation(
            ham,
            vwf,
            kernel,
            defination_observabels(lx, ly),
            meas_params;
            history_observables=[:E],
        )
        if is_root && results !== nothing
            write_column_ph_measure_outputs(folder, results)
        end
    else
        error("Unknown job: $(job)")
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_column_ph_backflow()
end
