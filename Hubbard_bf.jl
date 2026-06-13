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
using Utils: add_term_ij_nonPH, compute_eig_and_dU_reg1

include("Hubbard.jl")

struct ColumnHubbardNonPHParams
    lx::Int
    ly::Int
    bcx::Float64
    bcy::Float64
    x_boundary::Symbol
    chi1::Float64
    chi2::Float64
    mu::Dict{Symbol,Float64}
    mz::Dict{Symbol,Float64}
end

"""
用途: 构造 column-resolved nonPH Hubbard mean-field 参数对象。

参数:
- `lx, ly::Int`: 晶格尺寸。
- `bcx, bcy::Float64`: 边界条件因子。
- `x_boundary::Symbol`: `:pbc` 或 `:obc`。
- `chi1, chi2::Float64`: 最近邻和次近邻 hopping 参数。
- `mu, mz::Dict{Symbol, Float64}`: 按列保存的 `mu_x` 和 `mz_x`。

返回:
- `ColumnHubbardNonPHParams`: column-resolved nonPH 参数。
"""
function ColumnHubbardNonPHParams(;
    lx::Int,
    ly::Int,
    bcx::Float64=1.0,
    bcy::Float64=1.0,
    x_boundary::Symbol=:pbc,
    chi1::Float64=0.0,
    chi2::Float64=0.0,
    mu::Dict{Symbol,Float64}=Dict{Symbol,Float64}(),
    mz::Dict{Symbol,Float64}=Dict{Symbol,Float64}(),
)::ColumnHubbardNonPHParams
    if x_boundary != :pbc && x_boundary != :obc
        error("Unknown x_boundary=$(x_boundary). Expected :pbc or :obc.")
    end
    return ColumnHubbardNonPHParams(lx, ly, bcx, bcy, x_boundary, chi1, chi2, mu, mz)
end

"""
用途: 构造 column-resolved nonPH number-conserving mean-field Hamiltonian。

数学公式:
- hopping: `H_ij = -chi_ij` 同时作用于 up/down electron block。
- onsite field:
  `H_{i up,i up} += (-1)^(x+y) mz_x / 2 + mu_x / 2`,
  `H_{i down,i down} += -(-1)^(x+y) mz_x / 2 + mu_x / 2`。

参数:
- `params::ColumnHubbardNonPHParams`: column-resolved nonPH mean-field 参数。

返回:
- `Hermitian{Float64, Matrix{Float64}}`: 维度为 `2N_sites x 2N_sites` 的 Hamiltonian。
"""
function build_column_hubbard_nonph_hamiltonian(
    params::ColumnHubbardNonPHParams,
)
    lx = params.lx
    ly = params.ly
    n_sites = lx * ly
    hamiltonian = zeros(Float64, 2 * n_sites, 2 * n_sites)

    for x in 1:lx
        mz_x = get(params.mz, Symbol("mz_$(x)"), 0.0)
        mu_x = get(params.mu, Symbol("mu_$(x)"), 0.0)
        for y in 1:ly
            site_i = PartonSquare.xy_to_idx(x, y, ly)
            staggered_sign = (-1)^(x + y)

            site_y = y == ly ? PartonSquare.xy_to_idx(x, 1, ly) : PartonSquare.xy_to_idx(x, y + 1, ly)
            bc_y = y == ly ? params.bcy : 1.0

            if x < lx || params.x_boundary == :pbc
                site_x = x == lx ? PartonSquare.xy_to_idx(1, y, ly) : PartonSquare.xy_to_idx(x + 1, y, ly)
                site_pp = PartonSquare.xy_to_idx(x == lx ? 1 : x + 1, y == ly ? 1 : y + 1, ly)
                bc_x = x == lx ? params.bcx : 1.0
                bc_pp = (x == lx ? params.bcx : 1.0) * (y == ly ? params.bcy : 1.0)
                add_term_ij_nonPH(hamiltonian, site_i, site_x, -params.chi1 * bc_x)
                add_term_ij_nonPH(hamiltonian, site_i, site_pp, -params.chi2 * bc_pp)
            end

            add_term_ij_nonPH(hamiltonian, site_i, site_y, -params.chi1 * bc_y)

            if x > 1 || params.x_boundary == :pbc
                site_mp = PartonSquare.xy_to_idx(x == 1 ? lx : x - 1, y == ly ? 1 : y + 1, ly)
                bc_mp = (x == 1 ? params.bcx : 1.0) * (y == ly ? params.bcy : 1.0)
                add_term_ij_nonPH(hamiltonian, site_i, site_mp, -params.chi2 * bc_mp)
            end

            up_row = 2 * (site_i - 1) + 1
            down_row = up_row + 1
            hamiltonian[up_row, up_row] += staggered_sign * mz_x / 2 + mu_x / 2
            hamiltonian[down_row, down_row] += -staggered_sign * mz_x / 2 + mu_x / 2
        end
    end

    return Hermitian(hamiltonian + hamiltonian')
end

"""
用途: 构造 column-resolved nonPH Hamiltonian 对单个参数的导数矩阵。

参数:
- `params::ColumnHubbardNonPHParams`: 当前 mean-field 参数。
- `param_name::Symbol`: 参数名, 支持 `:chi2`, `:mz_x`, `:mu_x`。

返回:
- `Matrix{Float64}`: `dH/dp` 矩阵。
"""
function build_column_hubbard_nonph_dh_dparam(
    params::ColumnHubbardNonPHParams,
    param_name::Symbol,
)::Matrix{Float64}
    name_string = String(param_name)
    derivative_mu = Dict{Symbol,Float64}()
    derivative_mz = Dict{Symbol,Float64}()
    derivative_chi2 = 0.0

    if param_name == :chi2
        derivative_chi2 = 1.0
    elseif startswith(name_string, "mz_")
        derivative_mz[param_name] = 1.0
    elseif startswith(name_string, "mu_")
        derivative_mu[param_name] = 1.0
    else
        error("Unknown column nonPH mean-field parameter: $(param_name).")
    end

    derivative_params = ColumnHubbardNonPHParams(
        lx=params.lx,
        ly=params.ly,
        bcx=params.bcx,
        bcy=params.bcy,
        x_boundary=params.x_boundary,
        chi1=0.0,
        chi2=derivative_chi2,
        mu=derivative_mu,
        mz=derivative_mz,
    )
    return Matrix(build_column_hubbard_nonph_hamiltonian(derivative_params))
end

"""
用途: 生成 column-resolved nonPH determinant 的占据轨道和参数导数。

数学公式:
- 对 number-conserving Hamiltonian `H U = U epsilon` 对角化。
- nonPH determinant 取最低 `N_e` 个单粒子轨道。

参数:
- `params::ColumnHubbardNonPHParams`: mean-field 参数。
- `param_names::Vector{Symbol}`: 需要求导的 mean-field 参数名。
- `n_occupied_orbitals::Int`: 占据轨道数, 等于真实电子数。

返回:
- `(epsilon, occupied_orbitals, d_ut_params)`。
"""
function make_column_hubbard_nonph_ansatz_and_derivs(
    params::ColumnHubbardNonPHParams;
    param_names::Vector{Symbol}=Symbol[],
    n_occupied_orbitals::Int,
)
    hamiltonian = Matrix(build_column_hubbard_nonph_hamiltonian(params))
    hamiltonian_derivatives = Dict{Symbol,Matrix{Float64}}()
    for param_name in param_names
        hamiltonian_derivatives[param_name] = build_column_hubbard_nonph_dh_dparam(params, param_name)
    end

    epsilon, full_orbitals, _, orbital_derivatives = compute_eig_and_dU_reg1(
        hamiltonian,
        hamiltonian_derivatives,
    )
    if n_occupied_orbitals < 0 || n_occupied_orbitals > size(full_orbitals, 2)
        error("n_occupied_orbitals=$(n_occupied_orbitals) is outside 0:$(size(full_orbitals, 2)).")
    end

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
用途: 生成 column-resolved nonPH mean-field 参数名和初值。

参数:
- `ansatz::AbstractString`: `AFM` 或 `Stripe`。
- `lx, lambda::Int`: x 方向长度和 stripe 周期。
- `stripe_center::AbstractString`: `site` 或 `bond`。
- `mu_uniform, stripe_mu_amp, mz_amp, chi2::Float64`: 初态参数。
- `stripe_spin_peak_x::Float64`: spin envelope 峰值位置, `NaN` 时使用 `stripe_center`。

返回:
- `NamedTuple`: 包含 `wf_param_names`, `wf_init_params`。
"""
function build_column_nonph_mean_field_parameter_setup(
    ansatz::AbstractString,
    lx::Int,
    lambda::Int,
    stripe_center::AbstractString,
    mu_uniform::Float64,
    stripe_mu_amp::Float64,
    mz_amp::Float64,
    chi2::Float64,
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
            0.0,
            0.0,
            stripe_spin_peak_x,
        )
        for x in 1:lx
            push!(wf_param_names, Symbol("mz_$(x)"))
            push!(wf_param_names, Symbol("mu_$(x)"))
            push!(wf_init_params, stripe_column_params.mz[Symbol("mz_$(x)")])
            push!(wf_init_params, stripe_column_params.mu[Symbol("mu_$(x)")])
        end
    elseif ansatz == "AFM"
        for x in 1:lx
            push!(wf_param_names, Symbol("mz_$(x)"))
            push!(wf_param_names, Symbol("mu_$(x)"))
            push!(wf_init_params, mz_amp)
            push!(wf_init_params, mu_uniform)
        end
    else
        error("Unknown ansatz type: $(ansatz).")
    end
    return (; wf_param_names=wf_param_names, wf_init_params=wf_init_params)
end

"""
用途: 根据 NN/NNN bond 生成 backflow 使用的有向 source 数据。

参数:
- `bonds1, bonds2::Vector{Tuple{Int, Int}}`: 无向 bond 的代表方向。
- `t1, t2::Float64`: 对应 hopping 振幅。

返回:
- `Tuple{Vector{Tuple{Int, Int}}, Vector{Float64}}`: 有向 source bonds 和振幅。
"""
function build_column_backflow_source_data(
    bonds1::Vector{Tuple{Int,Int}},
    bonds2::Vector{Tuple{Int,Int}},
    t1::Float64,
    t2::Float64,
)
    source_bonds = Tuple{Int,Int}[]
    source_amplitudes = Float64[]
    for (site_i, site_j) in bonds1
        push!(source_bonds, (site_i, site_j))
        push!(source_amplitudes, t1)
        push!(source_bonds, (site_j, site_i))
        push!(source_amplitudes, t1)
    end
    for (site_i, site_j) in bonds2
        push!(source_bonds, (site_i, site_j))
        push!(source_amplitudes, t2)
        push!(source_bonds, (site_j, site_i))
        push!(source_amplitudes, t2)
    end
    return source_bonds, source_amplitudes
end

"""
用途: 构造 column nonPH 使用的 Eq.(5) composite backflow。

参数:
- `source_bonds, source_amplitudes`: backflow source 数据。
- `bf_epsilon, bf_eta1, bf_eta2, bf_eta3, bf_eta4::Float64`: Eq.(5) 参数。

返回:
- `CompositeBackflowTerm`: 按 `epsilon, eta1, eta2, eta3, eta4` 排列。
"""
function build_column_composite_backflow(
    source_bonds::Vector{Tuple{Int,Int}},
    source_amplitudes::Vector{Float64},
    bf_epsilon::Float64,
    bf_eta1::Float64,
    bf_eta2::Float64,
    bf_eta3::Float64,
    bf_eta4::Float64,
)
    epsilon_terms = [
        BackflowEpsilonTerm(
            param_name=:bf_epsilon,
            epsilon_bf=bf_epsilon,
            group_names=Symbol[:hubbard],
        ),
    ]
    hubbard_group = mfVMC.Backflow.build_directed_backflow_source_group(
        :hubbard,
        source_bonds,
        source_amplitudes,
        BackflowEta1DoublonHoleTerm(
            param_name=:bf_eta1,
            eta1_bf=bf_eta1,
        ),
        BackflowEta2SpinExchangeTerm(
            param_name=:bf_eta2,
            eta2_bf=bf_eta2,
        ),
        BackflowEta3DoublonSingleTerm(
            param_name=:bf_eta3,
            eta3_bf=bf_eta3,
        ),
        BackflowEta4SingleHoleTerm(
            param_name=:bf_eta4,
            eta4_bf=bf_eta4,
        ),
    )
    return CompositeBackflowTerm(epsilon_terms, [hubbard_group])
end

"""
用途: 根据开关构造 column nonPH backflow 对象。

参数:
- `enable_backflow::Bool`: 是否启用 backflow。
- 其余参数同 `build_column_composite_backflow`。

返回:
- `AbstractBackflowTerm`: 启用时为 composite backflow, 禁用时为 `NoBackflowTerm()`。
"""
function build_column_optional_backflow(
    enable_backflow::Bool,
    source_bonds::Vector{Tuple{Int,Int}},
    source_amplitudes::Vector{Float64},
    bf_epsilon::Float64,
    bf_eta1::Float64,
    bf_eta2::Float64,
    bf_eta3::Float64,
    bf_eta4::Float64,
)
    if !enable_backflow
        return NoBackflowTerm()
    end
    return build_column_composite_backflow(
        source_bonds,
        source_amplitudes,
        bf_epsilon,
        bf_eta1,
        bf_eta2,
        bf_eta3,
        bf_eta4,
    )
end

"""
用途: 解析布尔命令行字符串。

参数:
- `raw_value::AbstractString`: 支持 `true/false`, `1/0`, `yes/no`, `on/off`。
- `option_name::AbstractString`: 选项名, 用于错误信息。

返回:
- `Bool`: 解析结果。
"""
function parse_column_bool_flag(raw_value::AbstractString, option_name::AbstractString)::Bool
    normalized_value = lowercase(strip(raw_value))
    if normalized_value in ("true", "t", "1", "yes", "y", "on")
        return true
    elseif normalized_value in ("false", "f", "0", "no", "n", "off")
        return false
    end
    error("Invalid value for $(option_name): $(raw_value).")
end

#=
"""
用途: 更新 column nonPH determinant 波函数和参数导数。

参数:
- `vwf`: determinant 波函数。
- `param_names, params`: 完整参数名和值, 顺序为 mean-field, projector, backflow。
- `lx, ly, bcx, bcy, x_boundary`: 晶格和边界参数。
- `n_occupied_orbitals::Int`: nonPH 占据轨道数。
- `nparams_proj, nparams_backflow::Int`: projector/backflow 参数数量。

返回:
- `nothing`。
"""
const COLUMN_NONPH_ANSATZ_DOC_PLACEHOLDER = nothing
=#

"""
用途: 将 `measure` 任务的结果写入 Hubbard_bf.jl 使用的输出文件。

参数:
- `output_dir::AbstractString`: 输出目录, 函数内部会确保目录存在。
- `results`: `run_simulation` 返回的结果, 需要包含 `:means`, 可选包含 `:histories`。

返回:
- `nothing`。

公式:
- 对 `histories` 调用 `blocking_binning`, 得到每个观测量的平均值 `Mean`, 标准误差 `SE`,
  有效样本数 `N_eff`, 以及 integrated autocorrelation time `Tau_int`。
"""
function write_column_measure_outputs(output_dir::AbstractString, results)::Nothing
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
用途: 返回 `Hubbard_bf.jl` column nonPH 晶格中所有 site index 和晶胞坐标。

参数:
- `lx, ly::Int`: 晶格尺寸。

返回:
- `Vector{Tuple{Int, Int, Int}}`: 每个元素为 `(site, x, y)`, 其中 `site` 是 sampler site index,
  `(x, y)` 是晶格坐标。
"""
function build_column_nonph_site_coordinates(lx::Int, ly::Int)::Vector{Tuple{Int,Int,Int}}
    site_coordinates = Tuple{Int,Int,Int}[]
    for x in 1:lx, y in 1:ly
        site = hubbard_column_site_index(x, y, lx, ly)
        push!(site_coordinates, (site, x, y))
    end
    return site_coordinates
end

"""
用途: 计算 `Hubbard_bf.jl` column nonPH 的 longitudinal spin structure factor `Szz(q)`。

数学公式:
- `Szz(q) = (1 / N) * sum_{i,j} cos(qx * (x_i - x_j) + qy * (y_i - y_j))
   * Sz_i * Sz_j`。
- `qx = 2π * nx / lx`, `qy = 2π * ny / ly`。

参数:
- `vwf`: determinant 波函数, 需要提供 `vwf.sampler.state`。
- `site_coordinates::Vector{Tuple{Int, Int, Int}}`: site 和坐标列表。
- `lx, ly::Int`: 晶格尺寸。
- `nx, ny::Int`: momentum index, 对应 `2π * n / L`。

返回:
- `Float64`: 当前 Monte Carlo 构型上的 `Szz(q)` estimator。
"""
function measure_column_nonph_szz_structure_factor(
    vwf,
    site_coordinates::Vector{Tuple{Int,Int,Int}},
    lx::Int,
    ly::Int,
    nx::Int,
    ny::Int,
)::Float64
    n_sites = length(site_coordinates)
    n_sites > 0 || error("Column nonPH site list must not be empty.")
    qx = 2.0 * pi * nx / lx
    qy = 2.0 * pi * ny / ly
    state = vwf.sampler.state

    szz_value = 0.0
    for (site_i, x_i, y_i) in site_coordinates
        sz_i = get_Sz(state[site_i])
        for (site_j, x_j, y_j) in site_coordinates
            sz_j = get_Sz(state[site_j])
            phase = qx * (x_i - x_j) + qy * (y_i - y_j)
            szz_value += cos(phase) * sz_i * sz_j
        end
    end
    return szz_value / n_sites
end

"""
用途: 向 `Hubbard_bf.jl` measure observable 字典加入 `Szz(q)`。

参数:
- `observables::Dict{Symbol, Function}`: 待写入的观测量字典。
- `lx, ly::Int`: 晶格尺寸。

返回:
- `nothing`。
"""
function add_column_nonph_szz_structure_factor_observables!(
    observables::Dict{Symbol,Function},
    lx::Int,
    ly::Int,
)::Nothing
    site_coordinates = build_column_nonph_site_coordinates(lx, ly)
    for nx in 0:(lx - 1), ny in 0:(ly - 1)
        nx_local = nx
        ny_local = ny
        observable_key = Symbol("Szzq_$(nx_local)_$(ny_local)")
        observables[observable_key] = (model, vwf) -> measure_column_nonph_szz_structure_factor(
            vwf,
            site_coordinates,
            lx,
            ly,
            nx_local,
            ny_local,
        )
    end
    return nothing
end

"""
用途: 构造 `Hubbard_bf.jl` column nonPH measure 使用的 observables。

参数:
- `lx, ly::Int`: 晶格尺寸。

返回:
- `Dict{Symbol, Function}`: 包含 `:E`, 每个 site 的 `n/Sz`, 以及所有 `Szzq_nx_ny`。
"""
function defination_column_nonph_observabels(lx::Int, ly::Int)::Dict{Symbol,Function}
    observables = Dict{Symbol,Function}()
    observables[:E] = local_energy
    add_column_nonph_szz_structure_factor_observables!(observables, lx, ly)
    for x in 1:lx, y in 1:ly
        site = hubbard_column_site_index(x, y, lx, ly)
        spin_key = Symbol("Sz_$(x)_$(y)")
        density_key = Symbol("n_$(x)_$(y)")
        observables[spin_key] = (model, vwf) -> get_Sz(vwf.sampler.state[site])
        observables[density_key] = (model, vwf) -> begin
            site_state = vwf.sampler.state[site]
            n_up = (site_state & UP) != 0 ? 1.0 : 0.0
            n_down = (site_state & DN) != 0 ? 1.0 : 0.0
            return n_up + n_down
        end
    end
    return observables
end

"""
用途: 更新 column nonPH determinant 波函数和参数导数。

参数:
- `vwf`: determinant 波函数。
- `param_names, params`: 完整参数名和值, 顺序为 mean-field, projector, backflow。
- `lx, ly, bcx, bcy, x_boundary`: 晶格和边界参数。
- `n_occupied_orbitals::Int`: nonPH 占据轨道数。
- `nparams_proj, nparams_backflow::Int`: projector/backflow 参数数量。

返回:
- `nothing`。
"""
function update_column_nonph_ansatz!(
    vwf,
    param_names::Vector{Symbol},
    params::Vector{Float64},
    lx::Int,
    ly::Int,
    bcx::Float64,
    bcy::Float64,
    x_boundary::Symbol,
    n_occupied_orbitals::Int;
    nparams_proj::Int=0,
    nparams_backflow::Int=0,
)::Nothing
    total_param_count = length(param_names)
    nparams_wf = total_param_count - nparams_proj - nparams_backflow
    wf_param_names = param_names[1:nparams_wf]
    wf_param_values = params[1:nparams_wf]
    projector_param_names = param_names[(nparams_wf+1):(nparams_wf+nparams_proj)]
    projector_param_values = params[(nparams_wf+1):(nparams_wf+nparams_proj)]
    backflow_param_names = param_names[(nparams_wf+nparams_proj+1):end]
    backflow_param_values = params[(nparams_wf+nparams_proj+1):end]

    param_map = Dict{Symbol,Float64}(zip(wf_param_names, wf_param_values))
    mu = Dict{Symbol,Float64}()
    mz = Dict{Symbol,Float64}()
    for (param_name, param_value) in param_map
        name_string = String(param_name)
        if startswith(name_string, "mu_")
            mu[param_name] = param_value
        elseif startswith(name_string, "mz_")
            mz[param_name] = param_value
        elseif param_name == :chi2
            continue
        else
            error("Unknown column nonPH mean-field parameter: $(param_name).")
        end
    end

    nonph_params = ColumnHubbardNonPHParams(
        lx=lx,
        ly=ly,
        bcx=bcx,
        bcy=bcy,
        x_boundary=x_boundary,
        chi1=1.0,
        chi2=get(param_map, :chi2, 0.0),
        mu=mu,
        mz=mz,
    )
    _, gs_u, d_ut_params = make_column_hubbard_nonph_ansatz_and_derivs(
        nonph_params;
        param_names=wf_param_names,
        n_occupied_orbitals=n_occupied_orbitals,
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
用途: 解析 `Hubbard_bf.jl` 命令行参数。

参数:
- 无。读取 `ARGS`。

返回:
- `Dict{String, Any}`: ArgParse 结果。
"""
function parse_column_bf_commandline()
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
        arg_type = Int
        default = 4
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
    end
    return parse_args(settings)
end

"""
用途: 运行 column-resolved nonPH + backflow Hubbard VMC/SR 主流程。

参数:
- 无。所有参数来自命令行。

返回:
- `nothing`。
"""
function main_column_nonph_backflow()::Nothing
    args = parse_column_bf_commandline()
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

    mean_field_setup = build_column_nonph_mean_field_parameter_setup(
        args["ansatz"],
        lx,
        args["lambda"],
        args["stripe_center"],
        args["mu"],
        args["stripe_mu_amp"],
        args["mz"],
        args["chi2"],
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
    source_bonds, source_amplitudes = build_column_backflow_source_data(bonds1, bonds2, t1, t2)
    backflow = build_column_optional_backflow(
        parse_column_bool_flag(args["enable_backflow"], "--enable_backflow"),
        source_bonds,
        source_amplitudes,
        args["bf_epsilon"],
        args["bf_eta1"],
        args["bf_eta2"],
        args["bf_eta3"],
        args["bf_eta4"],
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
        init_params = build_init_params_from_json(args["init_params_json"], param_names)
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
        error("Invalid nonPH particle numbers: nup=$(nup), ndn=$(ndn), N_sites=$(n_sites).")
    end

    sampler = config_Hubbard(n_sites, nup, ndn; ifPH=false)
    init_config_Hubbard!(sampler)
    vwf = vwf_det(zeros(Float64, 2 * n_sites, nelec), sampler; backflow=backflow)
    set_projector!(vwf, projector)
    kernel = HubbardKernel(conserve_sz=true)

    if is_root
        println("Initial parameters: $(init_params)")
        println("column nonPH particle numbers: N_up=$(nup), N_down=$(ndn), N_e=$(nelec)")
    end

    update_column_nonph_ansatz!(
        vwf,
        param_names,
        init_params,
        lx,
        ly,
        bcx,
        bcy,
        x_boundary,
        nelec;
        nparams_proj=nparams_proj,
        nparams_backflow=nparams_backflow,
    )

    folder = "logs"
    mkpath(folder)
    if job == "SR"
        sr_params = SRParams(vmc_params=meas_params, n_steps=args["nSR"], lr=lr)
        exp_lr_func = build_exponential_lr_func(lr, lr_end, args["nSR"])
        update_vwf_func! = (vwf, params) -> update_column_nonph_ansatz!(
            vwf,
            param_names,
            params,
            lx,
            ly,
            bcx,
            bcy,
            x_boundary,
            nelec;
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
            defination_column_nonph_observabels(lx, ly),
            meas_params;
            history_observables=[:E],
        )
        if is_root && results !== nothing
            write_column_measure_outputs(folder, results)
        end
    else
        error("Unknown job: $(job)")
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_column_nonph_backflow()
end
