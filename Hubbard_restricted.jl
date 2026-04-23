using OrderedCollections

include("Hubbard.jl")

"""
用途: 构造按列均匀的 Hubbard mean-field 参数字典。

参数:
- `parameter_prefix::Symbol`: 参数名前缀, 例如 `:mu`、`:mz`、`:etad1`、`:etas1`。
- `lx::Int`: 晶格在 `x` 方向的列数。
- `parameter_value::Float64`: 每一列共享的参数取值。

返回:
- `Dict{Symbol, Float64}`: 形如 `parameter_prefix_x` 的列参数字典。
"""
function build_uniform_column_parameter_dict(
    parameter_prefix::Symbol,
    lx::Int,
    parameter_value::Float64,
)::Dict{Symbol, Float64}
    return Dict(
        Symbol("$(parameter_prefix)_$(x)") => parameter_value
        for x in 1:lx
    )
end

"""
用途: 构造受约束 AFM ansatz 的按列参数字典。

参数:
- `lx::Int`: 晶格在 `x` 方向的列数。
- `mu_uniform::Float64`: 平均 chemical potential。
- `mz_uniform::Float64`: AFM 包络振幅。
- `etad1_uniform::Float64`: 全局 d-wave pairing 参数。
- `etas1_uniform::Float64`: 全局 s-wave pairing 参数。

返回:
- `NamedTuple`: 包含 `etad1`、`etas1`、`mz`、`mu` 四个按列参数字典。
"""
function build_restricted_afm_column_params(
    lx::Int,
    mu_uniform::Float64,
    mz_uniform::Float64,
    etad1_uniform::Float64,
    etas1_uniform::Float64,
)
    return (
        etad1=build_uniform_column_parameter_dict(:etad1, lx, etad1_uniform),
        etas1=build_uniform_column_parameter_dict(:etas1, lx, etas1_uniform),
        mz=build_uniform_column_parameter_dict(:mz, lx, mz_uniform),
        mu=build_uniform_column_parameter_dict(:mu, lx, mu_uniform),
    )
end

"""
用途: 构造受约束 Stripe ansatz 的链式求导权重。

参数:
- `lx::Int`: 晶格在 `x` 方向的列数。
- `lambda::Int`: charge stripe 周期 `λ`。
- `stripe_center::AbstractString`: stripe 中心类型, 支持 `site` 或 `bond`。

返回:
- `NamedTuple`: 包含 charge、spin 与 pairing 对各全局参数的权重数组。

公式:
- `Q = 2π / λ`
- `charge_weight(x) = cos[Q * (x - x0)]`
- `spin_weight(x) = sin[Q / 2 * (x - x0)]`
- `pairing_x_weight(x) = |cos[Q / 2 * (x + 1/2 - x0)]|`
- `pairing_y_weight(x) = |cos[Q / 2 * (x - x0)]|`
"""
function build_restricted_stripe_chain_rule_weights(
    lx::Int,
    lambda::Int,
    stripe_center::AbstractString,
)
    if lambda <= 0
        error("lambda must be positive.")
    end

    stripe_center_offset = get_stripe_center_offset(stripe_center)
    stripe_wave_vector = 2.0 * pi / lambda

    charge_weights = Float64[]
    spin_weights = Float64[]
    pairing_x_weights = Float64[]
    pairing_y_weights = Float64[]
    etad1_to_etad1_weights = Float64[]
    etad1_to_etas1_weights = Float64[]
    etas1_to_etad1_weights = Float64[]
    etas1_to_etas1_weights = Float64[]

    for x in 1:lx
        x_coordinate = Float64(x)
        charge_weight = cos(stripe_wave_vector * (x_coordinate - stripe_center_offset))
        spin_weight = sin(stripe_wave_vector / 2.0 * (x_coordinate - stripe_center_offset))
        pairing_x_weight = abs(cos(stripe_wave_vector / 2.0 * (x_coordinate + 0.5 - stripe_center_offset)))
        pairing_y_weight = abs(cos(stripe_wave_vector / 2.0 * (x_coordinate - stripe_center_offset)))

        push!(charge_weights, charge_weight)
        push!(spin_weights, spin_weight)
        push!(pairing_x_weights, pairing_x_weight)
        push!(pairing_y_weights, pairing_y_weight)
        push!(etad1_to_etad1_weights, 0.5 * (pairing_y_weight + pairing_x_weight))
        push!(etad1_to_etas1_weights, 0.5 * (pairing_y_weight - pairing_x_weight))
        push!(etas1_to_etad1_weights, 0.5 * (pairing_y_weight - pairing_x_weight))
        push!(etas1_to_etas1_weights, 0.5 * (pairing_y_weight + pairing_x_weight))
    end

    return (
        charge_weights=charge_weights,
        spin_weights=spin_weights,
        pairing_x_weights=pairing_x_weights,
        pairing_y_weights=pairing_y_weights,
        etad1_to_etad1_weights=etad1_to_etad1_weights,
        etad1_to_etas1_weights=etad1_to_etas1_weights,
        etas1_to_etad1_weights=etas1_to_etad1_weights,
        etas1_to_etas1_weights=etas1_to_etas1_weights,
    )
end

"""
用途: 构造受约束 Stripe ansatz 的按列 mean-field 参数字典。

参数:
- `lx::Int`: 晶格在 `x` 方向的列数。
- `lambda::Int`: charge stripe 周期 `λ`。
- `stripe_center::AbstractString`: stripe 中心类型, 支持 `site` 或 `bond`。
- `mu_uniform::Float64`: 平均 chemical potential `μ`。
- `stripe_mu_amp::Float64`: charge modulation 振幅 `Δc`。
- `mz_amp::Float64`: spin modulation 振幅 `Δs`。
- `etad1_uniform::Float64`: 全局 d-wave pairing 参数。
- `etas1_uniform::Float64`: 全局 s-wave pairing 参数。

返回:
- `NamedTuple`: 包含 `etad1`、`etas1`、`mz`、`mu` 四个按列参数字典。
"""
function build_restricted_stripe_column_params(
    lx::Int,
    lambda::Int,
    stripe_center::AbstractString,
    mu_uniform::Float64,
    stripe_mu_amp::Float64,
    mz_amp::Float64,
    etad1_uniform::Float64,
    etas1_uniform::Float64,
)
    stripe_weights = build_restricted_stripe_chain_rule_weights(lx, lambda, stripe_center)
    x_bond_pairing_uniform = etas1_uniform - etad1_uniform
    y_bond_pairing_uniform = etas1_uniform + etad1_uniform

    etad1_by_x = Dict{Symbol, Float64}()
    etas1_by_x = Dict{Symbol, Float64}()
    mz_by_x = Dict{Symbol, Float64}()
    mu_by_x = Dict{Symbol, Float64}()

    for x in 1:lx
        x_bond_pairing = x_bond_pairing_uniform * stripe_weights.pairing_x_weights[x]
        y_bond_pairing = y_bond_pairing_uniform * stripe_weights.pairing_y_weights[x]

        etas1_by_x[Symbol("etas1_$(x)")] = (x_bond_pairing + y_bond_pairing) / 2.0
        etad1_by_x[Symbol("etad1_$(x)")] = (y_bond_pairing - x_bond_pairing) / 2.0
        mz_by_x[Symbol("mz_$(x)")] = mz_amp * stripe_weights.spin_weights[x]
        mu_by_x[Symbol("mu_$(x)")] = mu_uniform + stripe_mu_amp * stripe_weights.charge_weights[x]
    end

    return (; etad1=etad1_by_x, etas1=etas1_by_x, mz=mz_by_x, mu=mu_by_x)
end

"""
用途: 根据受约束 ansatz 构造实际的按列 Hubbard 参数字典。

参数:
- `lx::Int`: 晶格在 `x` 方向的列数。
- `ansatz::AbstractString`: ansatz 类型, 支持 `AFM` 或 `Stripe`。
- `lambda::Int`: stripe 周期 `λ`。
- `stripe_center::AbstractString`: stripe 中心类型。
- `mu_uniform::Float64`: 平均 chemical potential。
- `stripe_mu_amp::Float64`: charge stripe 振幅。
- `mz_uniform::Float64`: AFM 或 stripe 的自旋幅值。
- `etad1_uniform::Float64`: 全局 d-wave pairing 参数。
- `etas1_uniform::Float64`: 全局 s-wave pairing 参数。

返回:
- `NamedTuple`: 按列的 `etad1`、`etas1`、`mz`、`mu` 参数字典。
"""
function build_restricted_hubbard_column_params(
    lx::Int,
    ansatz::AbstractString,
    lambda::Int,
    stripe_center::AbstractString,
    mu_uniform::Float64,
    stripe_mu_amp::Float64,
    mz_uniform::Float64,
    etad1_uniform::Float64,
    etas1_uniform::Float64,
)
    ansatz_uppercase = uppercase(strip(ansatz))
    if ansatz_uppercase == "AFM"
        return build_restricted_afm_column_params(lx, mu_uniform, mz_uniform, etad1_uniform, etas1_uniform)
    elseif ansatz_uppercase == "STRIPE"
        return build_restricted_stripe_column_params(
            lx,
            lambda,
            stripe_center,
            mu_uniform,
            stripe_mu_amp,
            mz_uniform,
            etad1_uniform,
            etas1_uniform,
        )
    end
    error("Unknown ansatz type: $(ansatz). Expected 'AFM' or 'Stripe'.")
end

"""
用途: 返回受约束 Hubbard ansatz 的独立波函数参数名与初值。

参数:
- `args::Dict{String, Any}`: 命令行参数字典。

返回:
- `Tuple{Vector{Symbol}, Vector{Float64}}`: `(wf_param_names, wf_init_params)`。
"""
function build_restricted_wf_param_names_and_init_params(args)::Tuple{Vector{Symbol}, Vector{Float64}}
    ansatz_uppercase = uppercase(strip(args["ansatz"]))
    if ansatz_uppercase == "AFM"
        return (
            Symbol[:chi2, :etad1, :etas1, :mz, :mu],
            Float64[args["chi2"], args["etad1"], args["etas1"], args["mz"], args["mu"]],
        )
    elseif ansatz_uppercase == "STRIPE"
        return (
            Symbol[:chi2, :etad1, :etas1, :mz, :mu, :stripe_mu_amp],
            Float64[args["chi2"], args["etad1"], args["etas1"], args["mz"], args["mu"], args["stripe_mu_amp"]],
        )
    end
    error("Unknown ansatz type: $(args["ansatz"]). Expected 'AFM' or 'Stripe'.")
end

"""
用途: 返回构造链式导数所需的完整按列波函数参数名列表。

参数:
- `lx::Int`: 晶格在 `x` 方向的列数。

返回:
- `Vector{Symbol}`: 包含 `chi2` 与所有按列参数的名称列表。
"""
function build_expanded_hubbard_wf_param_names(lx::Int)::Vector{Symbol}
    expanded_param_names = Symbol[:chi2]
    for x in 1:lx
        push!(expanded_param_names, Symbol("etad1_$(x)"))
        push!(expanded_param_names, Symbol("etas1_$(x)"))
        push!(expanded_param_names, Symbol("mz_$(x)"))
        push!(expanded_param_names, Symbol("mu_$(x)"))
    end
    return expanded_param_names
end

"""
用途: 构造与参考导数矩阵同尺寸的零矩阵。

参数:
- `reference_matrix::AbstractMatrix{Float64}`: 参考导数矩阵。

返回:
- `Matrix{Float64}`: 零矩阵。
"""
function build_zero_derivative_matrix(reference_matrix::AbstractMatrix{Float64})::Matrix{Float64}
    return zeros(Float64, size(reference_matrix, 1), size(reference_matrix, 2))
end

"""
用途: 将按列参数的轨道导数按链式法则合成为受约束 ansatz 的轨道导数。

参数:
- `expanded_derivatives::OrderedDict{Symbol, Matrix{Float64}}`: 完整按列参数的轨道导数字典。
- `restricted_param_names::Vector{Symbol}`: 受约束 ansatz 的参数名列表。
- `ansatz::AbstractString`: ansatz 类型, 支持 `AFM` 或 `Stripe`。
- `lx::Int`: 晶格在 `x` 方向的列数。
- `lambda::Int`: stripe 周期 `λ`。
- `stripe_center::AbstractString`: stripe 中心类型。

返回:
- `OrderedDict{Symbol, Matrix{Float64}}`: 受约束 ansatz 的轨道导数字典。
"""
function combine_restricted_hubbard_derivatives(
    expanded_derivatives::OrderedDict{Symbol, Matrix{Float64}},
    restricted_param_names::Vector{Symbol},
    ansatz::AbstractString,
    lx::Int,
    lambda::Int,
    stripe_center::AbstractString,
)::OrderedDict{Symbol, Matrix{Float64}}
    reference_key = first(keys(expanded_derivatives))
    reference_matrix = expanded_derivatives[reference_key]
    restricted_derivatives = OrderedDict{Symbol, Matrix{Float64}}()
    ansatz_uppercase = uppercase(strip(ansatz))

    stripe_weights = ansatz_uppercase == "STRIPE" ? build_restricted_stripe_chain_rule_weights(lx, lambda, stripe_center) : nothing

    for param_name in restricted_param_names
        derivative_matrix = build_zero_derivative_matrix(reference_matrix)

        if param_name == :chi2
            derivative_matrix .= expanded_derivatives[:chi2]
        elseif ansatz_uppercase == "AFM"
            if param_name == :etad1
                for x in 1:lx
                    derivative_matrix .+= expanded_derivatives[Symbol("etad1_$(x)")]
                end
            elseif param_name == :etas1
                for x in 1:lx
                    derivative_matrix .+= expanded_derivatives[Symbol("etas1_$(x)")]
                end
            elseif param_name == :mz
                for x in 1:lx
                    derivative_matrix .+= expanded_derivatives[Symbol("mz_$(x)")]
                end
            elseif param_name == :mu
                for x in 1:lx
                    derivative_matrix .+= expanded_derivatives[Symbol("mu_$(x)")]
                end
            else
                error("Unknown restricted AFM parameter: $(param_name)")
            end
        elseif ansatz_uppercase == "STRIPE"
            if param_name == :etad1
                for x in 1:lx
                    derivative_matrix .+= stripe_weights.etad1_to_etad1_weights[x] .* expanded_derivatives[Symbol("etad1_$(x)")]
                    derivative_matrix .+= stripe_weights.etad1_to_etas1_weights[x] .* expanded_derivatives[Symbol("etas1_$(x)")]
                end
            elseif param_name == :etas1
                for x in 1:lx
                    derivative_matrix .+= stripe_weights.etas1_to_etad1_weights[x] .* expanded_derivatives[Symbol("etad1_$(x)")]
                    derivative_matrix .+= stripe_weights.etas1_to_etas1_weights[x] .* expanded_derivatives[Symbol("etas1_$(x)")]
                end
            elseif param_name == :mz
                for x in 1:lx
                    derivative_matrix .+= stripe_weights.spin_weights[x] .* expanded_derivatives[Symbol("mz_$(x)")]
                end
            elseif param_name == :mu
                for x in 1:lx
                    derivative_matrix .+= expanded_derivatives[Symbol("mu_$(x)")]
                end
            elseif param_name == :stripe_mu_amp
                for x in 1:lx
                    derivative_matrix .+= stripe_weights.charge_weights[x] .* expanded_derivatives[Symbol("mu_$(x)")]
                end
            else
                error("Unknown restricted Stripe parameter: $(param_name)")
            end
        else
            error("Unknown ansatz type: $(ansatz). Expected 'AFM' or 'Stripe'.")
        end

        restricted_derivatives[param_name] = derivative_matrix
    end

    return restricted_derivatives
end

"""
用途: 更新受约束 Hubbard ansatz 的波函数、投影器与 backflow 参数。

参数:
- `vwf`: determinant 波函数对象。
- `param_names::Vector{Symbol}`: 总参数名列表, 由波函数、投影器、backflow 参数拼接而成。
- `params::Vector{Float64}`: 与 `param_names` 对应的参数值向量。
- `lx`: 晶格在 `x` 方向的列数。
- `ly`: 晶格在 `y` 方向的列数。
- `bcx`: `x` 方向边界条件系数。
- `bcy`: `y` 方向边界条件系数。
- `target_sz::Int`: 目标总 `Sz`。
- `ansatz::AbstractString`: 受约束 ansatz 类型。
- `lambda::Int`: stripe 周期 `λ`。
- `stripe_center::AbstractString`: stripe 中心类型。
- `nparams_proj::Int`: projector 参数个数。
- `nparams_backflow::Int`: backflow 参数个数。

返回:
- 无返回值, 原地更新 `vwf`。
"""
function update_restricted_ansatz!(
    vwf,
    param_names::Vector{Symbol},
    params::Vector{Float64},
    lx,
    ly,
    bcx,
    bcy,
    target_sz::Int,
    ansatz::AbstractString,
    lambda::Int,
    stripe_center::AbstractString;
    nparams_proj::Int=0,
    nparams_backflow::Int=0,
)
    nparms = length(param_names)
    nparams_wf = nparms - nparams_proj - nparams_backflow
    wf_param_names = param_names[1:nparams_wf]
    wf_param_values = params[1:nparams_wf]
    projector_param_names = param_names[(nparams_wf + 1):(nparams_wf + nparams_proj)]
    projector_param_values = params[(nparams_wf + 1):(nparams_wf + nparams_proj)]
    backflow_param_names_local = param_names[(nparams_wf + nparams_proj + 1):end]
    backflow_param_values_local = params[(nparams_wf + nparams_proj + 1):end]

    wf_param_map = Dict{Symbol, Float64}(zip(wf_param_names, wf_param_values))
    chi2 = get(wf_param_map, :chi2, 0.0)
    etad1 = get(wf_param_map, :etad1, 0.0)
    etas1 = get(wf_param_map, :etas1, 0.0)
    mz = get(wf_param_map, :mz, 0.0)
    mu = get(wf_param_map, :mu, 0.0)
    stripe_mu_amp = get(wf_param_map, :stripe_mu_amp, 0.0)

    column_params = build_restricted_hubbard_column_params(
        lx,
        ansatz,
        lambda,
        stripe_center,
        mu,
        stripe_mu_amp,
        mz,
        etad1,
        etas1,
    )

    hubbard_params = PartonSquare.HubbardParams(
        Lx=lx,
        Ly=ly,
        bcx=bcx,
        bcy=bcy,
        chi1=1.0,
        etad1=column_params.etad1,
        etas1=column_params.etas1,
        chi2=chi2,
        mu=column_params.mu,
        mz=column_params.mz,
    )

    expanded_wf_param_names = build_expanded_hubbard_wf_param_names(lx)
    _, gs_u, expanded_derivatives = PartonSquare.make_ansatz_and_derivs(
        hubbard_params;
        param_names=expanded_wf_param_names,
        target_sz=target_sz,
    )
    restricted_derivatives = combine_restricted_hubbard_derivatives(
        expanded_derivatives,
        wf_param_names,
        ansatz,
        lx,
        lambda,
        stripe_center,
    )

    copyto!(vwf.base_gs_U, gs_u)
    copyto!(vwf.gs_U, gs_u)
    copyto!(vwf.backflow_u, gs_u)
    copyto!(vwf.gs_U_t, permutedims(gs_u))

    d_ut_matrix = zeros(Float64, size(gs_u, 2), size(gs_u, 1), length(wf_param_names))
    for (param_index, param_name) in enumerate(wf_param_names)
        d_ut_matrix[:, :, param_index] = restricted_derivatives[param_name]
    end

    update_vwf_params!(vwf, wf_param_names, d_ut_matrix)
    if !isempty(projector_param_names)
        update_vwf_projector_params!(vwf, projector_param_names, projector_param_values)
    end
    if !isempty(backflow_param_names_local)
        update_vwf_backflow_params!(vwf, backflow_param_names_local, backflow_param_values_local)
    end
    init_gswf!(vwf)
end

"""
用途: 运行受约束 Hubbard ansatz 的主程序。

参数:
- 无, 直接读取命令行参数。

返回:
- 无返回值。
"""
function main_restricted()
    args = parse_commandline()

    session = init_mpi_session()
    rank = session.rank
    is_root = (rank == session.root)

    lx = args["Lx"]
    ly = args["Ly"]
    bcx = args["bcx"]
    bcy = args["bcy"]
    target_sz = args["target_sz"]
    doping = args["doping"]
    lambda = args["lambda"]
    stripe_center = args["stripe_center"]
    n_mc = args["nMC"]
    w_mc = args["wMC"]
    r_mc = args["rMC"]
    d_mc = args["dMC"]
    n_steps = args["nSR"]
    lr = args["lr"]
    lr_end = args["lr_end"]
    if isnan(lr_end)
        lr_end = lr
    end

    t1 = args["t1"]
    t2 = args["t2"]
    u_value = args["U"]
    job = args["job"]
    ansatz = args["ansatz"]
    g = args["g"]
    vj1 = args["vj1"]
    vj2 = args["vj2"]
    init_params_json = args["init_params_json"]
    n_sites = lx * ly

    wf_param_names, wf_init_params = build_restricted_wf_param_names_and_init_params(args)

    meas_params = VMCParams(
        total_samples=n_mc,
        warmup_steps=w_mc,
        rebuild_every=r_mc,
        decorr_steps=d_mc,
        seed=args["seed"] + rank,
    )

    bonds1 = Tuple{Int, Int}[]
    bonds2 = Tuple{Int, Int}[]
    idx_local(x, y) = mod(x - 1, lx) * ly + mod(y - 1, ly) + 1
    for y in 1:ly, x in 1:lx
        site_index = idx_local(x, y)
        push!(bonds1, (site_index, idx_local(x + 1, y)))
        push!(bonds1, (site_index, idx_local(x, y + 1)))
        push!(bonds2, (site_index, idx_local(x + 1, y + 1)))
        push!(bonds2, (site_index, idx_local(x - 1, y + 1)))
    end

    site_to_neighbor_sites_j1 = [Int[] for _ in 1:n_sites]
    for (site_i, site_j) in bonds1
        if !(site_j in site_to_neighbor_sites_j1[site_i])
            push!(site_to_neighbor_sites_j1[site_i], site_j)
        end
        if !(site_i in site_to_neighbor_sites_j1[site_j])
            push!(site_to_neighbor_sites_j1[site_j], site_i)
        end
    end

    site_to_neighbor_sites_j2 = [Int[] for _ in 1:n_sites]
    for (site_i, site_j) in bonds2
        if !(site_j in site_to_neighbor_sites_j2[site_i])
            push!(site_to_neighbor_sites_j2[site_i], site_j)
        end
        if !(site_i in site_to_neighbor_sites_j2[site_j])
            push!(site_to_neighbor_sites_j2[site_j], site_i)
        end
    end

    projector = CompositeProjector([
        GutzwillerProjectorTerm(param_name=:g, g=g),
        JastrowProjectorTerm(param_name=:vj1, v=vj1, site_to_neighbor_sites=site_to_neighbor_sites_j1),
        JastrowProjectorTerm(param_name=:vj2, v=vj2, site_to_neighbor_sites=site_to_neighbor_sites_j2),
    ])
    backflow = NoBackflowTerm()
    proj_param_names = projector_param_names(projector)
    proj_init_params = projector_param_values(projector)
    nparams_proj = length(proj_param_names)
    backflow_param_name_list = backflow_param_names(backflow)
    backflow_init_params = backflow_param_values(backflow)
    nparams_backflow = length(backflow_param_name_list)

    init_params = vcat(wf_init_params, proj_init_params, backflow_init_params)
    param_names = vcat(wf_param_names, proj_param_names, backflow_param_name_list)

    if !isempty(init_params_json)
        init_params = build_init_params_from_json(init_params_json, param_names)
        if is_root
            println("Loaded initial parameters from json: $(init_params_json)")
        end
    end

    terms = OperatorTerm[]
    for (i, j) in bonds1
        push!(terms, OperatorTerm([:cdag_up, :c_up], [i, j], -t1))
        push!(terms, OperatorTerm([:cdag_up, :c_up], [j, i], -t1))
        push!(terms, OperatorTerm([:cdag_dn, :c_dn], [i, j], -t1))
        push!(terms, OperatorTerm([:cdag_dn, :c_dn], [j, i], -t1))
    end
    for (i, j) in bonds2
        push!(terms, OperatorTerm([:cdag_up, :c_up], [i, j], -t2))
        push!(terms, OperatorTerm([:cdag_up, :c_up], [j, i], -t2))
        push!(terms, OperatorTerm([:cdag_dn, :c_dn], [i, j], -t2))
        push!(terms, OperatorTerm([:cdag_dn, :c_dn], [j, i], -t2))
    end
    for i in 1:n_sites
        push!(terms, OperatorTerm([:n_up, :n_dn], [i, i], u_value))
    end
    ham = GeneralModel(n_sites, terms)

    nelec = Int(n_sites * (1 + doping))
    @assert (target_sz + nelec) % 2 == 0 "Wrong parity!"
    nup = (nelec + target_sz) ÷ 2
    ndn = nelec - nup
    sampler = config_Hubbard(n_sites, nup, ndn; ifPH=true)
    init_config_Hubbard!(sampler)

    vwf = vwf_det(zeros(Float64, 2 * n_sites, n_sites + target_sz), sampler; backflow=backflow)
    set_projector!(vwf, projector)
    kernel = HubbardKernel(conserve_sz=true)

    if is_root
        println("Initial parameters: $init_params")
    end
    update_restricted_ansatz!(
        vwf,
        param_names,
        init_params,
        lx,
        ly,
        bcx,
        bcy,
        target_sz,
        ansatz,
        lambda,
        stripe_center;
        nparams_proj=nparams_proj,
        nparams_backflow=nparams_backflow,
    )

    folder = "logs"
    mkpath(folder)

    if job == "SR"
        sr_params = SRParams(vmc_params=meas_params, n_steps=n_steps, lr=lr)
        exp_lr_func = build_exponential_lr_func(lr, lr_end, n_steps)

        update_vwf_func! = (vwf_local, params_local) -> update_restricted_ansatz!(
            vwf_local,
            param_names,
            params_local,
            lx,
            ly,
            bcx,
            bcy,
            target_sz,
            ansatz,
            lambda,
            stripe_center;
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
        observables = defination_observabels(lx, ly)
        history_observables = [:E]
        results = run_simulation(
            ham,
            vwf,
            kernel,
            observables,
            meas_params;
            history_observables=history_observables,
        )
        if is_root && results !== nothing
            means = results[:means]
            mean_dict = Dict{Symbol, Any}()
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

                txt_file = joinpath(folder, "block_binning.txt")
                open(txt_file, "w") do io
                    println(io, "# Observable\tMean\tSE\tN_eff\tTau_int")
                    for name in sort(collect(keys(mean_hist)))
                        mean_val = mean_hist[name]
                        se_val = se_dict[name]
                        n_eff_val = n_eff_dict[name]
                        tau_val = tau_int_dict[name]

                        if mean_val isa Number && se_val isa Number && n_eff_val isa Number && tau_val isa Number
                            @printf(
                                io,
                                "%s\t%.10f\t%.10f\t%.6f\t%.6f\n",
                                String(name),
                                mean_val,
                                se_val,
                                n_eff_val,
                                tau_val,
                            )
                        else
                            println(io, "$(String(name))\t$(mean_val)\t$(se_val)\t$(n_eff_val)\t$(tau_val)")
                        end
                    end
                end
            end

            json_file = joinpath(folder, "block_binning_mean.json")
            mean_dict_str = Dict{String, Any}()
            for (key, value) in mean_dict
                mean_dict_str[String(key)] = value
            end
            open(json_file, "w") do io
                JSON.print(io, mean_dict_str)
            end
        end
    else
        error("Unknown job type: $(job)")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_restricted()
end
