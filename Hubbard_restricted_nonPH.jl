include("Hubbard_restricted.jl")

using Utils: add_term_ij_nonPH, compute_eig_and_dU_reg1

"""
用途: 解析 nonPH restricted Hubbard 主程序的命令行参数。

参数:
- 无。参数来自 Julia 进程的 `ARGS`。

返回:
- `Dict{String, Any}`: `ArgParse.parse_args` 返回的参数字典。

说明:
- nonPH 版本删除 mean-field pairing 参数, 因此这里不提供 `etax` 和 `etay`。
- backflow 使用完整 Eq.(5) 的 `bf_epsilon`, `bf_eta1`, `bf_eta2`, `bf_eta3`。
"""
function parse_nonph_commandline()
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
        help = "Boundary condition phase in X (1.0 or -1.0)"
        arg_type = Float64
        default = 1.001
        "--bcy"
        help = "Boundary condition phase in Y (1.0 or -1.0)"
        arg_type = Float64
        default = 0.999
        "--chi2"
        help = "Next-nearest neighbor hopping in MF ansatz. Default follows --t2"
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
        "--target_sz"
        help = "Target total Sz, represented as N_up - N_down"
        arg_type = Int
        default = 0
        "--nMC"
        help = "Number of Monte Carlo total samples"
        arg_type = Int
        default = 10000
        "--wMC"
        help = "Number of Monte Carlo warming up steps"
        arg_type = Int
        default = 100
        "--rMC"
        help = "Number of rebuild interval"
        arg_type = Int
        default = 100
        "--dMC"
        help = "Number of Monte Carlo decorrelation sweeps"
        arg_type = Int
        default = 1
        "--seed"
        help = "Random seed"
        arg_type = Int
        default = 5423
        "--nSR"
        help = "Total steps for SR"
        arg_type = Int
        default = 50
        "--lr"
        help = "SR learn rate"
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
        help = "Comma-separated fixed parameter assignments, e.g. 'Delta_AF=1.0'"
        arg_type = String
        default = ""
        "--active_params"
        help = "Comma-separated parameter names optimized by SR. Empty means all non-fixed parameters."
        arg_type = String
        default = ""
        "--enable_backflow"
        help = "Whether to enable backflow terms. Accepts true/false, 1/0, yes/no. Default is true."
        arg_type = String
        default = "true"
        "--job"
        help = "Job to be done. Can be SR and measure"
        arg_type = String
        default = "SR"
        "--doping"
        help = "Doping level. This follows Hubbard_restricted.jl: N_e = N_sites * (1 + doping)."
        arg_type = Float64
        default = 0.125
        "--ansatz"
        help = "Ansatz type, can be 'AFM' or 'Stripe'"
        arg_type = String
        default = "Stripe"
        "--lambda"
        help = "Assumed stripe length"
        arg_type = Int
        default = 4
        "--stripe_center"
        help = "Stripe center type, can be 'site' or 'bond'"
        arg_type = String
        default = "site"
        "--g"
        help = "Gutzwiller projector parameter"
        arg_type = Float64
        default = 1.0
        "--bf_epsilon"
        help = "Eq.(5) backflow epsilon parameter"
        arg_type = Float64
        default = 1.0
        "--bf_eta1"
        help = "Eq.(5) backflow eta1 doublon-hole hopping parameter"
        arg_type = Float64
        default = 0.0
        "--bf_eta2"
        help = "Eq.(5) backflow eta2 spin-exchange hopping parameter"
        arg_type = Float64
        default = 0.0
        "--bf_eta3"
        help = "Eq.(5) backflow eta3 mixed virtual hopping parameter"
        arg_type = Float64
        default = 0.0
    end

    return parse_args(settings)
end

"""
用途: 保存 nonPH restricted Hubbard mean-field ansatz 的参数。

字段:
- `lx, ly::Int`: 晶格尺寸。
- `bcx, bcy::Float64`: 边界条件因子。
- `chi1, chi2::Float64`: 最近邻与次近邻 hopping 参数。
- `delta_af::Float64`: AF staggered field 振幅。
- `delta_c::Float64`: charge stripe field 振幅。
- `delta_s::Float64`: spin stripe field 振幅。
- `stripe_wavevector::Float64`: stripe 波矢 `Q`。
- `stripe_center_offset::Float64`: stripe 中心偏移 `x0`。
"""
struct RestrictedHubbardNonPHParams
    lx::Int
    ly::Int
    bcx::Float64
    bcy::Float64
    chi1::Float64
    chi2::Float64
    delta_af::Float64
    delta_c::Float64
    delta_s::Float64
    stripe_wavevector::Float64
    stripe_center_offset::Float64
end

"""
用途: 构造 `RestrictedHubbardNonPHParams` 参数对象。

参数:
- `lx, ly::Int`: 晶格尺寸。
- `bcx, bcy::Float64`: 边界条件因子。
- `chi1, chi2::Float64`: 最近邻与次近邻 hopping 参数。
- `delta_af, delta_c, delta_s::Float64`: AF/charge stripe/spin stripe 场。
- `stripe_wavevector::Float64`: stripe 波矢 `Q`。
- `stripe_center_offset::Float64`: stripe 中心偏移 `x0`。

返回:
- `RestrictedHubbardNonPHParams`: nonPH mean-field 参数对象。
"""
function RestrictedHubbardNonPHParams(;
    lx::Int,
    ly::Int,
    bcx::Float64=1.0,
    bcy::Float64=1.0,
    chi1::Float64=0.0,
    chi2::Float64=0.0,
    delta_af::Float64=0.0,
    delta_c::Float64=0.0,
    delta_s::Float64=0.0,
    stripe_wavevector::Float64=0.0,
    stripe_center_offset::Float64=0.0,
)
    return RestrictedHubbardNonPHParams(
        lx,
        ly,
        bcx,
        bcy,
        chi1,
        chi2,
        delta_af,
        delta_c,
        delta_s,
        stripe_wavevector,
        stripe_center_offset,
    )
end

"""
用途: 构造 nonPH restricted Hubbard 的 number-conserving mean-field Hamiltonian。

数学公式:
- 使用 spinful electron basis `row(i, up)=2i-1`, `row(i, down)=2i`。
- hopping 部分为 `H_ij = chi_ij` 同时作用在 up/down electron block, 不含 pairing block。
- 对角场为
  `H_{i up,i up} += (+1) * (-1)^(x+y) * m_i / 2 + rho_i / 2`,
  `H_{i down,i down} += (-1) * (-1)^(x+y) * m_i / 2 + rho_i / 2`。
- 其中 `rho_i = Delta_c cos(Q (x - x0))`,
  `m_i = Delta_AF + Delta_s sin(Q/2 * (x - x0))`。

参数:
- `params::RestrictedHubbardNonPHParams`: nonPH mean-field 参数。

返回:
- `Hermitian{Float64, Matrix{Float64}}`: 维度为 `(2N_sites, 2N_sites)` 的 Hermitian Hamiltonian。
"""
function build_restricted_hubbard_nonph_hamiltonian(
    params::RestrictedHubbardNonPHParams,
)
    lx = params.lx
    ly = params.ly
    n_sites = lx * ly
    hamiltonian = zeros(Float64, 2 * n_sites, 2 * n_sites)

    for x in 1:lx
        charge_field_x = params.delta_c * cos(params.stripe_wavevector * (x - params.stripe_center_offset))
        spin_field_x = params.delta_af + params.delta_s * sin(params.stripe_wavevector / 2 * (x - params.stripe_center_offset))

        for y in 1:ly
            site_i = PartonSquare.xy_to_idx(x, y, ly)
            staggered_sign = (-1)^(x + y)

            site_x = (x == lx) ? PartonSquare.xy_to_idx(1, y, ly) : PartonSquare.xy_to_idx(x + 1, y, ly)
            site_y = (y == ly) ? PartonSquare.xy_to_idx(x, 1, ly) : PartonSquare.xy_to_idx(x, y + 1, ly)
            bc_x = (x == lx) ? params.bcx : 1.0
            bc_y = (y == ly) ? params.bcy : 1.0

            site_pp = PartonSquare.xy_to_idx((x == lx) ? 1 : x + 1, (y == ly) ? 1 : y + 1, ly)
            site_mp = PartonSquare.xy_to_idx((x == 1) ? lx : x - 1, (y == ly) ? 1 : y + 1, ly)
            bc_pp = ((x == lx) ? params.bcx : 1.0) * ((y == ly) ? params.bcy : 1.0)
            bc_mp = ((x == 1) ? params.bcx : 1.0) * ((y == ly) ? params.bcy : 1.0)

            add_term_ij_nonPH(hamiltonian, site_i, site_x, -params.chi1 * bc_x)
            add_term_ij_nonPH(hamiltonian, site_i, site_y, -params.chi1 * bc_y)
            add_term_ij_nonPH(hamiltonian, site_i, site_pp, -params.chi2 * bc_pp)
            add_term_ij_nonPH(hamiltonian, site_i, site_mp, -params.chi2 * bc_mp)

            up_row = 2 * (site_i - 1) + 1
            down_row = up_row + 1
            hamiltonian[up_row, up_row] += staggered_sign * spin_field_x / 2 + charge_field_x / 2
            hamiltonian[down_row, down_row] += -staggered_sign * spin_field_x / 2 + charge_field_x / 2
        end
    end

    return Hermitian(hamiltonian + hamiltonian')
end

"""
用途: 构造 nonPH mean-field Hamiltonian 对单个参数的导数矩阵。

数学公式:
- 当前 nonPH Hamiltonian 对 `chi2`, `Delta_AF`, `Delta_c`, `Delta_s` 均为线性依赖。
- 因此 `dH/dp` 可通过只把目标参数置为 `1.0`, 其它可变参数置为 `0.0`,
  并保持 `Q, x0, bcx, bcy` 不变来得到。

参数:
- `params::RestrictedHubbardNonPHParams`: 当前 ansatz 参数, 提供尺寸、边界和 stripe 几何。
- `param_name::Symbol`: 目标参数名, 支持 `:chi2`, `:Delta_AF`, `:Delta_c`, `:Delta_s`。

返回:
- `Matrix{Float64}`: 与 Hamiltonian 同维度的 `dH/dp` 矩阵。
"""
function build_restricted_hubbard_nonph_dh_dparam(
    params::RestrictedHubbardNonPHParams,
    param_name::Symbol,
)::Matrix{Float64}
    derivative_params = RestrictedHubbardNonPHParams(
        lx=params.lx,
        ly=params.ly,
        bcx=params.bcx,
        bcy=params.bcy,
        chi1=0.0,
        chi2=(param_name == :chi2 ? 1.0 : 0.0),
        delta_af=(param_name == :Delta_AF ? 1.0 : 0.0),
        delta_c=(param_name == :Delta_c ? 1.0 : 0.0),
        delta_s=(param_name == :Delta_s ? 1.0 : 0.0),
        stripe_wavevector=params.stripe_wavevector,
        stripe_center_offset=params.stripe_center_offset,
    )
    return Matrix(build_restricted_hubbard_nonph_hamiltonian(derivative_params))
end

"""
用途: 生成 nonPH restricted Hubbard determinant 的占据轨道和 mean-field 参数导数。

数学公式:
- 先对 number-conserving spinful Hamiltonian `H` 对角化: `H U = U epsilon`。
- nonPH determinant 固定真实电子数 `N_e`, 因此取最低的 `n_occupied_orbitals=N_e` 个单粒子轨道。
- 对 mean-field 参数的 determinant 导数使用 `dU/dp`, 后续 SR 中进入
  `O_p = Tr(A^{-1} dA/dp)`。

参数:
- `params::RestrictedHubbardNonPHParams`: nonPH mean-field 参数。
- `param_names::Vector{Symbol}`: 需要求导的 mean-field 参数名。
- `n_occupied_orbitals::Int`: determinant 占据轨道数, nonPH 下等于真实电子数 `N_up + N_down`。

返回:
- `(epsilon, occupied_orbitals, dUt_params)`: 本征值, 占据轨道矩阵, 以及按参数名索引的转置导数矩阵。
"""
function make_restricted_hubbard_nonph_ansatz_and_derivs(
    params::RestrictedHubbardNonPHParams;
    param_names::Vector{Symbol}=Symbol[],
    n_occupied_orbitals::Int,
)
    hamiltonian = Matrix(build_restricted_hubbard_nonph_hamiltonian(params))
    hamiltonian_derivatives = Dict{Symbol,Matrix{Float64}}()
    for param_name in param_names
        hamiltonian_derivatives[param_name] = build_restricted_hubbard_nonph_dh_dparam(
            params,
            param_name,
        )
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
用途: 根据参数更新 nonPH restricted Hubbard determinant 波函数和参数导数。

参数:
- `vwf`: determinant 波函数对象。
- `param_names::Vector{Symbol}`: 完整参数名列表, 顺序为 mean-field, projector, backflow。
- `params::Vector{Float64}`: 与 `param_names` 对齐的完整参数值。
- `lx, ly::Int`: 晶格尺寸。
- `bcx, bcy::Float64`: 边界条件因子。
- `n_occupied_orbitals::Int`: nonPH determinant 占据轨道数, 等于真实电子数。
- `nparams_proj::Int`: projector 参数数量。
- `nparams_backflow::Int`: backflow 参数数量。
- `stripe_wavevector::Float64`: stripe 波矢 `Q`。
- `stripe_center_offset::Float64`: stripe 中心偏移 `x0`。
- `active_wf_param_names::Union{Nothing, Vector{Symbol}}`: 参与求导和 SR 的 mean-field 参数名; `nothing` 表示全部。

返回:
- `nothing`。
"""
function update_nonph_ansatz!(
    vwf,
    param_names::Vector{Symbol},
    params::Vector{Float64},
    lx::Int,
    ly::Int,
    bcx::Float64,
    bcy::Float64,
    n_occupied_orbitals::Int;
    nparams_proj::Int=0,
    nparams_backflow::Int=0,
    stripe_wavevector::Float64=0.0,
    stripe_center_offset::Float64=0.0,
    active_wf_param_names::Union{Nothing,Vector{Symbol}}=nothing,
)::Nothing
    total_param_count = length(param_names)
    nparams_wf = total_param_count - nparams_proj - nparams_backflow
    wf_param_names = param_names[1:nparams_wf]
    wf_param_values = params[1:nparams_wf]
    projector_param_names = param_names[(nparams_wf+1):(nparams_wf+nparams_proj)]
    projector_param_values = params[(nparams_wf+1):(nparams_wf+nparams_proj)]
    backflow_param_names = param_names[(nparams_wf+nparams_proj+1):end]
    backflow_param_values = params[(nparams_wf+nparams_proj+1):end]

    derivative_wf_param_names = active_wf_param_names === nothing ? wf_param_names : active_wf_param_names
    wf_param_name_set = Set(wf_param_names)
    for derivative_param_name in derivative_wf_param_names
        if !(derivative_param_name in wf_param_name_set)
            error("Active mean-field parameter $(derivative_param_name) is not in full mean-field parameter list.")
        end
    end

    param_map = Dict{Symbol,Float64}(zip(wf_param_names, wf_param_values))
    nonph_params = RestrictedHubbardNonPHParams(
        lx=lx,
        ly=ly,
        bcx=bcx,
        bcy=bcy,
        chi1=1.0,
        chi2=get(param_map, :chi2, 0.0),
        delta_af=get(param_map, :Delta_AF, 0.0),
        delta_c=get(param_map, :Delta_c, 0.0),
        delta_s=get(param_map, :Delta_s, 0.0),
        stripe_wavevector=stripe_wavevector,
        stripe_center_offset=stripe_center_offset,
    )

    _, gs_u, d_ut_params = make_restricted_hubbard_nonph_ansatz_and_derivs(
        nonph_params;
        param_names=derivative_wf_param_names,
        n_occupied_orbitals=n_occupied_orbitals,
    )

    copyto!(vwf.base_gs_U, gs_u)
    copyto!(vwf.gs_U, gs_u)
    copyto!(vwf.backflow_u, gs_u)
    copyto!(vwf.gs_U_t, permutedims(gs_u))

    d_ut_matrix = zeros(Float64, size(gs_u, 2), size(gs_u, 1), length(derivative_wf_param_names))
    for (param_index, param_name) in enumerate(derivative_wf_param_names)
        d_ut_matrix[:, :, param_index] = d_ut_params[param_name]
    end
    update_vwf_params!(vwf, derivative_wf_param_names, d_ut_matrix)

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
用途: 根据 `ansatz` 参数生成 nonPH mean-field 参数名、初值和 stripe 几何。

参数:
- `args::Dict{String, Any}`: 命令行参数字典。

返回:
- `NamedTuple`: 包含 `wf_param_names`, `wf_init_params`, `stripe_wavevector`, `stripe_center_offset`。
"""
function build_nonph_mean_field_parameter_setup(args)
    ansatz = args["ansatz"]
    if ansatz == "AFM"
        return (
            wf_param_names=[:chi2, :Delta_AF],
            wf_init_params=[args["chi2"], args["Delta_AF"]],
            stripe_wavevector=0.0,
            stripe_center_offset=0.0,
        )
    elseif ansatz == "Stripe"
        lambda = args["lambda"]
        stripe_center = args["stripe_center"]
        stripe_center_offset = if stripe_center == "site"
            0.0
        elseif stripe_center == "bond"
            0.5
        else
            error("Unknown stripe_center type: $(stripe_center)")
        end
        return (
            wf_param_names=[:chi2, :Delta_c, :Delta_s],
            wf_init_params=[args["chi2"], args["Delta_c"], args["Delta_s"]],
            stripe_wavevector=2π / lambda,
            stripe_center_offset=stripe_center_offset,
        )
    end
    error("Unknown ansatz type: $(ansatz)")
end

"""
用途: 运行 nonPH restricted Hubbard determinant VMC/SR 主流程。

参数:
- 无。所有配置来自命令行参数。

返回:
- `nothing`。

说明:
- sampler 使用 `config_Hubbard(...; ifPH=false)`, 因此 determinant row 直接对应真实电子轨道。
- determinant 列数取真实电子数 `N_e = N_sites * (1 + doping)`, 与当前 PH 脚本的 doping 约定保持一致。
"""
function main_nonph()::Nothing
    args = parse_nonph_commandline()

    session = init_mpi_session()
    rank = session.rank
    is_root = (rank == session.root)

    lx = args["Lx"]
    ly = args["Ly"]
    bcx = args["bcx"]
    bcy = args["bcy"]
    target_sz = args["target_sz"]
    doping = args["doping"]
    nmc = args["nMC"]
    wmc = args["wMC"]
    rmc = args["rMC"]
    dmc = args["dMC"]
    n_steps = args["nSR"]
    lr = args["lr"]
    lr_end = args["lr_end"]
    if isnan(lr_end)
        lr_end = lr
    end

    t1 = args["t1"]
    t2 = args["t2"]
    onsite_u = args["U"]
    job = args["job"]
    g = args["g"]
    bf_epsilon = args["bf_epsilon"]
    bf_eta1 = args["bf_eta1"]
    bf_eta2 = args["bf_eta2"]
    bf_eta3 = args["bf_eta3"]
    enable_backflow = parse_bool_flag(args["enable_backflow"], "--enable_backflow")
    init_params_json = args["init_params_json"]
    fixed_params_string = args["fixed_params"]
    active_params_string = args["active_params"]
    n_sites = lx * ly

    mean_field_setup = build_nonph_mean_field_parameter_setup(args)
    wf_param_names = mean_field_setup.wf_param_names
    wf_init_params = mean_field_setup.wf_init_params
    stripe_wavevector = mean_field_setup.stripe_wavevector
    stripe_center_offset = mean_field_setup.stripe_center_offset

    meas_params = VMCParams(
        total_samples=nmc,
        warmup_steps=wmc,
        rebuild_every=rmc,
        decorr_steps=dmc,
        seed=args["seed"] + rank,
    )

    bonds1 = Tuple{Int,Int}[]
    bonds2 = Tuple{Int,Int}[]
    local_idx(x, y) = mod(x - 1, lx) * ly + mod(y - 1, ly) + 1
    for y in 1:ly, x in 1:lx
        site_i = local_idx(x, y)
        push!(bonds1, (site_i, local_idx(x + 1, y)))
        push!(bonds1, (site_i, local_idx(x, y + 1)))
        push!(bonds2, (site_i, local_idx(x + 1, y + 1)))
        push!(bonds2, (site_i, local_idx(x - 1, y + 1)))
    end

    backflow_source_bonds = Tuple{Int,Int}[]
    backflow_source_amplitudes = Float64[]
    for (site_i, site_j) in bonds1
        push!(backflow_source_bonds, (site_i, site_j))
        push!(backflow_source_amplitudes, t1)
        push!(backflow_source_bonds, (site_j, site_i))
        push!(backflow_source_amplitudes, t1)
    end
    for (site_i, site_j) in bonds2
        push!(backflow_source_bonds, (site_i, site_j))
        push!(backflow_source_amplitudes, t2)
        push!(backflow_source_bonds, (site_j, site_i))
        push!(backflow_source_amplitudes, t2)
    end

    projector = build_restricted_projector(lx, ly, g)
    backflow = build_restricted_optional_backflow(
        enable_backflow,
        backflow_source_bonds,
        backflow_source_amplitudes,
        bf_epsilon,
        bf_eta1,
        bf_eta2,
        bf_eta3,
    )
    proj_param_names = projector_param_names(projector)
    proj_init_params = projector_param_values(projector)
    nparams_proj = length(proj_param_names)
    backflow_param_name_list = backflow_param_names(backflow)
    backflow_init_params = backflow_param_values(backflow)
    nparams_backflow = length(backflow_param_name_list)

    init_params = vcat(wf_init_params, proj_init_params, backflow_init_params)
    param_names = vcat(wf_param_names, proj_param_names, backflow_param_name_list)

    if !isempty(init_params_json)
        init_params = build_init_params_from_json_with_defaults(init_params_json, param_names, init_params)
        if is_root
            println("Loaded initial parameters from json: $(init_params_json)")
        end
    end

    fixed_param_values = parse_fixed_param_string(fixed_params_string)
    requested_active_param_names = parse_param_name_list(active_params_string)
    validate_fixed_mean_field_params!(wf_param_names, fixed_param_values)
    if !isempty(fixed_param_values)
        init_params = apply_fixed_params_to_values(param_names, init_params, fixed_param_values)
    end
    active_param_indices = build_active_param_indices(param_names, fixed_param_values, requested_active_param_names)
    if job == "SR" && isempty(active_param_indices)
        error("At least one parameter must remain active for SR optimization.")
    end

    uses_param_subset = length(active_param_indices) != length(param_names) || !isempty(requested_active_param_names)
    sr_param_names = uses_param_subset ? param_names[active_param_indices] : param_names
    sr_init_params = uses_param_subset ? init_params[active_param_indices] : init_params
    wf_param_name_set = Set(wf_param_names)
    projector_param_name_set = Set(proj_param_names)
    backflow_param_name_set = Set(backflow_param_name_list)
    active_wf_param_names = [name for name in sr_param_names if name in wf_param_name_set]
    active_projector_param_names = [name for name in sr_param_names if name in projector_param_name_set]
    active_backflow_param_names = [name for name in sr_param_names if name in backflow_param_name_set]
    set_active_sr_derivative_param_names!(
        proj_param_names,
        backflow_param_name_list;
        active_projector_param_names=uses_param_subset ? active_projector_param_names : nothing,
        active_backflow_param_names=uses_param_subset ? active_backflow_param_names : nothing,
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
    for site_i in 1:n_sites
        push!(terms, OperatorTerm([:n_up, :n_dn], [site_i, site_i], onsite_u))
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
        println("nonPH particle numbers: N_up=$(nup), N_down=$(ndn), N_e=$(nelec)")
        if !isempty(fixed_param_values)
            fixed_param_messages = [
                "$(String(param_name))=$(fixed_param_values[param_name])"
                for param_name in sort(collect(keys(fixed_param_values)); by=String)
            ]
            println("Fixed parameters: $(join(fixed_param_messages, ", "))")
        end
        if uses_param_subset
            active_param_names = param_names[active_param_indices]
            println("Active parameters: $(join(String.(active_param_names), ", "))")
        end
    end

    update_nonph_ansatz!(
        vwf,
        param_names,
        init_params,
        lx,
        ly,
        bcx,
        bcy,
        nelec;
        nparams_proj=nparams_proj,
        nparams_backflow=nparams_backflow,
        stripe_wavevector=stripe_wavevector,
        stripe_center_offset=stripe_center_offset,
    )

    folder = "logs"
    mkpath(folder)

    if job == "SR"
        sr_params = SRParams(vmc_params=meas_params, n_steps=n_steps, lr=lr)
        exp_lr_func = build_exponential_lr_func(lr, lr_end, n_steps)

        update_vwf_func! = (vwf, params) -> begin
            full_params = uses_param_subset ?
                          merge_active_params_into_full(init_params, active_param_indices, params) :
                          params
            derivative_wf_param_names = uses_param_subset ? active_wf_param_names : nothing
            update_nonph_ansatz!(
                vwf,
                param_names,
                full_params,
                lx,
                ly,
                bcx,
                bcy,
                nelec;
                nparams_proj=nparams_proj,
                nparams_backflow=nparams_backflow,
                stripe_wavevector=stripe_wavevector,
                stripe_center_offset=stripe_center_offset,
                active_wf_param_names=derivative_wf_param_names,
            )
        end

        run_sr_optimization(
            ham,
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
            mean_dict = Dict{Symbol,Any}()
            for (key, value) in means
                mean_dict[key] = value isa Number ? real(value) : value
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
            mean_dict_str = Dict{String,Any}()
            for (key, value) in mean_dict
                mean_dict_str[String(key)] = value
            end
            open(json_file, "w") do io
                JSON.print(io, mean_dict_str)
            end
        end
    else
        error("Unknown job: $(job)")
    end

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_nonph()
end
