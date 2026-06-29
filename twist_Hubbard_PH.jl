using MPI
using Printf
using LinearAlgebra
using ArgParse
using JSON

include("twist_Hubbard.jl")

"""
用途: 解析 twist Hubbard PH/no-backflow 主程序的命令行参数。

参数:
- 无。参数来自 Julia 进程的 `ARGS`。

返回:
- `Dict{String, Any}`: `ArgParse.parse_args` 返回的参数字典。

说明:
- PH determinant 默认允许通过 `--enable_backflow` 和 `bf_*` 参数启用 PH-aware backflow。
- mean-field 固定 `chi1x = 1`, 优化 `chi1y`, `chi2`, pairing `etax/etay`, `mu`,
  以及 AFM/Stripe order。
"""
function parse_twist_ph_commandline()
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
        "--tx"
        help = "Nearest-neighbor hopping amplitude in X direction"
        arg_type = Float64
        default = 1.0
        "--ty"
        help = "Nearest-neighbor hopping amplitude in Y direction"
        arg_type = Float64
        default = 1.0
        "--t2"
        help = "Next-nearest-neighbor hopping amplitude"
        arg_type = Float64
        default = -0.2
        "--U"
        help = "On-site interaction strength"
        arg_type = Float64
        default = 8.0
        "--bcx"
        help = "Mean-field boundary factor in X direction"
        arg_type = Float64
        default = 1.001
        "--bcy"
        help = "Mean-field boundary factor in Y direction"
        arg_type = Float64
        default = 0.999
        "--etax"
        help = "Singlet pairing amplitude on X bonds"
        arg_type = Float64
        default = 0.01
        "--etay"
        help = "Singlet pairing amplitude on Y bonds"
        arg_type = Float64
        default = 0.01
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
        help = "Chemical potential in PH mean-field"
        arg_type = Float64
        default = -3.0
        "--enable_backflow"
        help = "Enable twist Hubbard PH backflow"
        arg_type = String
        default = "true"
        "--bf_epsilon"
        help = "Backflow epsilon initial value"
        arg_type = Float64
        default = 1.0
        "--bf_eta1"
        help = "Backflow eta1 initial value"
        arg_type = Float64
        default = 0.0
        "--bf_eta2"
        help = "Backflow eta2 initial value"
        arg_type = Float64
        default = 0.0
        "--bf_eta3"
        help = "Backflow eta3 initial value"
        arg_type = Float64
        default = 0.0
        "--bf_eta4"
        help = "Backflow eta4 initial value"
        arg_type = Float64
        default = 0.0
        "--target_sz"
        help = "Target total Sz"
        arg_type = Int
        default = 0
        "--nMC"
        help = "Number of Monte Carlo total samples"
        arg_type = Int
        default = 10000
        "--wMC"
        help = "Number of Monte Carlo warmup steps"
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
        help = "SR learning rate"
        arg_type = Float64
        default = 0.04
        "--lr_end"
        help = "Target learning rate at the last SR step. Default follows --lr"
        arg_type = Float64
        default = NaN
        "--eigen_cutoff"
        help = "SR eigenvalue cutoff"
        arg_type = Float64
        default = 1.0e-4
        "--init_params_json"
        help = "Path to json file that provides initial parameters"
        arg_type = String
        default = ""
        "--fixed_params"
        help = "Comma-separated fixed parameter assignments, e.g. 'etax=0.0,g=1.0'"
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
        help = "Doping level, with N_e = N_sites * (1 + doping)"
        arg_type = Float64
        default = 0.125
        "--ansatz"
        help = "Ansatz type, can be 'AFM' or 'Stripe'"
        arg_type = String
        default = "Stripe"
        "--lambda"
        help = "Stripe wavelength"
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
        "--jastrow_dx_max"
        help = "Maximum x displacement for finite-distance Jastrow. Negative means Lx/2."
        arg_type = Int
        default = -1
        "--jastrow_dy_max"
        help = "Maximum y displacement for finite-distance Jastrow. Negative means Ly/2."
        arg_type = Int
        default = -1
    end

    return parse_args(settings)
end

"""
用途: 保存 twist Hubbard PH mean-field ansatz 的参数。

字段:
- `lx, ly::Int`: 晶格尺寸。
- `bcx, bcy::Float64`: mean-field 边界条件因子。
- `chi_x, chi_y::Float64`: x/y 最近邻 hopping, 其中主程序固定 `chi_x = 1`。
- `chi2::Float64`: 对角次近邻 hopping。
- `etax, etay::Float64`: singlet pairing 裸参数。
- `mu::Float64`: uniform chemical potential, 与 `Delta_c` 共同组成 charge field。
- `delta_af::Float64`: AFM staggered field 振幅。
- `delta_c, delta_s::Float64`: charge/spin stripe field 振幅。
- `stripe_wavevector::Float64`: stripe 波矢 `Q`。
- `stripe_center_offset::Float64`: stripe 中心偏移 `x0`。
"""
struct TwistHubbardPHParams
    lx::Int
    ly::Int
    bcx::Float64
    bcy::Float64
    chi_x::Float64
    chi_y::Float64
    chi2::Float64
    etax::Float64
    etay::Float64
    mu::Float64
    delta_af::Float64
    delta_c::Float64
    delta_s::Float64
    stripe_wavevector::Float64
    stripe_center_offset::Float64
end

"""
用途: 构造 `TwistHubbardPHParams` 参数对象。

参数:
- 关键字参数与 `TwistHubbardPHParams` 字段同名。

返回:
- `TwistHubbardPHParams`: 完整的 PH mean-field 参数对象。
"""
function TwistHubbardPHParams(;
    lx::Int,
    ly::Int,
    bcx::Float64=1.0,
    bcy::Float64=1.0,
    chi_x::Float64=1.0,
    chi_y::Float64=1.0,
    chi2::Float64=0.0,
    etax::Float64=0.0,
    etay::Float64=0.0,
    mu::Float64=0.0,
    delta_af::Float64=0.0,
    delta_c::Float64=0.0,
    delta_s::Float64=0.0,
    stripe_wavevector::Float64=0.0,
    stripe_center_offset::Float64=0.0,
)::TwistHubbardPHParams
    return TwistHubbardPHParams(
        lx,
        ly,
        bcx,
        bcy,
        chi_x,
        chi_y,
        chi2,
        etax,
        etay,
        mu,
        delta_af,
        delta_c,
        delta_s,
        stripe_wavevector,
        stripe_center_offset,
    )
end

"""
用途: 计算 Stripe ansatz 下某一列的 pairing modulation。

数学公式:
- `etax0(x) = etax * abs(cos(Q / 2 * (x + 0.5 - x0)))`。
- `etay0(x) = -etay * abs(cos(Q / 2 * (x - x0)))`。

参数:
- `params::TwistHubbardPHParams`: PH mean-field 参数。
- `x::Int`: 从 1 开始的 x 方向列坐标。

返回:
- `NamedTuple`: 包含 `etax0` 与 `etay0`。
"""
function compute_twist_ph_pairing_modulation(params::TwistHubbardPHParams, x::Int)
    wavevector = params.stripe_wavevector
    center_offset = params.stripe_center_offset
    etax0 = params.etax * abs(cos(wavevector / 2 * (x + 0.5 - center_offset)))
    etay0 = -params.etay * abs(cos(wavevector / 2 * (x - center_offset)))
    return (; etax0=etax0, etay0=etay0)
end

"""
用途: 构造 twist Hubbard 的 PH singlet-pairing mean-field Hamiltonian。

数学公式:
- basis 使用 PH Nambu 表象 `row(i, up electron)=2i-1`, `row(i, down hole)=2i`。
- 最近邻 hopping 通过 `add_term_ij_PH(H, i, j, -chi, eta)` 加入。
- x/y pairing 在 AFM 中等价为 uniform; 在 Stripe 中按
  `etax0 = etax * abs(cos(Q/2 * (x + 0.5 - x0)))`,
  `etay0 = -etay * abs(cos(Q/2 * (x - x0)))` 调制。
- PH onsite field 参考 `RestrictedHubbardParams`:
  `H_{i↑,i↑} += s_i m_i/2 + rho_i/2`,
  `H_{i↓h,i↓h} += s_i m_i/2 - rho_i/2`,
  其中 `s_i = (-1)^(x+y)`, `rho_i = mu + Delta_c cos(Q(x-x0))`,
  `m_i = Delta_AF + Delta_s sin(Q/2(x-x0))`。

参数:
- `params::TwistHubbardPHParams`: twist Hubbard PH mean-field 参数。

返回:
- `Hermitian{Float64, Matrix{Float64}}`: 维度为 `(2N_sites, 2N_sites)` 的 Hamiltonian。
"""
function build_twist_hubbard_ph_hamiltonian(
    params::TwistHubbardPHParams,
)
    lx = params.lx
    ly = params.ly
    n_sites = lx * ly
    hamiltonian = zeros(Float64, 2 * n_sites, 2 * n_sites)

    for x in 1:lx
        charge_field_x = params.mu + params.delta_c * cos(params.stripe_wavevector * (x - params.stripe_center_offset))
        spin_field_x = params.delta_af + params.delta_s * sin(params.stripe_wavevector / 2 * (x - params.stripe_center_offset))
        pairing = compute_twist_ph_pairing_modulation(params, x)

        for y in 1:ly
            site_i = twist_site_index(x, y, ly)
            site_x = twist_site_index(x == lx ? 1 : x + 1, y, ly)
            site_y = twist_site_index(x, y == ly ? 1 : y + 1, ly)
            site_pp = twist_site_index(x == lx ? 1 : x + 1, y == ly ? 1 : y + 1, ly)
            site_mp = twist_site_index(x == 1 ? lx : x - 1, y == ly ? 1 : y + 1, ly)
            bc_x = x == lx ? params.bcx : 1.0
            bc_y = y == ly ? params.bcy : 1.0
            bc_pp = ((x == lx) ? params.bcx : 1.0) * ((y == ly) ? params.bcy : 1.0)
            bc_mp = ((x == 1) ? params.bcx : 1.0) * ((y == ly) ? params.bcy : 1.0)

            add_term_ij_PH(hamiltonian, site_i, site_x, -params.chi_x * bc_x, pairing.etax0 * bc_x)
            add_term_ij_PH(hamiltonian, site_i, site_y, -params.chi_y * bc_y, pairing.etay0 * bc_y)
            add_term_ij_PH(hamiltonian, site_i, site_pp, -params.chi2 * bc_pp, 0.0)
            add_term_ij_PH(hamiltonian, site_i, site_mp, -params.chi2 * bc_mp, 0.0)

            row_up = 2 * site_i - 1
            row_down_hole = 2 * site_i
            staggered_sign = (-1)^(x + y)
            hamiltonian[row_up, row_up] += staggered_sign * spin_field_x / 2 + charge_field_x / 2
            hamiltonian[row_down_hole, row_down_hole] += staggered_sign * spin_field_x / 2 - charge_field_x / 2
        end
    end

    return Hermitian(hamiltonian + hamiltonian')
end

"""
用途: 构造 twist Hubbard PH Hamiltonian 对单个优化参数的导数矩阵。

数学公式:
- 当前参数都线性进入 Hamiltonian; `dH/dp` 通过只令目标参数为 `1.0`,
  其它可变 mean-field 参数为 `0.0`, 并保持 `Q, x0, bcx, bcy` 不变得到。

参数:
- `params::TwistHubbardPHParams`: 当前 ansatz 参数, 提供尺寸、边界和 stripe 几何。
- `param_name::Symbol`: 目标参数名, 支持 `:chi1y`, `:chi2`, `:etax`, `:etay`,
  `:Delta_AF`, `:Delta_c`, `:Delta_s`, `:mu`。

返回:
- `Matrix{Float64}`: 与 Hamiltonian 同维度的 `dH/dp` 矩阵。
"""
function build_twist_hubbard_ph_dh_dparam(
    params::TwistHubbardPHParams,
    param_name::Symbol,
)::Matrix{Float64}
    allowed_param_names = (:chi1y, :chi2, :etax, :etay, :Delta_AF, :Delta_c, :Delta_s, :mu)
    if !(param_name in allowed_param_names)
        error("Unknown twist Hubbard PH mean-field parameter: $(param_name).")
    end

    derivative_params = TwistHubbardPHParams(
        lx=params.lx,
        ly=params.ly,
        bcx=params.bcx,
        bcy=params.bcy,
        chi_x=0.0,
        chi_y=param_name == :chi1y ? 1.0 : 0.0,
        chi2=param_name == :chi2 ? 1.0 : 0.0,
        etax=param_name == :etax ? 1.0 : 0.0,
        etay=param_name == :etay ? 1.0 : 0.0,
        mu=param_name == :mu ? 1.0 : 0.0,
        delta_af=param_name == :Delta_AF ? 1.0 : 0.0,
        delta_c=param_name == :Delta_c ? 1.0 : 0.0,
        delta_s=param_name == :Delta_s ? 1.0 : 0.0,
        stripe_wavevector=params.stripe_wavevector,
        stripe_center_offset=params.stripe_center_offset,
    )
    return Matrix(build_twist_hubbard_ph_hamiltonian(derivative_params))
end

"""
用途: 对角化 twist Hubbard PH Hamiltonian 并生成 determinant orbitals 及参数导数。

参数:
- `params::TwistHubbardPHParams`: PH mean-field 参数。
- `param_names::Vector{Symbol}`: 需要计算导数的 mean-field 参数名。
- `target_sz::Int`: 目标 total Sz, PH determinant 占据轨道数为 `N_sites + target_sz`。

返回:
- `Tuple`: `(epsilon, occupied_orbitals, d_ut_params)`。
  `epsilon` 是全部本征值, `occupied_orbitals` 是 `(2N_sites, N_sites+target_sz)` 矩阵,
  `d_ut_params` 是参数名到转置导数矩阵 `dU'` 的映射。
"""
function make_twist_hubbard_ph_ansatz_and_derivs(
    params::TwistHubbardPHParams;
    param_names::Vector{Symbol}=Symbol[],
    target_sz::Int=0,
)
    hamiltonian = Matrix(build_twist_hubbard_ph_hamiltonian(params))
    hamiltonian_derivatives = Dict{Symbol,Matrix{Float64}}()
    for param_name in param_names
        hamiltonian_derivatives[param_name] = build_twist_hubbard_ph_dh_dparam(params, param_name)
    end

    epsilon, full_orbitals, _, orbital_derivatives = compute_eig_and_dU_reg1(
        hamiltonian,
        hamiltonian_derivatives,
    )
    n_sites = params.lx * params.ly
    n_occupied_orbitals = n_sites + target_sz
    if n_occupied_orbitals < 0 || n_occupied_orbitals > size(full_orbitals, 2)
        error("PH occupied orbitals N_sites + target_sz = $(n_occupied_orbitals) is outside 0:$(size(full_orbitals, 2)).")
    end

    if is_twist_root_rank()
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
用途: 根据参数更新 twist Hubbard PH determinant 波函数和参数导数。

参数:
- `vwf`: determinant 波函数对象。
- `param_names::Vector{Symbol}`: 完整参数名列表, 顺序为 mean-field, projector, backflow。
- `params::Vector{Float64}`: 与 `param_names` 对齐的完整参数值。
- `lx, ly::Int`: 晶格尺寸。
- `bcx, bcy::Float64`: mean-field 边界条件因子。
- `tx, ty::Float64`: physical hopping, 保留用于接口兼容; mean-field gauge 固定为 `chi_x=1`, `chi_y=chi1y`。
- `target_sz::Int`: 目标 total Sz。
- `nparams_proj::Int`: projector 参数数量。
- `nparams_backflow::Int`: backflow 参数数量。
- `stripe_wavevector::Float64`: stripe 波矢 `Q`。
- `stripe_center_offset::Float64`: stripe 中心偏移 `x0`。
- `active_wf_param_names::Union{Nothing, Vector{Symbol}}`: 参与求导和 SR 的 mean-field 参数名; `nothing` 表示全部。

返回:
- `nothing`。
"""
function update_twist_ph_ansatz!(
    vwf,
    param_names::Vector{Symbol},
    params::Vector{Float64},
    lx::Int,
    ly::Int,
    bcx::Float64,
    bcy::Float64,
    tx::Float64,
    ty::Float64,
    target_sz::Int;
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
    projector_param_names = nparams_proj > 0 ? param_names[(nparams_wf+1):(nparams_wf+nparams_proj)] : Symbol[]
    projector_param_values = nparams_proj > 0 ? params[(nparams_wf+1):(nparams_wf+nparams_proj)] : Float64[]
    backflow_param_names = nparams_backflow > 0 ? param_names[(nparams_wf+nparams_proj+1):total_param_count] : Symbol[]
    backflow_param_values = nparams_backflow > 0 ? params[(nparams_wf+nparams_proj+1):total_param_count] : Float64[]

    derivative_wf_param_names = active_wf_param_names === nothing ? wf_param_names : active_wf_param_names
    wf_param_name_set = Set(wf_param_names)
    for derivative_param_name in derivative_wf_param_names
        if !(derivative_param_name in wf_param_name_set)
            error("Active mean-field parameter $(derivative_param_name) is not in full mean-field parameter list.")
        end
    end

    param_map = Dict{Symbol,Float64}(zip(wf_param_names, wf_param_values))
    twist_params = TwistHubbardPHParams(
        lx=lx,
        ly=ly,
        bcx=bcx,
        bcy=bcy,
        chi_x=1.0,
        chi_y=get(param_map, :chi1y, compute_twist_chi1y_initial_value(tx, ty)),
        chi2=get(param_map, :chi2, 0.0),
        etax=get(param_map, :etax, 0.0),
        etay=get(param_map, :etay, 0.0),
        mu=get(param_map, :mu, 0.0),
        delta_af=get(param_map, :Delta_AF, 0.0),
        delta_c=get(param_map, :Delta_c, 0.0),
        delta_s=get(param_map, :Delta_s, 0.0),
        stripe_wavevector=stripe_wavevector,
        stripe_center_offset=stripe_center_offset,
    )

    _, gs_u, d_ut_params = make_twist_hubbard_ph_ansatz_and_derivs(
        twist_params;
        param_names=derivative_wf_param_names,
        target_sz=target_sz,
    )

    copyto!(vwf.base_gs_U, gs_u)
    copyto!(vwf.gs_U, gs_u)
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
用途: 根据 `ansatz` 参数生成 twist Hubbard PH mean-field 参数名、初值和 stripe 几何。

参数:
- `args::Dict{String, Any}`: 命令行参数字典, 或测试中构造的同结构字典。

返回:
- `NamedTuple`: 包含 `wf_param_names`, `wf_init_params`, `stripe_wavevector`, `stripe_center_offset`。
"""
function build_twist_ph_mean_field_parameter_setup(args)
    ansatz = args["ansatz"]
    chi1y_initial_value = compute_twist_chi1y_initial_value(args["tx"], args["ty"])
    chi2_initial_value = compute_twist_chi2_initial_value(args["tx"], args["t2"])
    if ansatz == "AFM"
        return (
            wf_param_names=[:chi1y, :chi2, :etax, :etay, :Delta_AF, :mu],
            wf_init_params=[
                chi1y_initial_value,
                chi2_initial_value,
                args["etax"],
                args["etay"],
                args["Delta_AF"],
                args["mu"],
            ],
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
            wf_param_names=[:chi1y, :chi2, :etax, :etay, :Delta_c, :Delta_s, :mu],
            wf_init_params=[
                chi1y_initial_value,
                chi2_initial_value,
                args["etax"],
                args["etay"],
                args["Delta_c"],
                args["Delta_s"],
                args["mu"],
            ],
            stripe_wavevector=2π / lambda,
            stripe_center_offset=stripe_center_offset,
        )
    end
    error("Unknown ansatz type: $(ansatz)")
end

"""
用途: 运行 twist Hubbard PH/no-backflow VMC/SR 主流程。

参数:
- 无。所有配置来自命令行参数。

返回:
- `nothing`。

说明:
- sampler 使用 `config_Hubbard(...; ifPH=true)`, determinant row set 为 up electron 加 down hole。
- determinant 列数取 `N_sites + target_sz`。
- 第一阶段固定使用 `NoBackflowTerm()`, 不构造或优化 backflow 参数。
"""
function main_twist_ph()::Nothing
    args = parse_twist_ph_commandline()

    session = init_mpi_session()
    rank = session.rank
    is_root = rank == session.root

    lx = args["Lx"]
    ly = args["Ly"]
    bcx = args["bcx"]
    bcy = args["bcy"]
    tx = args["tx"]
    ty = args["ty"]
    t2 = args["t2"]
    onsite_u = args["U"]
    target_sz = args["target_sz"]
    doping = args["doping"]
    nmc = args["nMC"]
    wmc = args["wMC"]
    rmc = args["rMC"]
    dmc = args["dMC"]
    n_steps = args["nSR"]
    lr = args["lr"]
    lr_end = isnan(args["lr_end"]) ? lr : args["lr_end"]
    eigen_cutoff = args["eigen_cutoff"]
    job = args["job"]
    init_params_json = args["init_params_json"]
    fixed_params_string = args["fixed_params"]
    active_params_string = args["active_params"]
    n_sites = lx * ly
    jastrow_dx_max = args["jastrow_dx_max"] < 0 ? div(lx, 2) : args["jastrow_dx_max"]
    jastrow_dy_max = args["jastrow_dy_max"] < 0 ? div(ly, 2) : args["jastrow_dy_max"]

    mean_field_setup = build_twist_ph_mean_field_parameter_setup(args)
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

    projector = build_twist_projector(
        lx,
        ly,
        args["g"];
        jastrow_dx_max=jastrow_dx_max,
        jastrow_dy_max=jastrow_dy_max,
    )
    hopping_bonds = build_twist_nearest_neighbor_bonds(lx, ly)
    source_bonds, source_amplitudes = build_twist_backflow_source_data(hopping_bonds, tx, ty, t2)
    backflow = build_twist_optional_backflow(
        parse_twist_bool_flag(args["enable_backflow"], "--enable_backflow"),
        source_bonds,
        source_amplitudes,
        args["bf_epsilon"],
        args["bf_eta1"],
        args["bf_eta2"],
        args["bf_eta3"],
        args["bf_eta4"];
        particle_hole_lower_block=true,
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
        init_params = build_twist_init_params_from_json_with_defaults(init_params_json, param_names, init_params)
        if is_root
            println("Loaded initial parameters from json: $(init_params_json)")
        end
    end

    fixed_param_values = parse_twist_fixed_param_string(fixed_params_string)
    requested_active_param_names = parse_twist_param_name_list(active_params_string)
    if !isempty(fixed_param_values)
        init_params = apply_twist_fixed_params_to_values(param_names, init_params, fixed_param_values)
    end
    active_param_indices = build_twist_active_param_indices(param_names, fixed_param_values, requested_active_param_names)
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
    set_active_twist_projector_derivative_param_names!(
        proj_param_names;
        active_projector_param_names=uses_param_subset ? active_projector_param_names : nothing,
    )
    set_active_twist_backflow_derivative_param_names!(
        backflow_param_name_list;
        active_backflow_param_names=uses_param_subset ? active_backflow_param_names : nothing,
    )

    term_setup = build_twist_hamiltonian_terms(lx, ly, tx, ty, t2, onsite_u)
    ham = GeneralModel(n_sites, term_setup.all_terms)

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
    n_occupied_orbitals = n_sites + target_sz
    if n_occupied_orbitals < 0 || n_occupied_orbitals > 2 * n_sites
        error("Invalid PH determinant column count N_sites + target_sz = $(n_occupied_orbitals).")
    end

    sampler = config_Hubbard(n_sites, nup, ndn; ifPH=true)
    init_config_Hubbard!(sampler)

    vwf = vwf_det(zeros(Float64, 2 * n_sites, n_occupied_orbitals), sampler; backflow=backflow)
    set_projector!(vwf, projector)
    kernel = HubbardKernel(conserve_sz=true)

    if is_root
        println("Initial parameters: $(init_params)")
        println("twist Hubbard PH particle numbers: N_up=$(nup), N_down=$(ndn), N_e=$(nelec), N_occ_PH=$(n_occupied_orbitals)")
        println("Nearest-neighbor hopping: tx=$(tx), ty=$(ty)")
        println("Backflow enabled: $(mfVMC.Backflow.uses_backflow(backflow))")
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

    update_twist_ph_ansatz!(
        vwf,
        param_names,
        init_params,
        lx,
        ly,
        bcx,
        bcy,
        tx,
        ty,
        target_sz;
        nparams_proj=nparams_proj,
        nparams_backflow=nparams_backflow,
        stripe_wavevector=stripe_wavevector,
        stripe_center_offset=stripe_center_offset,
    )

    folder = "logs"
    mkpath(folder)

    if job == "SR"
        sr_params = build_twist_sr_params(meas_params, n_steps, lr, eigen_cutoff)
        exp_lr_func = build_twist_exponential_lr_func(lr, lr_end, n_steps)

        update_vwf_func! = (vwf, params) -> begin
            full_params = uses_param_subset ?
                          merge_twist_active_params_into_full(init_params, active_param_indices, params) :
                          params
            derivative_wf_param_names = uses_param_subset ? active_wf_param_names : nothing
            update_twist_ph_ansatz!(
                vwf,
                param_names,
                full_params,
                lx,
                ly,
                bcx,
                bcy,
                tx,
                ty,
                target_sz;
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
            append_twist_inactive_params_to_json!(
                joinpath(folder, "min_params.json"),
                param_names,
                init_params,
                active_param_indices,
            )
        end
    elseif job == "measure"
        results = run_simulation(
            ham,
            vwf,
            kernel,
            definition_twist_observables(
                lx,
                ly;
                hopping_terms=term_setup.hopping_terms,
                interaction_terms=term_setup.interaction_terms,
                onsite_u=onsite_u,
            ),
            meas_params;
            history_observables=[:E, :E_hop, :E_int, :E_int_charge, :E_int_spin],
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

            mean_dict_str = Dict{String,Any}()
            for (key, value) in mean_dict
                mean_dict_str[String(key)] = value
            end
            open(joinpath(folder, "block_binning_mean.json"), "w") do io
                JSON.print(io, mean_dict_str)
            end
        end
    else
        error("Unknown job: $(job)")
    end

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_twist_ph()
end
