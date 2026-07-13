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
import mfVMC.Timing: @timed, ENABLE_TIMING, timing_reset!, timing_report

if !isdefined(@__MODULE__, :build_emery_general_model)
    include("Emery.jl")
end

"""
用途: 判断当前进程是否为 MPI root rank。

参数:
- 无。

返回:
- `Bool`: 未启用 MPI 或当前 rank 为 0 时返回 `true`。
"""
function is_emery_ph_root_rank()::Bool
    try
        if isdefined(mfVMC, :MPI) && mfVMC.MPI.Initialized()
            return mfVMC.MPI.Comm_rank(mfVMC.MPI.COMM_WORLD) == 0
        end
    catch
        return true
    end
    return true
end

struct ColumnEmeryPHParams
    lx::Int
    ly::Int
    bcx::Float64
    bcy::Float64
    chi1_dd::Float64
    chi1_dp::Float64
    chi1_pp::Float64
    etax_dd::Dict{Symbol,Float64}
    etay_dd::Dict{Symbol,Float64}
    mud::Dict{Symbol,Float64}
    mupx::Dict{Symbol,Float64}
    mupy::Dict{Symbol,Float64}
    mzd::Dict{Symbol,Float64}
    mzpx::Dict{Symbol,Float64}
    mzpy::Dict{Symbol,Float64}
end

"""
用途: 构造 column-resolved PH Emery mean-field 参数对象。

参数:
- `lx, ly::Int`: Cu 晶胞在 x/y 方向的数量。
- `bcx, bcy::Float64`: 边界条件因子, 当前 x 方向固定 OBC, `bcx` 仅保留为兼容字段。
- `chi1_dd, chi1_dp, chi1_pp::Float64`: Cu-Cu, Cu-O, O-O normal hopping 参数。
- `etax_dd, etay_dd::Dict{Symbol, Float64}`: Cu-Cu pairing 参数, 键名为 `etax_dd_x`, `etay_dd_x`。
- `mud/mupx/mupy/mzd/mzpx/mzpy`: 各 orbital 的 column-resolved onsite 参数。

返回:
- `ColumnEmeryPHParams`: 可用于构造 PH Nambu Hamiltonian 的参数对象。
"""
function ColumnEmeryPHParams(;
    lx::Int,
    ly::Int,
    bcx::Float64=1.0,
    bcy::Float64=1.0,
    chi1_dd::Float64=0.0,
    chi1_dp::Float64=0.0,
    chi1_pp::Float64=0.0,
    etax_dd::Dict{Symbol,Float64}=Dict{Symbol,Float64}(),
    etay_dd::Dict{Symbol,Float64}=Dict{Symbol,Float64}(),
    mud::Dict{Symbol,Float64}=Dict{Symbol,Float64}(),
    mupx::Dict{Symbol,Float64}=Dict{Symbol,Float64}(),
    mupy::Dict{Symbol,Float64}=Dict{Symbol,Float64}(),
    mzd::Dict{Symbol,Float64}=Dict{Symbol,Float64}(),
    mzpx::Dict{Symbol,Float64}=Dict{Symbol,Float64}(),
    mzpy::Dict{Symbol,Float64}=Dict{Symbol,Float64}(),
)::ColumnEmeryPHParams
    lx > 0 || error("lx must be positive, got $(lx).")
    ly > 0 || error("ly must be positive, got $(ly).")
    return ColumnEmeryPHParams(
        lx,
        ly,
        bcx,
        bcy,
        chi1_dd,
        chi1_dp,
        chi1_pp,
        etax_dd,
        etay_dd,
        mud,
        mupx,
        mupy,
        mzd,
        mzpx,
        mzpy,
    )
end

"""
用途: 计算 column Emery PH determinant 的占据轨道数。

数学公式:
- PH basis 使用 `(up electron, down hole)` 两个 row block。
- determinant 列数固定为 `N_sites + target_sz`, 其中 `N_sites = 3LxLy + Ly`。

参数:
- `lx, ly::Int`: Emery Cu 晶胞尺寸。
- `target_sz::Int`: 目标 total Sz。

返回:
- `Int`: PH determinant 占据轨道数。
"""
function compute_column_emery_ph_determinant_orbital_count(
    lx::Int,
    ly::Int,
    target_sz::Int,
)::Int
    n_sites = emery_n_sites(lx, ly)
    n_occupied_orbitals = n_sites + target_sz
    if n_occupied_orbitals < 0 || n_occupied_orbitals > 2 * n_sites
        error("Invalid PH occupied orbital count: $(n_occupied_orbitals), expected 0 <= count <= $(2 * n_sites).")
    end
    return n_occupied_orbitals
end

"""
用途: 在 PH Nambu Hamiltonian 中加入单个 Emery onsite 项。

参数:
- `hamiltonian::AbstractMatrix`: 待原地更新的 Hamiltonian。
- `site::Int`: 空间 site index, 1-based。
- `mu_value, mz_value::Float64`: onsite chemical potential 和 staggered field。
- `staggered_sign::Float64`: `(-1)^(x+y)`。

返回:
- `AbstractMatrix`: 原地修改后的 Hamiltonian。

公式:
- 因为 PH Hamiltonian 最后用 `H + H'` 闭合 Hermitian, 这里写入半强度:
  `H_{i up,i up} += (mu_i + s_i mz_i) / 2`,
  `H_{i down-hole,i down-hole} += (-mu_i + s_i mz_i) / 2`。
- 闭合后对角项变为 `mu_i + s_i mz_i` 和 `-mu_i + s_i mz_i`。
"""
function add_emery_ph_onsite!(
    hamiltonian::AbstractMatrix,
    site::Int,
    mu_value::Float64,
    mz_value::Float64,
    staggered_sign::Float64,
)
    hamiltonian[emery_spin_index(site, 1), emery_spin_index(site, 1)] += 0.5 * (mu_value + staggered_sign * mz_value)
    hamiltonian[emery_spin_index(site, 2), emery_spin_index(site, 2)] += 0.5 * (-mu_value + staggered_sign * mz_value)
    return hamiltonian
end

"""
用途: 构造 column-resolved PH Emery Nambu mean-field Hamiltonian。

参数:
- `params::ColumnEmeryPHParams`: column-resolved PH Emery mean-field 参数。

返回:
- `Hermitian{Float64, Matrix{Float64}}`: 维度为 `2N_sites x 2N_sites` 的 PH Hamiltonian。

公式:
- Nambu basis 为 `(c_{up}, c^dagger_{down})`。
- normal hopping 通过 `add_term_ij_PH(H, i, j, chi, eta)` 中的 `chi` 加入。
- Cu-Cu pairing 只放在 dd bond 上: x bond 使用 `+etax_dd_x`, y bond 使用 `-etay_dd_x`。
- Cu-O 和 O-O bond 的 pairing 为 0。
"""
function build_column_emery_ph_hamiltonian(
    params::ColumnEmeryPHParams,
)
    lx = params.lx
    ly = params.ly
    n_sites = emery_n_sites(lx, ly)
    hamiltonian = zeros(Float64, 2 * n_sites, 2 * n_sites)

    for x in 1:lx, y in 1:ly
        site_i = Emery_xyo_to_site_index(x, y, EMERY_ORB_D, lx, ly)
        if x < lx
            site_x = Emery_xyo_to_site_index(x + 1, y, EMERY_ORB_D, lx, ly)
            etax_value = get(params.etax_dd, Symbol("etax_dd_$(x)"), 0.0)
            add_term_ij_PH(hamiltonian, site_i, site_x, params.chi1_dd, etax_value)
        end
        y_next = y == ly ? 1 : y + 1
        bc_y = y == ly ? params.bcy : 1.0
        site_y = Emery_xyo_to_site_index(x, y_next, EMERY_ORB_D, lx, ly)
        etay_value = -get(params.etay_dd, Symbol("etay_dd_$(x)"), 0.0)
        add_term_ij_PH(hamiltonian, site_i, site_y, params.chi1_dd * bc_y, etay_value * bc_y)
    end

    for bond in build_emery_pd_bonds(lx, ly; amplitude=params.chi1_dp, bcy=params.bcy)
        add_term_ij_PH(hamiltonian, bond.i, bond.j, bond.coef, 0.0)
    end
    for bond in build_emery_pp_bonds(lx, ly; amplitude=params.chi1_pp, bcy=params.bcy)
        add_term_ij_PH(hamiltonian, bond.i, bond.j, bond.coef, 0.0)
    end

    for x in 1:lx, y in 1:ly
        staggered_sign = Float64((-1)^(x + y))
        d_site = Emery_xyo_to_site_index(x, y, EMERY_ORB_D, lx, ly)
        py_site = Emery_xyo_to_site_index(x, y, EMERY_ORB_PY, lx, ly)
        add_emery_ph_onsite!(
            hamiltonian,
            d_site,
            get(params.mud, Symbol("mud_$(x)"), 0.0),
            get(params.mzd, Symbol("mzd_$(x)"), 0.0),
            staggered_sign,
        )
        add_emery_ph_onsite!(
            hamiltonian,
            py_site,
            get(params.mupy, Symbol("mupy_$(x)"), 0.0),
            get(params.mzpy, Symbol("mzpy_$(x)"), 0.0),
            staggered_sign,
        )
    end
    for x in 0:lx, y in 1:ly
        staggered_sign = Float64((-1)^(x + y))
        px_site = Emery_xyo_to_site_index(x, y, EMERY_ORB_PX, lx, ly)
        add_emery_ph_onsite!(
            hamiltonian,
            px_site,
            get(params.mupx, Symbol("mupx_$(x)"), 0.0),
            get(params.mzpx, Symbol("mzpx_$(x)"), 0.0),
            staggered_sign,
        )
    end

    return Hermitian(hamiltonian + hamiltonian')
end

"""
用途: 构造 column-resolved PH Emery Hamiltonian 对单个参数的导数矩阵。

参数:
- `params::ColumnEmeryPHParams`: 当前 mean-field 参数。
- `param_name::Symbol`: 参数名, 支持 hopping, dd pairing 和按列 onsite 参数。

返回:
- `Matrix{Float64}`: `dH/dp` 矩阵。
"""
function build_column_emery_ph_dh_dparam(
    params::ColumnEmeryPHParams,
    param_name::Symbol,
)::Matrix{Float64}
    name_string = String(param_name)
    derivative_params_kwargs = Dict{Symbol,Any}(
        :lx => params.lx,
        :ly => params.ly,
        :bcx => params.bcx,
        :bcy => params.bcy,
    )

    if param_name in (:chi1_dd, :chi1_dp, :chi1_pp)
        derivative_params_kwargs[param_name] = 1.0
    elseif startswith(name_string, "etax_dd_")
        derivative_params_kwargs[:etax_dd] = Dict(param_name => 1.0)
    elseif startswith(name_string, "etay_dd_")
        derivative_params_kwargs[:etay_dd] = Dict(param_name => 1.0)
    elseif startswith(name_string, "mud_")
        derivative_params_kwargs[:mud] = Dict(param_name => 1.0)
    elseif startswith(name_string, "mzd_")
        derivative_params_kwargs[:mzd] = Dict(param_name => 1.0)
    elseif startswith(name_string, "mupx_")
        derivative_params_kwargs[:mupx] = Dict(param_name => 1.0)
    elseif startswith(name_string, "mzpx_")
        derivative_params_kwargs[:mzpx] = Dict(param_name => 1.0)
    elseif startswith(name_string, "mupy_")
        derivative_params_kwargs[:mupy] = Dict(param_name => 1.0)
    elseif startswith(name_string, "mzpy_")
        derivative_params_kwargs[:mzpy] = Dict(param_name => 1.0)
    else
        error("Unknown column Emery PH mean-field parameter: $(param_name).")
    end

    derivative_params = ColumnEmeryPHParams(; derivative_params_kwargs...)
    return Matrix(build_column_emery_ph_hamiltonian(derivative_params))
end

"""
用途: 从 mean-field 参数名和值构造 `ColumnEmeryPHParams`。

参数:
- `wf_param_names::Vector{Symbol}`: mean-field 参数名。
- `wf_param_values::Vector{Float64}`: 与参数名一一对应的参数值。
- `lx, ly::Int`: Cu 晶胞尺寸。
- `bcx, bcy::Float64`: 边界条件因子。
- `fixed_chi1_dp::Float64`: 当 `:chi1_dp` 不在优化参数中时使用的固定 Cu-O hopping。

返回:
- `ColumnEmeryPHParams`: 可直接用于构造 PH Emery mean-field Hamiltonian 的参数对象。
"""
function build_column_emery_ph_params_from_wf_params(
    wf_param_names::Vector{Symbol},
    wf_param_values::Vector{Float64},
    lx::Int,
    ly::Int,
    bcx::Float64,
    bcy::Float64;
    fixed_chi1_dp::Float64=0.0,
)::ColumnEmeryPHParams
    length(wf_param_names) == length(wf_param_values) ||
        error("wf_param_names and wf_param_values must have the same length.")

    param_map = Dict{Symbol,Float64}(zip(wf_param_names, wf_param_values))
    etax_dd = Dict{Symbol,Float64}()
    etay_dd = Dict{Symbol,Float64}()
    mud = Dict{Symbol,Float64}()
    mzd = Dict{Symbol,Float64}()
    mupx = Dict{Symbol,Float64}()
    mzpx = Dict{Symbol,Float64}()
    mupy = Dict{Symbol,Float64}()
    mzpy = Dict{Symbol,Float64}()

    for (param_name, param_value) in param_map
        name_string = String(param_name)
        if startswith(name_string, "etax_dd_")
            etax_dd[param_name] = param_value
        elseif startswith(name_string, "etay_dd_")
            etay_dd[param_name] = param_value
        elseif startswith(name_string, "mud_")
            mud[param_name] = param_value
        elseif startswith(name_string, "mzd_")
            mzd[param_name] = param_value
        elseif startswith(name_string, "mupx_")
            mupx[param_name] = param_value
        elseif startswith(name_string, "mzpx_")
            mzpx[param_name] = param_value
        elseif startswith(name_string, "mupy_")
            mupy[param_name] = param_value
        elseif startswith(name_string, "mzpy_")
            mzpy[param_name] = param_value
        elseif param_name in (:chi1_dd, :chi1_dp, :chi1_pp)
            continue
        else
            error("Unknown column Emery PH mean-field parameter: $(param_name).")
        end
    end

    return ColumnEmeryPHParams(
        lx=lx,
        ly=ly,
        bcx=bcx,
        bcy=bcy,
        chi1_dd=get(param_map, :chi1_dd, 0.0),
        chi1_dp=get(param_map, :chi1_dp, fixed_chi1_dp),
        chi1_pp=get(param_map, :chi1_pp, 0.0),
        etax_dd=etax_dd,
        etay_dd=etay_dd,
        mud=mud,
        mupx=mupx,
        mupy=mupy,
        mzd=mzd,
        mzpx=mzpx,
        mzpy=mzpy,
    )
end

"""
用途: 计算 Emery stripe 初始态中的 Cu-Cu pairing envelope。

参数:
- `x_coordinate::Float64`: pairing bond 使用的 x 坐标。
- `lambda::Int`: stripe 电荷调制周期 `lambda`。
- `stripe_center::AbstractString`: stripe 中心类型, 支持 `site` 或 `bond`。
- `pairing_amp::Float64`: pairing 调制振幅。
- `stripe_spin_peak_x::Float64`: spin envelope 峰值位置, `NaN` 时使用 `stripe_center`。

返回:
- `Float64`: `pairing_amp * abs(cos(Q / 2 * (x_coordinate - x0)))`, `Q = 2pi / lambda`。
"""
function compute_emery_stripe_dd_pairing_value(
    x_coordinate::Float64,
    lambda::Int,
    stripe_center::AbstractString,
    pairing_amp::Float64,
    stripe_spin_peak_x::Float64,
)::Float64
    lambda > 0 || error("lambda must be positive.")
    stripe_center_offset = if isnan(stripe_spin_peak_x)
        get_emery_stripe_center_offset(stripe_center)
    else
        stripe_spin_peak_x - lambda / 2.0
    end
    stripe_wave_vector = 2.0 * pi / lambda
    return pairing_amp * abs(cos(stripe_wave_vector / 2.0 * (x_coordinate - stripe_center_offset)))
end

"""
用途: 生成 column-resolved PH Emery mean-field 参数名和初值。

参数:
- `ansatz::AbstractString`: `AFM` 或 `Stripe`。
- `lx, lambda::Int`: Cu 晶胞 x 方向长度和 stripe 周期。
- `stripe_center::AbstractString`: `site` 或 `bond`。
- `mu_uniform, stripe_mu_amp, mz_amp::Float64`: 初态 onsite 参数。
- `chi1_dd, chi1_dp, chi1_pp::Float64`: mean-field hopping 初值, 其中 `chi1_dp` 固定为 gauge。
- `etax_dd, etay_dd::Float64`: Cu-Cu x/y pairing 初值或调制振幅。
- `stripe_spin_peak_x::Float64`: spin envelope 峰值位置, `NaN` 时使用 `stripe_center`。

返回:
- `NamedTuple`: 包含 `wf_param_names`, `wf_init_params`, `fixed_chi1_dp`。
"""
function build_column_emery_ph_mean_field_parameter_setup(
    ansatz::AbstractString,
    lx::Int,
    lambda::Int,
    stripe_center::AbstractString,
    mu_uniform::Float64,
    stripe_mu_amp::Float64,
    mz_amp::Float64,
    chi1_dd::Float64,
    chi1_dp::Float64,
    chi1_pp::Float64,
    etax_dd::Float64,
    etay_dd::Float64,
    stripe_spin_peak_x::Float64,
)
    wf_param_names = Symbol[:chi1_dd, :chi1_pp]
    wf_init_params = Float64[chi1_dd, chi1_pp]

    for x in 1:lx
        push!(wf_param_names, Symbol("etax_dd_$(x)"))
        push!(wf_param_names, Symbol("etay_dd_$(x)"))
        if ansatz == "Stripe"
            push!(wf_init_params, compute_emery_stripe_dd_pairing_value(Float64(x) + 0.5, lambda, stripe_center, etax_dd, stripe_spin_peak_x))
            push!(wf_init_params, compute_emery_stripe_dd_pairing_value(Float64(x), lambda, stripe_center, etay_dd, stripe_spin_peak_x))
        elseif ansatz == "AFM"
            push!(wf_init_params, etax_dd)
            push!(wf_init_params, etay_dd)
        else
            error("Unknown ansatz type: $(ansatz).")
        end
    end

    if ansatz == "Stripe"
        for x in 1:lx
            x_coordinate = Float64(x)
            push!(wf_param_names, Symbol("mud_$(x)"))
            push!(wf_param_names, Symbol("mzd_$(x)"))
            push!(wf_param_names, Symbol("mupy_$(x)"))
            push!(wf_param_names, Symbol("mzpy_$(x)"))
            push!(wf_init_params, compute_emery_stripe_mu_value(x_coordinate, lambda, stripe_center, mu_uniform, stripe_mu_amp, stripe_spin_peak_x))
            push!(wf_init_params, compute_emery_stripe_mzd_value(x_coordinate, lambda, stripe_center, mz_amp, stripe_spin_peak_x))
            push!(wf_init_params, compute_emery_stripe_mu_value(x_coordinate, lambda, stripe_center, mu_uniform, stripe_mu_amp, stripe_spin_peak_x))
            push!(wf_init_params, 0.0)
        end
        for x in 0:lx
            x_coordinate = Float64(x) + 0.5
            push!(wf_param_names, Symbol("mupx_$(x)"))
            push!(wf_param_names, Symbol("mzpx_$(x)"))
            push!(wf_init_params, compute_emery_stripe_mu_value(x_coordinate, lambda, stripe_center, mu_uniform, stripe_mu_amp, stripe_spin_peak_x))
            push!(wf_init_params, 0.0)
        end
    elseif ansatz == "AFM"
        for x in 1:lx
            push!(wf_param_names, Symbol("mud_$(x)"))
            push!(wf_param_names, Symbol("mzd_$(x)"))
            push!(wf_param_names, Symbol("mupy_$(x)"))
            push!(wf_param_names, Symbol("mzpy_$(x)"))
            push!(wf_init_params, mu_uniform)
            push!(wf_init_params, mz_amp)
            push!(wf_init_params, mu_uniform)
            push!(wf_init_params, 0.0)
        end
        for x in 0:lx
            push!(wf_param_names, Symbol("mupx_$(x)"))
            push!(wf_param_names, Symbol("mzpx_$(x)"))
            push!(wf_init_params, mu_uniform)
            push!(wf_init_params, 0.0)
        end
    end
    return (; wf_param_names=wf_param_names, wf_init_params=wf_init_params, fixed_chi1_dp=chi1_dp)
end

"""
用途: 用指定后缀构造 Emery directed split backflow 的 epsilon terms。

参数:
- `suffix::Symbol`: 参数后缀, 例如 `:up` 或 `:dn_hole`。
- `bf_epsilon_d, bf_epsilon_p::Float64`: d/p source 的 epsilon prefactor。

返回:
- `Vector{BackflowEpsilonTerm}`: d 和 p 两个 epsilon term。
"""
function build_column_emery_ph_epsilon_terms(
    suffix::Symbol,
    bf_epsilon_d::Float64,
    bf_epsilon_p::Float64,
)::Vector{mfVMC.BackflowEpsilonTerm}
    return mfVMC.BackflowEpsilonTerm[
        mfVMC.Backflow.BackflowEpsilonTerm(
            param_name=Symbol("bf_epsilon_d_$(suffix)"),
            epsilon_bf=bf_epsilon_d,
            group_names=Symbol[:dd, :dp],
        ),
        mfVMC.Backflow.BackflowEpsilonTerm(
            param_name=Symbol("bf_epsilon_p_$(suffix)"),
            epsilon_bf=bf_epsilon_p,
            group_names=Symbol[:pd, :pp],
        ),
    ]
end

"""
用途: 构造 Emery PH backflow 的单个 directed orbital source group。

参数:
- `group_name::Symbol`: `:dd`, `:dp`, `:pd` 或 `:pp`。
- `suffix::Symbol`: 参数后缀, 例如 `:up` 或 `:dn_hole`。
- `source_bonds, source_amplitudes`: 有向 source bond 和对应 hopping 振幅。
- `eta_values::NTuple{4, Float64}`: `eta1..eta4` 参数值。

返回:
- `DirectedBackflowSourceGroup`: 可传入 `CompositeBackflowTerm` 的 source group。

公式:
- upper row 使用 Eq.(5): `eta1 D_i H_j + eta2 n_iσ h_i-σ n_j-σ h_jσ
  + eta3 D_i n_j-σ h_jσ + eta4 n_iσ h_i-σ H_j`。
- PH lower row 的具体电子/空穴替换由 `CompositeBackflowTerm(particle_hole_lower_block=true)` 处理。
"""
function build_column_emery_ph_backflow_group(
    group_name::Symbol,
    suffix::Symbol,
    source_bonds::Vector{Tuple{Int,Int}},
    source_amplitudes::Vector{Float64},
    eta_values::NTuple{4,Float64},
)
    return mfVMC.Backflow.build_directed_backflow_source_group(
        group_name,
        source_bonds,
        source_amplitudes,
        mfVMC.BackflowEta1DoublonHoleTerm(Symbol("bf_eta1_$(group_name)_$(suffix)"), eta_values[1]),
        mfVMC.BackflowEta2SpinExchangeTerm(Symbol("bf_eta2_$(group_name)_$(suffix)"), eta_values[2]),
        mfVMC.BackflowEta3DoublonSingleTerm(Symbol("bf_eta3_$(group_name)_$(suffix)"), eta_values[3]),
        mfVMC.BackflowEta4SingleHoleTerm(Symbol("bf_eta4_$(group_name)_$(suffix)"), eta_values[4]),
    )
end

"""
用途: 构造 Emery PH split backflow, upper 和 down-hole lower row 使用独立参数。

参数:
- `dd/dp/pd/pp_source_bonds, dd/dp/pd/pp_source_amplitudes`: directed orbital class source 数据。
- `bf_*_up`: upper row 的 epsilon 和 eta 参数。
- `bf_*_dn_hole`: PH down-hole lower row 的 epsilon 和 eta 参数。

返回:
- `CompositeBackflowTerm`: 参数顺序为 upper 18 个后接 down-hole 18 个。
"""
function build_column_ph_directed_emery_backflow(
    dd_source_bonds::Vector{Tuple{Int,Int}},
    dd_source_amplitudes::Vector{Float64},
    dp_source_bonds::Vector{Tuple{Int,Int}},
    dp_source_amplitudes::Vector{Float64},
    pd_source_bonds::Vector{Tuple{Int,Int}},
    pd_source_amplitudes::Vector{Float64},
    pp_source_bonds::Vector{Tuple{Int,Int}},
    pp_source_amplitudes::Vector{Float64};
    bf_epsilon_d_up::Float64=1.0,
    bf_epsilon_p_up::Float64=1.0,
    bf_eta_dd_up::NTuple{4,Float64}=(0.0, 0.0, 0.0, 0.0),
    bf_eta_dp_up::NTuple{4,Float64}=(0.0, 0.0, 0.0, 0.0),
    bf_eta_pd_up::NTuple{4,Float64}=(0.0, 0.0, 0.0, 0.0),
    bf_eta_pp_up::NTuple{4,Float64}=(0.0, 0.0, 0.0, 0.0),
    bf_epsilon_d_dn_hole::Float64=1.0,
    bf_epsilon_p_dn_hole::Float64=1.0,
    bf_eta_dd_dn_hole::NTuple{4,Float64}=(0.0, 0.0, 0.0, 0.0),
    bf_eta_dp_dn_hole::NTuple{4,Float64}=(0.0, 0.0, 0.0, 0.0),
    bf_eta_pd_dn_hole::NTuple{4,Float64}=(0.0, 0.0, 0.0, 0.0),
    bf_eta_pp_dn_hole::NTuple{4,Float64}=(0.0, 0.0, 0.0, 0.0),
)::CompositeBackflowTerm
    upper_epsilon_terms = build_column_emery_ph_epsilon_terms(:up, bf_epsilon_d_up, bf_epsilon_p_up)
    lower_epsilon_terms = build_column_emery_ph_epsilon_terms(:dn_hole, bf_epsilon_d_dn_hole, bf_epsilon_p_dn_hole)

    upper_groups = [
        build_column_emery_ph_backflow_group(:dd, :up, dd_source_bonds, dd_source_amplitudes, bf_eta_dd_up),
        build_column_emery_ph_backflow_group(:dp, :up, dp_source_bonds, dp_source_amplitudes, bf_eta_dp_up),
        build_column_emery_ph_backflow_group(:pd, :up, pd_source_bonds, pd_source_amplitudes, bf_eta_pd_up),
        build_column_emery_ph_backflow_group(:pp, :up, pp_source_bonds, pp_source_amplitudes, bf_eta_pp_up),
    ]
    lower_groups = [
        build_column_emery_ph_backflow_group(:dd, :dn_hole, dd_source_bonds, dd_source_amplitudes, bf_eta_dd_dn_hole),
        build_column_emery_ph_backflow_group(:dp, :dn_hole, dp_source_bonds, dp_source_amplitudes, bf_eta_dp_dn_hole),
        build_column_emery_ph_backflow_group(:pd, :dn_hole, pd_source_bonds, pd_source_amplitudes, bf_eta_pd_dn_hole),
        build_column_emery_ph_backflow_group(:pp, :dn_hole, pp_source_bonds, pp_source_amplitudes, bf_eta_pp_dn_hole),
    ]

    return mfVMC.Backflow.CompositeBackflowTerm(
        upper_epsilon_terms,
        upper_groups;
        particle_hole_lower_block=true,
        lower_epsilon_terms=lower_epsilon_terms,
        lower_source_groups=lower_groups,
    )
end

"""
用途: 根据开关构造 Emery PH directed split backflow 对象。

参数:
- `enable_backflow::Bool`: 是否启用 backflow。
- `source_data`: `build_emery_backflow_source_data_by_directed_orbital_type` 的八个返回数组。

返回:
- `AbstractBackflowTerm`: 关闭时为 `NoBackflowTerm()`, 开启时为 36 参数 PH `CompositeBackflowTerm`。
"""
function build_column_optional_ph_directed_emery_backflow(
    enable_backflow::Bool,
    dd_source_bonds::Vector{Tuple{Int,Int}},
    dd_source_amplitudes::Vector{Float64},
    dp_source_bonds::Vector{Tuple{Int,Int}},
    dp_source_amplitudes::Vector{Float64},
    pd_source_bonds::Vector{Tuple{Int,Int}},
    pd_source_amplitudes::Vector{Float64},
    pp_source_bonds::Vector{Tuple{Int,Int}},
    pp_source_amplitudes::Vector{Float64};
    kwargs...,
)::AbstractBackflowTerm
    if !enable_backflow
        return NoBackflowTerm()
    end
    return build_column_ph_directed_emery_backflow(
        dd_source_bonds,
        dd_source_amplitudes,
        dp_source_bonds,
        dp_source_amplitudes,
        pd_source_bonds,
        pd_source_amplitudes,
        pp_source_bonds,
        pp_source_amplitudes;
        kwargs...,
    )
end

"""
用途: 对角化 column Emery PH Hamiltonian 并生成 determinant orbitals 及参数导数。

参数:
- `params::ColumnEmeryPHParams`: PH mean-field 参数。
- `param_names::Vector{Symbol}`: 需要计算导数的 mean-field 参数名。
- `target_sz::Int`: 目标 total Sz, PH determinant 占据轨道数为 `N_sites + target_sz`。

返回:
- `Tuple`: `(epsilon, occupied_orbitals, d_ut_params)`。
"""
function make_column_emery_ph_ansatz_and_derivs(
    params::ColumnEmeryPHParams;
    param_names::Vector{Symbol}=Symbol[],
    target_sz::Int=0,
)
    hamiltonian = Matrix(build_column_emery_ph_hamiltonian(params))
    hamiltonian_derivatives = Dict{Symbol,Matrix{Float64}}()
    for param_name in param_names
        hamiltonian_derivatives[param_name] = build_column_emery_ph_dh_dparam(params, param_name)
    end

    epsilon, full_orbitals, _, orbital_derivatives = compute_eig_and_dU_reg1(
        hamiltonian,
        hamiltonian_derivatives,
    )
    n_occupied_orbitals = compute_column_emery_ph_determinant_orbital_count(params.lx, params.ly, target_sz)

    if is_emery_ph_root_rank()
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
用途: 更新 column Emery PH determinant 波函数和参数导数。

参数:
- `vwf`: determinant 波函数对象。
- `param_names::Vector{Symbol}`: 完整参数名列表, 顺序为 mean-field, projector, backflow。
- `params::Vector{Float64}`: 与 `param_names` 对齐的完整参数值。
- `lx, ly, bcx, bcy`: Emery 晶格和边界参数。
- `target_sz::Int`: 目标 total Sz。
- `nparams_proj, nparams_backflow::Int`: projector/backflow 参数数量。
- `fixed_chi1_dp::Float64`: 固定的 Cu-O mean-field hopping。

返回:
- `nothing`。
"""
function update_column_emery_ph_ansatz!(
    vwf,
    param_names::Vector{Symbol},
    params::Vector{Float64},
    lx::Int,
    ly::Int,
    bcx::Float64,
    bcy::Float64,
    target_sz::Int;
    nparams_proj::Int=0,
    nparams_backflow::Int=0,
    fixed_chi1_dp::Float64=0.0,
)::Nothing
    total_param_count = length(param_names)
    nparams_wf = total_param_count - nparams_proj - nparams_backflow
    wf_param_names = param_names[1:nparams_wf]
    wf_param_values = params[1:nparams_wf]
    projector_param_names = param_names[(nparams_wf+1):(nparams_wf+nparams_proj)]
    projector_param_values = params[(nparams_wf+1):(nparams_wf+nparams_proj)]
    backflow_param_names = param_names[(nparams_wf+nparams_proj+1):total_param_count]
    backflow_param_values = params[(nparams_wf+nparams_proj+1):total_param_count]

    ph_params = build_column_emery_ph_params_from_wf_params(
        wf_param_names,
        wf_param_values,
        lx,
        ly,
        bcx,
        bcy;
        fixed_chi1_dp=fixed_chi1_dp,
    )
    _, occupied_orbitals, d_ut_params = make_column_emery_ph_ansatz_and_derivs(
        ph_params;
        param_names=wf_param_names,
        target_sz=target_sz,
    )

    copyto!(vwf.base_gs_U, occupied_orbitals)
    copyto!(vwf.gs_U, occupied_orbitals)
    copyto!(vwf.gs_U_t, permutedims(occupied_orbitals))
    d_ut_tensor = zeros(Float64, size(occupied_orbitals, 2), size(occupied_orbitals, 1), length(wf_param_names))
    for (param_index, param_name) in enumerate(wf_param_names)
        d_ut_tensor[:, :, param_index] = d_ut_params[param_name]
    end
    update_vwf_params!(vwf, wf_param_names, d_ut_tensor)

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
用途: 解析 `Emery_PH.jl` 命令行参数。

参数:
- 无。读取 `ARGS`。

返回:
- `Dict{String, Any}`: ArgParse 结果。
"""
function parse_column_emery_ph_commandline()
    settings = ArgParseSettings()
    @add_arg_table settings begin
        "--Lx"
        arg_type = Int
        default = 8
        "--Ly"
        arg_type = Int
        default = 3
        "--tpd"
        arg_type = Float64
        default = 1.0
        "--tpp"
        arg_type = Float64
        default = 0.0
        "--Delta_pd"
        arg_type = Float64
        default = 3.0
        "--Udd"
        arg_type = Float64
        default = 8.0
        "--Up"
        arg_type = Float64
        default = 0.0
        "--Vpd"
        arg_type = Float64
        default = 0.0
        "--Vpp"
        arg_type = Float64
        default = 0.0
        "--bcx"
        arg_type = Float64
        default = 1.0
        "--bcy"
        arg_type = Float64
        default = 1.0
        "--edge_pinning"
        arg_type = Float64
        default = 0.0
        "--chi1_dd"
        arg_type = Float64
        default = 0.0
        "--chi1_dp"
        arg_type = Float64
        default = 1.0
        "--chi1_pp"
        arg_type = Float64
        default = 0.0
        "--etax_dd"
        arg_type = Float64
        default = 0.01
        "--etay_dd"
        arg_type = Float64
        default = 0.01
        "--mz"
        arg_type = Float64
        default = 0.3
        "--mu"
        arg_type = Float64
        default = 0.0
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
        "--eigen_cutoff"
        arg_type = Float64
        default = 0.0
        "--init_params_json"
        arg_type = String
        default = ""
        "--job"
        arg_type = String
        default = "SR"
        "--doping"
        arg_type = String
        default = "0.125"
        "--ansatz"
        arg_type = String
        default = "Stripe"
        "--lambda"
        arg_type = Int
        default = 4
        "--stripe_center"
        arg_type = String
        default = "site"
        "--enable_orbital_gutzwiller"
        arg_type = String
        default = "true"
        "--g_d"
        arg_type = Float64
        default = 1.0
        "--g_p"
        arg_type = Float64
        default = 1.0
        "--vj_oo"
        arg_type = Float64
        default = 0.0
        "--vj_cuo"
        arg_type = Float64
        default = 0.0
        "--vj_cucu"
        arg_type = Float64
        default = 0.0
        "--enable_backflow"
        arg_type = String
        default = "true"
        "--enable_timing"
        arg_type = String
        default = "false"
        "--bf_epsilon_d"
        arg_type = Float64
        default = 1.0
        "--bf_epsilon_p"
        arg_type = Float64
        default = 1.0
        "--bf_eta1_dd"
        arg_type = Float64
        default = 0.0
        "--bf_eta2_dd"
        arg_type = Float64
        default = 0.0
        "--bf_eta3_dd"
        arg_type = Float64
        default = 0.0
        "--bf_eta4_dd"
        arg_type = Float64
        default = 0.0
        "--bf_eta1_dp"
        arg_type = Float64
        default = 0.0
        "--bf_eta2_dp"
        arg_type = Float64
        default = 0.0
        "--bf_eta3_dp"
        arg_type = Float64
        default = 0.0
        "--bf_eta4_dp"
        arg_type = Float64
        default = 0.0
        "--bf_eta1_pd"
        arg_type = Float64
        default = 0.0
        "--bf_eta2_pd"
        arg_type = Float64
        default = 0.0
        "--bf_eta3_pd"
        arg_type = Float64
        default = 0.0
        "--bf_eta4_pd"
        arg_type = Float64
        default = 0.0
        "--bf_eta1_pp"
        arg_type = Float64
        default = 0.0
        "--bf_eta2_pp"
        arg_type = Float64
        default = 0.0
        "--bf_eta3_pp"
        arg_type = Float64
        default = 0.0
        "--bf_eta4_pp"
        arg_type = Float64
        default = 0.0
    end
    return parse_args(settings)
end

"""
用途: 运行 column-resolved PH + backflow Emery VMC/SR 主流程。

参数:
- 无。所有参数来自命令行。

返回:
- `nothing`。
"""
function main_column_emery_ph_backflow()::Nothing
    args = parse_column_emery_ph_commandline()
    enable_timing = parse_column_bool_flag(args["enable_timing"], "--enable_timing")
    ENABLE_TIMING[] = enable_timing
    session = init_mpi_session()
    rank = session.rank
    is_root = rank == session.root

    lx = args["Lx"]
    ly = args["Ly"]
    bcx = args["bcx"]
    bcy = args["bcy"]
    n_sites = emery_n_sites(lx, ly)
    target_sz = args["target_sz"]
    doping = parse_column_doping_value(args["doping"], "--doping")
    lr = args["lr"]
    lr_end = isnan(args["lr_end"]) ? lr : args["lr_end"]
    job = args["job"]

    enable_timing && timing_reset!()

    mean_field_setup = build_column_emery_ph_mean_field_parameter_setup(
        args["ansatz"],
        lx,
        args["lambda"],
        args["stripe_center"],
        args["mu"],
        args["stripe_mu_amp"],
        args["mz"],
        args["chi1_dd"],
        args["chi1_dp"],
        args["chi1_pp"],
        args["etax_dd"],
        args["etay_dd"],
        args["stripe_spin_peak_x"],
    )
    wf_param_names = mean_field_setup.wf_param_names
    wf_init_params = mean_field_setup.wf_init_params
    fixed_chi1_dp = mean_field_setup.fixed_chi1_dp

    meas_params = VMCParams(
        total_samples=args["nMC"],
        warmup_steps=args["wMC"],
        rebuild_every=args["rMC"],
        decorr_steps=args["dMC"],
        seed=args["seed"] + rank,
    )

    source_data = build_emery_backflow_source_data_by_directed_orbital_type(
        lx,
        ly;
        tpd=args["tpd"],
        tpp=args["tpp"],
        bcy=bcy,
    )
    shared_eta_dd = (args["bf_eta1_dd"], args["bf_eta2_dd"], args["bf_eta3_dd"], args["bf_eta4_dd"])
    shared_eta_dp = (args["bf_eta1_dp"], args["bf_eta2_dp"], args["bf_eta3_dp"], args["bf_eta4_dp"])
    shared_eta_pd = (args["bf_eta1_pd"], args["bf_eta2_pd"], args["bf_eta3_pd"], args["bf_eta4_pd"])
    shared_eta_pp = (args["bf_eta1_pp"], args["bf_eta2_pp"], args["bf_eta3_pp"], args["bf_eta4_pp"])
    backflow = build_column_optional_ph_directed_emery_backflow(
        parse_column_bool_flag(args["enable_backflow"], "--enable_backflow"),
        source_data...;
        bf_epsilon_d_up=args["bf_epsilon_d"],
        bf_epsilon_p_up=args["bf_epsilon_p"],
        bf_eta_dd_up=shared_eta_dd,
        bf_eta_dp_up=shared_eta_dp,
        bf_eta_pd_up=shared_eta_pd,
        bf_eta_pp_up=shared_eta_pp,
        bf_epsilon_d_dn_hole=args["bf_epsilon_d"],
        bf_epsilon_p_dn_hole=args["bf_epsilon_p"],
        bf_eta_dd_dn_hole=shared_eta_dd,
        bf_eta_dp_dn_hole=shared_eta_dp,
        bf_eta_pd_dn_hole=shared_eta_pd,
        bf_eta_pp_dn_hole=shared_eta_pp,
    )
    projector = build_emery_density_jastrow_projector(
        lx,
        ly;
        enable_orbital_gutzwiller=parse_column_bool_flag(args["enable_orbital_gutzwiller"], "--enable_orbital_gutzwiller"),
        g_d=args["g_d"],
        g_p=args["g_p"],
        vj_oo=args["vj_oo"],
        vj_cuo=args["vj_cuo"],
        vj_cucu=args["vj_cucu"],
    )
    proj_param_names = projector_param_names(projector)
    proj_init_params = projector_param_values(projector)
    nparams_proj = length(proj_param_names)
    backflow_param_name_list = backflow_param_names(backflow)
    backflow_init_params = backflow_param_values(backflow)
    nparams_backflow = length(backflow_param_name_list)
    init_params = vcat(wf_init_params, proj_init_params, backflow_init_params)
    param_names = vcat(wf_param_names, proj_param_names, backflow_param_name_list)

    if !isempty(args["init_params_json"])
        init_params = build_column_init_params_from_json(args["init_params_json"], param_names, init_params)
        if is_root
            println("Loaded initial parameters from json: $(args["init_params_json"])")
        end
    end

    ham = @timed "build_emery_general_model" build_emery_general_model(
        lx,
        ly;
        tpd=args["tpd"],
        tpp=args["tpp"],
        Delta_pd=args["Delta_pd"],
        Udd=args["Udd"],
        Up=args["Up"],
        Vpd=args["Vpd"],
        Vpp=args["Vpp"],
    )
    if args["edge_pinning"] != 0.0
        for y in 1:ly
            left_site = Emery_xyo_to_site_index(1, y, EMERY_ORB_D, lx, ly)
            push!(ham.terms, OperatorTerm([:Sz], [left_site], args["edge_pinning"] * (-1)^(y + 1)))
        end
    end

    nelec = compute_emery_electron_count(lx, ly, doping)
    @assert (target_sz + nelec) % 2 == 0 "Wrong parity between target_sz and electron count."
    nup = (nelec + target_sz) ÷ 2
    ndn = nelec - nup
    if nup < 0 || ndn < 0 || nup > n_sites || ndn > n_sites
        error("Invalid PH particle numbers: nup=$(nup), ndn=$(ndn), N_sites=$(n_sites).")
    end
    n_occupied_orbitals = compute_column_emery_ph_determinant_orbital_count(lx, ly, target_sz)

    sampler = config_Hubbard(n_sites, nup, ndn; ifPH=true)
    init_config_Hubbard!(sampler)
    vwf = vwf_det(zeros(Float64, 2 * n_sites, n_occupied_orbitals), sampler; backflow=backflow)
    set_projector!(vwf, projector)
    kernel = HubbardKernel(conserve_sz=true)

    if is_root
        println("Initial parameters: $(init_params)")
        println("Fixed mean-field parameters: chi1_dp=$(fixed_chi1_dp)")
        println("column Emery PH particle numbers: N_up=$(nup), N_down=$(ndn), N_e=$(nelec), N_occ_PH=$(n_occupied_orbitals)")
        println("Backflow enabled: $(uses_backflow(backflow))")
    end

    update_column_emery_ph_ansatz!(
        vwf,
        param_names,
        init_params,
        lx,
        ly,
        bcx,
        bcy,
        target_sz;
        nparams_proj=nparams_proj,
        nparams_backflow=nparams_backflow,
        fixed_chi1_dp=fixed_chi1_dp,
    )

    folder = "logs"
    mkpath(folder)
    if job == "SR"
        sr_params = SRParams(
            vmc_params=meas_params,
            n_steps=args["nSR"],
            lr=lr,
            eigen_cutoff=args["eigen_cutoff"],
        )
        exp_lr_func = build_exponential_lr_func(lr, lr_end, args["nSR"])
        update_vwf_func! = (vwf, params) -> @timed "update_column_emery_ph_ansatz!" update_column_emery_ph_ansatz!(
            vwf,
            param_names,
            params,
            lx,
            ly,
            bcx,
            bcy,
            target_sz;
            nparams_proj=nparams_proj,
            nparams_backflow=nparams_backflow,
            fixed_chi1_dp=fixed_chi1_dp,
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
            if enable_timing
                println("[Timing] SR optimization complete. Printing timing report...")
                timing_report()
                open(joinpath(folder, "timing_report.txt"), "w") do io
                    timing_report(io)
                end
                println("[Timing] Timing report saved to $(joinpath(folder, "timing_report.txt")).")
            end
        end
    elseif job == "measure"
        results = run_simulation(
            ham,
            vwf,
            kernel,
            build_emery_observables(lx, ly),
            meas_params;
            history_observables=[:E],
        )
        if is_root && results !== nothing
            write_column_measure_outputs(folder, results)
            if enable_timing
                println("[Timing] Measurement complete. Printing timing report...")
                timing_report()
                open(joinpath(folder, "timing_report.txt"), "w") do io
                    timing_report(io)
                end
                println("[Timing] Timing report saved to $(joinpath(folder, "timing_report.txt")).")
            end
        end
    else
        error("Unknown job: $(job)")
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_column_emery_ph_backflow()
end
