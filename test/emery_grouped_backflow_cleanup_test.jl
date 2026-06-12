using Test
using LinearAlgebra

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "Emery.jl"))
using mfVMC.Sampler

function has_method_signature_containing(function_object, type_name::String)::Bool
    return any(method -> occursin(type_name, sprint(show, method.sig)), methods(function_object))
end

"""
用途: 构造一个最小 `BackflowEpsilonTerm`, 用于测试 group_names 字段.

参数:
- 无.

返回:
- `mfVMC.BackflowEpsilonTerm`: 带 group_names 的 epsilon term.
"""
function build_test_epsilon_term()
    return mfVMC.Backflow.BackflowEpsilonTerm(
        param_name=:bf_epsilon_test,
        epsilon_bf=1.2,
        group_names=Symbol[:dd, :dp],
    )
end

"""
用途: 构造只含一条 `dd` source bond 的最小 Emery directed backflow, 用于梯度测试.

参数:
- `bf_epsilon_d::Float64`: `bf_epsilon_d` 参数值.
- `bf_eta4_dd::Float64`: `bf_eta4_dd` 参数值.
- `dd_amplitude::Float64`: directed bond `(1, 2)` 的 hopping 振幅.

返回:
- `mfVMC.Backflow.CompositeBackflowTerm`: 只在 `dd` group 中含一条 source bond 的 backflow.
"""
function build_minimal_dd_eta4_backflow(;
    bf_epsilon_d::Float64,
    bf_eta4_dd::Float64,
    dd_amplitude::Float64=1.0,
)
    empty_bonds = Tuple{Int,Int}[]
    empty_amplitudes = Float64[]
    return build_column_directed_emery_backflow(
        Tuple{Int,Int}[(1, 2)], [dd_amplitude],
        empty_bonds, empty_amplitudes,
        empty_bonds, empty_amplitudes,
        empty_bonds, empty_amplitudes,
        bf_epsilon_d, 1.0,
        0.0, 0.0, 0.0, bf_eta4_dd,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    )
end

"""
用途: 对给定 backflow 和 occupied rows 计算 determinant 的 `log(abs(det(A)))`.

参数:
- `base_orbitals::Matrix{Float64}`: 裸轨道矩阵 `U_0`.
- `state_vector::Vector{Int8}`: 当前构型.
- `backflow`: 待测试的 backflow 对象.
- `electron_rows::Vector{Int}`: determinant 选取的 occupied row 列表.

返回:
- `Float64`: `log(abs(det(U_b[electron_rows, 1:N_e])))`.
"""
function compute_backflow_logabsdet(
    base_orbitals::Matrix{Float64},
    state_vector::Vector{Int8},
    backflow,
    electron_rows::Vector{Int},
)::Float64
    backflow_orbitals = mfVMC.build_backflow_orbitals(base_orbitals, state_vector, backflow)
    electron_count = length(electron_rows)
    slater_matrix = Matrix(backflow_orbitals[electron_rows, 1:electron_count])
    return log(abs(det(slater_matrix)))
end

"""
用途: 由 backflow 参数导数矩阵计算 determinant log-derivative.

参数:
- `base_orbitals::Matrix{Float64}`: 裸轨道矩阵 `U_0`.
- `state_vector::Vector{Int8}`: 当前构型.
- `backflow`: 待测试的 backflow 对象.
- `param_name::Symbol`: 目标 backflow 参数名.
- `electron_rows::Vector{Int}`: determinant 选取的 occupied row 列表.

返回:
- `Float64`: `Tr(A^{-1} dA/dp)`.
"""
function compute_backflow_param_log_derivative(
    base_orbitals::Matrix{Float64},
    state_vector::Vector{Int8},
    backflow,
    param_name::Symbol,
    electron_rows::Vector{Int},
)::Float64
    backflow_orbitals = mfVMC.build_backflow_orbitals(base_orbitals, state_vector, backflow)
    electron_count = length(electron_rows)
    slater_matrix = Matrix(backflow_orbitals[electron_rows, 1:electron_count])
    derivative_pairs = mfVMC.Backflow.build_backflow_derivative_orbitals(
        base_orbitals,
        state_vector,
        backflow,
    )
    pair_index = findfirst(pair -> first(pair) == param_name, derivative_pairs)
    pair_index === nothing && error("Missing backflow derivative for $(param_name).")
    derivative_matrix = Matrix(derivative_pairs[pair_index].second[electron_rows, 1:electron_count])
    return tr(inv(slater_matrix) * derivative_matrix)
end

"""
用途: 用 central finite difference 估计 determinant 的 `d log(abs(det(A))) / dp`.

参数:
- `evaluate_logabsdet::Function`: 输入参数值并返回 `log(abs(det(A(p))))` 的函数.
- `param_value::Float64`: 当前参数值 `p`.
- `step_size::Float64`: 有限差分步长 `h`.

返回:
- `Float64`: `(f(p+h)-f(p-h))/(2h)`.
"""
function compute_central_finite_difference_log_derivative(
    evaluate_logabsdet::Function,
    param_value::Float64,
    step_size::Float64,
)::Float64
    return (
        evaluate_logabsdet(param_value + step_size) -
        evaluate_logabsdet(param_value - step_size)
    ) / (2.0 * step_size)
end

@testset "Emery grouped backflow cleanup" begin
    @test !isdefined(mfVMC.Backflow, Symbol("fill_backflow_site_row_after_proposal_by_terms!"))
    @test !isdefined(mfVMC.Backflow, Symbol("fill_backflow_site_block_after_proposal_by_terms!"))

    # 验证已删除的旧 epsilon 专用函数不再存在.
    old_epsilon_helpers = [
        Symbol("add_backflow_correction_site_block_after_proposal!"),
        Symbol("add_backflow_correction_site_row_after_proposal!"),
        Symbol("compute_backflow_epsilon_row_mask"),
        Symbol("compute_backflow_epsilon_neighbor_data_signature"),
        Symbol("build_backflow_epsilon_neighbor_cache"),
        Symbol("add_backflow_correction_chain_rule_row!"),
        Symbol("add_backflow_correction_chain_rule_source_weights!"),
    ]
    for helper_name in old_epsilon_helpers
        @test !isdefined(mfVMC.Backflow, helper_name)
    end

    # 验证新的 eta-driven 辅助函数存在.
    new_helpers = [
        Symbol("fill_backflow_row_source_weights_from_state_getter!"),
        Symbol("fill_backflow_row_from_source_weights!"),
        Symbol("add_source_group_chain_rule_source_weights_and_track!"),
    ]
    for helper_name in new_helpers
        @test isdefined(mfVMC.Backflow, helper_name)
    end

    # 验证已废弃的直接 row 构建函数不再存在.
    obsolete_row_helpers = [
        Symbol("add_source_group_eta_contributions_and_track_activation!"),
        Symbol("add_epsilon_contributions_from_active_groups!"),
        Symbol("fill_backflow_site_row_from_state_getter!"),
    ]
    for helper_name in obsolete_row_helpers
        @test !isdefined(mfVMC.Backflow, helper_name)
    end

    dd_source_bonds,
    dd_source_amplitudes,
    dp_source_bonds,
    dp_source_amplitudes,
    pd_source_bonds,
    pd_source_amplitudes,
    pp_source_bonds,
    pp_source_amplitudes = build_emery_backflow_source_data_by_directed_orbital_type(
        2,
        2;
        tpd=1.0,
        tpp=0.2,
    )

    backflow = build_column_directed_emery_backflow(
        dd_source_bonds,
        dd_source_amplitudes,
        dp_source_bonds,
        dp_source_amplitudes,
        pd_source_bonds,
        pd_source_amplitudes,
        pp_source_bonds,
        pp_source_amplitudes,
        1.1,
        0.9,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
        1.3,
        1.4,
        1.5,
        1.6,
    )

    n_sites = emery_n_sites(2, 2)
    base_orbitals = reshape(collect(1.0:(8.0 * n_sites)), 2 * n_sites, 4)

    # 构造一个 site1 有 eta 贡献的状态:
    # site1=DB (3), site2=HOLE (0) -> eta1 通过 dd bond (1->2) 激活.
    state_vector = fill(Int8(0), n_sites)
    state_vector[1] = Int8(3)
    state_vector[2] = Int8(0)
    state_vector[3] = Int8(1)
    state_vector[4] = Int8(2)

    proposal = MoveProposal(
        1,
        2,
        Int8(3),
        Int8(0),
        Int8(1),
        Int8(2),
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )

    state_after = copy(state_vector)
    state_after[1] = Int8(1)
    state_after[2] = Int8(2)

    full_orbitals_after = mfVMC.build_backflow_orbitals(base_orbitals, state_after, backflow)
    site_block_buffer = zeros(Float64, 2, size(base_orbitals, 2))
    mfVMC.Backflow.fill_backflow_site_block_after_proposal!(
        site_block_buffer,
        base_orbitals,
        state_vector,
        backflow,
        proposal,
        1,
    )

    @test site_block_buffer == full_orbitals_after[1:2, :]
end

@testset "Eta-driven epsilon term structure" begin
    epsilon_term = build_test_epsilon_term()

    # BackflowEpsilonTerm 不再存储 site-neighbor mask 字段.
    @test !(:target_neighbors_by_source_site in fieldnames(typeof(epsilon_term)))
    @test !(:source_sites_by_target_neighbor in fieldnames(typeof(epsilon_term)))
    @test !(:source_sites in fieldnames(typeof(epsilon_term)))
    @test !(:neighbor_data_signature in fieldnames(typeof(epsilon_term)))
    @test !(:source_bonds in fieldnames(typeof(epsilon_term)))
    @test !(:source_amplitudes in fieldnames(typeof(epsilon_term)))

    # 验证新字段 group_names.
    @test epsilon_term.group_names == Symbol[:dd, :dp]
    @test epsilon_term.param_name == :bf_epsilon_test
    @test epsilon_term.epsilon_bf ≈ 1.2

    # 验证旧 keyword 构造函数 (source_bonds) 不再有效.
    @test_throws MethodError mfVMC.Backflow.BackflowEpsilonTerm(
        param_name=:bf_epsilon_test,
        epsilon_bf=1.2,
        source_bonds=Tuple{Int,Int}[(1, 2)],
    )
    @test_throws MethodError mfVMC.Backflow.BackflowEpsilonTerm(
        param_name=:bf_epsilon_test,
        epsilon_bf=1.2,
        source_bonds=Tuple{Int,Int}[(1, 2)],
        source_amplitudes=[1.0],
    )
end

@testset "Eta-driven epsilon activation" begin
    n_sites = 2
    n_orbitals = 2

    dd_bonds = Tuple{Int,Int}[(1, 2)]
    dd_amplitudes = [1.0]
    empty_bonds = Tuple{Int,Int}[]
    empty_amplitudes = Float64[]

    base_orbitals = ones(Float64, 2 * n_sites, n_orbitals)

    # Test 1: epsilon inactive when all eta parameters are zero.
    # state: site1=DB, site2=HOLE -> occupation allows hopping, but eta params are 0.
    bf_epsilon_d = 2.0
    backflow_zero_eta = build_column_directed_emery_backflow(
        dd_bonds, dd_amplitudes,
        empty_bonds, empty_amplitudes,
        empty_bonds, empty_amplitudes,
        empty_bonds, empty_amplitudes,
        bf_epsilon_d, 1.0,
        0.0, 0.0, 0.0, 0.0,  # dd: 全部为零
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    )

    state_db_hole = Int8[3, 0]
    backflow_orbitals = mfVMC.build_backflow_orbitals(base_orbitals, state_db_hole, backflow_zero_eta)
    @test backflow_orbitals ≈ base_orbitals  # 无 eta → 无 epsilon.

    # Test 2: epsilon active when eta contribution is nonzero.
    eta4_value = 0.5
    backflow_with_eta = build_column_directed_emery_backflow(
        dd_bonds, dd_amplitudes,
        empty_bonds, empty_amplitudes,
        empty_bonds, empty_amplitudes,
        empty_bonds, empty_amplitudes,
        bf_epsilon_d, 1.0,
        0.0, 0.0, 0.0, eta4_value,  # dd: 仅 eta4 非零
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    )

    # state: site1=UP, site2=HOLE -> eta4 (single-hole) 对于 UP row 激活.
    state_up_hole = Int8[1, 0]
    backflow_orbitals2 = mfVMC.build_backflow_orbitals(base_orbitals, state_up_hole, backflow_with_eta)
    # UP row (row 1): U_b = U_0 + eta4*t_ij*U_0(j) + (epsilon_d-1)*U_0(i)
    # = 1.0 + 0.5*1.0*1.0 + (2.0-1.0)*1.0 = 2.5
    @test backflow_orbitals2[1, 1] ≈ 2.5
    # DN row (row 2): n_dn=0, 无 eta 也无 epsilon.
    @test backflow_orbitals2[2, 1] ≈ 1.0

    # Test 3: epsilon inactive when eta parameter nonzero but bond amplitude is zero.
    zero_amp_bonds = Tuple{Int,Int}[(1, 2)]
    zero_amps = [0.0]
    backflow_zero_amp = build_column_directed_emery_backflow(
        zero_amp_bonds, zero_amps,
        empty_bonds, empty_amplitudes,
        empty_bonds, empty_amplitudes,
        empty_bonds, empty_amplitudes,
        bf_epsilon_d, 1.0,
        0.0, 0.0, 0.0, eta4_value,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    )
    backflow_orbitals3 = mfVMC.build_backflow_orbitals(base_orbitals, state_up_hole, backflow_zero_amp)
    @test backflow_orbitals3 ≈ base_orbitals

    # Test 4: epsilon sharing — bf_epsilon_d 响应 :dd 和 :dp, 但不响应 :pd 或 :pp.
    # 仅使用 dd group (包含 bf_epsilon_d).
    backflow_epsilon_d_active = build_column_directed_emery_backflow(
        dd_bonds, dd_amplitudes,
        empty_bonds, empty_amplitudes,  # dp: 空
        empty_bonds, empty_amplitudes,  # pd: 空
        empty_bonds, empty_amplitudes,  # pp: 空
        bf_epsilon_d, 1.0,
        0.0, 0.0, 0.0, eta4_value,  # dd: eta4 激活
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    )
    backflow_eps_d = mfVMC.build_backflow_orbitals(base_orbitals, state_up_hole, backflow_epsilon_d_active)
    @test backflow_eps_d[1, 1] ≈ 2.5  # epsilon_d 通过 dd 组激活

    # Test 5: bf_epsilon_p 不响应 :dd 或 :dp (group_names=[:pd, :pp]).
    backflow_epsilon_p_only = build_column_directed_emery_backflow(
        dd_bonds, dd_amplitudes,
        empty_bonds, empty_amplitudes,
        empty_bonds, empty_amplitudes,
        empty_bonds, empty_amplitudes,
        1.0, bf_epsilon_d,              # epsilon_d=1.0 (无效果), epsilon_p=2.0
        0.0, 0.0, 0.0, eta4_value,     # dd: eta4 激活, 但 epsilon_p 不应响应
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    )
    backflow_eps_p = mfVMC.build_backflow_orbitals(base_orbitals, state_up_hole, backflow_epsilon_p_only)
    # 仅 eta4 贡献 (无 epsilon), 因为 epsilon_p 只响应 :pd/:pp.
    # eta4 coefficient = 1.0 * (0.0*...) = 0.0... wait let me re-check.
    # bond_amplitude = 1.0, eta4_factor = n_up * h_dn * H_j = 1*1*1 = 1, eta4_value = 0.5
    # coefficient = 1.0 * 0.5 * 1.0 = 0.5
    # U_b(1,:) = U_0(1,:) + 0.5*U_0(3,:) = 1.0 + 0.5 = 1.5
    @test backflow_eps_p[1, 1] ≈ 1.5  # epsilon_p 不应激活 (dd 不在 [:pd,:pp] 中)
end

@testset "Eta-driven epsilon derivative activation" begin
    n_sites = 2
    n_orbitals = 2
    dd_bonds = Tuple{Int,Int}[(1, 2)]
    dd_amplitudes = [1.0]
    empty_bonds = Tuple{Int,Int}[]
    empty_amplitudes = Float64[]
    base_orbitals = ones(Float64, 2 * n_sites, n_orbitals)
    state_up_hole = Int8[1, 0]

    backflow_zero_eta = build_column_directed_emery_backflow(
        dd_bonds, dd_amplitudes,
        empty_bonds, empty_amplitudes,
        empty_bonds, empty_amplitudes,
        empty_bonds, empty_amplitudes,
        2.0, 1.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    )
    zero_eta_derivatives = mfVMC.Backflow.build_backflow_derivative_orbitals(
        base_orbitals,
        state_up_hole,
        backflow_zero_eta,
    )
    epsilon_d_index = findfirst(pair -> first(pair) == :bf_epsilon_d, zero_eta_derivatives)
    eta4_dd_index = findfirst(pair -> first(pair) == :bf_eta4_dd, zero_eta_derivatives)
    @test epsilon_d_index !== nothing
    @test eta4_dd_index !== nothing
    @test zero_eta_derivatives[epsilon_d_index].second == zeros(Float64, size(base_orbitals))
    @test zero_eta_derivatives[eta4_dd_index].second[1, :] == base_orbitals[3, :]

    backflow_active_eta = build_column_directed_emery_backflow(
        dd_bonds, dd_amplitudes,
        empty_bonds, empty_amplitudes,
        empty_bonds, empty_amplitudes,
        empty_bonds, empty_amplitudes,
        2.0, 1.0,
        0.0, 0.0, 0.0, 0.5,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    )
    active_eta_derivatives = mfVMC.Backflow.build_backflow_derivative_orbitals(
        base_orbitals,
        state_up_hole,
        backflow_active_eta,
    )
    epsilon_d_active_index = findfirst(pair -> first(pair) == :bf_epsilon_d, active_eta_derivatives)
    @test active_eta_derivatives[epsilon_d_active_index].second[1, :] == base_orbitals[1, :]
    @test active_eta_derivatives[epsilon_d_active_index].second[2:end, :] == zeros(Float64, 3, n_orbitals)

    zero_amp_backflow = build_column_directed_emery_backflow(
        dd_bonds, [0.0],
        empty_bonds, empty_amplitudes,
        empty_bonds, empty_amplitudes,
        empty_bonds, empty_amplitudes,
        2.0, 1.0,
        0.0, 0.0, 0.0, 0.5,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    )
    zero_amp_derivatives = mfVMC.Backflow.build_backflow_derivative_orbitals(
        base_orbitals,
        state_up_hole,
        zero_amp_backflow,
    )
    epsilon_d_zero_amp_index = findfirst(pair -> first(pair) == :bf_epsilon_d, zero_amp_derivatives)
    @test zero_amp_derivatives[epsilon_d_zero_amp_index].second == zeros(Float64, size(base_orbitals))
end

@testset "Eta-driven determinant gradient correctness" begin
    base_orbitals = [
        2.0 0.3
        3.0 0.4
        5.0 0.5
        7.0 0.6
    ]
    state_up_hole = Int8[1, 0]
    electron_rows = [1]
    step_size = 1.0e-6

    # 默认点 epsilon=1, eta=0: epsilon 不激活, 但 eta4 的线性响应应该仍然存在.
    backflow_default = build_minimal_dd_eta4_backflow(
        bf_epsilon_d=1.0,
        bf_eta4_dd=0.0,
        dd_amplitude=1.2,
    )
    eta_default_analytic = compute_backflow_param_log_derivative(
        base_orbitals,
        state_up_hole,
        backflow_default,
        :bf_eta4_dd,
        electron_rows,
    )
    eta_default_numeric = compute_central_finite_difference_log_derivative(
        eta4_value -> compute_backflow_logabsdet(
            base_orbitals,
            state_up_hole,
            build_minimal_dd_eta4_backflow(
                bf_epsilon_d=1.0,
                bf_eta4_dd=eta4_value,
                dd_amplitude=1.2,
            ),
            electron_rows,
        ),
        0.0,
        step_size,
    )
    @test eta_default_analytic ≈ eta_default_numeric rtol = 1.0e-8 atol = 1.0e-8

    epsilon_default_analytic = compute_backflow_param_log_derivative(
        base_orbitals,
        state_up_hole,
        backflow_default,
        :bf_epsilon_d,
        electron_rows,
    )
    epsilon_default_numeric = compute_central_finite_difference_log_derivative(
        epsilon_value -> compute_backflow_logabsdet(
            base_orbitals,
            state_up_hole,
            build_minimal_dd_eta4_backflow(
                bf_epsilon_d=epsilon_value,
                bf_eta4_dd=0.0,
                dd_amplitude=1.2,
            ),
            electron_rows,
        ),
        1.0,
        step_size,
    )
    @test epsilon_default_analytic ≈ 0.0 atol = 1.0e-12
    @test epsilon_default_analytic ≈ epsilon_default_numeric rtol = 1.0e-8 atol = 1.0e-8

    # 激活点 eta!=0: epsilon 与 eta4 的 determinant log-derivative 都应匹配有限差分.
    active_epsilon = 1.3
    active_eta4 = 0.4
    active_backflow = build_minimal_dd_eta4_backflow(
        bf_epsilon_d=active_epsilon,
        bf_eta4_dd=active_eta4,
        dd_amplitude=1.2,
    )

    eta_active_analytic = compute_backflow_param_log_derivative(
        base_orbitals,
        state_up_hole,
        active_backflow,
        :bf_eta4_dd,
        electron_rows,
    )
    eta_active_numeric = compute_central_finite_difference_log_derivative(
        eta4_value -> compute_backflow_logabsdet(
            base_orbitals,
            state_up_hole,
            build_minimal_dd_eta4_backflow(
                bf_epsilon_d=active_epsilon,
                bf_eta4_dd=eta4_value,
                dd_amplitude=1.2,
            ),
            electron_rows,
        ),
        active_eta4,
        step_size,
    )
    @test eta_active_analytic ≈ eta_active_numeric rtol = 1.0e-8 atol = 1.0e-8

    epsilon_active_analytic = compute_backflow_param_log_derivative(
        base_orbitals,
        state_up_hole,
        active_backflow,
        :bf_epsilon_d,
        electron_rows,
    )
    epsilon_active_numeric = compute_central_finite_difference_log_derivative(
        epsilon_value -> compute_backflow_logabsdet(
            base_orbitals,
            state_up_hole,
            build_minimal_dd_eta4_backflow(
                bf_epsilon_d=epsilon_value,
                bf_eta4_dd=active_eta4,
                dd_amplitude=1.2,
            ),
            electron_rows,
        ),
        active_epsilon,
        step_size,
    )
    @test epsilon_active_analytic ≈ epsilon_active_numeric rtol = 1.0e-8 atol = 1.0e-8
end

@testset "Proposal row-block consistency" begin
    n_sites = 2
    n_orbitals = 2
    dd_bonds = Tuple{Int,Int}[(1, 2)]
    dd_amplitudes = [1.0]
    empty_bonds = Tuple{Int,Int}[]
    empty_amplitudes = Float64[]

    # 构造 eta 和 epsilon 同时激活的配置.
    backflow = build_column_directed_emery_backflow(
        dd_bonds, dd_amplitudes,
        empty_bonds, empty_amplitudes,
        empty_bonds, empty_amplitudes,
        empty_bonds, empty_amplitudes,
        1.5, 1.0,                       # epsilon_d=1.5, epsilon_p=1.0 (无效果)
        0.0, 0.0, 0.0, 0.5,            # dd: eta4=0.5
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    )

    base_orbitals = ones(Float64, 2 * n_sites, n_orbitals)
    state_vector = Int8[1, 0]  # site1=UP, site2=HOLE

    # proposal: site1 UP->UP, site2 HOLE->DN.
    # proposal 后 state_after: site1=UP, site2=DN.
    # For UP row at site 1: n_up=1, site2 is DN (not HOLE), so eta4 factor = n_up*h_dn*H_j = 1*1*0 = 0.
    # 也就是说 proposal 后 eta 不激活, epsilon 也不激活.
    # 那么全重建应该等于 base.
    proposal_on = MoveProposal(
        1, 2,
        Int8(1), Int8(1),    # site1: current=UP, new=UP (不变)
        Int8(0), Int8(2),    # site2: current=HOLE, new=DN
        0, 0, 0, 0, 0, 0, 0,
    )

    state_after = copy(state_vector)
    state_after[1] = proposal_on.new_state1
    state_after[2] = proposal_on.new_state2

    full_orbitals_after = mfVMC.build_backflow_orbitals(base_orbitals, state_after, backflow)
    site_block_buffer = zeros(Float64, 2, size(base_orbitals, 2))
    mfVMC.Backflow.fill_backflow_site_block_after_proposal!(
        site_block_buffer, base_orbitals, state_vector, backflow, proposal_on, 1,
    )
    @test site_block_buffer[1, :] == full_orbitals_after[1, :]
    @test site_block_buffer[2, :] == full_orbitals_after[2, :]

    # 另一个 proposal: site1 UP->UP, site2 HOLE->HOLE (不变),
    # 此时 eta 应激活 (eta4: UP + HOLE)
    proposal_eta = MoveProposal(
        1, 2,
        Int8(1), Int8(1),    # site1: UP -> UP (不变)
        Int8(0), Int8(0),    # site2: HOLE -> HOLE (不变)
        0, 0, 0, 0, 0, 0, 0,
    )

    state_after_eta = copy(state_vector)
    state_after_eta[1] = proposal_eta.new_state1
    state_after_eta[2] = proposal_eta.new_state2

    full_orbitals_after_eta = mfVMC.build_backflow_orbitals(base_orbitals, state_after_eta, backflow)
    site_block_buffer_eta = zeros(Float64, 2, size(base_orbitals, 2))
    mfVMC.Backflow.fill_backflow_site_block_after_proposal!(
        site_block_buffer_eta, base_orbitals, state_vector, backflow, proposal_eta, 1,
    )
    @test site_block_buffer_eta[1, :] == full_orbitals_after_eta[1, :]
    @test site_block_buffer_eta[2, :] == full_orbitals_after_eta[2, :]
end

@testset "Backflow source weights use eta-driven epsilon activation" begin
    source_row_indices = zeros(Int, 4)
    source_row_weights = zeros(Float64, 4)

    active_backflow = build_minimal_dd_eta4_backflow(
        bf_epsilon_d=1.5,
        bf_eta4_dd=0.5,
        dd_amplitude=1.0,
    )
    active_state = Int8[1, 0]
    active_count = mfVMC.Backflow.fill_backflow_row_source_weights_from_state_getter!(
        source_row_indices,
        source_row_weights,
        active_state,
        active_backflow,
        1,
        site_index -> active_state[site_index],
    )
    @test active_count == 2
    @test source_row_indices[1:active_count] == [1, 3]
    @test source_row_weights[1:active_count] ≈ [1.5, 0.5]

    fill!(source_row_indices, 0)
    fill!(source_row_weights, 0.0)
    zero_eta_backflow = build_minimal_dd_eta4_backflow(
        bf_epsilon_d=1.5,
        bf_eta4_dd=0.0,
        dd_amplitude=1.0,
    )
    zero_eta_count = mfVMC.Backflow.fill_backflow_row_source_weights_from_state_getter!(
        source_row_indices,
        source_row_weights,
        active_state,
        zero_eta_backflow,
        1,
        site_index -> active_state[site_index],
    )
    @test zero_eta_count == 1
    @test source_row_indices[1:zero_eta_count] == [1]
    @test source_row_weights[1:zero_eta_count] ≈ [1.0]

    fill!(source_row_indices, 0)
    fill!(source_row_weights, 0.0)
    no_up_state = Int8[2, 0]
    no_up_count = mfVMC.Backflow.fill_backflow_row_source_weights_from_state_getter!(
        source_row_indices,
        source_row_weights,
        no_up_state,
        active_backflow,
        1,
        site_index -> no_up_state[site_index],
    )
    @test no_up_count == 1
    @test source_row_indices[1:no_up_count] == [1]
    @test source_row_weights[1:no_up_count] ≈ [1.0]
end

@testset "Backflow chain-rule row materializes from source weights" begin
    input_derivative_orbitals = [
        2.0 0.2
        3.0 0.3
        5.0 0.5
        7.0 0.7
    ]
    state_up_hole = Int8[1, 0]
    backflow = build_minimal_dd_eta4_backflow(
        bf_epsilon_d=1.5,
        bf_eta4_dd=0.5,
        dd_amplitude=1.0,
    )

    source_row_indices = zeros(Int, 4)
    source_row_weights = zeros(Float64, 4)
    source_count = mfVMC.Backflow.fill_backflow_row_source_weights_from_state_getter!(
        source_row_indices,
        source_row_weights,
        state_up_hole,
        backflow,
        1,
        site_index -> state_up_hole[site_index],
    )

    expected_row = zeros(Float64, size(input_derivative_orbitals, 2))
    mfVMC.Backflow.fill_backflow_row_from_source_weights!(
        expected_row,
        input_derivative_orbitals,
        source_row_indices,
        source_row_weights,
        source_count,
    )

    actual_row = zeros(Float64, size(input_derivative_orbitals, 2))
    mfVMC.Backflow.fill_backflow_chain_rule_row!(
        actual_row,
        input_derivative_orbitals,
        state_up_hole,
        backflow,
        1,
    )

    @test actual_row ≈ expected_row
    @test actual_row ≈ 1.5 .* input_derivative_orbitals[1, :] .+ 0.5 .* input_derivative_orbitals[3, :]
end

@testset "Backflow full and proposal rows materialize from source weights" begin
    base_orbitals = [
        2.0 0.2
        3.0 0.3
        5.0 0.5
        7.0 0.7
    ]
    state_vector = Int8[1, 0]
    backflow = build_minimal_dd_eta4_backflow(
        bf_epsilon_d=1.5,
        bf_eta4_dd=0.5,
        dd_amplitude=1.0,
    )

    source_row_indices = zeros(Int, 4)
    source_row_weights = zeros(Float64, 4)
    source_count = mfVMC.Backflow.fill_backflow_row_source_weights_from_state_getter!(
        source_row_indices,
        source_row_weights,
        state_vector,
        backflow,
        1,
        site_index -> state_vector[site_index],
    )
    expected_row = zeros(Float64, size(base_orbitals, 2))
    mfVMC.Backflow.fill_backflow_row_from_source_weights!(
        expected_row,
        base_orbitals,
        source_row_indices,
        source_row_weights,
        source_count,
    )

    full_orbitals = mfVMC.build_backflow_orbitals(base_orbitals, state_vector, backflow)
    @test full_orbitals[1, :] ≈ expected_row

    proposal = MoveProposal(
        1, 2,
        Int8(1), Int8(0),    # old_state1=UP, old_state2=HOLE
        Int8(1), Int8(0),    # new_state1=UP, new_state2=HOLE (不变)
        0, 0, 0, 0, 0, 0, 0,
    )
    proposal_row = zeros(Float64, size(base_orbitals, 2))
    mfVMC.Backflow.fill_grouped_source_composite_site_row_after_proposal!(
        proposal_row,
        base_orbitals,
        state_vector,
        backflow,
        proposal,
        1,
        1,
    )
    @test proposal_row ≈ expected_row
end

@testset "Backflow proposal row validates site row inputs" begin
    base_orbitals = [
        2.0 0.2
        3.0 0.3
        5.0 0.5
        7.0 0.7
    ]
    state_vector = Int8[1, 0]
    backflow = build_minimal_dd_eta4_backflow(
        bf_epsilon_d=1.5,
        bf_eta4_dd=0.5,
        dd_amplitude=1.0,
    )
    proposal = MoveProposal(
        1, 2,
        Int8(1), Int8(0),
        Int8(1), Int8(0),
        0, 0, 0, 0, 0, 0, 0,
    )

    valid_row_buffer = zeros(Float64, size(base_orbitals, 2))
    short_row_buffer = zeros(Float64, size(base_orbitals, 2) - 1)

    @test_throws ErrorException mfVMC.Backflow.fill_backflow_site_row_after_proposal!(
        valid_row_buffer,
        base_orbitals,
        state_vector,
        backflow,
        proposal,
        1,
        3,
    )
    @test_throws ErrorException mfVMC.Backflow.fill_backflow_site_row_after_proposal!(
        valid_row_buffer,
        base_orbitals,
        state_vector,
        backflow,
        proposal,
        3,
        1,
    )
    @test_throws ErrorException mfVMC.Backflow.fill_backflow_site_row_after_proposal!(
        short_row_buffer,
        base_orbitals,
        state_vector,
        backflow,
        proposal,
        1,
        1,
    )
end

"""
用途: 构造一个小尺寸 Emery grouped backflow determinant 测试对象。

参数:
- 无。

返回:
- `vwf_det`: 已设置 grouped Emery backflow 的 determinant 波函数对象。
"""
function build_grouped_emery_backflow_vwf_fixture()
    lx = 2
    ly = 2
    n_sites = emery_n_sites(lx, ly)
    state_chars = join(("Dhud"[mod1(site_index, 4)] for site_index in 1:n_sites))
    sampler = mfVMC.Sampler.init_config_Hubbard_by_state_char(state_chars)

    dd_source_bonds,
    dd_source_amplitudes,
    dp_source_bonds,
    dp_source_amplitudes,
    pd_source_bonds,
    pd_source_amplitudes,
    pp_source_bonds,
    pp_source_amplitudes = build_emery_backflow_source_data_by_directed_orbital_type(
        lx,
        ly;
        tpd=1.0,
        tpp=0.45,
        bcy=1.0,
    )
    backflow = build_column_directed_emery_backflow(
        dd_source_bonds,
        dd_source_amplitudes,
        dp_source_bonds,
        dp_source_amplitudes,
        pd_source_bonds,
        pd_source_amplitudes,
        pp_source_bonds,
        pp_source_amplitudes,
        0.91,
        1.08,
        0.11,
        -0.07,
        0.05,
        -0.03,
        -0.13,
        0.17,
        -0.19,
        0.23,
        0.29,
        -0.31,
        0.37,
        -0.41,
        -0.43,
        0.47,
        -0.53,
        0.59,
    )

    n_rows = 2 * n_sites
    n_electrons = length(sampler.electron_locs)
    base_orbitals = [
        0.2 * row_index + 0.07 * orbital_index + 0.003 * row_index * orbital_index
        for row_index in 1:n_rows, orbital_index in 1:n_electrons
    ]
    return mfVMC.vwf_det(base_orbitals, sampler; backflow=backflow)
end

@testset "Backflow rebuild materializes only occupied rows" begin
    vwf = build_grouped_emery_backflow_vwf_fixture()
    init_gswf!(vwf)

    full_orbitals = mfVMC.Backflow.build_backflow_orbitals(
        vwf.base_gs_U,
        vwf.sampler.state,
        vwf.backflow,
    )
    expected_slater = full_orbitals[vwf.sampler.electron_locs, :]

    @test vwf.awf_mat_t ≈ transpose(expected_slater)
    @test vwf.awf_val ≈ det(expected_slater)
    @test vwf.awf_inv ≈ inv(expected_slater)
end

@testset "vwf_det does not store full backflow cache" begin
    vwf = build_grouped_emery_backflow_vwf_fixture()
    @test !hasproperty(vwf, :backflow_u)
end

@testset "Backflow rejects gs_U_t rank1 path" begin
    vwf = build_grouped_emery_backflow_vwf_fixture()
    init_gswf!(vwf)
    @test_throws ErrorException mfVMC.VMC.ratio_rank1(vwf, 1, vwf.sampler.electron_locs[1])
end
