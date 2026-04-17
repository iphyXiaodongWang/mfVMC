module Backflow

using ..Sampler

export AbstractBackflowTerm, NoBackflowTerm, Eq4BackflowTerm
export uses_backflow
export backflow_param_names, backflow_param_values, backflow_param_count
export update_backflow_params!
export compute_doublon_hole_masks, compute_recombination_mask
export build_backflow_orbitals, build_eq4_backflow_orbitals
export build_backflow_derivative_orbitals, build_eq4_backflow_derivative_orbitals

"""
用途: 管理 determinant 路线中的 backflow correlation 轨道修正。

当前实现范围:
- 第一阶段只实现 PRB 2008 中 `Eq.(4)` 的核心 orbital dressing。
- 当前项目的 `Hubbard.jl` 使用 `ConfigurationPH`, 即内部基底不是
  `up electron + down electron`, 而是 `up electron + down hole`。
- 因此本模块中的 backflow 公式不是直接作用在“物理电子轨道”上, 而是作用在
  当前 determinant 实际使用的 PH 轨道基底上。

当前代码采用的 `Eq.(4)` 对应写法为:

- `U_b(a, i, k; x) = [1 + (epsilon_bf - 1) * xi_i(x)] * U_0(a, i, k)
   + eta_bf * sum_j[t_ij * D_i(x) * H_j(x) * U_0(a, j, k)]`

其中:
- `a` 表示当前代码中的内部通道, 即 `up` 或 `dn_hole`。
- `i, j` 为格点指标, `k` 为轨道指标。
- `U_0` 为裸轨道矩阵, `U_b` 为构型依赖的 backflow 轨道矩阵。
- `D_i(x) = 1` 当且仅当 site `i` 为 doublon, 否则为 0。
- `H_i(x) = 1` 当且仅当 site `i` 为 hole, 否则为 0。
- `xi_i(x) = 1` 当且仅当存在某个相邻 `j` 使得 `D_i(x) * H_j(x) = 1`,
  否则为 0。

实现约定:
- `source_bonds` 使用有向键 `(i, j)` 表示 `D_i * H_j` 通道。
- `source_amplitudes[n]` 对应该键的 `t_ij`。
- 若物理模型需要无向键, 调用方应显式传入 `(i, j)` 与 `(j, i)` 两条有向键。

后续阶段规划:
- 在 determinant 路线中加入 proposal 后的精确重建比值 `Psi(x') / Psi(x)`。
- 加入 SR 所需的 backflow 参数对数导数:
  `partial_alpha log Psi = Tr[A^{-1} * partial_alpha A]`。
- 若性能成为瓶颈, 再评估局域受影响行的快更新, 第一阶段不提前做复杂优化。
"""

abstract type AbstractBackflowTerm end


"""
用途: 表示未启用 backflow 的空对象。

参数:
- 无。

返回:
- `NoBackflowTerm`, 用于复用统一接口。
"""
struct NoBackflowTerm <: AbstractBackflowTerm
end


"""
用途: 保存 `Eq.(4)` backflow 所需的全局参数与键列表。

数学公式:
- `U_b = [1 + (epsilon_bf - 1) * xi_i] * U_0
   + eta_bf * sum_j[t_ij * D_i * H_j * U_0(j)]`。

参数:
- `param_name_epsilon::Symbol`: `epsilon_bf` 的参数名。
- `param_name_eta::Symbol`: `eta_bf` 的参数名。
- `epsilon_bf::Float64`: `Eq.(4)` 中的 `epsilon`。
- `eta_bf::Float64`: `Eq.(4)` 中的 `eta`。
- `source_bonds::Vector{Tuple{Int, Int}}`: 有向键 `(i, j)` 列表, 表示 `D_i * H_j`。
- `source_amplitudes::Vector{Float64}`: 与 `source_bonds` 对齐的 `t_ij` 振幅列表。

返回:
- `Eq4BackflowTerm` 实例。
"""
mutable struct Eq4BackflowTerm <: AbstractBackflowTerm
    param_name_epsilon::Symbol
    param_name_eta::Symbol
    epsilon_bf::Float64
    eta_bf::Float64
    source_bonds::Vector{Tuple{Int, Int}}
    source_amplitudes::Vector{Float64}
end


"""
用途: 构造 `Eq.(4)` backflow 参数对象, 并为每条键提供默认振幅 `1.0`。

参数:
- `param_name_epsilon::Symbol`: `epsilon_bf` 的参数名。
- `param_name_eta::Symbol`: `eta_bf` 的参数名。
- `epsilon_bf::Real`: `Eq.(4)` 中的 `epsilon`。
- `eta_bf::Real`: `Eq.(4)` 中的 `eta`。
- `source_bonds::Vector{Tuple{Int, Int}}`: 有向键 `(i, j)` 列表。
- `source_amplitudes::Vector{<:Real}`: 每条键对应的 `t_ij`, 默认全为 `1.0`。

返回:
- `Eq4BackflowTerm` 实例。
"""
function Eq4BackflowTerm(;
    param_name_epsilon::Symbol=:bf_epsilon,
    param_name_eta::Symbol=:bf_eta,
    epsilon_bf::Real=1.0,
    eta_bf::Real=0.0,
    source_bonds::Vector{Tuple{Int, Int}}=Tuple{Int, Int}[],
    source_amplitudes::Vector{<:Real}=ones(Float64, length(source_bonds)),
)
    if length(source_bonds) != length(source_amplitudes)
        error("Length mismatch: source_bonds has $(length(source_bonds)) entries, but source_amplitudes has $(length(source_amplitudes)).")
    end

    return Eq4BackflowTerm(
        param_name_epsilon,
        param_name_eta,
        Float64(epsilon_bf),
        Float64(eta_bf),
        copy(source_bonds),
        Float64.(source_amplitudes),
    )
end


"""
用途: 判断是否真的启用了非平凡 backflow。

参数:
- `backflow_term::AbstractBackflowTerm`: backflow 对象。

返回:
- `Bool`: 若不是 `NoBackflowTerm`, 返回 `true`。
"""
uses_backflow(::NoBackflowTerm) = false
uses_backflow(::Eq4BackflowTerm) = true


"""
用途: 返回 backflow 参数名列表。

参数:
- `backflow_term::AbstractBackflowTerm`: backflow 对象。

返回:
- `Vector{Symbol}`: 参数名列表。
"""
backflow_param_names(::NoBackflowTerm) = Symbol[]
function backflow_param_names(backflow_term::Eq4BackflowTerm)
    return Symbol[backflow_term.param_name_epsilon, backflow_term.param_name_eta]
end


"""
用途: 返回 backflow 参数值列表。

参数:
- `backflow_term::AbstractBackflowTerm`: backflow 对象。

返回:
- `Vector{Float64}`: 参数值列表。
"""
backflow_param_values(::NoBackflowTerm) = Float64[]
function backflow_param_values(backflow_term::Eq4BackflowTerm)
    return Float64[backflow_term.epsilon_bf, backflow_term.eta_bf]
end


"""
用途: 返回 backflow 参数总数。

参数:
- `backflow_term::AbstractBackflowTerm`: backflow 对象。

返回:
- `Int`: 参数个数。
"""
function backflow_param_count(backflow_term::AbstractBackflowTerm)
    return length(backflow_param_names(backflow_term))
end


"""
用途: 按名称批量更新 `Eq.(4)` backflow 参数。

参数:
- `backflow_term::Eq4BackflowTerm`: backflow 参数对象。
- `param_names::Vector{Symbol}`: 参数名列表。
- `param_values::Vector{<:Real}`: 参数值列表。

返回:
- `nothing`。
"""
function update_backflow_params!(
    ::NoBackflowTerm,
    param_names::Vector{Symbol},
    param_values::Vector{<:Real},
)
    if !isempty(param_names) || !isempty(param_values)
        error("NoBackflowTerm does not accept any parameters.")
    end
    return nothing
end

function update_backflow_params!(
    backflow_term::Eq4BackflowTerm,
    param_names::Vector{Symbol},
    param_values::Vector{<:Real},
)
    if length(param_names) != length(param_values)
        error("Length mismatch: param_names has $(length(param_names)) entries, but param_values has $(length(param_values)).")
    end

    for (param_name, param_value) in zip(param_names, param_values)
        if param_name == backflow_term.param_name_epsilon
            backflow_term.epsilon_bf = Float64(param_value)
        elseif param_name == backflow_term.param_name_eta
            backflow_term.eta_bf = Float64(param_value)
        else
            error("Unknown backflow parameter name: $param_name")
        end
    end

    return nothing
end


"""
用途: 按对象内部顺序更新 backflow 参数。

参数:
- `backflow_term::AbstractBackflowTerm`: backflow 对象。
- `param_values::Vector{<:Real}`: 参数值列表。

返回:
- `nothing`。
"""
function update_backflow_params!(
    backflow_term::AbstractBackflowTerm,
    param_values::Vector{<:Real},
)
    expected_names = backflow_param_names(backflow_term)
    if length(expected_names) != length(param_values)
        error("Length mismatch: expected $(length(expected_names)) backflow parameters, got $(length(param_values)).")
    end
    return update_backflow_params!(backflow_term, expected_names, param_values)
end


"""
用途: 从当前采样构型中提取 doublon 与 hole 指示函数。

数学公式:
- `D_i = 1`, 当且仅当 site `i` 为 `DB`。
- `H_i = 1`, 当且仅当 site `i` 为 `HOLE`。

参数:
- `state_vector::Vector{Int8}`: 站点状态编码数组。

返回:
- `Tuple{Vector{Float64}, Vector{Float64}}`: `(doublon_mask, hole_mask)`。
"""
function compute_doublon_hole_masks(state_vector::Vector{Int8})
    doublon_mask = Float64[state_code == DB ? 1.0 : 0.0 for state_code in state_vector]
    hole_mask = Float64[state_code == HOLE ? 1.0 : 0.0 for state_code in state_vector]
    return doublon_mask, hole_mask
end


"""
用途: 计算 `Eq.(4)` 中的 `xi_i(x)` 重组掩码。

数学公式:
- `xi_i(x) = 1`, 当且仅当存在某个有向键 `(i, j)` 满足 `D_i(x) * H_j(x) = 1`。
- 否则 `xi_i(x) = 0`。

参数:
- `backflow_term::Eq4BackflowTerm`: backflow 参数对象。
- `doublon_mask::Vector{Float64}`: `D_i` 列表。
- `hole_mask::Vector{Float64}`: `H_i` 列表。

返回:
- `Vector{Float64}`: `xi_i(x)` 数组。
"""
function compute_recombination_mask(
    backflow_term::Eq4BackflowTerm,
    doublon_mask::Vector{Float64},
    hole_mask::Vector{Float64},
)
    if length(doublon_mask) != length(hole_mask)
        error("Mask length mismatch: doublon_mask has $(length(doublon_mask)) entries, but hole_mask has $(length(hole_mask)).")
    end

    recombination_mask = zeros(Float64, length(doublon_mask))
    for (site_i, site_j) in backflow_term.source_bonds
        if doublon_mask[site_i] > 0.5 && hole_mask[site_j] > 0.5
            recombination_mask[site_i] = 1.0
        end
    end

    return recombination_mask
end


"""
用途: 校验轨道矩阵与站点数是否匹配当前 PH 基底。

参数:
- `base_orbitals::AbstractMatrix`: 裸轨道矩阵, 其行数必须为 `2 * N_sites`。
- `n_sites::Int`: 站点数。

返回:
- `nothing`。若维度不匹配则抛出 error。
"""
function validate_orbital_dimensions(base_orbitals::AbstractMatrix, n_sites::Int)
    expected_rows = 2 * n_sites
    if size(base_orbitals, 1) != expected_rows
        error("Orbital row mismatch: expected $expected_rows rows for PH basis, got $(size(base_orbitals, 1)).")
    end
    return nothing
end


"""
用途: 在 `NoBackflowTerm` 情况下直接返回裸轨道副本。

参数:
- `base_orbitals::AbstractMatrix{T}`: 裸轨道矩阵。
- `state_vector::Vector{Int8}`: 当前构型, 此处仅用于接口统一。
- `backflow_term::NoBackflowTerm`: 空 backflow 对象。

返回:
- `Matrix{T}`: 与 `base_orbitals` 相同的轨道矩阵副本。
"""
function build_backflow_orbitals(
    base_orbitals::AbstractMatrix{T},
    state_vector::Vector{Int8},
    ::NoBackflowTerm,
) where {T}
    validate_orbital_dimensions(base_orbitals, length(state_vector))
    return Matrix{T}(base_orbitals)
end


"""
用途: 构造 `Eq.(4)` 的构型依赖 backflow 轨道矩阵。

数学公式:
- `U_b(a, i, k; x) = [1 + (epsilon_bf - 1) * xi_i(x)] * U_0(a, i, k)
   + eta_bf * sum_j[t_ij * D_i(x) * H_j(x) * U_0(a, j, k)]`。

实现说明:
- 对每个站点 `i`, 同时更新两条内部行:
  `row_up = 2 * (i - 1) + 1`,
  `row_dn_hole = 2 * (i - 1) + 2`。
- 第一阶段统一对这两个内部通道施加同一组 backflow 系数。

参数:
- `base_orbitals::AbstractMatrix{T}`: 裸轨道矩阵 `U_0`。
- `state_vector::Vector{Int8}`: 当前 Monte Carlo 构型。
- `backflow_term::Eq4BackflowTerm`: `Eq.(4)` backflow 参数对象。

返回:
- `Matrix{T}`: 构型依赖的 `U_b(x)`。
"""
function build_eq4_backflow_orbitals(
    base_orbitals::AbstractMatrix{T},
    state_vector::Vector{Int8},
    backflow_term::Eq4BackflowTerm,
) where {T}
    n_sites = length(state_vector)
    validate_orbital_dimensions(base_orbitals, n_sites)

    doublon_mask, hole_mask = compute_doublon_hole_masks(state_vector)
    recombination_mask = compute_recombination_mask(backflow_term, doublon_mask, hole_mask)

    backflow_orbitals = Matrix{T}(base_orbitals)

    for site_index in 1:n_sites
        row_up = 2 * (site_index - 1) + 1
        row_dn_hole = 2 * (site_index - 1) + 2
        prefactor = one(T) + T(backflow_term.epsilon_bf - 1.0) * T(recombination_mask[site_index])

        @views backflow_orbitals[row_up, :] .*= prefactor
        @views backflow_orbitals[row_dn_hole, :] .*= prefactor
    end

    for (bond_index, (site_i, site_j)) in enumerate(backflow_term.source_bonds)
        if doublon_mask[site_i] > 0.5 && hole_mask[site_j] > 0.5
            row_i_up = 2 * (site_i - 1) + 1
            row_i_dn_hole = 2 * (site_i - 1) + 2
            row_j_up = 2 * (site_j - 1) + 1
            row_j_dn_hole = 2 * (site_j - 1) + 2
            bond_amplitude = T(backflow_term.source_amplitudes[bond_index])
            eta_value = T(backflow_term.eta_bf)

            @views backflow_orbitals[row_i_up, :] .+= eta_value * bond_amplitude .* base_orbitals[row_j_up, :]
            @views backflow_orbitals[row_i_dn_hole, :] .+= eta_value * bond_amplitude .* base_orbitals[row_j_dn_hole, :]
        end
    end

    return backflow_orbitals
end


"""
用途: 统一入口, 按 backflow 对象类型构造轨道矩阵。

参数:
- `base_orbitals::AbstractMatrix{T}`: 裸轨道矩阵。
- `state_vector::Vector{Int8}`: 当前构型。
- `backflow_term::AbstractBackflowTerm`: backflow 对象。

返回:
- `Matrix{T}`: 处理后的轨道矩阵。
"""
function build_backflow_orbitals(
    base_orbitals::AbstractMatrix{T},
    state_vector::Vector{Int8},
    backflow_term::Eq4BackflowTerm,
) where {T}
    return build_eq4_backflow_orbitals(base_orbitals, state_vector, backflow_term)
end


"""
用途: 在 `NoBackflowTerm` 情况下返回空的导数轨道列表。

参数:
- `base_orbitals::AbstractMatrix{T}`: 裸轨道矩阵。
- `state_vector::Vector{Int8}`: 当前构型。
- `backflow_term::NoBackflowTerm`: 空 backflow 对象。

返回:
- `Vector{Pair{Symbol, Matrix{T}}}`: 空列表。
"""
function build_backflow_derivative_orbitals(
    base_orbitals::AbstractMatrix{T},
    state_vector::Vector{Int8},
    ::NoBackflowTerm,
) where {T}
    validate_orbital_dimensions(base_orbitals, length(state_vector))
    return Pair{Symbol, Matrix{T}}[]
end


"""
用途: 构造 `Eq.(4)` 对 `epsilon_bf` 与 `eta_bf` 的轨道导数矩阵。

数学公式:
- `partial U_b / partial epsilon_bf = xi_i * U_0(i)`。
- `partial U_b / partial eta_bf = sum_j[t_ij * D_i * H_j * U_0(j)]`。

参数:
- `base_orbitals::AbstractMatrix{T}`: 裸轨道矩阵 `U_0`。
- `state_vector::Vector{Int8}`: 当前 Monte Carlo 构型。
- `backflow_term::Eq4BackflowTerm`: `Eq.(4)` backflow 参数对象。

返回:
- `Tuple{Matrix{T}, Matrix{T}}`: `(d_orbitals_epsilon, d_orbitals_eta)`。
"""
function build_eq4_backflow_derivative_orbitals(
    base_orbitals::AbstractMatrix{T},
    state_vector::Vector{Int8},
    backflow_term::Eq4BackflowTerm,
) where {T}
    n_sites = length(state_vector)
    validate_orbital_dimensions(base_orbitals, n_sites)

    doublon_mask, hole_mask = compute_doublon_hole_masks(state_vector)
    recombination_mask = compute_recombination_mask(backflow_term, doublon_mask, hole_mask)

    d_orbitals_epsilon = zeros(T, size(base_orbitals))
    d_orbitals_eta = zeros(T, size(base_orbitals))

    for site_index in 1:n_sites
        if recombination_mask[site_index] > 0.5
            row_up = 2 * (site_index - 1) + 1
            row_dn_hole = 2 * (site_index - 1) + 2
            @views d_orbitals_epsilon[row_up, :] .= base_orbitals[row_up, :]
            @views d_orbitals_epsilon[row_dn_hole, :] .= base_orbitals[row_dn_hole, :]
        end
    end

    for (bond_index, (site_i, site_j)) in enumerate(backflow_term.source_bonds)
        if doublon_mask[site_i] > 0.5 && hole_mask[site_j] > 0.5
            row_i_up = 2 * (site_i - 1) + 1
            row_i_dn_hole = 2 * (site_i - 1) + 2
            row_j_up = 2 * (site_j - 1) + 1
            row_j_dn_hole = 2 * (site_j - 1) + 2
            bond_amplitude = T(backflow_term.source_amplitudes[bond_index])

            @views d_orbitals_eta[row_i_up, :] .+= bond_amplitude .* base_orbitals[row_j_up, :]
            @views d_orbitals_eta[row_i_dn_hole, :] .+= bond_amplitude .* base_orbitals[row_j_dn_hole, :]
        end
    end

    return d_orbitals_epsilon, d_orbitals_eta
end


"""
用途: 统一返回 backflow 参数顺序对应的轨道导数矩阵列表。

参数:
- `base_orbitals::AbstractMatrix{T}`: 裸轨道矩阵。
- `state_vector::Vector{Int8}`: 当前构型。
- `backflow_term::AbstractBackflowTerm`: backflow 对象。

返回:
- `Vector{Pair{Symbol, Matrix{T}}}`: 参数名到轨道导数矩阵的有序列表。
"""
function build_backflow_derivative_orbitals(
    base_orbitals::AbstractMatrix{T},
    state_vector::Vector{Int8},
    backflow_term::Eq4BackflowTerm,
) where {T}
    d_orbitals_epsilon, d_orbitals_eta = build_eq4_backflow_derivative_orbitals(
        base_orbitals,
        state_vector,
        backflow_term,
    )

    return Pair{Symbol, Matrix{T}}[
        backflow_term.param_name_epsilon => d_orbitals_epsilon,
        backflow_term.param_name_eta => d_orbitals_eta,
    ]
end


end # module
