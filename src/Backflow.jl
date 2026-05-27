module Backflow

using ..Sampler

export AbstractBackflowTerm, AbstractBackflowCorrectionTerm
export NoBackflowTerm, Eq4BackflowTerm
export CompositeBackflowTerm
export BackflowEpsilonTerm, BackflowEta1DoublonHoleTerm
export BackflowEta2SpinExchangeTerm, BackflowEta3MixedVirtualHopTerm
export uses_backflow
export backflow_param_names, backflow_param_values, backflow_param_count
export update_backflow_params!
export compute_doublon_hole_masks, compute_recombination_mask
export build_eq4_backflow_graph_cache
export build_backflow_orbitals, build_eq4_backflow_orbitals
export fill_backflow_chain_rule_orbitals!
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
abstract type AbstractBackflowCorrectionTerm end

const BACKFLOW_EPSILON_MASK_TERMS = (:eta1, :eta2, :eta3)


"""
用途: 保存 Eq.(5) 单个 backflow correction term 的公共 source 数据。

参数:
- `source_bonds::Vector{Tuple{Int, Int}}`: 有向键 `(i, j)` 列表。
- `source_amplitudes::Vector{<:Real}`: 与有向键对齐的 hopping 振幅。

返回:
- `NamedTuple`: 包含复制后的 `source_bonds`, `source_amplitudes`, 内容签名与图缓存。
"""
function build_backflow_correction_source_cache(
    source_bonds::Vector{Tuple{Int, Int}},
    source_amplitudes::Vector{<:Real},
)
    if length(source_bonds) != length(source_amplitudes)
        error("Length mismatch: source_bonds has $(length(source_bonds)) entries, but source_amplitudes has $(length(source_amplitudes)).")
    end

    source_bonds_copy = copy(source_bonds)
    source_amplitudes_copy = Float64.(source_amplitudes)
    graph_cache = build_eq4_backflow_graph_cache(source_bonds_copy)
    source_data_signature = compute_eq4_backflow_source_data_signature(
        source_bonds_copy,
        source_amplitudes_copy,
    )

    return (
        source_bonds=source_bonds_copy,
        source_amplitudes=source_amplitudes_copy,
        source_data_signature=source_data_signature,
        outgoing_bond_indices_by_source=graph_cache.outgoing_bond_indices_by_source,
        incoming_source_sites_by_target=graph_cache.incoming_source_sites_by_target,
    )
end

"""
用途: 规范化 `epsilon` prefactor 使用的 virtual hopping mask 通道列表。

参数:
- `epsilon_mask_terms::AbstractVector{Symbol}`: 允许的元素为 `:eta1`, `:eta2`, `:eta3`。

返回:
- `Vector{Symbol}`: 去重后且保持输入顺序的 mask 通道列表。
"""
function normalize_backflow_epsilon_mask_terms(
    epsilon_mask_terms::AbstractVector{Symbol},
)::Vector{Symbol}
    normalized_terms = Symbol[]
    for mask_term in epsilon_mask_terms
        if !(mask_term in BACKFLOW_EPSILON_MASK_TERMS)
            error("Unknown epsilon mask term $(mask_term). Allowed terms are :eta1, :eta2, and :eta3.")
        end
        if !(mask_term in normalized_terms)
            push!(normalized_terms, mask_term)
        end
    end
    return normalized_terms
end


"""
用途: 保存 `Eq.(4)` backflow 的图缓存, 便于快速定位受影响站点。

参数:
- `outgoing_bond_indices_by_source::Vector{Vector{Int}}`: 按 source site 存储的 bond 索引列表。
- `incoming_source_sites_by_target::Vector{Vector{Int}}`: 按 target site 存储的 source site 去重列表。

返回:
- `NamedTuple`: 包含上述两个缓存数组。
"""
function build_eq4_backflow_graph_cache(source_bonds::Vector{Tuple{Int, Int}})
    max_site_index = 0
    for (bond_index, (site_i, site_j)) in enumerate(source_bonds)
        if site_i < 1 || site_j < 1
            error("Invalid source_bonds[$bond_index] = ($(site_i), $(site_j)): site indices must be positive.")
        end
        max_site_index = max(max_site_index, site_i, site_j)
    end

    outgoing_bond_indices_by_source = [Int[] for _ in 1:max_site_index]
    incoming_source_sites_by_target = [Int[] for _ in 1:max_site_index]

    for (bond_index, (source_site, target_site)) in enumerate(source_bonds)
        push!(outgoing_bond_indices_by_source[source_site], bond_index)
        push!(incoming_source_sites_by_target[target_site], source_site)
    end

    for source_site in 1:max_site_index
        sort!(unique!(outgoing_bond_indices_by_source[source_site]))
        sort!(unique!(incoming_source_sites_by_target[source_site]))
    end

    return (
        outgoing_bond_indices_by_source=outgoing_bond_indices_by_source,
        incoming_source_sites_by_target=incoming_source_sites_by_target,
    )
end


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

结构约定:
- `source_bonds` 与 `source_amplitudes` 在构造后禁止原地修改。
- 若需要修改 bond 拓扑或 `t_ij`, 应重新构造新的 `Eq4BackflowTerm`。

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
    source_data_signature::UInt
    outgoing_bond_indices_by_source::Vector{Vector{Int}}
    incoming_source_sites_by_target::Vector{Vector{Int}}
end


"""
用途: Eq.(5) 中的 `epsilon` backflow correction term。

数学公式:
- `delta U_epsilon(i, sigma) = (epsilon_bf - 1) * xi_{i,sigma} * U_0(i, sigma)`。

字段:
- `param_name::Symbol`: 参数名。
- `epsilon_bf::Float64`: `epsilon` 参数值。
- `epsilon_mask_terms::Vector{Symbol}`: 控制 `xi_{i,sigma}` 的 virtual hopping 通道。
- `source_bonds::Vector{Tuple{Int, Int}}`: 有向键 `(i, j)` 列表。
- `source_amplitudes::Vector{Float64}`: 与有向键对齐的 hopping 振幅。
"""
mutable struct BackflowEpsilonTerm <: AbstractBackflowCorrectionTerm
    param_name::Symbol
    epsilon_bf::Float64
    epsilon_mask_terms::Vector{Symbol}
    source_bonds::Vector{Tuple{Int, Int}}
    source_amplitudes::Vector{Float64}
    source_data_signature::UInt
    outgoing_bond_indices_by_source::Vector{Vector{Int}}
    incoming_source_sites_by_target::Vector{Vector{Int}}
end

"""
用途: Eq.(5) 中的 `eta1` doublon-hole backflow correction term。

数学公式:
- `delta U_eta1(i, sigma) = eta1_bf * sum_j t_ij * D_i * H_j * U_0(j, sigma)`。

字段:
- `param_name::Symbol`: 参数名。
- `eta1_bf::Float64`: `eta1` 参数值。
- `source_bonds::Vector{Tuple{Int, Int}}`: 有向键 `(i, j)` 列表。
- `source_amplitudes::Vector{Float64}`: 与有向键对齐的 hopping 振幅。
"""
mutable struct BackflowEta1DoublonHoleTerm <: AbstractBackflowCorrectionTerm
    param_name::Symbol
    eta1_bf::Float64
    source_bonds::Vector{Tuple{Int, Int}}
    source_amplitudes::Vector{Float64}
    source_data_signature::UInt
    outgoing_bond_indices_by_source::Vector{Vector{Int}}
    incoming_source_sites_by_target::Vector{Vector{Int}}
end

"""
用途: Eq.(5) 中的 `eta2` spin-exchange backflow correction term。

数学公式:
- `delta U_eta2(i, sigma) = eta2_bf * sum_j t_ij *
   n_i_sigma h_i_-sigma n_j_-sigma h_j_sigma * U_0(j, sigma)`。

字段:
- `param_name::Symbol`: 参数名。
- `eta2_bf::Float64`: `eta2` 参数值。
- `source_bonds::Vector{Tuple{Int, Int}}`: 有向键 `(i, j)` 列表。
- `source_amplitudes::Vector{Float64}`: 与有向键对齐的 hopping 振幅。
"""
mutable struct BackflowEta2SpinExchangeTerm <: AbstractBackflowCorrectionTerm
    param_name::Symbol
    eta2_bf::Float64
    source_bonds::Vector{Tuple{Int, Int}}
    source_amplitudes::Vector{Float64}
    source_data_signature::UInt
    outgoing_bond_indices_by_source::Vector{Vector{Int}}
    incoming_source_sites_by_target::Vector{Vector{Int}}
end

"""
用途: Eq.(5) 中的 `eta3` mixed virtual hopping backflow correction term。

数学公式:
- `delta U_eta3(i, sigma) = eta3_bf * sum_j t_ij *
   (D_i n_j_-sigma h_j_sigma + n_i_sigma h_i_-sigma H_j) * U_0(j, sigma)`。

字段:
- `param_name::Symbol`: 参数名。
- `eta3_bf::Float64`: `eta3` 参数值。
- `source_bonds::Vector{Tuple{Int, Int}}`: 有向键 `(i, j)` 列表。
- `source_amplitudes::Vector{Float64}`: 与有向键对齐的 hopping 振幅。
"""
mutable struct BackflowEta3MixedVirtualHopTerm <: AbstractBackflowCorrectionTerm
    param_name::Symbol
    eta3_bf::Float64
    source_bonds::Vector{Tuple{Int, Int}}
    source_amplitudes::Vector{Float64}
    source_data_signature::UInt
    outgoing_bond_indices_by_source::Vector{Vector{Int}}
    incoming_source_sites_by_target::Vector{Vector{Int}}
end

"""
用途: 组合多个 Eq.(5) backflow correction terms。

字段:
- `terms::Vector{AbstractBackflowCorrectionTerm}`: 按参数顺序保存的 backflow correction term 列表。
"""
mutable struct CompositeBackflowTerm <: AbstractBackflowTerm
    terms::Vector{AbstractBackflowCorrectionTerm}
    function CompositeBackflowTerm(terms::Vector{<:AbstractBackflowCorrectionTerm})
        return new(AbstractBackflowCorrectionTerm[terms...])
    end
end


"""
用途: 基于 `source_bonds` 与 `source_amplitudes` 的内容生成一致性签名。

参数:
- `source_bonds::Vector{Tuple{Int, Int}}`: 有向键 `(i, j)` 列表。
- `source_amplitudes::Vector{<:Real}`: 与有向键对齐的 `t_ij` 列表。

返回:
- `UInt`: 用于检测原地修改的内容签名。
"""
function compute_eq4_backflow_source_data_signature(
    source_bonds::Vector{Tuple{Int, Int}},
    source_amplitudes::Vector{<:Real},
)
    source_data_signature = hash(length(source_bonds))
    for source_bond in source_bonds
        source_data_signature = hash(source_bond, source_data_signature)
    end

    source_data_signature = hash(length(source_amplitudes), source_data_signature)
    for source_amplitude in source_amplitudes
        source_data_signature = hash(Float64(source_amplitude), source_data_signature)
    end

    return source_data_signature
end


"""
用途: 校验 `Eq4BackflowTerm` 的 source 数据在构造后未被原地修改。

参数:
- `backflow_term::Eq4BackflowTerm`: 待校验的 backflow 参数对象。

返回:
- `nothing`。若检测到原地修改则抛出 error。
"""
function validate_eq4_backflow_source_data!(
    backflow_term::Eq4BackflowTerm,
)
    current_signature = compute_eq4_backflow_source_data_signature(
        backflow_term.source_bonds,
        backflow_term.source_amplitudes,
    )

    if current_signature != backflow_term.source_data_signature
        error("Eq4BackflowTerm source_bonds/source_amplitudes were mutated after construction. Please rebuild Eq4BackflowTerm instead of modifying it in place.")
    end

    return nothing
end


"""
用途: 构造带图缓存的 `Eq.(4)` backflow 参数对象。

参数:
- `param_name_epsilon::Symbol`: `epsilon_bf` 的参数名。
- `param_name_eta::Symbol`: `eta_bf` 的参数名。
- `epsilon_bf::Real`: `Eq.(4)` 中的 `epsilon`。
- `eta_bf::Real`: `Eq.(4)` 中的 `eta`。
- `source_bonds::Vector{Tuple{Int, Int}}`: 有向键 `(i, j)` 列表。
- `source_amplitudes::Vector{<:Real}`: 每条键对应的 `t_ij`。

返回:
- `Eq4BackflowTerm` 实例。
"""
function build_eq4_backflow_term_with_cache(
    param_name_epsilon::Symbol,
    param_name_eta::Symbol,
    epsilon_bf::Real,
    eta_bf::Real,
    source_bonds::Vector{Tuple{Int, Int}},
    source_amplitudes::Vector{<:Real},
)
    if length(source_bonds) != length(source_amplitudes)
        error("Length mismatch: source_bonds has $(length(source_bonds)) entries, but source_amplitudes has $(length(source_amplitudes)).")
    end

    source_bonds_copy = copy(source_bonds)
    source_amplitudes_copy = Float64.(source_amplitudes)
    graph_cache = build_eq4_backflow_graph_cache(source_bonds_copy)
    source_data_signature = compute_eq4_backflow_source_data_signature(
        source_bonds_copy,
        source_amplitudes_copy,
    )

    return Eq4BackflowTerm(
        param_name_epsilon,
        param_name_eta,
        Float64(epsilon_bf),
        Float64(eta_bf),
        source_bonds_copy,
        source_amplitudes_copy,
        source_data_signature,
        graph_cache.outgoing_bond_indices_by_source,
        graph_cache.incoming_source_sites_by_target,
    )
end


"""
用途: 兼容旧的 positional 构造方式, 并自动补齐图缓存。

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
function Eq4BackflowTerm(
    param_name_epsilon::Symbol,
    param_name_eta::Symbol,
    epsilon_bf::Real,
    eta_bf::Real,
    source_bonds::Vector{Tuple{Int, Int}},
    source_amplitudes::Vector{<:Real},
)
    return build_eq4_backflow_term_with_cache(
        param_name_epsilon,
        param_name_eta,
        epsilon_bf,
        eta_bf,
        source_bonds,
        source_amplitudes,
    )
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
    return build_eq4_backflow_term_with_cache(
        param_name_epsilon,
        param_name_eta,
        epsilon_bf,
        eta_bf,
        source_bonds,
        source_amplitudes,
    )
end


"""
用途: 构造 Eq.(5) 的 `epsilon` backflow correction term。

参数:
- `param_name::Symbol`: 参数名, 默认 `:bf_epsilon`。
- `epsilon_bf::Real`: `epsilon` 参数值。
- `epsilon_mask_terms::AbstractVector{Symbol}`: 打开 `epsilon` prefactor 的 virtual hopping 通道,
  允许 `:eta1`, `:eta2`, `:eta3`。
- `source_bonds::Vector{Tuple{Int, Int}}`: 有向键 `(i, j)` 列表。
- `source_amplitudes::Vector{<:Real}`: 每条有向键对应的 hopping 振幅。

返回:
- `BackflowEpsilonTerm`: 带 source 缓存的 correction term。
"""
function BackflowEpsilonTerm(;
    param_name::Symbol=:bf_epsilon,
    epsilon_bf::Real=1.0,
    epsilon_mask_terms::AbstractVector{Symbol}=Symbol[:eta1],
    source_bonds::Vector{Tuple{Int, Int}}=Tuple{Int, Int}[],
    source_amplitudes::Vector{<:Real}=ones(Float64, length(source_bonds)),
)
    source_cache = build_backflow_correction_source_cache(source_bonds, source_amplitudes)
    return BackflowEpsilonTerm(
        param_name,
        Float64(epsilon_bf),
        normalize_backflow_epsilon_mask_terms(epsilon_mask_terms),
        source_cache.source_bonds,
        source_cache.source_amplitudes,
        source_cache.source_data_signature,
        source_cache.outgoing_bond_indices_by_source,
        source_cache.incoming_source_sites_by_target,
    )
end

"""
用途: 构造 Eq.(5) 的 `eta1` doublon-hole backflow correction term。

参数:
- `param_name::Symbol`: 参数名, 默认 `:bf_eta1`。
- `eta1_bf::Real`: `eta1` 参数值。
- `source_bonds::Vector{Tuple{Int, Int}}`: 有向键 `(i, j)` 列表。
- `source_amplitudes::Vector{<:Real}`: 每条有向键对应的 hopping 振幅。

返回:
- `BackflowEta1DoublonHoleTerm`: 带 source 缓存的 correction term。
"""
function BackflowEta1DoublonHoleTerm(;
    param_name::Symbol=:bf_eta1,
    eta1_bf::Real=0.0,
    source_bonds::Vector{Tuple{Int, Int}}=Tuple{Int, Int}[],
    source_amplitudes::Vector{<:Real}=ones(Float64, length(source_bonds)),
)
    source_cache = build_backflow_correction_source_cache(source_bonds, source_amplitudes)
    return BackflowEta1DoublonHoleTerm(
        param_name,
        Float64(eta1_bf),
        source_cache.source_bonds,
        source_cache.source_amplitudes,
        source_cache.source_data_signature,
        source_cache.outgoing_bond_indices_by_source,
        source_cache.incoming_source_sites_by_target,
    )
end

"""
用途: 构造 Eq.(5) 的 `eta2` spin-exchange backflow correction term。

参数:
- `param_name::Symbol`: 参数名, 默认 `:bf_eta2`。
- `eta2_bf::Real`: `eta2` 参数值。
- `source_bonds::Vector{Tuple{Int, Int}}`: 有向键 `(i, j)` 列表。
- `source_amplitudes::Vector{<:Real}`: 每条有向键对应的 hopping 振幅。

返回:
- `BackflowEta2SpinExchangeTerm`: 带 source 缓存的 correction term。
"""
function BackflowEta2SpinExchangeTerm(;
    param_name::Symbol=:bf_eta2,
    eta2_bf::Real=0.0,
    source_bonds::Vector{Tuple{Int, Int}}=Tuple{Int, Int}[],
    source_amplitudes::Vector{<:Real}=ones(Float64, length(source_bonds)),
)
    source_cache = build_backflow_correction_source_cache(source_bonds, source_amplitudes)
    return BackflowEta2SpinExchangeTerm(
        param_name,
        Float64(eta2_bf),
        source_cache.source_bonds,
        source_cache.source_amplitudes,
        source_cache.source_data_signature,
        source_cache.outgoing_bond_indices_by_source,
        source_cache.incoming_source_sites_by_target,
    )
end

"""
用途: 构造 Eq.(5) 的 `eta3` mixed virtual hopping backflow correction term。

参数:
- `param_name::Symbol`: 参数名, 默认 `:bf_eta3`。
- `eta3_bf::Real`: `eta3` 参数值。
- `source_bonds::Vector{Tuple{Int, Int}}`: 有向键 `(i, j)` 列表。
- `source_amplitudes::Vector{<:Real}`: 每条有向键对应的 hopping 振幅。

返回:
- `BackflowEta3MixedVirtualHopTerm`: 带 source 缓存的 correction term。
"""
function BackflowEta3MixedVirtualHopTerm(;
    param_name::Symbol=:bf_eta3,
    eta3_bf::Real=0.0,
    source_bonds::Vector{Tuple{Int, Int}}=Tuple{Int, Int}[],
    source_amplitudes::Vector{<:Real}=ones(Float64, length(source_bonds)),
)
    source_cache = build_backflow_correction_source_cache(source_bonds, source_amplitudes)
    return BackflowEta3MixedVirtualHopTerm(
        param_name,
        Float64(eta3_bf),
        source_cache.source_bonds,
        source_cache.source_amplitudes,
        source_cache.source_data_signature,
        source_cache.outgoing_bond_indices_by_source,
        source_cache.incoming_source_sites_by_target,
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
uses_backflow(::CompositeBackflowTerm) = true

"""
用途: 返回单个 Eq.(5) correction term 的参数名。

参数:
- `correction_term::AbstractBackflowCorrectionTerm`: backflow correction term。

返回:
- `Symbol`: correction term 的参数名。
"""
backflow_correction_param_name(correction_term::AbstractBackflowCorrectionTerm)::Symbol = correction_term.param_name

"""
用途: 返回单个 Eq.(5) correction term 的参数值。

参数:
- `correction_term::AbstractBackflowCorrectionTerm`: backflow correction term。

返回:
- `Float64`: correction term 的参数值。
"""
function backflow_correction_param_value(correction_term::BackflowEpsilonTerm)::Float64
    return correction_term.epsilon_bf
end
function backflow_correction_param_value(correction_term::BackflowEta1DoublonHoleTerm)::Float64
    return correction_term.eta1_bf
end
function backflow_correction_param_value(correction_term::BackflowEta2SpinExchangeTerm)::Float64
    return correction_term.eta2_bf
end
function backflow_correction_param_value(correction_term::BackflowEta3MixedVirtualHopTerm)::Float64
    return correction_term.eta3_bf
end


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
function backflow_param_names(backflow_term::CompositeBackflowTerm)
    return Symbol[backflow_correction_param_name(term) for term in backflow_term.terms]
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
function backflow_param_values(backflow_term::CompositeBackflowTerm)
    return Float64[backflow_correction_param_value(term) for term in backflow_term.terms]
end


"""
用途: 收集一次 proposal 会影响到的站点索引。

规则:
- 先包含 proposal 涉及的改动站点。
- 再包含所有其 outgoing bonds 会指向这些改动站点, 且当前构型中为 doublon 的 source site。
- 返回结果按站点索引升序排列, 且不重复。

参数:
- `state_vector::Vector{Int8}`: 当前构型的状态数组, 用于提供站点总数并做边界检查。
- `backflow_term::Eq4BackflowTerm`: backflow 图缓存对象。
- `proposal::MoveProposal`: Monte Carlo proposal。

返回:
- `Vector{Int}`: 受影响的站点索引列表。
"""
function collect_affected_site_indices(
    state_vector::Vector{Int8},
    backflow_term::Eq4BackflowTerm,
    proposal::MoveProposal,
)
    validate_eq4_backflow_source_data!(backflow_term)

    n_sites = length(state_vector)
    affected_site_set = Set{Int}()

    if 1 <= proposal.site1 <= n_sites
        push!(affected_site_set, proposal.site1)
    end
    if 1 <= proposal.site2 <= n_sites
        push!(affected_site_set, proposal.site2)
    end

    max_target_site = min(n_sites, length(backflow_term.incoming_source_sites_by_target))
    for target_site in (proposal.site1, proposal.site2)
        if 1 <= target_site <= max_target_site
            for source_site in backflow_term.incoming_source_sites_by_target[target_site]
                if !(1 <= source_site <= n_sites)
                    error("Backflow cache source site $source_site is out of bounds for a state vector with $n_sites sites.")
                end
                if source_site != proposal.site1 && source_site != proposal.site2 && state_vector[source_site] != DB
                    continue
                end
                push!(affected_site_set, source_site)
            end
        end
    end

    return sort!(collect(affected_site_set))
end

"""
用途: 向站点索引列表中追加一个尚未出现的站点, 用于局域更新热路径的小集合去重。

参数:
- `site_indices::Vector{Int}`: 已收集的站点索引列表。
- `site_index::Int`: 候选站点索引。

返回:
- `nothing`。
"""
function push_unique_site_index!(
    site_indices::Vector{Int},
    site_index::Int,
)
    for existing_site_index in site_indices
        if existing_site_index == site_index
            return nothing
        end
    end
    push!(site_indices, site_index)
    return nothing
end

"""
用途: 收集一次 proposal 会影响到的组合式 Eq.(5) backflow 站点索引。

规则:
- 先包含 proposal 直接改变的站点。
- 对每个 correction term, 若某个改变站点是有向键 `(i, j)` 的 target `j`,
  则 source `i` 对应的轨道行也会依赖改变后的 `j`, 因而加入受影响列表。
- 返回结果按站点索引升序排列, 且不重复。

参数:
- `state_vector::Vector{Int8}`: 当前构型的状态数组, 用于提供站点总数并做边界检查。
- `backflow_term::CompositeBackflowTerm`: 组合式 Eq.(5) backflow 对象。
- `proposal::MoveProposal`: Monte Carlo proposal。

返回:
- `Vector{Int}`: 受影响的站点索引列表。
"""
function collect_affected_site_indices(
    state_vector::Vector{Int8},
    backflow_term::CompositeBackflowTerm,
    proposal::MoveProposal,
)
    n_sites = length(state_vector)
    affected_sites = Int[]

    for changed_site in (proposal.site1, proposal.site2)
        if 1 <= changed_site <= n_sites
            push_unique_site_index!(affected_sites, changed_site)
        end
    end

    for correction_term in backflow_term.terms
        max_target_site = min(n_sites, length(correction_term.incoming_source_sites_by_target))
        for changed_site in (proposal.site1, proposal.site2)
            if !(1 <= changed_site <= max_target_site)
                continue
            end

            for source_site in correction_term.incoming_source_sites_by_target[changed_site]
                if !(1 <= source_site <= n_sites)
                    error("Backflow cache source site $source_site is out of bounds for a state vector with $n_sites sites.")
                end
                push_unique_site_index!(affected_sites, source_site)
            end
        end
    end

    return sort!(affected_sites)
end


"""
用途: 返回某个站点在 proposal 提交后的状态编码。

参数:
- `state_vector::Vector{Int8}`: proposal 提交前的构型状态数组。
- `proposal::MoveProposal`: Monte Carlo proposal。
- `site_index::Int`: 需要查询的站点编号。

返回:
- `Int8`: proposal 提交后的站点状态编码。
"""
@inline function get_site_state_after_proposal(
    state_vector::Vector{Int8},
    proposal::MoveProposal,
    site_index::Int,
)
    if site_index == proposal.site1
        return proposal.new_state1
    elseif site_index == proposal.site2
        return proposal.new_state2
    end

    return state_vector[site_index]
end


"""
用途: 将单个受影响站点在 proposal 提交后的局域 backflow 行块写入预分配 buffer。

数学公式:
- `U_b(a, i, k; x') = [1 + (epsilon_bf - 1) * xi_i(x')] * U_0(a, i, k)
   + eta_bf * sum_j[t_ij * D_i(x') * H_j(x') * U_0(a, j, k)]`。

参数:
- `site_block_buffer::AbstractMatrix{T}`: 输出 buffer, 形状必须为 `2 x N_orb`。
- `base_orbitals::AbstractMatrix{T}`: 裸轨道矩阵 `U_0`。
- `state_vector::Vector{Int8}`: proposal 提交前的构型状态数组。
- `backflow_term::Eq4BackflowTerm`: `Eq.(4)` backflow 参数对象。
- `proposal::MoveProposal`: Monte Carlo proposal。
- `site_index::Int`: 待写入的站点编号。

返回:
- `nothing`。
"""
function fill_eq4_backflow_site_block_after_proposal!(
    site_block_buffer::AbstractMatrix{T},
    base_orbitals::AbstractMatrix{T},
    state_vector::Vector{Int8},
    backflow_term::Eq4BackflowTerm,
    proposal::MoveProposal,
    site_index::Int,
) where {T}
    if size(site_block_buffer, 1) != 2 || size(site_block_buffer, 2) != size(base_orbitals, 2)
        error("Local site block buffer must have shape (2, $(size(base_orbitals, 2))), got $(size(site_block_buffer)).")
    end

    row_up = 2 * (site_index - 1) + 1
    row_dn_hole = 2 * (site_index - 1) + 2
    copyto!(@view(site_block_buffer[1, :]), @view(base_orbitals[row_up, :]))
    copyto!(@view(site_block_buffer[2, :]), @view(base_orbitals[row_dn_hole, :]))

    site_state_after = get_site_state_after_proposal(state_vector, proposal, site_index)
    is_doublon_after = site_state_after == DB
    xi_value = false

    if site_index <= length(backflow_term.outgoing_bond_indices_by_source) && is_doublon_after
        for bond_index in backflow_term.outgoing_bond_indices_by_source[site_index]
            (_, target_site) = backflow_term.source_bonds[bond_index]
            if get_site_state_after_proposal(state_vector, proposal, target_site) == HOLE
                xi_value = true
                break
            end
        end
    end

    prefactor = one(T) + T(backflow_term.epsilon_bf - 1.0) * T(xi_value)
    site_block_buffer .*= prefactor

    if site_index <= length(backflow_term.outgoing_bond_indices_by_source) && is_doublon_after
        eta_value = T(backflow_term.eta_bf)
        for bond_index in backflow_term.outgoing_bond_indices_by_source[site_index]
            (_, target_site) = backflow_term.source_bonds[bond_index]
            if get_site_state_after_proposal(state_vector, proposal, target_site) != HOLE
                continue
            end

            target_row_up = 2 * (target_site - 1) + 1
            target_row_dn_hole = 2 * (target_site - 1) + 2
            bond_amplitude = T(backflow_term.source_amplitudes[bond_index])

            @views site_block_buffer[1, :] .+= eta_value * bond_amplitude .* base_orbitals[target_row_up, :]
            @views site_block_buffer[2, :] .+= eta_value * bond_amplitude .* base_orbitals[target_row_dn_hole, :]
        end
    end

    return nothing
end

"""
用途: 用统一接口写入 Eq.(4) backflow 在 proposal 提交后的局域站点行块。

参数:
- `site_block_buffer::AbstractMatrix{T}`: 输出 buffer, 形状必须为 `2 x N_orb`。
- `base_orbitals::AbstractMatrix{T}`: 裸轨道矩阵 `U_0`。
- `state_vector::Vector{Int8}`: proposal 提交前的构型状态数组。
- `backflow_term::Eq4BackflowTerm`: `Eq.(4)` backflow 参数对象。
- `proposal::MoveProposal`: Monte Carlo proposal。
- `site_index::Int`: 待写入的站点编号。

返回:
- `nothing`。
"""
function fill_backflow_site_block_after_proposal!(
    site_block_buffer::AbstractMatrix{T},
    base_orbitals::AbstractMatrix{T},
    state_vector::Vector{Int8},
    backflow_term::Eq4BackflowTerm,
    proposal::MoveProposal,
    site_index::Int,
) where {T}
    return fill_eq4_backflow_site_block_after_proposal!(
        site_block_buffer,
        base_orbitals,
        state_vector,
        backflow_term,
        proposal,
        site_index,
    )
end

"""
用途: 将 Eq.(5) 的 `epsilon` correction term 在 proposal 后对单个站点行块的贡献累加到 buffer。

数学公式:
- `delta U_epsilon(i, sigma; x') = (epsilon_bf - 1) * xi_{i,sigma}(x') * U_0(i, sigma)`。

参数:
- `site_block_buffer::AbstractMatrix{T}`: 待累加的 `2 x N_orb` 站点行块。
- `base_orbitals::AbstractMatrix{T}`: 裸轨道矩阵 `U_0`。
- `state_vector::Vector{Int8}`: proposal 提交前的构型状态数组。
- `correction_term::BackflowEpsilonTerm`: `epsilon` correction term。
- `proposal::MoveProposal`: Monte Carlo proposal。
- `site_index::Int`: 待写入的站点编号。

返回:
- `nothing`。
"""
function add_backflow_correction_site_block_after_proposal!(
    site_block_buffer::AbstractMatrix{T},
    base_orbitals::AbstractMatrix{T},
    state_vector::Vector{Int8},
    correction_term::BackflowEpsilonTerm,
    proposal::MoveProposal,
    site_index::Int,
) where {T}
    site_state_after = get_site_state_after_proposal(state_vector, proposal, site_index)
    if site_index > length(correction_term.outgoing_bond_indices_by_source)
        return nothing
    end

    epsilon_shift = T(correction_term.epsilon_bf - 1.0)
    for row_offset in 1:2
        spin = backflow_spin_from_row_offset(row_offset)
        xi_value = false
        for bond_index in correction_term.outgoing_bond_indices_by_source[site_index]
            (_, target_site) = correction_term.source_bonds[bond_index]
            target_state_after = get_site_state_after_proposal(state_vector, proposal, target_site)
            if is_backflow_epsilon_row_active(
                site_state_after,
                target_state_after,
                spin,
                correction_term.epsilon_mask_terms,
            )
                xi_value = true
                break
            end
        end
        if xi_value
            row_index = 2 * (site_index - 1) + row_offset
            @views site_block_buffer[row_offset, :] .+= epsilon_shift .* base_orbitals[row_index, :]
        end
    end

    return nothing
end

"""
用途: 将 Eq.(5) 的 `eta1` correction term 在 proposal 后对单个站点行块的贡献累加到 buffer。

数学公式:
- `delta U_eta1(i, sigma; x') = eta1_bf * sum_j t_ij * D_i(x') * H_j(x') * U_0(j, sigma)`。

参数:
- `site_block_buffer::AbstractMatrix{T}`: 待累加的 `2 x N_orb` 站点行块。
- `base_orbitals::AbstractMatrix{T}`: 裸轨道矩阵 `U_0`。
- `state_vector::Vector{Int8}`: proposal 提交前的构型状态数组。
- `correction_term::BackflowEta1DoublonHoleTerm`: `eta1` correction term。
- `proposal::MoveProposal`: Monte Carlo proposal。
- `site_index::Int`: 待写入的站点编号。

返回:
- `nothing`。
"""
function add_backflow_correction_site_block_after_proposal!(
    site_block_buffer::AbstractMatrix{T},
    base_orbitals::AbstractMatrix{T},
    state_vector::Vector{Int8},
    correction_term::BackflowEta1DoublonHoleTerm,
    proposal::MoveProposal,
    site_index::Int,
) where {T}
    site_state_after = get_site_state_after_proposal(state_vector, proposal, site_index)
    if site_state_after != DB || site_index > length(correction_term.outgoing_bond_indices_by_source)
        return nothing
    end

    eta1_value = T(correction_term.eta1_bf)
    for bond_index in correction_term.outgoing_bond_indices_by_source[site_index]
        (_, target_site) = correction_term.source_bonds[bond_index]
        if get_site_state_after_proposal(state_vector, proposal, target_site) != HOLE
            continue
        end

        target_row_up = 2 * (target_site - 1) + 1
        target_row_dn_hole = target_row_up + 1
        bond_amplitude = T(correction_term.source_amplitudes[bond_index])
        @views site_block_buffer[1, :] .+= eta1_value * bond_amplitude .* base_orbitals[target_row_up, :]
        @views site_block_buffer[2, :] .+= eta1_value * bond_amplitude .* base_orbitals[target_row_dn_hole, :]
    end

    return nothing
end

"""
用途: 将 Eq.(5) 的 `eta2` correction term 在 proposal 后对单个站点行块的贡献累加到 buffer。

数学公式:
- `delta U_eta2(i, sigma; x') = eta2_bf * sum_j t_ij *
   n_i_sigma h_i_-sigma n_j_-sigma h_j_sigma * U_0(j, sigma)`。

参数:
- `site_block_buffer::AbstractMatrix{T}`: 待累加的 `2 x N_orb` 站点行块。
- `base_orbitals::AbstractMatrix{T}`: 裸轨道矩阵 `U_0`。
- `state_vector::Vector{Int8}`: proposal 提交前的构型状态数组。
- `correction_term::BackflowEta2SpinExchangeTerm`: `eta2` correction term。
- `proposal::MoveProposal`: Monte Carlo proposal。
- `site_index::Int`: 待写入的站点编号。

返回:
- `nothing`。
"""
function add_backflow_correction_site_block_after_proposal!(
    site_block_buffer::AbstractMatrix{T},
    base_orbitals::AbstractMatrix{T},
    state_vector::Vector{Int8},
    correction_term::BackflowEta2SpinExchangeTerm,
    proposal::MoveProposal,
    site_index::Int,
) where {T}
    if site_index > length(correction_term.outgoing_bond_indices_by_source)
        return nothing
    end

    state_i = get_site_state_after_proposal(state_vector, proposal, site_index)
    eta2_value = T(correction_term.eta2_bf)
    for bond_index in correction_term.outgoing_bond_indices_by_source[site_index]
        (_, target_site) = correction_term.source_bonds[bond_index]
        state_j = get_site_state_after_proposal(state_vector, proposal, target_site)
        bond_amplitude = T(correction_term.source_amplitudes[bond_index])
        for row_offset in 1:2
            spin = backflow_spin_from_row_offset(row_offset)
            opposite_spin = backflow_opposite_spin(spin)
            eta2_factor =
                backflow_n_sigma(state_i, spin) *
                backflow_h_sigma(state_i, opposite_spin) *
                backflow_n_sigma(state_j, opposite_spin) *
                backflow_h_sigma(state_j, spin)
            if eta2_factor == 0.0
                continue
            end
            target_row = 2 * (target_site - 1) + row_offset
            @views site_block_buffer[row_offset, :] .+= eta2_value * bond_amplitude * T(eta2_factor) .* base_orbitals[target_row, :]
        end
    end

    return nothing
end

"""
用途: 将 Eq.(5) 的 `eta3` correction term 在 proposal 后对单个站点行块的贡献累加到 buffer。

数学公式:
- `delta U_eta3(i, sigma; x') = eta3_bf * sum_j t_ij *
   (D_i n_j_-sigma h_j_sigma + n_i_sigma h_i_-sigma H_j) * U_0(j, sigma)`。

参数:
- `site_block_buffer::AbstractMatrix{T}`: 待累加的 `2 x N_orb` 站点行块。
- `base_orbitals::AbstractMatrix{T}`: 裸轨道矩阵 `U_0`。
- `state_vector::Vector{Int8}`: proposal 提交前的构型状态数组。
- `correction_term::BackflowEta3MixedVirtualHopTerm`: `eta3` correction term。
- `proposal::MoveProposal`: Monte Carlo proposal。
- `site_index::Int`: 待写入的站点编号。

返回:
- `nothing`。
"""
function add_backflow_correction_site_block_after_proposal!(
    site_block_buffer::AbstractMatrix{T},
    base_orbitals::AbstractMatrix{T},
    state_vector::Vector{Int8},
    correction_term::BackflowEta3MixedVirtualHopTerm,
    proposal::MoveProposal,
    site_index::Int,
) where {T}
    if site_index > length(correction_term.outgoing_bond_indices_by_source)
        return nothing
    end

    state_i = get_site_state_after_proposal(state_vector, proposal, site_index)
    eta3_value = T(correction_term.eta3_bf)
    for bond_index in correction_term.outgoing_bond_indices_by_source[site_index]
        (_, target_site) = correction_term.source_bonds[bond_index]
        state_j = get_site_state_after_proposal(state_vector, proposal, target_site)
        bond_amplitude = T(correction_term.source_amplitudes[bond_index])
        for row_offset in 1:2
            spin = backflow_spin_from_row_offset(row_offset)
            opposite_spin = backflow_opposite_spin(spin)
            eta3_factor =
                (state_i == DB ? 1.0 : 0.0) *
                backflow_n_sigma(state_j, opposite_spin) *
                backflow_h_sigma(state_j, spin) +
                backflow_n_sigma(state_i, spin) *
                backflow_h_sigma(state_i, opposite_spin) *
                (state_j == HOLE ? 1.0 : 0.0)
            if eta3_factor == 0.0
                continue
            end
            target_row = 2 * (target_site - 1) + row_offset
            @views site_block_buffer[row_offset, :] .+= eta3_value * bond_amplitude * T(eta3_factor) .* base_orbitals[target_row, :]
        end
    end

    return nothing
end

"""
用途: 写入组合式 Eq.(5) backflow 在 proposal 提交后的局域站点行块。

数学公式:
- `U_b(i, sigma; x') = U_0(i, sigma) + sum_m delta U_m(i, sigma; x')`,
  其中 `m` 遍历 `epsilon, eta1, eta2, eta3` 等 correction terms。

参数:
- `site_block_buffer::AbstractMatrix{T}`: 输出 buffer, 形状必须为 `2 x N_orb`。
- `base_orbitals::AbstractMatrix{T}`: 裸轨道矩阵 `U_0`。
- `state_vector::Vector{Int8}`: proposal 提交前的构型状态数组。
- `backflow_term::CompositeBackflowTerm`: 组合式 Eq.(5) backflow 对象。
- `proposal::MoveProposal`: Monte Carlo proposal。
- `site_index::Int`: 待写入的站点编号。

返回:
- `nothing`。
"""
function fill_backflow_site_block_after_proposal!(
    site_block_buffer::AbstractMatrix{T},
    base_orbitals::AbstractMatrix{T},
    state_vector::Vector{Int8},
    backflow_term::CompositeBackflowTerm,
    proposal::MoveProposal,
    site_index::Int,
) where {T}
    validate_orbital_dimensions(base_orbitals, length(state_vector))
    if size(site_block_buffer, 1) != 2 || size(site_block_buffer, 2) != size(base_orbitals, 2)
        error("Local site block buffer must have shape (2, $(size(base_orbitals, 2))), got $(size(site_block_buffer)).")
    end
    if !(1 <= site_index <= length(state_vector))
        error("Affected site $site_index is out of bounds for a state vector with $(length(state_vector)) sites.")
    end

    row_up = 2 * (site_index - 1) + 1
    row_dn_hole = row_up + 1
    copyto!(@view(site_block_buffer[1, :]), @view(base_orbitals[row_up, :]))
    copyto!(@view(site_block_buffer[2, :]), @view(base_orbitals[row_dn_hole, :]))

    for correction_term in backflow_term.terms
        add_backflow_correction_site_block_after_proposal!(
            site_block_buffer,
            base_orbitals,
            state_vector,
            correction_term,
            proposal,
            site_index,
        )
    end

    return nothing
end


"""
用途: 仅为单个受影响站点构造 proposal 提交后的局域 backflow 行块。

数学公式:
- `U_b(a, i, k; x') = [1 + (epsilon_bf - 1) * xi_i(x')] * U_0(a, i, k)
   + eta_bf * sum_j[t_ij * D_i(x') * H_j(x') * U_0(a, j, k)]`。

参数:
- `base_orbitals::AbstractMatrix{T}`: 裸轨道矩阵 `U_0`。
- `state_vector::Vector{Int8}`: proposal 提交前的构型状态数组。
- `backflow_term::Eq4BackflowTerm`: `Eq.(4)` backflow 参数对象。
- `proposal::MoveProposal`: Monte Carlo proposal。
- `site_index::Int`: 待构造局域行块的站点编号。

返回:
- `Matrix{T}`: 形状为 `2 x N_orb` 的局域站点行块。
"""
function build_eq4_backflow_site_block_after_proposal(
    base_orbitals::AbstractMatrix{T},
    state_vector::Vector{Int8},
    backflow_term::Eq4BackflowTerm,
    proposal::MoveProposal,
    site_index::Int,
) where {T}
    site_block = Matrix{T}(undef, 2, size(base_orbitals, 2))
    fill_eq4_backflow_site_block_after_proposal!(
        site_block,
        base_orbitals,
        state_vector,
        backflow_term,
        proposal,
        site_index,
    )
    return site_block
end


"""
用途: 为所有受影响站点批量构造 proposal 提交后的局域 backflow 行块。

参数:
- `base_orbitals::AbstractMatrix{T}`: 裸轨道矩阵 `U_0`。
- `state_vector::Vector{Int8}`: proposal 提交前的构型状态数组。
- `backflow_term::Eq4BackflowTerm`: `Eq.(4)` backflow 参数对象。
- `proposal::MoveProposal`: Monte Carlo proposal。
- `affected_sites::Vector{Int}`: 已排序的受影响站点列表。

返回:
- `Vector{Pair{Int, Matrix{T}}}`: `site_index => site_block` 的有序列表。
"""
function build_local_backflow_site_blocks_after_proposal(
    base_orbitals::AbstractMatrix{T},
    state_vector::Vector{Int8},
    backflow_term::Eq4BackflowTerm,
    proposal::MoveProposal,
    affected_sites::Vector{Int},
) where {T}
    validate_eq4_backflow_source_data!(backflow_term)
    validate_orbital_dimensions(base_orbitals, length(state_vector))

    site_blocks = Vector{Pair{Int, Matrix{T}}}(undef, length(affected_sites))
    for (block_index, site_index) in enumerate(affected_sites)
        if !(1 <= site_index <= length(state_vector))
            error("Affected site $site_index is out of bounds for a state vector with $(length(state_vector)) sites.")
        end

        site_blocks[block_index] = site_index => build_eq4_backflow_site_block_after_proposal(
            base_orbitals,
            state_vector,
            backflow_term,
            proposal,
            site_index,
        )
    end

    return site_blocks
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
用途: 按名称更新单个 Eq.(5) correction term 参数。

参数:
- `correction_term::AbstractBackflowCorrectionTerm`: 待更新的 correction term。
- `param_name::Symbol`: 参数名。
- `param_value::Real`: 参数值。

返回:
- `Bool`: 若参数名匹配并完成更新则返回 `true`, 否则返回 `false`。
"""
function update_backflow_correction_param!(
    correction_term::BackflowEpsilonTerm,
    param_name::Symbol,
    param_value::Real,
)::Bool
    if param_name != correction_term.param_name
        return false
    end
    correction_term.epsilon_bf = Float64(param_value)
    return true
end
function update_backflow_correction_param!(
    correction_term::BackflowEta1DoublonHoleTerm,
    param_name::Symbol,
    param_value::Real,
)::Bool
    if param_name != correction_term.param_name
        return false
    end
    correction_term.eta1_bf = Float64(param_value)
    return true
end
function update_backflow_correction_param!(
    correction_term::BackflowEta2SpinExchangeTerm,
    param_name::Symbol,
    param_value::Real,
)::Bool
    if param_name != correction_term.param_name
        return false
    end
    correction_term.eta2_bf = Float64(param_value)
    return true
end
function update_backflow_correction_param!(
    correction_term::BackflowEta3MixedVirtualHopTerm,
    param_name::Symbol,
    param_value::Real,
)::Bool
    if param_name != correction_term.param_name
        return false
    end
    correction_term.eta3_bf = Float64(param_value)
    return true
end

"""
用途: 按名称批量更新组合式 Eq.(5) backflow 参数。

参数:
- `backflow_term::CompositeBackflowTerm`: 组合式 backflow 对象。
- `param_names::Vector{Symbol}`: 参数名列表。
- `param_values::Vector{<:Real}`: 参数值列表。

返回:
- `nothing`。若存在未知参数名会抛出异常。
"""
function update_backflow_params!(
    backflow_term::CompositeBackflowTerm,
    param_names::Vector{Symbol},
    param_values::Vector{<:Real},
)
    if length(param_names) != length(param_values)
        error("Length mismatch: param_names has $(length(param_names)) entries, but param_values has $(length(param_values)).")
    end

    for (param_name, param_value) in zip(param_names, param_values)
        is_updated = false
        for correction_term in backflow_term.terms
            if update_backflow_correction_param!(correction_term, param_name, param_value)
                is_updated = true
                break
            end
        end
        if !is_updated
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
    validate_eq4_backflow_source_data!(backflow_term)

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
用途: 校验 Eq.(5) correction term 的 source 数据在构造后未被原地修改。

参数:
- `correction_term::AbstractBackflowCorrectionTerm`: 待校验的 correction term。

返回:
- `nothing`。若检测到原地修改则抛出异常。
"""
function validate_backflow_correction_source_data!(
    correction_term::AbstractBackflowCorrectionTerm,
)
    current_signature = compute_eq4_backflow_source_data_signature(
        correction_term.source_bonds,
        correction_term.source_amplitudes,
    )

    if current_signature != correction_term.source_data_signature
        error("Backflow correction term source_bonds/source_amplitudes were mutated after construction. Please rebuild the correction term instead of modifying it in place.")
    end

    return nothing
end

"""
用途: 将 PH 内部行偏移映射为方案 A 使用的物理自旋标签。

参数:
- `row_offset::Int`: 站点内行偏移, `1` 为 up 通道, `2` 为 down 通道。

返回:
- `Int8`: `UP` 或 `DN`。
"""
function backflow_spin_from_row_offset(row_offset::Int)::Int8
    if row_offset == 1
        return UP
    elseif row_offset == 2
        return DN
    end
    error("row_offset must be 1 or 2, got $(row_offset).")
end

"""
用途: 返回给定物理自旋的反向自旋。

参数:
- `spin::Int8`: `UP` 或 `DN`。

返回:
- `Int8`: 反向自旋。
"""
function backflow_opposite_spin(spin::Int8)::Int8
    if spin == UP
        return DN
    elseif spin == DN
        return UP
    end
    error("spin must be UP or DN, got $(spin).")
end

"""
用途: 计算物理占据数 `n_{i,sigma}`。

参数:
- `state_code::Int8`: 站点物理状态编码。
- `spin::Int8`: `UP` 或 `DN`。

返回:
- `Float64`: 若站点含有该自旋则为 `1.0`, 否则为 `0.0`。
"""
function backflow_n_sigma(state_code::Int8, spin::Int8)::Float64
    return (state_code & spin) != 0 ? 1.0 : 0.0
end

"""
用途: 计算物理空穴因子 `h_{i,sigma} = 1 - n_{i,sigma}`。

参数:
- `state_code::Int8`: 站点物理状态编码。
- `spin::Int8`: `UP` 或 `DN`。

返回:
- `Float64`: `1 - n_{i,sigma}`。
"""
function backflow_h_sigma(state_code::Int8, spin::Int8)::Float64
    return 1.0 - backflow_n_sigma(state_code, spin)
end

"""
用途: 计算 Eq.(5) 中 `eta2` 对给定 `(i, j, sigma)` 的局域 virtual hopping 因子。

数学公式:
- `eta2_factor = n_{i,sigma} h_{i,-sigma} n_{j,-sigma} h_{j,sigma}`。

参数:
- `state_i::Int8`: source site `i` 的物理状态编码。
- `state_j::Int8`: target site `j` 的物理状态编码。
- `spin::Int8`: 当前行对应的物理自旋 `sigma`。

返回:
- `Float64`: 因子取值, 当前实现中为 `0.0` 或 `1.0`。
"""
function compute_eta2_virtual_hopping_factor(
    state_i::Int8,
    state_j::Int8,
    spin::Int8,
)::Float64
    opposite_spin = backflow_opposite_spin(spin)
    return backflow_n_sigma(state_i, spin) *
           backflow_h_sigma(state_i, opposite_spin) *
           backflow_n_sigma(state_j, opposite_spin) *
           backflow_h_sigma(state_j, spin)
end

"""
用途: 计算 Eq.(5) 中 `eta3` 对给定 `(i, j, sigma)` 的 mixed virtual hopping 因子。

数学公式:
- `eta3_factor = D_i n_{j,-sigma} h_{j,sigma} + n_{i,sigma} h_{i,-sigma} H_j`。

参数:
- `state_i::Int8`: source site `i` 的物理状态编码。
- `state_j::Int8`: target site `j` 的物理状态编码。
- `spin::Int8`: 当前行对应的物理自旋 `sigma`。

返回:
- `Float64`: 因子取值, 当前实现中为 `0.0` 或 `1.0`。
"""
function compute_eta3_virtual_hopping_factor(
    state_i::Int8,
    state_j::Int8,
    spin::Int8,
)::Float64
    opposite_spin = backflow_opposite_spin(spin)
    return (state_i == DB ? 1.0 : 0.0) *
           backflow_n_sigma(state_j, opposite_spin) *
           backflow_h_sigma(state_j, spin) +
           backflow_n_sigma(state_i, spin) *
           backflow_h_sigma(state_i, opposite_spin) *
           (state_j == HOLE ? 1.0 : 0.0)
end

"""
用途: 判断 `epsilon` prefactor 是否应在某个 `(i, sigma)` 行打开。

数学公式:
- 若 `epsilon_mask_terms` 包含 `:eta1`, 则检查 `D_i H_j`。
- 若包含 `:eta2`, 则检查
  `n_{i,sigma} h_{i,-sigma} n_{j,-sigma} h_{j,sigma}`。
- 若包含 `:eta3`, 则检查
  `D_i n_{j,-sigma} h_{j,sigma} + n_{i,sigma} h_{i,-sigma} H_j`。

参数:
- `state_i::Int8`: source site `i` 的物理状态编码。
- `state_j::Int8`: target site `j` 的物理状态编码。
- `spin::Int8`: 当前行对应的物理自旋 `sigma`。
- `epsilon_mask_terms::Vector{Symbol}`: 参与 `epsilon` mask 的 virtual hopping 通道。

返回:
- `Bool`: 任一指定通道在该有向键上激活时返回 `true`。
"""
function is_backflow_epsilon_row_active(
    state_i::Int8,
    state_j::Int8,
    spin::Int8,
    epsilon_mask_terms::Vector{Symbol},
)::Bool
    for mask_term in epsilon_mask_terms
        if mask_term == :eta1
            if state_i == DB && state_j == HOLE
                return true
            end
        elseif mask_term == :eta2
            if compute_eta2_virtual_hopping_factor(state_i, state_j, spin) != 0.0
                return true
            end
        elseif mask_term == :eta3
            if compute_eta3_virtual_hopping_factor(state_i, state_j, spin) != 0.0
                return true
            end
        else
            error("Unknown epsilon mask term $(mask_term).")
        end
    end
    return false
end

"""
用途: 为 `BackflowEpsilonTerm` 构造 spin-resolved `xi_{i,sigma}` 行掩码。

数学公式:
- `xi_{i,sigma} = 1`, 当且仅当存在有向键 `(i, j)` 使得
  `epsilon_mask_terms` 指定的任意 virtual hopping 因子非零。

参数:
- `state_vector::Vector{Int8}`: 当前 Monte Carlo 构型。
- `correction_term::BackflowEpsilonTerm`: `epsilon` correction term。

返回:
- `Vector{Bool}`: 长度为 `2 * N_sites` 的行掩码, 第 `2i-1` 行为 up,
  第 `2i` 行为 down。
"""
function compute_backflow_epsilon_row_mask(
    state_vector::Vector{Int8},
    correction_term::BackflowEpsilonTerm,
)::Vector{Bool}
    epsilon_row_mask = falses(2 * length(state_vector))
    for (site_i, site_j) in correction_term.source_bonds
        state_i = state_vector[site_i]
        state_j = state_vector[site_j]
        for row_offset in 1:2
            spin = backflow_spin_from_row_offset(row_offset)
            if is_backflow_epsilon_row_active(
                state_i,
                state_j,
                spin,
                correction_term.epsilon_mask_terms,
            )
                row_i = 2 * (site_i - 1) + row_offset
                epsilon_row_mask[row_i] = true
            end
        end
    end
    return epsilon_row_mask
end

"""
用途: 将 Eq.(5) 的 `epsilon` correction term 加到 backflow 轨道矩阵。

数学公式:
- `delta U_epsilon(i, sigma) = (epsilon_bf - 1) * xi_{i,sigma} * U_0(i, sigma)`。

参数:
- `backflow_orbitals::AbstractMatrix{T}`: 待累加的 backflow 轨道矩阵。
- `base_orbitals::AbstractMatrix{T}`: 裸轨道矩阵。
- `state_vector::Vector{Int8}`: 当前构型。
- `correction_term::BackflowEpsilonTerm`: `epsilon` correction term。

返回:
- `nothing`。
"""
function add_backflow_correction_orbitals!(
    backflow_orbitals::AbstractMatrix{T},
    base_orbitals::AbstractMatrix{T},
    state_vector::Vector{Int8},
    correction_term::BackflowEpsilonTerm,
) where {T}
    validate_backflow_correction_source_data!(correction_term)
    epsilon_row_mask = compute_backflow_epsilon_row_mask(state_vector, correction_term)
    epsilon_shift = T(correction_term.epsilon_bf - 1.0)
    for row_index in eachindex(epsilon_row_mask)
        if !epsilon_row_mask[row_index]
            continue
        end
        @views backflow_orbitals[row_index, :] .+= epsilon_shift .* base_orbitals[row_index, :]
    end

    return nothing
end

"""
用途: 将 Eq.(5) 的 `eta1` correction term 加到 backflow 轨道矩阵。

数学公式:
- `delta U_eta1(i, sigma) = eta1_bf * sum_j t_ij * D_i * H_j * U_0(j, sigma)`。

参数:
- `backflow_orbitals::AbstractMatrix{T}`: 待累加的 backflow 轨道矩阵。
- `base_orbitals::AbstractMatrix{T}`: 裸轨道矩阵。
- `state_vector::Vector{Int8}`: 当前构型。
- `correction_term::BackflowEta1DoublonHoleTerm`: `eta1` correction term。

返回:
- `nothing`。
"""
function add_backflow_correction_orbitals!(
    backflow_orbitals::AbstractMatrix{T},
    base_orbitals::AbstractMatrix{T},
    state_vector::Vector{Int8},
    correction_term::BackflowEta1DoublonHoleTerm,
) where {T}
    validate_backflow_correction_source_data!(correction_term)
    eta1_value = T(correction_term.eta1_bf)
    for (bond_index, (site_i, site_j)) in enumerate(correction_term.source_bonds)
        if state_vector[site_i] != DB || state_vector[site_j] != HOLE
            continue
        end
        bond_amplitude = T(correction_term.source_amplitudes[bond_index])
        for row_offset in 1:2
            row_i = 2 * (site_i - 1) + row_offset
            row_j = 2 * (site_j - 1) + row_offset
            @views backflow_orbitals[row_i, :] .+= eta1_value * bond_amplitude .* base_orbitals[row_j, :]
        end
    end

    return nothing
end

"""
用途: 将 Eq.(5) 的 `eta2` correction term 加到 backflow 轨道矩阵。

数学公式:
- `delta U_eta2(i, sigma) = eta2_bf * sum_j t_ij *
   n_i_sigma h_i_-sigma n_j_-sigma h_j_sigma * U_0(j, sigma)`。

参数:
- `backflow_orbitals::AbstractMatrix{T}`: 待累加的 backflow 轨道矩阵。
- `base_orbitals::AbstractMatrix{T}`: 裸轨道矩阵。
- `state_vector::Vector{Int8}`: 当前构型。
- `correction_term::BackflowEta2SpinExchangeTerm`: `eta2` correction term。

返回:
- `nothing`。
"""
function add_backflow_correction_orbitals!(
    backflow_orbitals::AbstractMatrix{T},
    base_orbitals::AbstractMatrix{T},
    state_vector::Vector{Int8},
    correction_term::BackflowEta2SpinExchangeTerm,
) where {T}
    validate_backflow_correction_source_data!(correction_term)
    eta2_value = T(correction_term.eta2_bf)
    for (bond_index, (site_i, site_j)) in enumerate(correction_term.source_bonds)
        state_i = state_vector[site_i]
        state_j = state_vector[site_j]
        bond_amplitude = T(correction_term.source_amplitudes[bond_index])
        for row_offset in 1:2
            spin = backflow_spin_from_row_offset(row_offset)
            opposite_spin = backflow_opposite_spin(spin)
            eta2_factor =
                backflow_n_sigma(state_i, spin) *
                backflow_h_sigma(state_i, opposite_spin) *
                backflow_n_sigma(state_j, opposite_spin) *
                backflow_h_sigma(state_j, spin)
            if eta2_factor == 0.0
                continue
            end
            row_i = 2 * (site_i - 1) + row_offset
            row_j = 2 * (site_j - 1) + row_offset
            @views backflow_orbitals[row_i, :] .+= eta2_value * bond_amplitude * T(eta2_factor) .* base_orbitals[row_j, :]
        end
    end

    return nothing
end

"""
用途: 将 Eq.(5) 的 `eta3` correction term 加到 backflow 轨道矩阵。

数学公式:
- `delta U_eta3(i, sigma) = eta3_bf * sum_j t_ij *
   (D_i n_j_-sigma h_j_sigma + n_i_sigma h_i_-sigma H_j) * U_0(j, sigma)`。

参数:
- `backflow_orbitals::AbstractMatrix{T}`: 待累加的 backflow 轨道矩阵。
- `base_orbitals::AbstractMatrix{T}`: 裸轨道矩阵。
- `state_vector::Vector{Int8}`: 当前构型。
- `correction_term::BackflowEta3MixedVirtualHopTerm`: `eta3` correction term。

返回:
- `nothing`。
"""
function add_backflow_correction_orbitals!(
    backflow_orbitals::AbstractMatrix{T},
    base_orbitals::AbstractMatrix{T},
    state_vector::Vector{Int8},
    correction_term::BackflowEta3MixedVirtualHopTerm,
) where {T}
    validate_backflow_correction_source_data!(correction_term)
    eta3_value = T(correction_term.eta3_bf)
    for (bond_index, (site_i, site_j)) in enumerate(correction_term.source_bonds)
        state_i = state_vector[site_i]
        state_j = state_vector[site_j]
        bond_amplitude = T(correction_term.source_amplitudes[bond_index])
        for row_offset in 1:2
            spin = backflow_spin_from_row_offset(row_offset)
            opposite_spin = backflow_opposite_spin(spin)
            eta3_factor =
                (state_i == DB ? 1.0 : 0.0) *
                backflow_n_sigma(state_j, opposite_spin) *
                backflow_h_sigma(state_j, spin) +
                backflow_n_sigma(state_i, spin) *
                backflow_h_sigma(state_i, opposite_spin) *
                (state_j == HOLE ? 1.0 : 0.0)
            if eta3_factor == 0.0
                continue
            end
            row_i = 2 * (site_i - 1) + row_offset
            row_j = 2 * (site_j - 1) + row_offset
            @views backflow_orbitals[row_i, :] .+= eta3_value * bond_amplitude * T(eta3_factor) .* base_orbitals[row_j, :]
        end
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
用途: 构造组合式 Eq.(5) backflow 的构型依赖轨道矩阵。

数学公式:
- `U_b = U_0 + sum_m delta U_m`, 其中每个 `delta U_m` 由一个
  `AbstractBackflowCorrectionTerm` 提供。

参数:
- `base_orbitals::AbstractMatrix{T}`: 裸轨道矩阵 `U_0`。
- `state_vector::Vector{Int8}`: 当前 Monte Carlo 构型。
- `backflow_term::CompositeBackflowTerm`: 组合式 backflow 对象。

返回:
- `Matrix{T}`: 构型依赖的 `U_b(x)`。
"""
function build_backflow_orbitals(
    base_orbitals::AbstractMatrix{T},
    state_vector::Vector{Int8},
    backflow_term::CompositeBackflowTerm,
) where {T}
    validate_orbital_dimensions(base_orbitals, length(state_vector))
    backflow_orbitals = Matrix{T}(base_orbitals)
    for correction_term in backflow_term.terms
        add_backflow_correction_orbitals!(
            backflow_orbitals,
            base_orbitals,
            state_vector,
            correction_term,
        )
    end
    return backflow_orbitals
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
用途: 检查 chain rule 输出矩阵与输入导数轨道矩阵尺寸一致。

参数:
- `output_orbitals::AbstractMatrix`: 待写入的输出矩阵。
- `input_derivative_orbitals::AbstractMatrix`: 输入的裸轨道导数矩阵。

返回:
- `nothing`。若尺寸不一致则抛出 error。
"""
function validate_chain_rule_output_dimensions(
    output_orbitals::AbstractMatrix,
    input_derivative_orbitals::AbstractMatrix,
)
    if size(output_orbitals) != size(input_derivative_orbitals)
        error("Chain-rule output shape mismatch: expected $(size(input_derivative_orbitals)), got $(size(output_orbitals)).")
    end
    return nothing
end

"""
用途: 在 `NoBackflowTerm` 情况下计算 mean-field 参数的 chain rule 轨道导数。

数学公式:
- 无 backflow 时 `U_b = U_0`, 因此 `dU_b / dp = dU_0 / dp`。

参数:
- `output_orbitals::AbstractMatrix{T}`: 输出矩阵, 写入 `dU_b / dp`。
- `input_derivative_orbitals::AbstractMatrix{T}`: 输入裸轨道导数 `dU_0 / dp`。
- `state_vector::Vector{Int8}`: 当前构型, 此处仅用于接口统一。
- `backflow_term::NoBackflowTerm`: 空 backflow 对象。

返回:
- `nothing`。
"""
function fill_backflow_chain_rule_orbitals!(
    output_orbitals::AbstractMatrix{T},
    input_derivative_orbitals::AbstractMatrix{T},
    state_vector::Vector{Int8},
    ::NoBackflowTerm,
) where {T}
    validate_orbital_dimensions(input_derivative_orbitals, length(state_vector))
    validate_chain_rule_output_dimensions(output_orbitals, input_derivative_orbitals)
    copyto!(output_orbitals, input_derivative_orbitals)
    return nothing
end

"""
用途: 计算 Eq.(5) composite backflow 对 mean-field 参数的 chain rule 轨道导数。

数学公式:
- 若 `U_b = B_x[U_0]`, 且固定当前构型 `x` 与 backflow 参数,
  则 `dU_b / dp = B_x[dU_0 / dp]`。
- 这里 `B_x` 是 `U_0 + sum_m delta U_m` 对裸轨道矩阵的线性变换。

参数:
- `output_orbitals::AbstractMatrix{T}`: 输出矩阵, 写入 `dU_b / dp`。
- `input_derivative_orbitals::AbstractMatrix{T}`: 输入裸轨道导数 `dU_0 / dp`。
- `state_vector::Vector{Int8}`: 当前 Monte Carlo 构型。
- `backflow_term::CompositeBackflowTerm`: 组合式 Eq.(5) backflow 对象。

返回:
- `nothing`。
"""
function fill_backflow_chain_rule_orbitals!(
    output_orbitals::AbstractMatrix{T},
    input_derivative_orbitals::AbstractMatrix{T},
    state_vector::Vector{Int8},
    backflow_term::CompositeBackflowTerm,
) where {T}
    validate_orbital_dimensions(input_derivative_orbitals, length(state_vector))
    validate_chain_rule_output_dimensions(output_orbitals, input_derivative_orbitals)
    copyto!(output_orbitals, input_derivative_orbitals)
    for correction_term in backflow_term.terms
        add_backflow_correction_orbitals!(
            output_orbitals,
            input_derivative_orbitals,
            state_vector,
            correction_term,
        )
    end
    return nothing
end

"""
用途: 计算 Eq.(4) backflow 对 mean-field 参数的 chain rule 轨道导数。

数学公式:
- `dU_b(a, i, k) / dp =
   [1 + (epsilon_bf - 1) * xi_i] * dU_0(a, i, k) / dp
   + eta_bf * sum_j[t_ij * D_i * H_j * dU_0(a, j, k) / dp]`。

参数:
- `output_orbitals::AbstractMatrix{T}`: 输出矩阵, 写入 `dU_b / dp`。
- `input_derivative_orbitals::AbstractMatrix{T}`: 输入裸轨道导数 `dU_0 / dp`。
- `state_vector::Vector{Int8}`: 当前 Monte Carlo 构型。
- `backflow_term::Eq4BackflowTerm`: Eq.(4) backflow 参数对象。

返回:
- `nothing`。
"""
function fill_backflow_chain_rule_orbitals!(
    output_orbitals::AbstractMatrix{T},
    input_derivative_orbitals::AbstractMatrix{T},
    state_vector::Vector{Int8},
    backflow_term::Eq4BackflowTerm,
) where {T}
    n_sites = length(state_vector)
    validate_eq4_backflow_source_data!(backflow_term)
    validate_orbital_dimensions(input_derivative_orbitals, n_sites)
    validate_chain_rule_output_dimensions(output_orbitals, input_derivative_orbitals)

    doublon_mask, hole_mask = compute_doublon_hole_masks(state_vector)
    recombination_mask = compute_recombination_mask(backflow_term, doublon_mask, hole_mask)

    copyto!(output_orbitals, input_derivative_orbitals)

    for site_index in 1:n_sites
        prefactor = one(T) + T(backflow_term.epsilon_bf - 1.0) * T(recombination_mask[site_index])
        row_up = 2 * (site_index - 1) + 1
        row_dn_hole = row_up + 1
        @views output_orbitals[row_up, :] .*= prefactor
        @views output_orbitals[row_dn_hole, :] .*= prefactor
    end

    eta_value = T(backflow_term.eta_bf)
    for (bond_index, (site_i, site_j)) in enumerate(backflow_term.source_bonds)
        if doublon_mask[site_i] > 0.5 && hole_mask[site_j] > 0.5
            bond_amplitude = T(backflow_term.source_amplitudes[bond_index])
            row_i_up = 2 * (site_i - 1) + 1
            row_i_dn_hole = row_i_up + 1
            row_j_up = 2 * (site_j - 1) + 1
            row_j_dn_hole = row_j_up + 1

            @views output_orbitals[row_i_up, :] .+= eta_value * bond_amplitude .* input_derivative_orbitals[row_j_up, :]
            @views output_orbitals[row_i_dn_hole, :] .+= eta_value * bond_amplitude .* input_derivative_orbitals[row_j_dn_hole, :]
        end
    end

    return nothing
end


"""
用途: 将 Eq.(5) 的 `epsilon` correction term 对 `epsilon_bf` 的导数累加到导数轨道矩阵。

数学公式:
- `partial U_b(i, sigma) / partial epsilon_bf = xi_{i,sigma} * U_0(i, sigma)`。

参数:
- `derivative_orbitals::AbstractMatrix{T}`: 待累加的导数轨道矩阵。
- `base_orbitals::AbstractMatrix{T}`: 裸轨道矩阵 `U_0`。
- `state_vector::Vector{Int8}`: 当前 Monte Carlo 构型。
- `correction_term::BackflowEpsilonTerm`: `epsilon` correction term。

返回:
- `nothing`。
"""
function add_backflow_correction_derivative_orbitals!(
    derivative_orbitals::AbstractMatrix{T},
    base_orbitals::AbstractMatrix{T},
    state_vector::Vector{Int8},
    correction_term::BackflowEpsilonTerm,
) where {T}
    validate_backflow_correction_source_data!(correction_term)
    epsilon_row_mask = compute_backflow_epsilon_row_mask(state_vector, correction_term)
    for row_index in eachindex(epsilon_row_mask)
        if !epsilon_row_mask[row_index]
            continue
        end
        copyto!(@view(derivative_orbitals[row_index, :]), @view(base_orbitals[row_index, :]))
    end

    return nothing
end

"""
用途: 将 Eq.(5) 的 `eta1` correction term 对 `eta1_bf` 的导数累加到导数轨道矩阵。

数学公式:
- `partial U_b(i, sigma) / partial eta1_bf =
   sum_j t_ij * D_i * H_j * U_0(j, sigma)`。

参数:
- `derivative_orbitals::AbstractMatrix{T}`: 待累加的导数轨道矩阵。
- `base_orbitals::AbstractMatrix{T}`: 裸轨道矩阵 `U_0`。
- `state_vector::Vector{Int8}`: 当前 Monte Carlo 构型。
- `correction_term::BackflowEta1DoublonHoleTerm`: `eta1` correction term。

返回:
- `nothing`。
"""
function add_backflow_correction_derivative_orbitals!(
    derivative_orbitals::AbstractMatrix{T},
    base_orbitals::AbstractMatrix{T},
    state_vector::Vector{Int8},
    correction_term::BackflowEta1DoublonHoleTerm,
) where {T}
    validate_backflow_correction_source_data!(correction_term)
    for (bond_index, (site_i, site_j)) in enumerate(correction_term.source_bonds)
        if state_vector[site_i] != DB || state_vector[site_j] != HOLE
            continue
        end
        bond_amplitude = T(correction_term.source_amplitudes[bond_index])
        for row_offset in 1:2
            row_i = 2 * (site_i - 1) + row_offset
            row_j = 2 * (site_j - 1) + row_offset
            @views derivative_orbitals[row_i, :] .+= bond_amplitude .* base_orbitals[row_j, :]
        end
    end

    return nothing
end

"""
用途: 将 Eq.(5) 的 `eta2` correction term 对 `eta2_bf` 的导数累加到导数轨道矩阵。

数学公式:
- `partial U_b(i, sigma) / partial eta2_bf =
   sum_j t_ij * n_i_sigma h_i_-sigma n_j_-sigma h_j_sigma * U_0(j, sigma)`。

参数:
- `derivative_orbitals::AbstractMatrix{T}`: 待累加的导数轨道矩阵。
- `base_orbitals::AbstractMatrix{T}`: 裸轨道矩阵 `U_0`。
- `state_vector::Vector{Int8}`: 当前 Monte Carlo 构型。
- `correction_term::BackflowEta2SpinExchangeTerm`: `eta2` correction term。

返回:
- `nothing`。
"""
function add_backflow_correction_derivative_orbitals!(
    derivative_orbitals::AbstractMatrix{T},
    base_orbitals::AbstractMatrix{T},
    state_vector::Vector{Int8},
    correction_term::BackflowEta2SpinExchangeTerm,
) where {T}
    validate_backflow_correction_source_data!(correction_term)
    for (bond_index, (site_i, site_j)) in enumerate(correction_term.source_bonds)
        state_i = state_vector[site_i]
        state_j = state_vector[site_j]
        bond_amplitude = T(correction_term.source_amplitudes[bond_index])
        for row_offset in 1:2
            spin = backflow_spin_from_row_offset(row_offset)
            opposite_spin = backflow_opposite_spin(spin)
            eta2_factor =
                backflow_n_sigma(state_i, spin) *
                backflow_h_sigma(state_i, opposite_spin) *
                backflow_n_sigma(state_j, opposite_spin) *
                backflow_h_sigma(state_j, spin)
            if eta2_factor == 0.0
                continue
            end
            row_i = 2 * (site_i - 1) + row_offset
            row_j = 2 * (site_j - 1) + row_offset
            @views derivative_orbitals[row_i, :] .+= bond_amplitude * T(eta2_factor) .* base_orbitals[row_j, :]
        end
    end

    return nothing
end

"""
用途: 将 Eq.(5) 的 `eta3` correction term 对 `eta3_bf` 的导数累加到导数轨道矩阵。

数学公式:
- `partial U_b(i, sigma) / partial eta3_bf =
   sum_j t_ij * (D_i n_j_-sigma h_j_sigma + n_i_sigma h_i_-sigma H_j) * U_0(j, sigma)`。

参数:
- `derivative_orbitals::AbstractMatrix{T}`: 待累加的导数轨道矩阵。
- `base_orbitals::AbstractMatrix{T}`: 裸轨道矩阵 `U_0`。
- `state_vector::Vector{Int8}`: 当前 Monte Carlo 构型。
- `correction_term::BackflowEta3MixedVirtualHopTerm`: `eta3` correction term。

返回:
- `nothing`。
"""
function add_backflow_correction_derivative_orbitals!(
    derivative_orbitals::AbstractMatrix{T},
    base_orbitals::AbstractMatrix{T},
    state_vector::Vector{Int8},
    correction_term::BackflowEta3MixedVirtualHopTerm,
) where {T}
    validate_backflow_correction_source_data!(correction_term)
    for (bond_index, (site_i, site_j)) in enumerate(correction_term.source_bonds)
        state_i = state_vector[site_i]
        state_j = state_vector[site_j]
        bond_amplitude = T(correction_term.source_amplitudes[bond_index])
        for row_offset in 1:2
            spin = backflow_spin_from_row_offset(row_offset)
            opposite_spin = backflow_opposite_spin(spin)
            eta3_factor =
                (state_i == DB ? 1.0 : 0.0) *
                backflow_n_sigma(state_j, opposite_spin) *
                backflow_h_sigma(state_j, spin) +
                backflow_n_sigma(state_i, spin) *
                backflow_h_sigma(state_i, opposite_spin) *
                (state_j == HOLE ? 1.0 : 0.0)
            if eta3_factor == 0.0
                continue
            end
            row_i = 2 * (site_i - 1) + row_offset
            row_j = 2 * (site_j - 1) + row_offset
            @views derivative_orbitals[row_i, :] .+= bond_amplitude * T(eta3_factor) .* base_orbitals[row_j, :]
        end
    end

    return nothing
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
用途: 统一返回组合式 Eq.(5) backflow 参数顺序对应的轨道导数矩阵列表。

数学公式:
- `partial U_b / partial p_m = partial delta U_m / partial p_m`,
  其中 `p_m` 是对应 correction term 的唯一参数。

参数:
- `base_orbitals::AbstractMatrix{T}`: 裸轨道矩阵 `U_0`。
- `state_vector::Vector{Int8}`: 当前 Monte Carlo 构型。
- `backflow_term::CompositeBackflowTerm`: 组合式 Eq.(5) backflow 对象。

返回:
- `Vector{Pair{Symbol, Matrix{T}}}`: 参数名到轨道导数矩阵的有序列表。
"""
function build_backflow_derivative_orbitals(
    base_orbitals::AbstractMatrix{T},
    state_vector::Vector{Int8},
    backflow_term::CompositeBackflowTerm,
) where {T}
    validate_orbital_dimensions(base_orbitals, length(state_vector))
    derivative_pairs = Pair{Symbol, Matrix{T}}[]
    for correction_term in backflow_term.terms
        derivative_orbitals = zeros(T, size(base_orbitals))
        add_backflow_correction_derivative_orbitals!(
            derivative_orbitals,
            base_orbitals,
            state_vector,
            correction_term,
        )
        push!(
            derivative_pairs,
            backflow_correction_param_name(correction_term) => derivative_orbitals,
        )
    end

    return derivative_pairs
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
