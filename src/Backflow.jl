module Backflow

using ..Sampler

export AbstractBackflowTerm, AbstractBackflowCorrectionTerm
export NoBackflowTerm
export CompositeBackflowTerm
export BackflowEpsilonTerm, BackflowEta1DoublonHoleTerm
export BackflowEta2SpinExchangeTerm, BackflowEta3MixedVirtualHopTerm
export BackflowEta3DoublonSingleTerm, BackflowEta4SingleHoleTerm
export uses_backflow
export backflow_param_names, backflow_param_values, backflow_param_count
export update_backflow_params!
export compute_doublon_hole_masks
export build_backflow_orbitals
export fill_backflow_chain_rule_orbitals!
export build_backflow_derivative_orbitals

"""
用途: 管理 determinant 路线中的 backflow correlation 轨道修正。

当前实现范围:
- 当前实现保留组合式 Eq.(5) backflow correction terms: `epsilon`, `eta1`, `eta2`, `eta3`。
- 当前实现只假设轨道矩阵采用每个 site 两行的 spin-resolved layout:
  第一行为 `UP`, 第二行为 `DN`。
- 该 layout 可由不同 determinant 表示共用, 具体行的物理解释由调用方决定。

当前代码采用的组合式写法为:

- `U_b(i, sigma, k; x) = U_0(i, sigma, k) + sum_m delta U_m(i, sigma, k; x)`,
  其中 `m` 遍历 `epsilon`, `eta1`, `eta2`, `eta3` correction terms。

其中:
- `i, j` 为格点指标, `sigma` 为物理自旋标签, `k` 为轨道指标。
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

const BACKFLOW_EPSILON_MASK_TERMS = (:eta1, :eta2, :eta3, :eta3_doublon_single, :eta4, :eta4_single_hole)


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
    graph_cache = build_backflow_source_graph_cache(source_bonds_copy)
    source_data_signature = compute_backflow_source_data_signature(
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
            error("Unknown epsilon mask term $(mask_term). Allowed terms are :eta1, :eta2, :eta3, :eta3_doublon_single, :eta4, and :eta4_single_hole.")
        end
        if !(mask_term in normalized_terms)
            push!(normalized_terms, mask_term)
        end
    end
    return normalized_terms
end


"""
用途: 保存 backflow source graph 的图缓存, 便于快速定位受影响站点。

参数:
- `outgoing_bond_indices_by_source::Vector{Vector{Int}}`: 按 source site 存储的 bond 索引列表。
- `incoming_source_sites_by_target::Vector{Vector{Int}}`: 按 target site 存储的 source site 去重列表。

返回:
- `NamedTuple`: 包含上述两个缓存数组。
"""
function build_backflow_source_graph_cache(source_bonds::Vector{Tuple{Int, Int}})
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
用途: 三带 Emery split backflow 中的 `eta3` doublon-single correction term。

数学公式:
- `delta U_eta3(i, sigma) = eta3_bf * sum_j t_ij *
   D_i n_j_-sigma h_j_sigma * U_0(j, sigma)`。

字段:
- `param_name::Symbol`: 参数名。
- `eta3_bf::Float64`: split `eta3` 参数值。
- `source_bonds::Vector{Tuple{Int, Int}}`: 有向键 `(i, j)` 列表。
- `source_amplitudes::Vector{Float64}`: 与有向键对齐的 hopping 振幅。
"""
mutable struct BackflowEta3DoublonSingleTerm <: AbstractBackflowCorrectionTerm
    param_name::Symbol
    eta3_bf::Float64
    source_bonds::Vector{Tuple{Int, Int}}
    source_amplitudes::Vector{Float64}
    source_data_signature::UInt
    outgoing_bond_indices_by_source::Vector{Vector{Int}}
    incoming_source_sites_by_target::Vector{Vector{Int}}
end

"""
用途: 三带 Emery split backflow 中的 `eta4` single-hole correction term。

数学公式:
- `delta U_eta4(i, sigma) = eta4_bf * sum_j t_ij *
   n_i_sigma h_i_-sigma H_j * U_0(j, sigma)`。

字段:
- `param_name::Symbol`: 参数名。
- `eta4_bf::Float64`: split `eta4` 参数值。
- `source_bonds::Vector{Tuple{Int, Int}}`: 有向键 `(i, j)` 列表。
- `source_amplitudes::Vector{Float64}`: 与有向键对齐的 hopping 振幅。
"""
mutable struct BackflowEta4SingleHoleTerm <: AbstractBackflowCorrectionTerm
    param_name::Symbol
    eta4_bf::Float64
    source_bonds::Vector{Tuple{Int, Int}}
    source_amplitudes::Vector{Float64}
    source_data_signature::UInt
    outgoing_bond_indices_by_source::Vector{Vector{Int}}
    incoming_source_sites_by_target::Vector{Vector{Int}}
end

"""
用途: 缓存一个 directed split backflow 组内共享 source graph 的 `eta1/eta2/eta3/eta4` term。

字段:
- `suffix::Symbol`: directed 组名, 例如 `:dp`, `:pd`, `:pp`。
- `eta1_term/eta2_term/eta3_term/eta4_term`: 该组内四个 split backflow correction term。

说明:
- 该结构只在 `CompositeBackflowTerm` 构造阶段建立一次, 用于避免 MCMC 热循环中反复通过
  参数名字符串匹配查找同一组 term。
"""
struct DirectedSplitBackflowGroup
    suffix::Symbol
    eta1_term::BackflowEta1DoublonHoleTerm
    eta2_term::BackflowEta2SpinExchangeTerm
    eta3_term::BackflowEta3DoublonSingleTerm
    eta4_term::BackflowEta4SingleHoleTerm
end

"""
用途: 合并多个 backflow correction term 的 incoming source graph。

参数:
- `terms::Vector{<:AbstractBackflowCorrectionTerm}`: composite backflow 中的 correction terms。

返回:
- `Vector{Vector{Int}}`: `incoming_source_sites_by_target[target]` 保存所有会影响该 target 的 source site,
  已按 target 分组并排序去重。
"""
function build_composite_incoming_source_sites_by_target(
    terms::Vector{<:AbstractBackflowCorrectionTerm},
)::Vector{Vector{Int}}
    max_target_site = 0
    for correction_term in terms
        max_target_site = max(max_target_site, length(correction_term.incoming_source_sites_by_target))
    end

    incoming_source_sites_by_target = [Int[] for _ in 1:max_target_site]
    for correction_term in terms
        for target_site in eachindex(correction_term.incoming_source_sites_by_target)
            target_sources = incoming_source_sites_by_target[target_site]
            for source_site in correction_term.incoming_source_sites_by_target[target_site]
                if !(source_site in target_sources)
                    push!(target_sources, source_site)
                end
            end
        end
    end

    for target_sources in incoming_source_sites_by_target
        sort!(target_sources)
    end
    return incoming_source_sites_by_target
end

"""
用途: 从 composite backflow terms 中取出指定类型的 correction term。

参数:
- `term_type::Type{T}`: 目标 correction term 类型。
- `terms::Vector{AbstractBackflowCorrectionTerm}`: composite 中的 correction term 列表。

返回:
- `Union{Nothing, T}`: 若存在对应 correction term, 返回该对象, 否则返回 `nothing`。
"""
function find_composite_correction_term(
    term_type::Type{T},
    terms::Vector{AbstractBackflowCorrectionTerm},
)::Union{Nothing,T} where {T<:AbstractBackflowCorrectionTerm}
    for correction_term in terms
        if correction_term isa T
            return correction_term
        end
    end
    return nothing
end

"""
用途: 判断 composite backflow 的所有 correction terms 是否共享同一套 source 数据。

参数:
- `terms::Vector{AbstractBackflowCorrectionTerm}`: composite 中的 correction term 列表。

返回:
- `Bool`: 所有 correction terms 的 `source_data_signature` 相同时返回 `true`。
"""
function has_shared_backflow_source_data(
    terms::Vector{AbstractBackflowCorrectionTerm},
)::Bool
    isempty(terms) && return false
    reference_signature = terms[1].source_data_signature
    for correction_term in terms
        if correction_term.source_data_signature != reference_signature
            return false
        end
    end
    return true
end

"""
用途: 判断单个 correction term 是否被 shared-source fused 局域快路径支持。

参数:
- `correction_term::AbstractBackflowCorrectionTerm`: 待判断的 backflow correction term。

返回:
- `Bool`: 只有旧 Eq.(5) 的 `epsilon/eta1/eta2/mixed eta3` 返回 `true`。
"""
function supports_shared_source_fused_site_block(
    correction_term::AbstractBackflowCorrectionTerm,
)::Bool
    return correction_term isa BackflowEpsilonTerm ||
           correction_term isa BackflowEta1DoublonHoleTerm ||
           correction_term isa BackflowEta2SpinExchangeTerm ||
           correction_term isa BackflowEta3MixedVirtualHopTerm
end

"""
用途: 判断 composite backflow 是否可以使用 shared-source fused 局域快路径。

参数:
- `backflow_term::CompositeBackflowTerm`: 组合式 backflow 对象。

返回:
- `Bool`: source 数据共享且所有 term 都属于旧 Eq.(5) fused 支持集合时返回 `true`。
"""
function can_use_shared_source_fused_site_block(
    backflow_term,
)::Bool
    return backflow_term.has_shared_source_data &&
           all(supports_shared_source_fused_site_block, backflow_term.terms)
end

"""
用途: 判断参数名是否属于指定 directed backflow 后缀组。

参数:
- `param_name::Symbol`: backflow 参数名, 例如 `:bf_eta1_dp`。
- `suffix::Symbol`: directed 组名, 例如 `:dp`, `:pd`, `:pp`。

返回:
- `Bool`: 参数名以 `_<suffix>` 结尾时返回 `true`。
"""
function backflow_param_name_has_suffix(param_name::Symbol, suffix::Symbol)::Bool
    return endswith(String(param_name), "_" * String(suffix))
end

"""
用途: 按 directed 后缀查找某一类 split backflow correction term。

参数:
- `term_type::Type{T}`: 目标 correction term 类型。
- `terms::Vector{AbstractBackflowCorrectionTerm}`: composite 中的 term 列表。
- `suffix::Symbol`: directed 组名, 例如 `:dp`, `:pd`, `:pp`。

返回:
- `Union{Nothing, T}`: 找到唯一匹配 term 时返回该 term, 否则返回 `nothing`。
"""
function find_composite_correction_term_by_suffix(
    term_type::Type{T},
    terms::Vector{AbstractBackflowCorrectionTerm},
    suffix::Symbol,
)::Union{Nothing,T} where {T<:AbstractBackflowCorrectionTerm}
    matched_term = nothing
    for correction_term in terms
        if correction_term isa T && backflow_param_name_has_suffix(correction_term.param_name, suffix)
            matched_term !== nothing && return nothing
            matched_term = correction_term
        end
    end
    return matched_term
end

"""
用途: 从 term 列表中收集一个 directed split backflow 组内的 `eta1/eta2/eta3/eta4` term。

参数:
- `terms::Vector{AbstractBackflowCorrectionTerm}`: composite 中的 term 列表。
- `suffix::Symbol`: directed 组名, 例如 `:dp`, `:pd`, `:pp`。

返回:
- `Union{Nothing, DirectedSplitBackflowGroup}`: 若四个 term 都存在且共享同一 source 数据,
  返回缓存后的 directed group, 否则返回 `nothing`。
"""
function collect_directed_split_backflow_group(
    terms::Vector{AbstractBackflowCorrectionTerm},
    suffix::Symbol,
)::Union{Nothing,DirectedSplitBackflowGroup}
    eta1_term = find_composite_correction_term_by_suffix(
        BackflowEta1DoublonHoleTerm,
        terms,
        suffix,
    )
    eta2_term = find_composite_correction_term_by_suffix(
        BackflowEta2SpinExchangeTerm,
        terms,
        suffix,
    )
    eta3_term = find_composite_correction_term_by_suffix(
        BackflowEta3DoublonSingleTerm,
        terms,
        suffix,
    )
    eta4_term = find_composite_correction_term_by_suffix(
        BackflowEta4SingleHoleTerm,
        terms,
        suffix,
    )
    if eta1_term === nothing || eta2_term === nothing || eta3_term === nothing || eta4_term === nothing
        return nothing
    end

    reference_signature = eta1_term.source_data_signature
    if eta2_term.source_data_signature != reference_signature ||
       eta3_term.source_data_signature != reference_signature ||
       eta4_term.source_data_signature != reference_signature
        return nothing
    end

    return DirectedSplitBackflowGroup(suffix, eta1_term, eta2_term, eta3_term, eta4_term)
end

"""
用途: 在 `CompositeBackflowTerm` 构造阶段建立 directed split backflow 组缓存。

参数:
- `terms::Vector{AbstractBackflowCorrectionTerm}`: composite 中的 term 列表。

返回:
- `Tuple{Bool, Vector{DirectedSplitBackflowGroup}}`: 第一项表示是否可以使用 grouped fused
  site-block path, 第二项为按 `dp/pd/pp` 顺序保存的组缓存。
"""
function build_directed_split_backflow_groups(
    terms::Vector{AbstractBackflowCorrectionTerm},
)::Tuple{Bool,Vector{DirectedSplitBackflowGroup}}
    groups = DirectedSplitBackflowGroup[]
    for suffix in (:dp, :pd, :pp)
        group = collect_directed_split_backflow_group(terms, suffix)
        group === nothing && return false, DirectedSplitBackflowGroup[]
        push!(groups, group)
    end

    epsilon_count = count(correction_term -> correction_term isa BackflowEpsilonTerm, terms)
    if epsilon_count > 1
        return false, DirectedSplitBackflowGroup[]
    end

    expected_term_count = 4 * length(groups) + epsilon_count
    if length(terms) != expected_term_count
        return false, DirectedSplitBackflowGroup[]
    end

    return true, groups
end

"""
用途: 判断 composite backflow 是否可以使用 directed grouped fused 局域快路径。

参数:
- `backflow_term::CompositeBackflowTerm`: 组合式 backflow 对象。

返回:
- `Bool`: 当对象由可选 `epsilon` 加上 `dp/pd/pp` 三组 split `eta1..eta4` 组成,
  且每组内部 source 数据一致时返回 `true`。
"""
function can_use_grouped_source_fused_site_block(
    backflow_term,
)::Bool
    return backflow_term.has_grouped_source_data
end

"""
用途: 组合多个 Eq.(5) backflow correction terms。

字段:
- `terms::Vector{AbstractBackflowCorrectionTerm}`: 按参数顺序保存的 backflow correction term 列表。
- `incoming_source_sites_by_target::Vector{Vector{Int}}`: composite 级别合并后的 incoming source graph,
  用于 proposal 局域更新时一次性收集 affected sites。
- `has_shared_source_data::Bool`: 所有 correction terms 是否共享同一套 source 数据。
- `shared_source_bonds, shared_source_amplitudes`: shared-source fast path 使用的有向键与振幅。
- `shared_outgoing_bond_indices_by_source`: shared-source fast path 使用的 outgoing graph。
- `has_grouped_source_data::Bool`: 是否存在可用于 directed grouped fused path 的 `dp/pd/pp` 组缓存。
- `grouped_source_groups::Vector{DirectedSplitBackflowGroup}`: directed grouped fused path 使用的组缓存。
"""
mutable struct CompositeBackflowTerm <: AbstractBackflowTerm
    terms::Vector{AbstractBackflowCorrectionTerm}
    incoming_source_sites_by_target::Vector{Vector{Int}}
    has_shared_source_data::Bool
    shared_source_bonds::Vector{Tuple{Int,Int}}
    shared_source_amplitudes::Vector{Float64}
    shared_outgoing_bond_indices_by_source::Vector{Vector{Int}}
    has_grouped_source_data::Bool
    grouped_source_groups::Vector{DirectedSplitBackflowGroup}
    epsilon_term::Union{Nothing,BackflowEpsilonTerm}
    eta1_term::Union{Nothing,BackflowEta1DoublonHoleTerm}
    eta2_term::Union{Nothing,BackflowEta2SpinExchangeTerm}
    eta3_term::Union{Nothing,BackflowEta3MixedVirtualHopTerm}
    function CompositeBackflowTerm(terms::Vector{<:AbstractBackflowCorrectionTerm})
        term_list = AbstractBackflowCorrectionTerm[terms...]
        has_shared_source_data = has_shared_backflow_source_data(term_list)
        shared_source_bonds = has_shared_source_data ? term_list[1].source_bonds : Tuple{Int,Int}[]
        shared_source_amplitudes = has_shared_source_data ? term_list[1].source_amplitudes : Float64[]
        shared_outgoing_bond_indices_by_source = has_shared_source_data ? term_list[1].outgoing_bond_indices_by_source : Vector{Int}[]
        has_grouped_source_data, grouped_source_groups = build_directed_split_backflow_groups(term_list)
        return new(
            term_list,
            build_composite_incoming_source_sites_by_target(term_list),
            has_shared_source_data,
            shared_source_bonds,
            shared_source_amplitudes,
            shared_outgoing_bond_indices_by_source,
            has_grouped_source_data,
            grouped_source_groups,
            find_composite_correction_term(BackflowEpsilonTerm, term_list),
            find_composite_correction_term(BackflowEta1DoublonHoleTerm, term_list),
            find_composite_correction_term(BackflowEta2SpinExchangeTerm, term_list),
            find_composite_correction_term(BackflowEta3MixedVirtualHopTerm, term_list),
        )
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
function compute_backflow_source_data_signature(
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
用途: 构造 split 版本的 `eta3` doublon-single backflow correction term。

参数:
- `param_name::Symbol`: 参数名, 默认 `:bf_eta3`。
- `eta3_bf::Real`: split `eta3` 参数值。
- `source_bonds::Vector{Tuple{Int, Int}}`: 有向键 `(i, j)` 列表。
- `source_amplitudes::Vector{<:Real}`: 每条有向键对应的 hopping 振幅。

返回:
- `BackflowEta3DoublonSingleTerm`: 带 source 缓存的 correction term。
"""
function BackflowEta3DoublonSingleTerm(;
    param_name::Symbol=:bf_eta3,
    eta3_bf::Real=0.0,
    source_bonds::Vector{Tuple{Int, Int}}=Tuple{Int, Int}[],
    source_amplitudes::Vector{<:Real}=ones(Float64, length(source_bonds)),
)
    source_cache = build_backflow_correction_source_cache(source_bonds, source_amplitudes)
    return BackflowEta3DoublonSingleTerm(
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
用途: 构造 split 版本的 `eta4` single-hole backflow correction term。

参数:
- `param_name::Symbol`: 参数名, 默认 `:bf_eta4`。
- `eta4_bf::Real`: split `eta4` 参数值。
- `source_bonds::Vector{Tuple{Int, Int}}`: 有向键 `(i, j)` 列表。
- `source_amplitudes::Vector{<:Real}`: 每条有向键对应的 hopping 振幅。

返回:
- `BackflowEta4SingleHoleTerm`: 带 source 缓存的 correction term。
"""
function BackflowEta4SingleHoleTerm(;
    param_name::Symbol=:bf_eta4,
    eta4_bf::Real=0.0,
    source_bonds::Vector{Tuple{Int, Int}}=Tuple{Int, Int}[],
    source_amplitudes::Vector{<:Real}=ones(Float64, length(source_bonds)),
)
    source_cache = build_backflow_correction_source_cache(source_bonds, source_amplitudes)
    return BackflowEta4SingleHoleTerm(
        param_name,
        Float64(eta4_bf),
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
function backflow_correction_param_value(correction_term::BackflowEta3DoublonSingleTerm)::Float64
    return correction_term.eta3_bf
end
function backflow_correction_param_value(correction_term::BackflowEta4SingleHoleTerm)::Float64
    return correction_term.eta4_bf
end


"""
用途: 返回 backflow 参数名列表。

参数:
- `backflow_term::AbstractBackflowTerm`: backflow 对象。

返回:
- `Vector{Symbol}`: 参数名列表。
"""
backflow_param_names(::NoBackflowTerm) = Symbol[]
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
function backflow_param_values(backflow_term::CompositeBackflowTerm)
    return Float64[backflow_correction_param_value(term) for term in backflow_term.terms]
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
- 若某个改变站点是有向键 `(i, j)` 的 target `j`, 则 source `i` 对应的轨道行
  也会依赖改变后的 `j`, 因而加入受影响列表。
- 使用 `CompositeBackflowTerm` 预先合并的 incoming source graph, 避免对多个 correction term
  重复收集同一组 source sites。
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

    max_target_site = min(n_sites, length(backflow_term.incoming_source_sites_by_target))
    for changed_site in (proposal.site1, proposal.site2)
        if !(1 <= changed_site <= max_target_site)
            continue
        end

        for source_site in backflow_term.incoming_source_sites_by_target[changed_site]
            if !(1 <= source_site <= n_sites)
                error("Backflow cache source site $source_site is out of bounds for a state vector with $n_sites sites.")
            end
            push_unique_site_index!(affected_sites, source_site)
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
        target_row_down = target_row_up + 1
        bond_amplitude = T(correction_term.source_amplitudes[bond_index])
        @views site_block_buffer[1, :] .+= eta1_value * bond_amplitude .* base_orbitals[target_row_up, :]
        @views site_block_buffer[2, :] .+= eta1_value * bond_amplitude .* base_orbitals[target_row_down, :]
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
用途: 将 split `eta3` doublon-single correction term 在 proposal 后对单个站点行块的贡献累加到 buffer。

数学公式:
- `delta U_eta3(i, sigma; x') = eta3_bf * sum_j t_ij *
   D_i(x') n_{j,-sigma}(x') h_{j,sigma}(x') U_0(j, sigma)`。

参数:
- `site_block_buffer::AbstractMatrix{T}`: 待累加的 `2 x N_orb` 站点行块。
- `base_orbitals::AbstractMatrix{T}`: 裸轨道矩阵 `U_0`。
- `state_vector::Vector{Int8}`: proposal 提交前的构型状态数组。
- `correction_term::BackflowEta3DoublonSingleTerm`: split `eta3` correction term。
- `proposal::MoveProposal`: Monte Carlo proposal。
- `site_index::Int`: 待写入的站点编号。

返回:
- `nothing`。
"""
function add_backflow_correction_site_block_after_proposal!(
    site_block_buffer::AbstractMatrix{T},
    base_orbitals::AbstractMatrix{T},
    state_vector::Vector{Int8},
    correction_term::BackflowEta3DoublonSingleTerm,
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
            eta3_factor = compute_eta3_doublon_single_factor(state_i, state_j, spin)
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
用途: 将 split `eta4` single-hole correction term 在 proposal 后对单个站点行块的贡献累加到 buffer。

数学公式:
- `delta U_eta4(i, sigma; x') = eta4_bf * sum_j t_ij *
   n_{i,sigma}(x') h_{i,-sigma}(x') H_j(x') U_0(j, sigma)`。

参数:
- `site_block_buffer::AbstractMatrix{T}`: 待累加的 `2 x N_orb` 站点行块。
- `base_orbitals::AbstractMatrix{T}`: 裸轨道矩阵 `U_0`。
- `state_vector::Vector{Int8}`: proposal 提交前的构型状态数组。
- `correction_term::BackflowEta4SingleHoleTerm`: split `eta4` correction term。
- `proposal::MoveProposal`: Monte Carlo proposal。
- `site_index::Int`: 待写入的站点编号。

返回:
- `nothing`。
"""
function add_backflow_correction_site_block_after_proposal!(
    site_block_buffer::AbstractMatrix{T},
    base_orbitals::AbstractMatrix{T},
    state_vector::Vector{Int8},
    correction_term::BackflowEta4SingleHoleTerm,
    proposal::MoveProposal,
    site_index::Int,
) where {T}
    if site_index > length(correction_term.outgoing_bond_indices_by_source)
        return nothing
    end

    state_i = get_site_state_after_proposal(state_vector, proposal, site_index)
    eta4_value = T(correction_term.eta4_bf)
    for bond_index in correction_term.outgoing_bond_indices_by_source[site_index]
        (_, target_site) = correction_term.source_bonds[bond_index]
        state_j = get_site_state_after_proposal(state_vector, proposal, target_site)
        bond_amplitude = T(correction_term.source_amplitudes[bond_index])
        for row_offset in 1:2
            spin = backflow_spin_from_row_offset(row_offset)
            eta4_factor = compute_eta4_single_hole_factor(state_i, state_j, spin)
            if eta4_factor == 0.0
                continue
            end
            target_row = 2 * (target_site - 1) + row_offset
            @views site_block_buffer[row_offset, :] .+= eta4_value * bond_amplitude * T(eta4_factor) .* base_orbitals[target_row, :]
        end
    end

    return nothing
end

"""
用途: 校验局域 site block 尺寸并写入裸轨道基准行。

参数:
- `site_block_buffer::AbstractMatrix{T}`: 输出 buffer, 形状必须为 `2 x N_orb`。
- `base_orbitals::AbstractMatrix{T}`: 裸轨道矩阵 `U_0`。
- `state_vector::Vector{Int8}`: proposal 提交前的构型状态数组。
- `site_index::Int`: 待写入的站点编号。

返回:
- `Tuple{Int, Int}`: `(row_up, row_down)`, 即该 site 在 spin-resolved 轨道矩阵中的两行。
"""
function initialize_site_block_base_after_proposal!(
    site_block_buffer::AbstractMatrix{T},
    base_orbitals::AbstractMatrix{T},
    state_vector::Vector{Int8},
    site_index::Int,
)::Tuple{Int,Int} where {T}
    validate_orbital_dimensions(base_orbitals, length(state_vector))
    if size(site_block_buffer, 1) != 2 || size(site_block_buffer, 2) != size(base_orbitals, 2)
        error("Local site block buffer must have shape (2, $(size(base_orbitals, 2))), got $(size(site_block_buffer)).")
    end
    if !(1 <= site_index <= length(state_vector))
        error("Affected site $site_index is out of bounds for a state vector with $(length(state_vector)) sites.")
    end

    row_up = 2 * (site_index - 1) + 1
    row_down = row_up + 1
    copyto!(@view(site_block_buffer[1, :]), @view(base_orbitals[row_up, :]))
    copyto!(@view(site_block_buffer[2, :]), @view(base_orbitals[row_down, :]))
    return row_up, row_down
end

"""
用途: 使用逐 correction term 的旧逻辑写入 proposal 后的局域 site block。

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
function fill_backflow_site_block_after_proposal_by_terms!(
    site_block_buffer::AbstractMatrix{T},
    base_orbitals::AbstractMatrix{T},
    state_vector::Vector{Int8},
    backflow_term::CompositeBackflowTerm,
    proposal::MoveProposal,
    site_index::Int,
) where {T}
    initialize_site_block_base_after_proposal!(
        site_block_buffer,
        base_orbitals,
        state_vector,
        site_index,
    )
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
用途: 对共享 source 数据的 composite backflow 使用 fused 逻辑写入 proposal 后的局域 site block。

参数:
- `site_block_buffer::AbstractMatrix{T}`: 输出 buffer, 形状必须为 `2 x N_orb`。
- `base_orbitals::AbstractMatrix{T}`: 裸轨道矩阵 `U_0`。
- `state_vector::Vector{Int8}`: proposal 提交前的构型状态数组。
- `backflow_term::CompositeBackflowTerm`: 组合式 Eq.(5) backflow 对象, 需满足 `has_shared_source_data == true`。
- `proposal::MoveProposal`: Monte Carlo proposal。
- `site_index::Int`: 待写入的站点编号。

返回:
- `nothing`。

公式:
- 对同一条有向键 `(i,j)` 一次性计算 Eq.(5) 中 `epsilon`, `eta1`, `eta2`, `eta3`
  对 site `i` 的贡献, 避免四个 correction term 分别扫描同一套 outgoing bonds。
"""
function fill_shared_source_composite_site_block_after_proposal!(
    site_block_buffer::AbstractMatrix{T},
    base_orbitals::AbstractMatrix{T},
    state_vector::Vector{Int8},
    backflow_term::CompositeBackflowTerm,
    proposal::MoveProposal,
    site_index::Int,
) where {T}
    row_up, row_down = initialize_site_block_base_after_proposal!(
        site_block_buffer,
        base_orbitals,
        state_vector,
        site_index,
    )
    if !backflow_term.has_shared_source_data ||
       site_index > length(backflow_term.shared_outgoing_bond_indices_by_source)
        return nothing
    end

    state_i = get_site_state_after_proposal(state_vector, proposal, site_index)
    epsilon_term = backflow_term.epsilon_term
    eta1_term = backflow_term.eta1_term
    eta2_term = backflow_term.eta2_term
    eta3_term = backflow_term.eta3_term
    epsilon_shift = epsilon_term === nothing ? zero(T) : T(epsilon_term.epsilon_bf - 1.0)
    eta1_value = eta1_term === nothing ? zero(T) : T(eta1_term.eta1_bf)
    eta2_value = eta2_term === nothing ? zero(T) : T(eta2_term.eta2_bf)
    eta3_value = eta3_term === nothing ? zero(T) : T(eta3_term.eta3_bf)
    epsilon_up_is_active = false
    epsilon_down_is_active = false

    for bond_index in backflow_term.shared_outgoing_bond_indices_by_source[site_index]
        (_, target_site) = backflow_term.shared_source_bonds[bond_index]
        state_j = get_site_state_after_proposal(state_vector, proposal, target_site)
        bond_amplitude = T(backflow_term.shared_source_amplitudes[bond_index])
        target_row_up = 2 * (target_site - 1) + 1
        target_row_down = target_row_up + 1

        if epsilon_term !== nothing && epsilon_shift != zero(T)
            if !epsilon_up_is_active &&
               is_backflow_epsilon_row_active(
                   state_i,
                   state_j,
                   UP,
                   epsilon_term.epsilon_mask_terms,
               )
                epsilon_up_is_active = true
            end
            if !epsilon_down_is_active &&
               is_backflow_epsilon_row_active(
                   state_i,
                   state_j,
                   DN,
                   epsilon_term.epsilon_mask_terms,
               )
                epsilon_down_is_active = true
            end
        end

        eta1_factor = state_i == DB && state_j == HOLE ? one(T) : zero(T)
        if eta1_value != zero(T) && eta1_factor != zero(T)
            coefficient = eta1_value * bond_amplitude
            @views site_block_buffer[1, :] .+= coefficient .* base_orbitals[target_row_up, :]
            @views site_block_buffer[2, :] .+= coefficient .* base_orbitals[target_row_down, :]
        end

        for row_offset in 1:2
            spin = backflow_spin_from_row_offset(row_offset)
            opposite_spin = backflow_opposite_spin(spin)
            eta2_factor =
                backflow_n_sigma(state_i, spin) *
                backflow_h_sigma(state_i, opposite_spin) *
                backflow_n_sigma(state_j, opposite_spin) *
                backflow_h_sigma(state_j, spin)
            eta3_factor =
                (state_i == DB ? 1.0 : 0.0) *
                backflow_n_sigma(state_j, opposite_spin) *
                backflow_h_sigma(state_j, spin) +
                backflow_n_sigma(state_i, spin) *
                backflow_h_sigma(state_i, opposite_spin) *
                (state_j == HOLE ? 1.0 : 0.0)
            coefficient =
                eta2_value * bond_amplitude * T(eta2_factor) +
                eta3_value * bond_amplitude * T(eta3_factor)
            if coefficient == zero(T)
                continue
            end
            target_row = row_offset == 1 ? target_row_up : target_row_down
            @views site_block_buffer[row_offset, :] .+= coefficient .* base_orbitals[target_row, :]
        end
    end

    if epsilon_shift != zero(T)
        if epsilon_up_is_active
            @views site_block_buffer[1, :] .+= epsilon_shift .* base_orbitals[row_up, :]
        end
        if epsilon_down_is_active
            @views site_block_buffer[2, :] .+= epsilon_shift .* base_orbitals[row_down, :]
        end
    end

    return nothing
end

"""
用途: 对 directed split backflow 使用 grouped fused 逻辑写入 proposal 后的局域 site block。

数学公式:
- 对每个 directed 组 `g in {dp, pd, pp}` 和有向键 `(i,j)` 一次性计算
  `eta1_g D_i H_j U_0(j,sigma)`,
  `eta2_g n_{i,sigma} h_{i,-sigma} n_{j,-sigma} h_{j,sigma} U_0(j,sigma)`,
  `eta3_g D_i n_{j,-sigma} h_{j,sigma} U_0(j,sigma)`,
  `eta4_g n_{i,sigma} h_{i,-sigma} H_j U_0(j,sigma)`。
- `epsilon` 项仍按自己的 source graph 判断 `xi_{i,sigma}`, 因为 Emery 中它通常是
  `dp/pd/pp` source graph 的并集。

参数:
- `site_block_buffer::AbstractMatrix{T}`: 输出 buffer, 形状必须为 `2 x N_orb`。
- `base_orbitals::AbstractMatrix{T}`: 裸轨道矩阵 `U_0`。
- `state_vector::Vector{Int8}`: proposal 提交前的构型状态数组。
- `backflow_term::CompositeBackflowTerm`: directed split 组合式 backflow 对象。
- `proposal::MoveProposal`: Monte Carlo proposal。
- `site_index::Int`: 待写入的站点编号。

返回:
- `nothing`。
"""
function fill_grouped_source_composite_site_block_after_proposal!(
    site_block_buffer::AbstractMatrix{T},
    base_orbitals::AbstractMatrix{T},
    state_vector::Vector{Int8},
    backflow_term::CompositeBackflowTerm,
    proposal::MoveProposal,
    site_index::Int,
) where {T}
    row_up, row_down = initialize_site_block_base_after_proposal!(
        site_block_buffer,
        base_orbitals,
        state_vector,
        site_index,
    )
    state_i = get_site_state_after_proposal(state_vector, proposal, site_index)

    epsilon_term = backflow_term.epsilon_term
    if epsilon_term !== nothing && site_index <= length(epsilon_term.outgoing_bond_indices_by_source)
        epsilon_shift = T(epsilon_term.epsilon_bf - 1.0)
        if epsilon_shift != zero(T)
            epsilon_up_is_active = false
            epsilon_down_is_active = false
            for bond_index in epsilon_term.outgoing_bond_indices_by_source[site_index]
                (_, target_site) = epsilon_term.source_bonds[bond_index]
                state_j = get_site_state_after_proposal(state_vector, proposal, target_site)
                if !epsilon_up_is_active &&
                   is_backflow_epsilon_row_active(
                       state_i,
                       state_j,
                       UP,
                       epsilon_term.epsilon_mask_terms,
                   )
                    epsilon_up_is_active = true
                end
                if !epsilon_down_is_active &&
                   is_backflow_epsilon_row_active(
                       state_i,
                       state_j,
                       DN,
                       epsilon_term.epsilon_mask_terms,
                   )
                    epsilon_down_is_active = true
                end
                epsilon_up_is_active && epsilon_down_is_active && break
            end

            if epsilon_up_is_active
                @views site_block_buffer[1, :] .+= epsilon_shift .* base_orbitals[row_up, :]
            end
            if epsilon_down_is_active
                @views site_block_buffer[2, :] .+= epsilon_shift .* base_orbitals[row_down, :]
            end
        end
    end

    for grouped_terms in backflow_term.grouped_source_groups
        eta1_term = grouped_terms.eta1_term
        eta2_term = grouped_terms.eta2_term
        eta3_term = grouped_terms.eta3_term
        eta4_term = grouped_terms.eta4_term
        if site_index > length(eta1_term.outgoing_bond_indices_by_source)
            continue
        end

        eta1_value = T(eta1_term.eta1_bf)
        eta2_value = T(eta2_term.eta2_bf)
        eta3_value = T(eta3_term.eta3_bf)
        eta4_value = T(eta4_term.eta4_bf)
        for bond_index in eta1_term.outgoing_bond_indices_by_source[site_index]
            (_, target_site) = eta1_term.source_bonds[bond_index]
            state_j = get_site_state_after_proposal(state_vector, proposal, target_site)
            bond_amplitude = T(eta1_term.source_amplitudes[bond_index])
            target_row_up = 2 * (target_site - 1) + 1
            target_row_down = target_row_up + 1

            if eta1_value != zero(T) && state_i == DB && state_j == HOLE
                coefficient = eta1_value * bond_amplitude
                @views site_block_buffer[1, :] .+= coefficient .* base_orbitals[target_row_up, :]
                @views site_block_buffer[2, :] .+= coefficient .* base_orbitals[target_row_down, :]
            end

            for row_offset in 1:2
                spin = backflow_spin_from_row_offset(row_offset)
                eta2_factor = compute_eta2_virtual_hopping_factor(state_i, state_j, spin)
                eta3_factor = compute_eta3_doublon_single_factor(state_i, state_j, spin)
                eta4_factor = compute_eta4_single_hole_factor(state_i, state_j, spin)
                coefficient =
                    bond_amplitude *
                    (
                        eta2_value * T(eta2_factor) +
                        eta3_value * T(eta3_factor) +
                        eta4_value * T(eta4_factor)
                    )
                if coefficient == zero(T)
                    continue
                end

                target_row = row_offset == 1 ? target_row_up : target_row_down
                @views site_block_buffer[row_offset, :] .+= coefficient .* base_orbitals[target_row, :]
            end
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
    if can_use_shared_source_fused_site_block(backflow_term)
        return fill_shared_source_composite_site_block_after_proposal!(
            site_block_buffer,
            base_orbitals,
            state_vector,
            backflow_term,
            proposal,
            site_index,
        )
    end
    if can_use_grouped_source_fused_site_block(backflow_term)
        return fill_grouped_source_composite_site_block_after_proposal!(
            site_block_buffer,
            base_orbitals,
            state_vector,
            backflow_term,
            proposal,
            site_index,
        )
    end
    return fill_backflow_site_block_after_proposal_by_terms!(
        site_block_buffer,
        base_orbitals,
        state_vector,
        backflow_term,
        proposal,
        site_index,
    )
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
用途: 按名称批量更新 backflow 参数。

参数:
- `backflow_term::AbstractBackflowTerm`: backflow 参数对象。
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
function update_backflow_correction_param!(
    correction_term::BackflowEta3DoublonSingleTerm,
    param_name::Symbol,
    param_value::Real,
)::Bool
    if param_name != correction_term.param_name
        return false
    end
    correction_term.eta3_bf = Float64(param_value)
    return true
end
function update_backflow_correction_param!(
    correction_term::BackflowEta4SingleHoleTerm,
    param_name::Symbol,
    param_value::Real,
)::Bool
    if param_name != correction_term.param_name
        return false
    end
    correction_term.eta4_bf = Float64(param_value)
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
用途: 校验轨道矩阵与每个 site 两行的 spin-resolved layout 是否匹配。

参数:
- `base_orbitals::AbstractMatrix`: 裸轨道矩阵, 其行数必须为 `2 * N_sites`。
- `n_sites::Int`: 站点数。

返回:
- `nothing`。若维度不匹配则抛出 error。
"""
function validate_orbital_dimensions(base_orbitals::AbstractMatrix, n_sites::Int)
    expected_rows = 2 * n_sites
    if size(base_orbitals, 1) != expected_rows
        error("Orbital row mismatch: expected $expected_rows rows for two-spin-row layout, got $(size(base_orbitals, 1)).")
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
    current_signature = compute_backflow_source_data_signature(
        correction_term.source_bonds,
        correction_term.source_amplitudes,
    )

    if current_signature != correction_term.source_data_signature
        error("Backflow correction term source_bonds/source_amplitudes were mutated after construction. Please rebuild the correction term instead of modifying it in place.")
    end

    return nothing
end

"""
用途: 将每个 site 内部的行偏移映射为物理自旋标签。

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
    return compute_eta3_doublon_single_factor(state_i, state_j, spin) +
           compute_eta4_single_hole_factor(state_i, state_j, spin)
end

"""
用途: 计算 split Eq.(5) 中 `eta3` 对给定 `(i, j, sigma)` 的 doublon-single 因子。

数学公式:
- `eta3_factor = D_i n_{j,-sigma} h_{j,sigma}`。

参数:
- `state_i::Int8`: source site `i` 的物理状态编码。
- `state_j::Int8`: target site `j` 的物理状态编码。
- `spin::Int8`: 当前行对应的物理自旋 `sigma`。

返回:
- `Float64`: 因子取值, 当前实现中为 `0.0` 或 `1.0`。
"""
function compute_eta3_doublon_single_factor(
    state_i::Int8,
    state_j::Int8,
    spin::Int8,
)::Float64
    opposite_spin = backflow_opposite_spin(spin)
    return (state_i == DB ? 1.0 : 0.0) *
           backflow_n_sigma(state_j, opposite_spin) *
           backflow_h_sigma(state_j, spin)
end

"""
用途: 计算 split Eq.(5) 中 `eta4` 对给定 `(i, j, sigma)` 的 single-hole 因子。

数学公式:
- `eta4_factor = n_{i,sigma} h_{i,-sigma} H_j`。

参数:
- `state_i::Int8`: source site `i` 的物理状态编码。
- `state_j::Int8`: target site `j` 的物理状态编码。
- `spin::Int8`: 当前行对应的物理自旋 `sigma`。

返回:
- `Float64`: 因子取值, 当前实现中为 `0.0` 或 `1.0`。
"""
function compute_eta4_single_hole_factor(
    state_i::Int8,
    state_j::Int8,
    spin::Int8,
)::Float64
    opposite_spin = backflow_opposite_spin(spin)
    return backflow_n_sigma(state_i, spin) *
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
        elseif mask_term == :eta3_doublon_single
            if compute_eta3_doublon_single_factor(state_i, state_j, spin) != 0.0
                return true
            end
        elseif mask_term == :eta4 || mask_term == :eta4_single_hole
            if compute_eta4_single_hole_factor(state_i, state_j, spin) != 0.0
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
用途: 将 split `eta3` doublon-single correction term 加到 backflow 轨道矩阵。

数学公式:
- `delta U_eta3(i, sigma) = eta3_bf * sum_j t_ij *
   D_i n_j_-sigma h_j_sigma * U_0(j, sigma)`。

参数:
- `backflow_orbitals::AbstractMatrix{T}`: 待累加的 backflow 轨道矩阵。
- `base_orbitals::AbstractMatrix{T}`: 裸轨道矩阵。
- `state_vector::Vector{Int8}`: 当前构型。
- `correction_term::BackflowEta3DoublonSingleTerm`: split `eta3` correction term。

返回:
- `nothing`。
"""
function add_backflow_correction_orbitals!(
    backflow_orbitals::AbstractMatrix{T},
    base_orbitals::AbstractMatrix{T},
    state_vector::Vector{Int8},
    correction_term::BackflowEta3DoublonSingleTerm,
) where {T}
    validate_backflow_correction_source_data!(correction_term)
    eta3_value = T(correction_term.eta3_bf)
    for (bond_index, (site_i, site_j)) in enumerate(correction_term.source_bonds)
        state_i = state_vector[site_i]
        state_j = state_vector[site_j]
        bond_amplitude = T(correction_term.source_amplitudes[bond_index])
        for row_offset in 1:2
            spin = backflow_spin_from_row_offset(row_offset)
            eta3_factor = compute_eta3_doublon_single_factor(state_i, state_j, spin)
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
用途: 将 split `eta4` single-hole correction term 加到 backflow 轨道矩阵。

数学公式:
- `delta U_eta4(i, sigma) = eta4_bf * sum_j t_ij *
   n_i_sigma h_i_-sigma H_j * U_0(j, sigma)`。

参数:
- `backflow_orbitals::AbstractMatrix{T}`: 待累加的 backflow 轨道矩阵。
- `base_orbitals::AbstractMatrix{T}`: 裸轨道矩阵。
- `state_vector::Vector{Int8}`: 当前构型。
- `correction_term::BackflowEta4SingleHoleTerm`: split `eta4` correction term。

返回:
- `nothing`。
"""
function add_backflow_correction_orbitals!(
    backflow_orbitals::AbstractMatrix{T},
    base_orbitals::AbstractMatrix{T},
    state_vector::Vector{Int8},
    correction_term::BackflowEta4SingleHoleTerm,
) where {T}
    validate_backflow_correction_source_data!(correction_term)
    eta4_value = T(correction_term.eta4_bf)
    for (bond_index, (site_i, site_j)) in enumerate(correction_term.source_bonds)
        state_i = state_vector[site_i]
        state_j = state_vector[site_j]
        bond_amplitude = T(correction_term.source_amplitudes[bond_index])
        for row_offset in 1:2
            spin = backflow_spin_from_row_offset(row_offset)
            eta4_factor = compute_eta4_single_hole_factor(state_i, state_j, spin)
            if eta4_factor == 0.0
                continue
            end
            row_i = 2 * (site_i - 1) + row_offset
            row_j = 2 * (site_j - 1) + row_offset
            @views backflow_orbitals[row_i, :] .+= eta4_value * bond_amplitude * T(eta4_factor) .* base_orbitals[row_j, :]
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
用途: 将 split `eta3` correction term 对 `eta3_bf` 的导数累加到导数轨道矩阵。

数学公式:
- `partial U_b(i, sigma) / partial eta3_bf =
   sum_j t_ij * D_i n_j_-sigma h_j_sigma * U_0(j, sigma)`。

参数:
- `derivative_orbitals::AbstractMatrix{T}`: 待累加的导数轨道矩阵。
- `base_orbitals::AbstractMatrix{T}`: 裸轨道矩阵 `U_0`。
- `state_vector::Vector{Int8}`: 当前 Monte Carlo 构型。
- `correction_term::BackflowEta3DoublonSingleTerm`: split `eta3` correction term。

返回:
- `nothing`。
"""
function add_backflow_correction_derivative_orbitals!(
    derivative_orbitals::AbstractMatrix{T},
    base_orbitals::AbstractMatrix{T},
    state_vector::Vector{Int8},
    correction_term::BackflowEta3DoublonSingleTerm,
) where {T}
    validate_backflow_correction_source_data!(correction_term)
    for (bond_index, (site_i, site_j)) in enumerate(correction_term.source_bonds)
        state_i = state_vector[site_i]
        state_j = state_vector[site_j]
        bond_amplitude = T(correction_term.source_amplitudes[bond_index])
        for row_offset in 1:2
            spin = backflow_spin_from_row_offset(row_offset)
            eta3_factor = compute_eta3_doublon_single_factor(state_i, state_j, spin)
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
用途: 将 split `eta4` correction term 对 `eta4_bf` 的导数累加到导数轨道矩阵。

数学公式:
- `partial U_b(i, sigma) / partial eta4_bf =
   sum_j t_ij * n_i_sigma h_i_-sigma H_j * U_0(j, sigma)`。

参数:
- `derivative_orbitals::AbstractMatrix{T}`: 待累加的导数轨道矩阵。
- `base_orbitals::AbstractMatrix{T}`: 裸轨道矩阵 `U_0`。
- `state_vector::Vector{Int8}`: 当前 Monte Carlo 构型。
- `correction_term::BackflowEta4SingleHoleTerm`: split `eta4` correction term。

返回:
- `nothing`。
"""
function add_backflow_correction_derivative_orbitals!(
    derivative_orbitals::AbstractMatrix{T},
    base_orbitals::AbstractMatrix{T},
    state_vector::Vector{Int8},
    correction_term::BackflowEta4SingleHoleTerm,
) where {T}
    validate_backflow_correction_source_data!(correction_term)
    for (bond_index, (site_i, site_j)) in enumerate(correction_term.source_bonds)
        state_i = state_vector[site_i]
        state_j = state_vector[site_j]
        bond_amplitude = T(correction_term.source_amplitudes[bond_index])
        for row_offset in 1:2
            spin = backflow_spin_from_row_offset(row_offset)
            eta4_factor = compute_eta4_single_hole_factor(state_i, state_j, spin)
            if eta4_factor == 0.0
                continue
            end
            row_i = 2 * (site_i - 1) + row_offset
            row_j = 2 * (site_j - 1) + row_offset
            @views derivative_orbitals[row_i, :] .+= bond_amplitude * T(eta4_factor) .* base_orbitals[row_j, :]
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


end # module
