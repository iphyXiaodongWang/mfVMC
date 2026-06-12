module Backflow

using ..Sampler

export AbstractBackflowTerm, AbstractBackflowCorrectionTerm
export NoBackflowTerm
export CompositeBackflowTerm
export DirectedBackflowSourceGroup
export BackflowEpsilonTerm, BackflowEta1DoublonHoleTerm
export BackflowEta2SpinExchangeTerm
export BackflowEta3DoublonSingleTerm, BackflowEta4SingleHoleTerm
export build_directed_backflow_source_group
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
- 当前实现保留 Emery directed split backflow correction terms:
  `epsilon`, `eta1`, `eta2`, `eta3_doublon_single`, `eta4`。
- 当前实现只假设轨道矩阵采用每个 site 两行的 spin-resolved layout:
  第一行为 `UP`, 第二行为 `DN`。
- 该 layout 可由不同 determinant 表示共用, 具体行的物理解释由调用方决定。

当前代码采用的组合式写法为:

- `U_b(i, sigma, k; x) = U_0(i, sigma, k) + sum_m delta U_m(i, sigma, k; x)`,
  其中 `m` 遍历 `epsilon`, `eta1`, `eta2`, `eta3_doublon_single`, `eta4`
  correction terms。

其中:
- `i, j` 为格点指标, `sigma` 为物理自旋标签, `k` 为轨道指标。
- `U_0` 为裸轨道矩阵, `U_b` 为构型依赖的 backflow 轨道矩阵。
- `D_i(x) = 1` 当且仅当 site `i` 为 doublon, 否则为 0。
- `H_i(x) = 1` 当且仅当 site `i` 为 hole, 否则为 0。
- epsilon 的激活由 eta contribution 实际数值驱动: 只有当某行对应的 source group
  中有至少一条有向键产生非零 eta coefficient 时, epsilon 才激活。

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


"""
用途: 保存 Eq.(5) 单个 backflow correction term 的公共 source 数据。

参数:
- `source_bonds::Vector{Tuple{Int, Int}}`: 有向键 `(i, j)` 列表。
- `source_amplitudes::Vector{<:Real}`: 与有向键对齐的 hopping 振幅。

返回:
- `NamedTuple`: 包含复制后的 `source_bonds`, `source_amplitudes`, 内容签名与图缓存。
"""
function build_backflow_correction_source_cache(
    source_bonds::Vector{Tuple{Int,Int}},
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
用途: 保存 backflow source graph 的图缓存, 便于快速定位受影响站点。

参数:
- `outgoing_bond_indices_by_source::Vector{Vector{Int}}`: 按 source site 存储的 bond 索引列表。
- `incoming_source_sites_by_target::Vector{Vector{Int}}`: 按 target site 存储的 source site 去重列表。

返回:
- `NamedTuple`: 包含上述两个缓存数组。
"""
function build_backflow_source_graph_cache(source_bonds::Vector{Tuple{Int,Int}})
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
用途: Eq.(5) 中的 `epsilon` backflow correction term.

数学公式:
- `delta U_epsilon(i, sigma) = (epsilon_bf - 1) * U_0(i, sigma)`.
- epsilon 仅在该行对应的 source group 中至少有一条有向键产生非零 eta contribution 时激活.
- 激活判断不再依赖独立的 site-neighbor occupancy mask, 而是由 eta contribution 的实际数值驱动.

字段:
- `param_name::Symbol`: 参数名.
- `epsilon_bf::Float64`: `epsilon` 参数值.
- `group_names::Vector{Symbol}`: 该 epsilon term 控制的 source group 名称列表,
  例如 `[:dd, :dp]` 表示 `bf_epsilon_d` 由 `dd` 和 `dp` 组的 eta contribution 激活.
"""
mutable struct BackflowEpsilonTerm <: AbstractBackflowCorrectionTerm
    param_name::Symbol
    epsilon_bf::Float64
    group_names::Vector{Symbol}
end

"""
用途: Eq.(5) 中的 `eta1` doublon-hole backflow correction term。

数学公式:
- `delta U_eta1(i, sigma) = eta1_bf * sum_j t_ij * D_i * H_j * U_0(j, sigma)`。

字段:
- `param_name::Symbol`: 参数名。
- `eta1_bf::Float64`: `eta1` 参数值。

说明:
- source 数据现在由 `DirectedBackflowSourceGroup` 持有, eta term 变为轻量参数对象。
"""
mutable struct BackflowEta1DoublonHoleTerm <: AbstractBackflowCorrectionTerm
    param_name::Symbol
    eta1_bf::Float64
end

"""
用途: Eq.(5) 中的 `eta2` spin-exchange backflow correction term。

数学公式:
- `delta U_eta2(i, sigma) = eta2_bf * sum_j t_ij *
   n_i_sigma h_i_-sigma n_j_-sigma h_j_sigma * U_0(j, sigma)`。

字段:
- `param_name::Symbol`: 参数名。
- `eta2_bf::Float64`: `eta2` 参数值。

说明:
- source 数据现在由 `DirectedBackflowSourceGroup` 持有, eta term 变为轻量参数对象。
"""
mutable struct BackflowEta2SpinExchangeTerm <: AbstractBackflowCorrectionTerm
    param_name::Symbol
    eta2_bf::Float64
end

"""
用途: 三带 Emery split backflow 中的 `eta3` doublon-single correction term。

数学公式:
- `delta U_eta3(i, sigma) = eta3_bf * sum_j t_ij *
   D_i n_j_-sigma h_j_sigma * U_0(j, sigma)`。

字段:
- `param_name::Symbol`: 参数名。
- `eta3_bf::Float64`: split `eta3` 参数值。

说明:
- source 数据现在由 `DirectedBackflowSourceGroup` 持有, eta term 变为轻量参数对象。
"""
mutable struct BackflowEta3DoublonSingleTerm <: AbstractBackflowCorrectionTerm
    param_name::Symbol
    eta3_bf::Float64
end

"""
用途: 三带 Emery split backflow 中的 `eta4` single-hole correction term。

数学公式:
- `delta U_eta4(i, sigma) = eta4_bf * sum_j t_ij *
   n_i_sigma h_i_-sigma H_j * U_0(j, sigma)`。

字段:
- `param_name::Symbol`: 参数名。
- `eta4_bf::Float64`: split `eta4` 参数值。

说明:
- source 数据现在由 `DirectedBackflowSourceGroup` 持有, eta term 变为轻量参数对象。
"""
mutable struct BackflowEta4SingleHoleTerm <: AbstractBackflowCorrectionTerm
    param_name::Symbol
    eta4_bf::Float64
end

"""
用途: 保存一个 directed split backflow source group 中共享的 source graph 和四个 eta term。

字段:
- `group_name::Symbol`: directed 组名, 例如 `:dd`, `:dp`, `:pd`, `:pp`。
- `source_bonds::Vector{Tuple{Int,Int}}`: 有向键 `(i, j)` 列表。
- `source_amplitudes::Vector{Float64}`: 与有向键对齐的 hopping 振幅。
- `source_data_signature::UInt`: source 数据内容签名, 用于检测原地修改。
- `outgoing_bond_indices_by_source::Vector{Vector{Int}}`: 按 source site 存储的 bond 索引列表。
- `incoming_source_sites_by_target::Vector{Vector{Int}}`: 按 target site 存储的 source site 去重列表。
- `eta1_term::BackflowEta1DoublonHoleTerm`: 该组内的 eta1 doublon-hole term。
- `eta2_term::BackflowEta2SpinExchangeTerm`: 该组内的 eta2 spin-exchange term。
- `eta3_term::BackflowEta3DoublonSingleTerm`: 该组内的 eta3 doublon-single term。
- `eta4_term::BackflowEta4SingleHoleTerm`: 该组内的 eta4 single-hole term。

说明:
- 每个 source group 拥有共享的 `source_bonds`/`source_amplitudes`/graph cache,
  eta term 只保存参数名和参数值, 不再重复保存 source 数据。
"""
struct DirectedBackflowSourceGroup
    group_name::Symbol
    source_bonds::Vector{Tuple{Int,Int}}
    source_amplitudes::Vector{Float64}
    source_data_signature::UInt
    outgoing_bond_indices_by_source::Vector{Vector{Int}}
    incoming_source_sites_by_target::Vector{Vector{Int}}
    eta1_term::BackflowEta1DoublonHoleTerm
    eta2_term::BackflowEta2SpinExchangeTerm
    eta3_term::BackflowEta3DoublonSingleTerm
    eta4_term::BackflowEta4SingleHoleTerm
end

"""
用途: 构造一个 directed backflow source group。

参数:
- `group_name::Symbol`: directed 组名, 例如 `:dd`, `:dp`, `:pd`, `:pp`。
- `source_bonds::Vector{Tuple{Int,Int}}`: 有向键 `(i, j)` 列表。
- `source_amplitudes::Vector{<:Real}`: 每条有向键对应的 hopping 振幅。
- `eta1_term::BackflowEta1DoublonHoleTerm`: 该组的 eta1 term。
- `eta2_term::BackflowEta2SpinExchangeTerm`: 该组的 eta2 term。
- `eta3_term::BackflowEta3DoublonSingleTerm`: 该组的 eta3 term。
- `eta4_term::BackflowEta4SingleHoleTerm`: 该组的 eta4 term。

返回:
- `DirectedBackflowSourceGroup`: 带 source 图缓存的 directed group。
"""
function build_directed_backflow_source_group(
    group_name::Symbol,
    source_bonds::Vector{Tuple{Int,Int}},
    source_amplitudes::Vector{<:Real},
    eta1_term::BackflowEta1DoublonHoleTerm,
    eta2_term::BackflowEta2SpinExchangeTerm,
    eta3_term::BackflowEta3DoublonSingleTerm,
    eta4_term::BackflowEta4SingleHoleTerm,
)::DirectedBackflowSourceGroup
    source_cache = build_backflow_correction_source_cache(source_bonds, source_amplitudes)
    return DirectedBackflowSourceGroup(
        group_name,
        source_cache.source_bonds,
        source_cache.source_amplitudes,
        source_cache.source_data_signature,
        source_cache.outgoing_bond_indices_by_source,
        source_cache.incoming_source_sites_by_target,
        eta1_term,
        eta2_term,
        eta3_term,
        eta4_term,
    )
end

"""
用途: 从 directed source groups 合并 composite 级别的 incoming source graph.

参数:
- `source_groups::Vector{DirectedBackflowSourceGroup}`: directed source groups.

返回:
- `Vector{Vector{Int}}`: composite 级别合并后的 incoming source graph.

说明:
- epsilon terms 不再持有独立的 site-neighbor mask, incoming graph 只来自 eta source groups.
"""
function build_composite_incoming_from_groups(
    source_groups::Vector{DirectedBackflowSourceGroup},
)::Vector{Vector{Int}}
    max_target_site = 0
    for source_group in source_groups
        max_target_site = max(max_target_site, length(source_group.incoming_source_sites_by_target))
    end

    incoming_source_sites_by_target = [Int[] for _ in 1:max_target_site]

    for source_group in source_groups
        for target_site in eachindex(source_group.incoming_source_sites_by_target)
            target_sources = incoming_source_sites_by_target[target_site]
            for source_site in source_group.incoming_source_sites_by_target[target_site]
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
用途: 组合多个 Eq.(5) backflow correction terms。

字段:
- `epsilon_terms::Vector{BackflowEpsilonTerm}`: composite 中所有 epsilon correction term,
  允许不同 source 子图使用不同 epsilon 参数。
- `source_groups::Vector{DirectedBackflowSourceGroup}`: directed split backflow source groups,
  每个 group 持有共享的 source 数据和四个 eta term。
- `terms::Vector{AbstractBackflowCorrectionTerm}`: 按参数顺序展开的 term 列表,
  由 `epsilon_terms` 和 `source_groups` 的 eta term 自动生成。
- `incoming_source_sites_by_target::Vector{Vector{Int}}`: composite 级别合并后的 incoming source graph,
  用于 proposal 局域更新时一次性收集 affected sites。
"""
struct CompositeBackflowTerm <: AbstractBackflowTerm
    epsilon_terms::Vector{BackflowEpsilonTerm}
    source_groups::Vector{DirectedBackflowSourceGroup}
    terms::Vector{AbstractBackflowCorrectionTerm}
    incoming_source_sites_by_target::Vector{Vector{Int}}

    function CompositeBackflowTerm(
        epsilon_terms::Vector{BackflowEpsilonTerm},
        source_groups::Vector{DirectedBackflowSourceGroup},
    )
        term_list = AbstractBackflowCorrectionTerm[]
        for epsilon_term in epsilon_terms
            push!(term_list, epsilon_term)
        end
        for source_group in source_groups
            push!(term_list, source_group.eta1_term)
            push!(term_list, source_group.eta2_term)
            push!(term_list, source_group.eta3_term)
            push!(term_list, source_group.eta4_term)
        end

        return new(
            epsilon_terms,
            source_groups,
            term_list,
            build_composite_incoming_from_groups(source_groups),
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
    source_bonds::Vector{Tuple{Int,Int}},
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
用途: 构造 Eq.(5) 的 `epsilon` backflow correction term.

参数:
- `param_name::Symbol`: 参数名, 默认 `:bf_epsilon`.
- `epsilon_bf::Real`: `epsilon` 参数值.
- `group_names::Vector{Symbol}`: 该 epsilon term 控制的 source group 名称列表, 默认为空.

返回:
- `BackflowEpsilonTerm`: 轻量 correction term, 不带独立的 site-neighbor mask.
"""
function BackflowEpsilonTerm(;
    param_name::Symbol=:bf_epsilon,
    epsilon_bf::Real=1.0,
    group_names::Vector{Symbol}=Symbol[],
)
    return BackflowEpsilonTerm(
        param_name,
        Float64(epsilon_bf),
        group_names,
    )
end

"""
用途: 构造 Eq.(5) 的 `eta1` doublon-hole backflow correction term。

参数:
- `param_name::Symbol`: 参数名, 默认 `:bf_eta1`。
- `eta1_bf::Real`: `eta1` 参数值。

返回:
- `BackflowEta1DoublonHoleTerm`: 轻量 correction term, source 数据由 `DirectedBackflowSourceGroup` 持有。
"""
function BackflowEta1DoublonHoleTerm(;
    param_name::Symbol=:bf_eta1,
    eta1_bf::Real=0.0,
)
    return BackflowEta1DoublonHoleTerm(param_name, Float64(eta1_bf))
end

"""
用途: 构造 Eq.(5) 的 `eta2` spin-exchange backflow correction term。

参数:
- `param_name::Symbol`: 参数名, 默认 `:bf_eta2`。
- `eta2_bf::Real`: `eta2` 参数值。

返回:
- `BackflowEta2SpinExchangeTerm`: 轻量 correction term, source 数据由 `DirectedBackflowSourceGroup` 持有。
"""
function BackflowEta2SpinExchangeTerm(;
    param_name::Symbol=:bf_eta2,
    eta2_bf::Real=0.0,
)
    return BackflowEta2SpinExchangeTerm(param_name, Float64(eta2_bf))
end

"""
用途: 构造 split 版本的 `eta3` doublon-single backflow correction term。

参数:
- `param_name::Symbol`: 参数名, 默认 `:bf_eta3`。
- `eta3_bf::Real`: split `eta3` 参数值。

返回:
- `BackflowEta3DoublonSingleTerm`: 轻量 correction term, source 数据由 `DirectedBackflowSourceGroup` 持有。
"""
function BackflowEta3DoublonSingleTerm(;
    param_name::Symbol=:bf_eta3,
    eta3_bf::Real=0.0,
)
    return BackflowEta3DoublonSingleTerm(param_name, Float64(eta3_bf))
end

"""
用途: 构造 split 版本的 `eta4` single-hole backflow correction term。

参数:
- `param_name::Symbol`: 参数名, 默认 `:bf_eta4`。
- `eta4_bf::Real`: split `eta4` 参数值。

返回:
- `BackflowEta4SingleHoleTerm`: 轻量 correction term, source 数据由 `DirectedBackflowSourceGroup` 持有。
"""
function BackflowEta4SingleHoleTerm(;
    param_name::Symbol=:bf_eta4,
    eta4_bf::Real=0.0,
)
    return BackflowEta4SingleHoleTerm(param_name, Float64(eta4_bf))
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
用途: 校验局域 site row 尺寸并写入裸轨道基准行。

参数:
- `site_row_buffer::AbstractVector{T}`: 输出 buffer, 长度必须等于轨道数。
- `base_orbitals::AbstractMatrix{T}`: 裸轨道矩阵 `U_0`。
- `state_vector::Vector{Int8}`: proposal 提交前的构型状态数组。
- `site_index::Int`: 待写入的站点编号。
- `row_offset::Int`: 站点内部自旋行偏移, `1` 为 up, `2` 为 down。

返回:
- `Int`: 该 site row 在 spin-resolved 轨道矩阵中的全局行号。
"""
function initialize_site_row_base_after_proposal!(
    site_row_buffer::AbstractVector{T},
    base_orbitals::AbstractMatrix{T},
    state_vector::Vector{Int8},
    site_index::Int,
    row_offset::Int,
)::Int where {T}
    validate_orbital_dimensions(base_orbitals, length(state_vector))
    if length(site_row_buffer) != size(base_orbitals, 2)
        error("Local site row buffer must have length $(size(base_orbitals, 2)), got $(length(site_row_buffer)).")
    end
    if !(1 <= site_index <= length(state_vector))
        error("Affected site $site_index is out of bounds for a state vector with $(length(state_vector)) sites.")
    end
    if row_offset != 1 && row_offset != 2
        error("row_offset must be 1 or 2, got $(row_offset).")
    end

    row_index = 2 * (site_index - 1) + row_offset
    copyto!(site_row_buffer, @view(base_orbitals[row_index, :]))
    return row_index
end

"""
用途: 计算 source site `site_index` 在给定 row_offset 和 states 时, 对各 source group
的 eta contribution 系数并累加到 output_row, 同时收集产生非零 eta 的 group 名称。

数学公式:
- `eta_coefficient = t_ij * (
      eta1_bf * eta1_factor + eta2_bf * eta2_factor
      + eta3_bf * eta3_factor + eta4_bf * eta4_factor)`.
- 只要系数非零, 就将对应的 group_name 记录到 active_group_names 集合。

参数:
- `output_row::AbstractVector{T}`: 待累加的轨道行.
- `base_orbitals::AbstractMatrix{T}`: 裸轨道矩阵 `U_0`.
- `state_i::Int8`: source site 的状态编码.
- `get_state_j::Function`: `(site_index) -> Int8`, 获取 target site 的状态.
- `source_group::DirectedBackflowSourceGroup`: directed source group.
- `site_index::Int`: source site 编号.
- `row_offset::Int`: 站点内部自旋行偏移.
- `active_group_names::Set{Symbol}`: 用于收集中产生非零 eta 的 group 名称.

返回:
- `nothing`.

说明:
- proposal 路径传入 `get_site_state_after_proposal` 闭包, 全重建路径传入 `state_j -> state_vector[site_j]` 函数.
"""
function add_source_group_eta_contributions_and_track_activation!(
    output_row::AbstractVector{T},
    base_orbitals::AbstractMatrix{T},
    state_i::Int8,
    get_state_j::Function,
    source_group::DirectedBackflowSourceGroup,
    site_index::Int,
    row_offset::Int,
    active_group_names::Set{Symbol},
) where {T}
    if site_index > length(source_group.outgoing_bond_indices_by_source)
        return nothing
    end

    eta1_value = T(source_group.eta1_term.eta1_bf)
    eta2_value = T(source_group.eta2_term.eta2_bf)
    eta3_value = T(source_group.eta3_term.eta3_bf)
    eta4_value = T(source_group.eta4_term.eta4_bf)
    spin = backflow_spin_from_row_offset(row_offset)
    has_eta = false

    for bond_index in source_group.outgoing_bond_indices_by_source[site_index]
        (_, target_site) = source_group.source_bonds[bond_index]
        state_j = get_state_j(target_site)
        bond_amplitude = T(source_group.source_amplitudes[bond_index])
        eta_contribution = compute_backflow_eta_contribution(
            state_i,
            state_j,
            spin,
            bond_amplitude,
            eta1_value,
            eta2_value,
            eta3_value,
            eta4_value,
        )
        coefficient = eta_contribution.coefficient
        if coefficient == zero(T)
            continue
        end

        target_row = 2 * (target_site - 1) + row_offset
        @views output_row .+= coefficient .* base_orbitals[target_row, :]
        has_eta = true
    end

    if has_eta
        push!(active_group_names, source_group.group_name)
    end

    return nothing
end

"""
用途: 在 eta 贡献计算完成后, 根据 active_group_names 添加 epsilon correction。

数学公式:
- 对每个 epsilon term, 若其 group_names 中至少有一个在 active_group_names 中,
  则 `output_row += (epsilon_bf - 1.0) * base_orbitals[row_index, :]`.

参数:
- `output_row::AbstractVector{T}`: 待累加的轨道行.
- `base_orbitals::AbstractMatrix{T}`: 裸轨道矩阵 `U_0`.
- `epsilon_terms::Vector{BackflowEpsilonTerm}`: composite 中的所有 epsilon term.
- `row_index::Int`: 当前全局行号.
- `active_group_names::Set{Symbol}`: 产生了非零 eta 贡献的 group 名称集合.

返回:
- `nothing`.
"""
function add_epsilon_contributions_from_active_groups!(
    output_row::AbstractVector{T},
    base_orbitals::AbstractMatrix{T},
    epsilon_terms::Vector{BackflowEpsilonTerm},
    row_index::Int,
    active_group_names::Set{Symbol},
) where {T}
    for epsilon_term in epsilon_terms
        epsilon_shift = T(epsilon_term.epsilon_bf - 1.0)
        if epsilon_shift == zero(T)
            continue
        end
        for group_name in epsilon_term.group_names
            if group_name in active_group_names
                @views output_row .+= epsilon_shift .* base_orbitals[row_index, :]
                break
            end
        end
    end
    return nothing
end

"""
用途: 使用统一 eta-driven epsilon 逻辑构造单个 `(site, spin)` backflow row。

数学公式:
- `U_b(i,sigma) = U_0(i,sigma) + sum_g sum_j eta_coeff_g(i,j,sigma) U_0(j,sigma)
  + sum_e I_e(i,sigma) (epsilon_e - 1) U_0(i,sigma)`.
- `I_e(i,sigma)=1` 当且仅当该 epsilon term 控制的任一 source group 对该 row
  产生非零 eta coefficient。

参数:
- `output_row::AbstractVector{T}`: 输出 row buffer, 长度等于轨道数.
- `base_orbitals::AbstractMatrix{T}`: 裸轨道矩阵 `U_0`.
- `backflow_term::CompositeBackflowTerm`: directed split 组合式 backflow 对象.
- `site_index::Int`: source site 编号.
- `row_offset::Int`: site 内部自旋行偏移, `1` 为 up, `2` 为 down.
- `get_state::Function`: `(site_index) -> Int8`, 用于读取当前或 proposal 后的 site 状态.

返回:
- `Int`: 当前 row 在 spin-resolved 轨道矩阵中的全局行号.
"""
function fill_backflow_site_row_from_state_getter!(
    output_row::AbstractVector{T},
    base_orbitals::AbstractMatrix{T},
    backflow_term::CompositeBackflowTerm,
    site_index::Int,
    row_offset::Int,
    get_state::Function,
)::Int where {T}
    row_index = 2 * (site_index - 1) + row_offset
    copyto!(output_row, @view(base_orbitals[row_index, :]))

    state_i = get_state(site_index)
    spin = backflow_spin_from_row_offset(row_offset)
    if backflow_n_sigma(state_i, spin) == 0.0
        return row_index
    end

    active_group_names = Set{Symbol}()
    for source_group in backflow_term.source_groups
        add_source_group_eta_contributions_and_track_activation!(
            output_row,
            base_orbitals,
            state_i,
            get_state,
            source_group,
            site_index,
            row_offset,
            active_group_names,
        )
    end

    add_epsilon_contributions_from_active_groups!(
        output_row,
        base_orbitals,
        backflow_term.epsilon_terms,
        row_index,
        active_group_names,
    )

    return row_index
end

"""
用途: 对 directed split backflow 使用 eta-driven epsilon 逻辑写入 proposal 后的单个 occupied row。

数学公式:
- 对固定 `(i,sigma)` 只计算该 row 需要的 eta1/eta2/eta3/eta4 contribution。
- epsilon 仅在对应 source group 产生了非零 eta contribution 时激活。

参数:
- `site_row_buffer::AbstractVector{T}`: 输出 buffer, 长度必须等于轨道数。
- `base_orbitals::AbstractMatrix{T}`: 裸轨道矩阵 `U_0`。
- `state_vector::Vector{Int8}`: proposal 提交前的构型状态数组。
- `backflow_term::CompositeBackflowTerm`: directed split 组合式 backflow 对象。
- `proposal::MoveProposal`: Monte Carlo proposal。
- `site_index::Int`: 待写入的站点编号。
- `row_offset::Int`: 站点内部自旋行偏移, `1` 为 up, `2` 为 down。

返回:
- `nothing`。
"""
function fill_grouped_source_composite_site_row_after_proposal!(
    site_row_buffer::AbstractVector{T},
    base_orbitals::AbstractMatrix{T},
    state_vector::Vector{Int8},
    backflow_term::CompositeBackflowTerm,
    proposal::MoveProposal,
    site_index::Int,
    row_offset::Int,
) where {T}
    initialize_site_row_base_after_proposal!(
        site_row_buffer,
        base_orbitals,
        state_vector,
        site_index,
        row_offset,
    )
    fill_backflow_site_row_from_state_getter!(
        site_row_buffer,
        base_orbitals,
        backflow_term,
        site_index,
        row_offset,
        site_j -> get_site_state_after_proposal(state_vector, proposal, site_j),
    )

    return nothing
end

"""
用途: 写入组合式 Eq.(5) backflow 在 proposal 提交后的单个 occupied row。

参数:
- `site_row_buffer::AbstractVector{T}`: 输出 buffer, 长度必须等于轨道数。
- `base_orbitals::AbstractMatrix{T}`: 裸轨道矩阵 `U_0`。
- `state_vector::Vector{Int8}`: proposal 提交前的构型状态数组。
- `backflow_term::CompositeBackflowTerm`: 组合式 Eq.(5) backflow 对象。
- `proposal::MoveProposal`: Monte Carlo proposal。
- `site_index::Int`: 待写入的站点编号。
- `row_offset::Int`: 站点内部自旋行偏移, `1` 为 up, `2` 为 down。

返回:
- `nothing`。
"""
function fill_backflow_site_row_after_proposal!(
    site_row_buffer::AbstractVector{T},
    base_orbitals::AbstractMatrix{T},
    state_vector::Vector{Int8},
    backflow_term::CompositeBackflowTerm,
    proposal::MoveProposal,
    site_index::Int,
    row_offset::Int,
) where {T}
    return fill_grouped_source_composite_site_row_after_proposal!(
        site_row_buffer,
        base_orbitals,
        state_vector,
        backflow_term,
        proposal,
        site_index,
        row_offset,
    )
end

"""
用途: 对 directed split backflow 使用 eta-driven epsilon 逻辑写入 proposal 后的局域 site block。

数学公式:
- 对每个 directed 组 `g in {dd, dp, pd, pp}` 和有向键 `(i,j)` 一次性计算
  `eta1_g D_i H_j U_0(j,sigma)`,
  `eta2_g n_{i,sigma} h_{i,-sigma} n_{j,-sigma} h_{j,sigma} U_0(j,sigma)`,
  `eta3_g D_i n_{j,-sigma} h_{j,sigma} U_0(j,sigma)`,
  `eta4_g n_{i,sigma} h_{i,-sigma} H_j U_0(j,sigma)`。
- epsilon 仅在对应 source group 产生了非零 eta contribution 时激活。

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
    initialize_site_block_base_after_proposal!(
        site_block_buffer,
        base_orbitals,
        state_vector,
        site_index,
    )
    get_state = site_j -> get_site_state_after_proposal(state_vector, proposal, site_j)
    fill_backflow_site_row_from_state_getter!(
        @view(site_block_buffer[1, :]),
        base_orbitals,
        backflow_term,
        site_index,
        1,
        get_state,
    )
    fill_backflow_site_row_from_state_getter!(
        @view(site_block_buffer[2, :]),
        base_orbitals,
        backflow_term,
        site_index,
        2,
        get_state,
    )

    return nothing
end

"""
用途: 写入组合式 Eq.(5) backflow 在 proposal 提交后的局域站点行块。

数学公式:
- `U_b(i, sigma; x') = U_0(i, sigma) + sum_m delta U_m(i, sigma; x')`,
  其中 `m` 遍历 `epsilon, eta1, eta2, eta3_doublon_single, eta4` correction terms。

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
    return fill_grouped_source_composite_site_block_after_proposal!(
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
用途: 统一计算单条有向 source bond 对某个 `(i, sigma)` row 的 eta factors 与实际 eta coefficient。

数学公式:
- `coefficient = t_ij * (eta1 * f1 + eta2 * f2 + eta3 * f3 + eta4 * f4)`,
  其中 `f1 = D_i H_j`,
  `f2 = n_{i,sigma} h_{i,-sigma} n_{j,-sigma} h_{j,sigma}`,
  `f3 = D_i n_{j,-sigma} h_{j,sigma}`,
  `f4 = n_{i,sigma} h_{i,-sigma} H_j`.

参数:
- `state_i::Int8`: source site `i` 的物理状态编码.
- `state_j::Int8`: target site `j` 的物理状态编码.
- `spin::Int8`: 当前 row 对应的物理自旋 `sigma`.
- `bond_amplitude::T`: 有向键 hopping 振幅 `t_ij`.
- `eta1_value, eta2_value, eta3_value, eta4_value::T`: 当前 source group 的 eta 参数值.

返回:
- `NamedTuple`: 包含 `coefficient`, `eta1_factor`, `eta2_factor`,
  `eta3_factor`, `eta4_factor`, 均为类型 `T`.
"""
function compute_backflow_eta_contribution(
    state_i::Int8,
    state_j::Int8,
    spin::Int8,
    bond_amplitude::T,
    eta1_value::T,
    eta2_value::T,
    eta3_value::T,
    eta4_value::T,
) where {T}
    eta1_factor = (state_i == DB && state_j == HOLE) ? one(T) : zero(T)
    eta2_factor = T(compute_eta2_virtual_hopping_factor(state_i, state_j, spin))
    eta3_factor = T(compute_eta3_doublon_single_factor(state_i, state_j, spin))
    eta4_factor = T(compute_eta4_single_hole_factor(state_i, state_j, spin))
    coefficient =
        bond_amplitude *
        (
            eta1_value * eta1_factor +
            eta2_value * eta2_factor +
            eta3_value * eta3_factor +
            eta4_value * eta4_factor
        )
    return (
        coefficient=coefficient,
        eta1_factor=eta1_factor,
        eta2_factor=eta2_factor,
        eta3_factor=eta3_factor,
        eta4_factor=eta4_factor,
    )
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
用途: 校验 DirectedBackflowSourceGroup 的 source 数据在构造后未被原地修改。

参数:
- `source_group::DirectedBackflowSourceGroup`: 待校验的 source group。

返回:
- `nothing`。若检测到原地修改则抛出异常。
"""
function validate_backflow_source_group_data!(
    source_group::DirectedBackflowSourceGroup,
)
    current_signature = compute_backflow_source_data_signature(
        source_group.source_bonds,
        source_group.source_amplitudes,
    )

    if current_signature != source_group.source_data_signature
        error("Backflow source group $(source_group.group_name) source_bonds/source_amplitudes were mutated after construction. Please rebuild the source group instead of modifying it in place.")
    end

    return nothing
end

"""
用途: 构造组合式 Eq.(5) backflow 的构型依赖轨道矩阵。

数学公式:
- `U_b = U_0 + sum_m delta U_m`, 其中每个 `delta U_m` 由一个
  `AbstractBackflowCorrectionTerm` 提供。
- epsilon 由 eta contribution 的实际数值驱动: 只有当某行对应的 source group
  中有至少一条有向键产生非零 eta coefficient 时, epsilon 才激活。

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
    n_sites = length(state_vector)

    for site_i in 1:n_sites
        for row_offset in 1:2
            row_i = 2 * (site_i - 1) + row_offset
            fill_backflow_site_row_from_state_getter!(
                @view(backflow_orbitals[row_i, :]),
                base_orbitals,
                backflow_term,
                site_i,
                row_offset,
                site_j -> state_vector[site_j],
            )
        end
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
- epsilon 由 eta contribution 的实际数值驱动。

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
    n_sites = length(state_vector)

    for site_i in 1:n_sites
        state_i = state_vector[site_i]
        for row_offset in 1:2
            spin = backflow_spin_from_row_offset(row_offset)
            if backflow_n_sigma(state_i, spin) == 0.0
                continue
            end
            row_i = 2 * (site_i - 1) + row_offset

            active_group_names = Set{Symbol}()
            for source_group in backflow_term.source_groups
                add_source_group_eta_contributions_and_track_activation!(
                    @view(output_orbitals[row_i, :]),
                    input_derivative_orbitals,
                    state_i,
                    site_j -> state_vector[site_j],
                    source_group,
                    site_i,
                    row_offset,
                    active_group_names,
                )
            end

            add_epsilon_contributions_from_active_groups!(
                @view(output_orbitals[row_i, :]),
                input_derivative_orbitals,
                backflow_term.epsilon_terms,
                row_i,
                active_group_names,
            )
        end
    end

    return nothing
end

"""
用途: 校验 chain-rule 单行输出并写入裸导数轨道基准行。

参数:
- `output_row::AbstractVector{T}`: 输出 buffer, 长度必须等于轨道数。
- `input_derivative_orbitals::AbstractMatrix{T}`: 输入裸轨道导数 `dU_0 / dp`。
- `state_vector::Vector{Int8}`: 当前构型。
- `row_index::Int`: 需要计算的 spin-resolved 全局行号。

返回:
- `Tuple{Int, Int}`: `(site_index, row_offset)`, 其中 `row_offset=1` 为 up, `2` 为 down。
"""
function initialize_backflow_chain_rule_row!(
    output_row::AbstractVector{T},
    input_derivative_orbitals::AbstractMatrix{T},
    state_vector::Vector{Int8},
    row_index::Int,
)::Tuple{Int,Int} where {T}
    validate_orbital_dimensions(input_derivative_orbitals, length(state_vector))
    if length(output_row) != size(input_derivative_orbitals, 2)
        error("Chain-rule row buffer length $(length(output_row)) != orbital count $(size(input_derivative_orbitals, 2)).")
    end
    if !(1 <= row_index <= size(input_derivative_orbitals, 1))
        error("Chain-rule row index $row_index is out of bounds for $(size(input_derivative_orbitals, 1)) rows.")
    end

    site_index = div(row_index + 1, 2)
    row_offset = isodd(row_index) ? 1 : 2
    copyto!(output_row, @view(input_derivative_orbitals[row_index, :]))
    return site_index, row_offset
end

"""
用途: 只计算指定 occupied row 的 backflow chain-rule 轨道导数。

数学公式:
- 若 `U_b = B_x[U_0]`, 则对裸导数矩阵 `dU_0 / dp`,
  本函数返回单行 `(B_x[dU_0 / dp])[row_index, :]`。
- epsilon 由 eta contribution 的实际数值驱动。

参数:
- `output_row::AbstractVector{T}`: 输出 buffer, 长度必须等于轨道数。
- `input_derivative_orbitals::AbstractMatrix{T}`: 输入裸轨道导数 `dU_0 / dp`。
- `state_vector::Vector{Int8}`: 当前 Monte Carlo 构型。
- `backflow_term::CompositeBackflowTerm`: 组合式 Eq.(5) backflow 对象。
- `row_index::Int`: 需要计算的 spin-resolved 全局行号。

返回:
- `nothing`。
"""
function fill_backflow_chain_rule_row!(
    output_row::AbstractVector{T},
    input_derivative_orbitals::AbstractMatrix{T},
    state_vector::Vector{Int8},
    backflow_term::CompositeBackflowTerm,
    row_index::Int,
) where {T}
    row_count = 2 * length(state_vector)
    source_row_indices = Vector{Int}(undef, row_count)
    source_row_weights = Vector{T}(undef, row_count)
    source_count = fill_backflow_row_source_weights_from_state_getter!(
        source_row_indices,
        source_row_weights,
        state_vector,
        backflow_term,
        row_index,
        site_index -> state_vector[site_index],
    )
    fill_backflow_row_from_source_weights!(
        output_row,
        input_derivative_orbitals,
        source_row_indices,
        source_row_weights,
        source_count,
    )
    return nothing
end

"""
用途: 在无 backflow 情况下只复制指定 row 的裸轨道导数。

参数:
- `output_row::AbstractVector{T}`: 输出 buffer, 长度必须等于轨道数。
- `input_derivative_orbitals::AbstractMatrix{T}`: 输入裸轨道导数 `dU_0 / dp`。
- `state_vector::Vector{Int8}`: 当前构型, 此处仅用于接口统一。
- `backflow_term::NoBackflowTerm`: 空 backflow 对象。
- `row_index::Int`: 需要复制的 spin-resolved 全局行号。

返回:
- `nothing`。
"""
function fill_backflow_chain_rule_row!(
    output_row::AbstractVector{T},
    input_derivative_orbitals::AbstractMatrix{T},
    state_vector::Vector{Int8},
    ::NoBackflowTerm,
    row_index::Int,
) where {T}
    initialize_backflow_chain_rule_row!(
        output_row,
        input_derivative_orbitals,
        state_vector,
        row_index,
    )
    return nothing
end

"""
用途: 初始化指定 row 的 backflow chain-rule source-row 权重列表。

参数:
- `source_row_indices::AbstractVector{Int}`: 输出 source row 编号 buffer, 长度至少为 `2 * length(state_vector)`。
- `source_row_weights::AbstractVector{T}`: 输出 source row 权重 buffer, 与 `source_row_indices` 对齐。
- `state_vector::Vector{Int8}`: 当前 Monte Carlo 构型。
- `row_index::Int`: 需要展开的 spin-resolved 全局行号。

返回:
- `Tuple{Int, Int, Int}`: `(site_index, row_offset, source_count)`。

公式:
- 初始项为恒等变换, 即 `dU_b(row) = 1 * dU_0(row) + ...`。
"""
function initialize_backflow_chain_rule_source_weights!(
    source_row_indices::AbstractVector{Int},
    source_row_weights::AbstractVector{T},
    state_vector::Vector{Int8},
    row_index::Int,
)::Tuple{Int,Int,Int} where {T}
    row_count = 2 * length(state_vector)
    if length(source_row_indices) < row_count
        error("source_row_indices length $(length(source_row_indices)) < required row count $row_count.")
    end
    if length(source_row_weights) < row_count
        error("source_row_weights length $(length(source_row_weights)) < required row count $row_count.")
    end
    if !(1 <= row_index <= row_count)
        error("Chain-rule row index $row_index is out of bounds for $row_count rows.")
    end

    site_index = div(row_index + 1, 2)
    row_offset = isodd(row_index) ? 1 : 2
    source_row_indices[1] = row_index
    source_row_weights[1] = one(T)
    return site_index, row_offset, 1
end

"""
用途: 根据 source row 权重列表 materialize 一个 backflow row。

参数:
- `output_row::AbstractVector{T}`: 输出 buffer, 长度等于 orbital 列数。
- `input_orbitals::AbstractMatrix{T}`: 被线性组合的输入轨道矩阵, 可以是 `U_0` 或 `dU_0/dp`。
- `source_row_indices::AbstractVector{Int}`: source row 编号 buffer。
- `source_row_weights::AbstractVector{T}`: source row 权重 buffer。
- `source_count::Int`: 有效 source row 数量。

返回:
- `nothing`。

公式:
- `output_row[:] = sum_{k=1}^{source_count} source_row_weights[k] * input_orbitals[source_row_indices[k], :]`。
"""
function fill_backflow_row_from_source_weights!(
    output_row::AbstractVector{T},
    input_orbitals::AbstractMatrix{T},
    source_row_indices::AbstractVector{Int},
    source_row_weights::AbstractVector{T},
    source_count::Int,
) where {T}
    fill!(output_row, zero(T))
    for source_offset in 1:source_count
        source_row_index = source_row_indices[source_offset]
        source_weight = source_row_weights[source_offset]
        @views output_row .+= source_weight .* input_orbitals[source_row_index, :]
    end
    return nothing
end

"""
用途: 将一个 source row 权重累加到权重列表, 若 source row 已存在则合并权重。

参数:
- `source_row_indices::AbstractVector{Int}`: source row 编号 buffer。
- `source_row_weights::AbstractVector{T}`: source row 权重 buffer。
- `source_count::Int`: 当前已使用的 source row 数量。
- `source_row_index::Int`: 待累加的 source row 编号。
- `source_weight::T`: 待累加的权重。

返回:
- `Int`: 更新后的 source row 数量。
"""
function add_backflow_chain_rule_source_weight!(
    source_row_indices::AbstractVector{Int},
    source_row_weights::AbstractVector{T},
    source_count::Int,
    source_row_index::Int,
    source_weight::T,
)::Int where {T}
    if source_weight == zero(T)
        return source_count
    end
    @inbounds for source_offset in 1:source_count
        if source_row_indices[source_offset] == source_row_index
            source_row_weights[source_offset] += source_weight
            return source_count
        end
    end
    if source_count >= length(source_row_indices) || source_count >= length(source_row_weights)
        error("Backflow source weight buffer is too small for source row $source_row_index.")
    end

    next_source_count = source_count + 1
    source_row_indices[next_source_count] = source_row_index
    source_row_weights[next_source_count] = source_weight
    return next_source_count
end

"""
用途: 使用 grouped source 逻辑将单个 source group 的 chain rule source-weight 贡献累加到列表, 同时跟踪产生了非零 eta 的 group 名称.

参数:
- `source_row_indices::AbstractVector{Int}`: source row 编号 buffer.
- `source_row_weights::AbstractVector{T}`: source row 权重 buffer.
- `source_count::Int`: 当前 source row 数量.
- `state_i::Int8`: source site 的状态编码.
- `get_state_j::Function`: `(site_index) -> Int8`, 用于读取 target site 状态.
- `source_group::DirectedBackflowSourceGroup`: directed source group.
- `site_index::Int`: 当前 site 编号.
- `row_offset::Int`: 当前 site 内部自旋行偏移.
- `active_group_names::Set{Symbol}`: 输出, 产生了非零 eta 的 group 名称集合.

返回:
- `Int`: 更新后的 source row 数量.
"""
function add_source_group_chain_rule_source_weights_and_track!(
    source_row_indices::AbstractVector{Int},
    source_row_weights::AbstractVector{T},
    source_count::Int,
    state_i::Int8,
    get_state_j::Function,
    source_group::DirectedBackflowSourceGroup,
    site_index::Int,
    row_offset::Int,
    active_group_names::Set{Symbol},
)::Int where {T}
    if site_index > length(source_group.outgoing_bond_indices_by_source)
        return source_count
    end

    has_eta = false
    spin = backflow_spin_from_row_offset(row_offset)
    eta1_value = T(source_group.eta1_term.eta1_bf)
    eta2_value = T(source_group.eta2_term.eta2_bf)
    eta3_value = T(source_group.eta3_term.eta3_bf)
    eta4_value = T(source_group.eta4_term.eta4_bf)

    for bond_index in source_group.outgoing_bond_indices_by_source[site_index]
        (_, target_site) = source_group.source_bonds[bond_index]
        state_j = get_state_j(target_site)
        target_row = 2 * (target_site - 1) + row_offset
        bond_amplitude = T(source_group.source_amplitudes[bond_index])
        eta_contribution = compute_backflow_eta_contribution(
            state_i,
            state_j,
            spin,
            bond_amplitude,
            eta1_value,
            eta2_value,
            eta3_value,
            eta4_value,
        )
        coefficient = eta_contribution.coefficient
        if coefficient == zero(T)
            continue
        end

        source_count = add_backflow_chain_rule_source_weight!(
            source_row_indices,
            source_row_weights,
            source_count,
            target_row,
            coefficient,
        )
        has_eta = true
    end

    if has_eta
        push!(active_group_names, source_group.group_name)
    end

    return source_count
end

"""
用途: 将指定 output row 的 backflow chain-rule 展开为 source rows 与权重。

参数:
- `source_row_indices::AbstractVector{Int}`: 输出 source row 编号 buffer。
- `source_row_weights::AbstractVector{T}`: 输出 source row 权重 buffer。
- `state_vector::Vector{Int8}`: 当前 Monte Carlo 构型。
- `backflow_term::CompositeBackflowTerm`: 组合式 Eq.(5) backflow 对象。
- `row_index::Int`: 需要展开的 spin-resolved 全局行号。

返回:
- `Int`: 有效 source row 数量。

公式:
- 若 `(B_x[dU_0])(row,:) = sum_s w_s dU_0(s,:)`, 本函数返回所有 `(s,w_s)`。
- epsilon 只在对应 source group 产生非零 eta contribution 时才贡献权重。
"""
function fill_backflow_chain_rule_source_weights!(
    source_row_indices::AbstractVector{Int},
    source_row_weights::AbstractVector{T},
    state_vector::Vector{Int8},
    backflow_term::CompositeBackflowTerm,
    row_index::Int,
)::Int where {T}
    return fill_backflow_row_source_weights_from_state_getter!(
        source_row_indices,
        source_row_weights,
        state_vector,
        backflow_term,
        row_index,
        site_index -> state_vector[site_index],
    )
end

"""
用途: 根据给定 state getter 将一个 backflow output row 展开为 source row 权重列表。

参数:
- `source_row_indices::AbstractVector{Int}`: 输出 source row 编号 buffer, 长度至少为 `2 * length(state_vector)`。
- `source_row_weights::AbstractVector{T}`: 输出 source row 权重 buffer, 与 `source_row_indices` 对齐。
- `state_vector::Vector{Int8}`: 当前 Monte Carlo 构型, 用于确定总 site 数和 row 边界。
- `backflow_term::CompositeBackflowTerm`: Emery grouped backflow 对象。
- `row_index::Int`: 需要展开的 spin-resolved 全局行号。
- `get_state::Function`: `(site_index) -> Int8`, 用于读取当前或 proposal 后的 site 状态。

返回:
- `Int`: 有效 source row 数量。

公式:
- `U_b(row,:) = sum_s w_s * U_0(s,:)`。
- 初始恒等项为 `w_row = 1`。
- eta 项对 target row 增加 `t_ij * eta_k * f_k(x)`。
- epsilon 项只在对应 group 存在非零 eta contribution 时对本 row 增加 `epsilon_bf - 1`。
"""
function fill_backflow_row_source_weights_from_state_getter!(
    source_row_indices::AbstractVector{Int},
    source_row_weights::AbstractVector{T},
    state_vector::Vector{Int8},
    backflow_term::CompositeBackflowTerm,
    row_index::Int,
    get_state::Function,
)::Int where {T}
    site_index, row_offset, source_count = initialize_backflow_chain_rule_source_weights!(
        source_row_indices,
        source_row_weights,
        state_vector,
        row_index,
    )
    state_i = get_state(site_index)
    spin = backflow_spin_from_row_offset(row_offset)

    if backflow_n_sigma(state_i, spin) == 0.0
        return source_count
    end

    active_group_names = Set{Symbol}()
    for source_group in backflow_term.source_groups
        source_count = add_source_group_chain_rule_source_weights_and_track!(
            source_row_indices,
            source_row_weights,
            source_count,
            state_i,
            get_state,
            source_group,
            site_index,
            row_offset,
            active_group_names,
        )
    end

    for epsilon_term in backflow_term.epsilon_terms
        epsilon_shift = T(epsilon_term.epsilon_bf - 1.0)
        if epsilon_shift == zero(T)
            continue
        end
        for group_name in epsilon_term.group_names
            if group_name in active_group_names
                source_count = add_backflow_chain_rule_source_weight!(
                    source_row_indices,
                    source_row_weights,
                    source_count,
                    row_index,
                    epsilon_shift,
                )
                break
            end
        end
    end

    return source_count
end

"""
用途: 在无 backflow 情况下将指定 output row 展开为单个同名 source row。

参数:
- `source_row_indices::AbstractVector{Int}`: 输出 source row 编号 buffer。
- `source_row_weights::AbstractVector{T}`: 输出 source row 权重 buffer。
- `state_vector::Vector{Int8}`: 当前 Monte Carlo 构型。
- `backflow_term::NoBackflowTerm`: 空 backflow 对象。
- `row_index::Int`: 需要展开的 spin-resolved 全局行号。

返回:
- `Int`: 有效 source row 数量, 固定为 `1`。
"""
function fill_backflow_chain_rule_source_weights!(
    source_row_indices::AbstractVector{Int},
    source_row_weights::AbstractVector{T},
    state_vector::Vector{Int8},
    ::NoBackflowTerm,
    row_index::Int,
)::Int where {T}
    _, _, source_count = initialize_backflow_chain_rule_source_weights!(
        source_row_indices,
        source_row_weights,
        state_vector,
        row_index,
    )
    return source_count
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
    return Pair{Symbol,Matrix{T}}[]
end


"""
用途: 统一返回组合式 Eq.(5) backflow 参数顺序对应的轨道导数矩阵列表。

数学公式:
- `partial U_b / partial p_m = partial delta U_m / partial p_m`,
  其中 `p_m` 是对应 correction term 的唯一参数。
- epsilon 参数导数 `partial U_b / partial epsilon_bf = U_0(row)` 只在
  对应 source group 中有非零 eta contribution 时填入。

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
    n_sites = length(state_vector)

    # Build a mapping from group_name -> set of epsilon term indices for efficient lookup.
    group_to_epsilon_indices = Dict{Symbol,Vector{Int}}()
    for (epsilon_index, epsilon_term) in enumerate(backflow_term.epsilon_terms)
        for group_name in epsilon_term.group_names
            if !haskey(group_to_epsilon_indices, group_name)
                group_to_epsilon_indices[group_name] = Int[]
            end
            push!(group_to_epsilon_indices[group_name], epsilon_index)
        end
    end

    # Pre-allocate eta derivative matrices and epsilon derivative matrix tracker.
    # We iterate site-by-site, row-by-row to find which rows have nonzero eta.
    # For epsilon derivatives, we first compute the full eta scan, then fill.
    epsilon_derivative = zeros(T, size(base_orbitals))

    # Track which epsilon rows are active (binary per epsilon term per row).
    # Use a bitset approach: per-epsilon-term boolean row mask.
    n_epsilon = length(backflow_term.epsilon_terms)
    n_rows = size(base_orbitals, 1)
    epsilon_active_rows = [falses(n_rows) for _ in 1:n_epsilon]

    eta1_deriv_array = Matrix{T}[]
    eta2_deriv_array = Matrix{T}[]
    eta3_deriv_array = Matrix{T}[]
    eta4_deriv_array = Matrix{T}[]
    eta1_param_names = Symbol[]
    eta2_param_names = Symbol[]
    eta3_param_names = Symbol[]
    eta4_param_names = Symbol[]

    for source_group in backflow_term.source_groups
        eta1_deriv = zeros(T, size(base_orbitals))
        eta2_deriv = zeros(T, size(base_orbitals))
        eta3_deriv = zeros(T, size(base_orbitals))
        eta4_deriv = zeros(T, size(base_orbitals))
        eta1_value = T(source_group.eta1_term.eta1_bf)
        eta2_value = T(source_group.eta2_term.eta2_bf)
        eta3_value = T(source_group.eta3_term.eta3_bf)
        eta4_value = T(source_group.eta4_term.eta4_bf)

        for (bond_index, (site_i, site_j)) in enumerate(source_group.source_bonds)
            state_i = state_vector[site_i]
            state_j = state_vector[site_j]
            bond_amplitude = T(source_group.source_amplitudes[bond_index])

            for row_offset in 1:2
                spin = backflow_spin_from_row_offset(row_offset)
                row_i = 2 * (site_i - 1) + row_offset
                row_j = 2 * (site_j - 1) + row_offset

                eta_contribution = compute_backflow_eta_contribution(
                    state_i,
                    state_j,
                    spin,
                    bond_amplitude,
                    eta1_value,
                    eta2_value,
                    eta3_value,
                    eta4_value,
                )

                if eta_contribution.eta1_factor != zero(T)
                    @views eta1_deriv[row_i, :] .+= bond_amplitude .* base_orbitals[row_j, :]
                end

                if eta_contribution.eta2_factor != zero(T)
                    @views eta2_deriv[row_i, :] .+= bond_amplitude * eta_contribution.eta2_factor .* base_orbitals[row_j, :]
                end

                if eta_contribution.eta3_factor != zero(T)
                    @views eta3_deriv[row_i, :] .+= bond_amplitude * eta_contribution.eta3_factor .* base_orbitals[row_j, :]
                end

                if eta_contribution.eta4_factor != zero(T)
                    @views eta4_deriv[row_i, :] .+= bond_amplitude * eta_contribution.eta4_factor .* base_orbitals[row_j, :]
                end

                # Mark epsilon terms owned by this group as active for this row.
                if eta_contribution.coefficient != zero(T) && haskey(group_to_epsilon_indices, source_group.group_name)
                    for epsilon_index in group_to_epsilon_indices[source_group.group_name]
                        epsilon_active_rows[epsilon_index][row_i] = true
                    end
                end
            end
        end

        push!(eta1_deriv_array, eta1_deriv)
        push!(eta2_deriv_array, eta2_deriv)
        push!(eta3_deriv_array, eta3_deriv)
        push!(eta4_deriv_array, eta4_deriv)
        push!(eta1_param_names, source_group.eta1_term.param_name)
        push!(eta2_param_names, source_group.eta2_term.param_name)
        push!(eta3_param_names, source_group.eta3_term.param_name)
        push!(eta4_param_names, source_group.eta4_term.param_name)
    end

    # Build the full derivative pairs list in term order:
    # epsilon terms first, then eta1/eta2/eta3/eta4 for each group.
    derivative_pairs = Pair{Symbol,Matrix{T}}[]

    for (epsilon_index, epsilon_term) in enumerate(backflow_term.epsilon_terms)
        epsilon_deriv = zeros(T, size(base_orbitals))
        active_mask = epsilon_active_rows[epsilon_index]
        for row_i in eachindex(active_mask)
            if active_mask[row_i]
                copyto!(@view(epsilon_deriv[row_i, :]), @view(base_orbitals[row_i, :]))
            end
        end
        push!(derivative_pairs, epsilon_term.param_name => epsilon_deriv)
    end

    for sg_idx in eachindex(backflow_term.source_groups)
        push!(derivative_pairs, eta1_param_names[sg_idx] => eta1_deriv_array[sg_idx])
        push!(derivative_pairs, eta2_param_names[sg_idx] => eta2_deriv_array[sg_idx])
        push!(derivative_pairs, eta3_param_names[sg_idx] => eta3_deriv_array[sg_idx])
        push!(derivative_pairs, eta4_param_names[sg_idx] => eta4_deriv_array[sg_idx])
    end

    return derivative_pairs
end


end # module
