# Emery group-based backflow 设计方案

## 背景

当前 `CompositeBackflowTerm` 的主要构造方式是传入 flat `terms::Vector{AbstractBackflowCorrectionTerm}`。对于 Emery directed split backflow, `Emery.jl` 先构造 `epsilon_d`, `epsilon_p`, 以及 `dd/dp/pd/pp` 四组下的 `eta1/eta2/eta3/eta4` 共 16 个 eta term, 然后 `Backflow.jl` 再通过 `build_directed_split_backflow_groups` 按参数名后缀反向拼回 directed group。

这个流程存在几个问题:

- Emery 的 source group 信息本来已经在 `Emery.jl` 中显式存在, 但当前实现先拆散再反推, 数据流不直观。
- `build_directed_split_backflow_groups` 固定假设 `dd/dp/pd/pp` 和 `eta1..eta4`, 但它被放在通用 `CompositeBackflowTerm` 构造器内, 容易把 Emery-specific 规则泄漏到共享抽象中。
- `eta1..eta4` term 各自重复保存同一组 `source_bonds/source_amplitudes/source graph cache`, 实际 ownership 不清晰。

## 目标

本轮目标是把 Emery directed split backflow 的核心表示改成 group-based:

- `CompositeBackflowTerm` 以 source group 为核心组织 eta correction terms。
- 每个 source group 负责保存共享的 `source_bonds`, `source_amplitudes`, `source_data_signature`, `outgoing_bond_indices_by_source`, `incoming_source_sites_by_target`。
- `BackflowEta1DoublonHoleTerm`, `BackflowEta2SpinExchangeTerm`, `BackflowEta3DoublonSingleTerm`, `BackflowEta4SingleHoleTerm` 不再保存 source 数据, 只保存参数名和参数值。
- `BackflowEpsilonTerm` 暂时保持当前结构, 仍然自己保存 source 数据和 source graph cache。
- `Emery.jl` 构造 backflow 时直接构造 `dd/dp/pd/pp` 四个 source group, 不再依赖参数名后缀反推 group。

## 非目标

- 本轮暂不迁移 `Hubbard.jl`, `Hubbard_bf.jl`, `Hubbard_restricted.jl` 的普通 Hubbard backflow。
- 本轮不保证普通 Hubbard 的 legacy `CompositeBackflowTerm([terms...])` 路径继续可用。
- 本轮不改变 Emery 参数顺序和命令行参数名称。
- 本轮不把 `epsilon` source 数据迁移到 group。

## 核心结构

新增或重命名一个固定字段式 source group, 建议名称为 `DirectedBackflowSourceGroup`:

```julia
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
```

eta term 变成轻量参数对象:

```julia
mutable struct BackflowEta1DoublonHoleTerm <: AbstractBackflowCorrectionTerm
    param_name::Symbol
    eta1_bf::Float64
end
```

`eta2/eta3/eta4` 同理, 分别只保存 `param_name` 和对应参数值。

`CompositeBackflowTerm` 建议调整为:

```julia
struct CompositeBackflowTerm <: AbstractBackflowTerm
    epsilon_terms::Vector{BackflowEpsilonTerm}
    source_groups::Vector{DirectedBackflowSourceGroup}
    terms::Vector{AbstractBackflowCorrectionTerm}
    incoming_source_sites_by_target::Vector{Vector{Int}}
end
```

其中 `terms` 是从 `epsilon_terms` 和 `source_groups` 展开的派生列表, 用来兼容现有参数接口:

```text
epsilon_terms,
group1 eta1, group1 eta2, group1 eta3, group1 eta4,
group2 eta1, group2 eta2, group2 eta3, group2 eta4,
...
```

Emery 参数顺序保持为:

```text
bf_epsilon_d, bf_epsilon_p,
bf_eta1_dd, bf_eta2_dd, bf_eta3_dd, bf_eta4_dd,
bf_eta1_dp, bf_eta2_dp, bf_eta3_dp, bf_eta4_dp,
bf_eta1_pd, bf_eta2_pd, bf_eta3_pd, bf_eta4_pd,
bf_eta1_pp, bf_eta2_pp, bf_eta3_pp, bf_eta4_pp
```

## Emery 构造流程

`Emery.jl` 中 `build_column_directed_emery_backflow` 保留现有入参和外部行为, 内部改为:

1. 构造 `epsilon_d` 和 `epsilon_p`, 暂时沿用当前 `BackflowEpsilonTerm`。
2. 用 `dd_source_bonds/dd_source_amplitudes` 构造 `:dd` group。
3. 用 `dp_source_bonds/dp_source_amplitudes` 构造 `:dp` group。
4. 用 `pd_source_bonds/pd_source_amplitudes` 构造 `:pd` group。
5. 用 `pp_source_bonds/pp_source_amplitudes` 构造 `:pp` group。
6. 调用新的 group-based `CompositeBackflowTerm(epsilon_terms, source_groups)` 构造器。

这样 Emery 不再需要 `build_directed_split_backflow_groups`, `collect_directed_split_backflow_group`, `find_composite_correction_term_by_suffix`, `backflow_param_name_has_suffix`。

## 计算路径

`epsilon` 计算路径保持基于 `BackflowEpsilonTerm`。

eta 计算路径改成基于 `DirectedBackflowSourceGroup`:

- 遍历 `backflow_term.source_groups`。
- 从 group 读取 source graph 和 bond amplitude。
- 从 `group.eta1_term` 到 `group.eta4_term` 读取参数值。
- 在同一次 source bond loop 中累加 `eta1/eta2/eta3/eta4` contribution。

需要迁移的主要函数:

- `build_backflow_orbitals`
- `fill_backflow_site_row_after_proposal!`
- `fill_backflow_site_block_after_proposal!`
- `fill_backflow_chain_rule_orbitals!`
- `fill_backflow_chain_rule_row!`
- `fill_backflow_chain_rule_source_weights!`
- `build_backflow_derivative_orbitals`

proposal 后的 row/block 路径已经接近 group-based, 主要改为读取 `group.source_bonds` 和 `group.outgoing_bond_indices_by_source`。full build, chain rule, parameter derivative 需要新增 group-based helper 或改写现有 helper。

## 修改文件

预计修改:

- `src/Backflow.jl`
  - 新增 `DirectedBackflowSourceGroup`。
  - 调整 `BackflowEta1DoublonHoleTerm`, `BackflowEta2SpinExchangeTerm`, `BackflowEta3DoublonSingleTerm`, `BackflowEta4SingleHoleTerm` 字段。
  - 新增 group-based constructor/helper。
  - 调整 `CompositeBackflowTerm` 字段和构造器。
  - 将 eta 相关 full build, proposal update, chain rule, derivative 路径迁移到 group-based source 数据。

- `Emery.jl`
  - 调整 `build_column_directed_emery_backflow` 内部构造流程。
  - 保持外部参数顺序和命令行参数名称不变。

预计新增或调整测试:

- `test/backflow_composite_terms_test.jl`
- `test/tmp_emery_directed_backflow_gradient_correctness.jl`
- `test/tmp_emery_backflow_energy_correctness.jl`
- 必要时新增一个 Emery group constructor 小测试。

## 依赖顺序

1. 先新增 group 类型和 group 构造 helper。
2. 调整 eta term 字段和轻量 constructor。
3. 调整 `CompositeBackflowTerm(epsilon_terms, source_groups)` 构造器和 `terms` 展开逻辑。
4. 修改 Emery 构造函数, 让 Emery 使用 group-based constructor。
5. 迁移 proposal row/block 更新路径。
6. 迁移 full `build_backflow_orbitals` 路径。
7. 迁移 chain rule 路径。
8. 迁移 backflow 参数导数路径。
9. 跑 Emery 相关测试, 再清理不再使用的 suffix 反推函数。

这个顺序可以支持渐进开发: 每一步只处理一个依赖层, 避免同时改数据结构和所有数学路径导致定位困难。

## 测试要点

自动化测试重点:

- Emery group-based constructor 的参数顺序保持不变。
- `backflow_param_names(backflow)` 与当前 Emery 顺序一致。
- `backflow_param_values(backflow)` 与输入参数一致。
- `update_backflow_params!` 能正确更新 epsilon 和四组 eta 参数。
- group 中 `source_bonds/source_amplitudes` 缓存与输入一致。
- `build_backflow_orbitals` 与旧实现的数值结果一致。
- proposal 后 row/block 局域重算与 full rebuild 一致。
- chain rule 结果与 finite difference 或旧测试基准一致。
- backflow 参数导数测试继续通过。

建议执行:

```powershell
julia test\backflow_composite_terms_test.jl
julia test\tmp_emery_backflow_energy_correctness.jl
julia test\tmp_emery_directed_backflow_gradient_correctness.jl
julia test\backflow_chain_rule_selected_rows_test.jl
```

人工检查:

- `Emery.jl` 中 `build_column_directed_emery_backflow` 不再构造 16 个带 source 数据的 eta term。
- `src/Backflow.jl` 中 eta term 不再拥有 `source_bonds/source_amplitudes` 字段。
- Emery 参数顺序和命令行参数名称没有变化。

## 风险与限制

- 普通 Hubbard 路径暂不纳入本轮目标, 可能因为 eta term 字段变化或 `CompositeBackflowTerm` 构造器变化而需要后续迁移。
- `epsilon` 暂时保留旧 source ownership, 因此短期内会存在两种 source ownership: epsilon 自持 source, eta group 持 source。
- 如果某些共享 helper 仍默认从 eta term 读取 `source_bonds`, 编译或测试会暴露遗漏迁移点。
- 删除旧 suffix 反推函数应放在 Emery 测试通过之后, 避免调试阶段缺少回退参考。

## 自检

- 已满足“只针对 Emery 验收”的边界, 并明确普通 Hubbard 为非目标。
- 已保留 `epsilon` 当前结构, 没有把 epsilon 迁移到 group。
- 已保持 Emery 参数顺序不变。
- 已按技术依赖顺序拆解实现步骤。
- 已列出修改文件和测试要点。
