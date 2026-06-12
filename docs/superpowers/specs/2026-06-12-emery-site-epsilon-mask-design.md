# Emery site-based epsilon mask 设计方案

## 背景

上一轮 Emery directed split backflow 已经把 `eta1/eta2/eta3/eta4` 的 source 数据迁移到 `DirectedBackflowSourceGroup` 中, 但 `BackflowEpsilonTerm` 仍然保留旧结构:

- 自己保存 `source_bonds/source_amplitudes`。
- 使用 `epsilon_mask_terms = [:eta1, :eta2, :eta3_doublon_single, :eta4]` 判断 `epsilon` 是否激活。
- 通过逐 bond 检查 eta channel 是否非零来构造 `xi_{i,sigma}`。

这个结构在新的 group-based backflow 中有两个问题:

- `epsilon` 本质上是 backflow 后本轨道 `U_0(i,sigma)` 的剩余 prefactor, 不是一个带 hopping amplitude 的 bond contribution。
- 当前 mask 复用了 eta channel 名称, 但实际语义可以简化成 site-row 条件, 继续保留 `epsilon_mask_terms` 会让代码难读且容易误解。

## 目标

本轮目标是只针对 Emery backflow 路径, 将 `epsilon` 逻辑改成 site-based:

- 保留现有两个参数 `bf_epsilon_d` 和 `bf_epsilon_p`。
- `bf_epsilon_d` 仍作用在 source 为 Cu/d 的 site, 邻接关系由 `dd + dp` 构造。
- `bf_epsilon_p` 仍作用在 source 为 O/p 的 site, 邻接关系由 `pd + pp` 构造。
- `BackflowEpsilonTerm` 内部不再保存 `source_bonds/source_amplitudes`。
- `BackflowEpsilonTerm` 内部保存 site 与 target neighbor 的邻接表。
- 删除 `epsilon_mask_terms` 逻辑, 用统一的 site-row 激活条件判断 `xi_{i,sigma}`。

新的激活条件为:

```julia
xi_{i,sigma} = 1
```

当且仅当:

```julia
n_{i,sigma} = 1
```

并且存在至少一个 target neighbor `j`, 满足:

```julia
h_{j,sigma} = 1
```

也就是:

```julia
site i 上有 sigma 电子, 且存在一个 target neighbor j 上没有同自旋 sigma 电子。
```

## 非目标

- 本轮不改变 Emery 的 backflow 参数数量。
- 本轮不把 `bf_epsilon_d/bf_epsilon_p` 拆成 `dd/dp/pd/pp` 四个 group-level epsilon 参数。
- 本轮不改变 `eta1/eta2/eta3/eta4` 的公式和 group-based 计算路径。
- 本轮不迁移 Hubbard backflow。
- 本轮不引入新的物理参数或命令行参数。

## 公式等价关系

当前 `epsilon_mask_terms` 默认打开四个 eta channel:

```julia
eta1: D_i H_j
eta2: n_{i,sigma} h_{i,-sigma} n_{j,-sigma} h_{j,sigma}
eta3: D_i n_{j,-sigma} h_{j,sigma}
eta4: n_{i,sigma} h_{i,-sigma} H_j
```

这四种情况的并集等价于:

```julia
n_{i,sigma} h_{j,sigma}
```

原因是:

- `n_{i,sigma} = 1` 时, source site `i` 只能是 single-`sigma` 或 doublon。
- `h_{j,sigma} = 1` 时, target site `j` 只能是 hole 或 single-opposite-spin。
- 这四种 source/target 组合刚好对应 `eta4`, `eta2`, `eta1`, `eta3`。

因此新的 `epsilon` mask 不再需要逐个 eta channel 判断, 只需要检查 `n_{i,sigma}` 和 target neighbor 上的 `h_{j,sigma}`。

## 核心结构

`BackflowEpsilonTerm` 建议改成:

```julia
mutable struct BackflowEpsilonTerm <: AbstractBackflowCorrectionTerm
    param_name::Symbol
    epsilon_bf::Float64
    source_sites::Vector{Int}
    target_neighbors_by_source_site::Vector{Vector{Int}}
    source_sites_by_target_neighbor::Vector{Vector{Int}}
    neighbor_data_signature::UInt
end
```

字段含义:

- `source_sites`: 使用这个 epsilon term 的 source site 列表。
- `target_neighbors_by_source_site[i]`: source site `i` 对应的 target neighbor 列表。
- `source_sites_by_target_neighbor[j]`: 当 target site `j` 状态变化时, 哪些 source site 的 epsilon mask 可能受影响。
- `neighbor_data_signature`: 邻接数据签名, 用于检测构造后被手动修改的情况。

`target_neighbors_by_source_site` 使用 dense adjacency 表示:

```julia
target_neighbors_by_source_site[i] == Int[]
```

表示 site `i` 不属于这个 epsilon term, 或者属于该 term 但没有 target neighbor。实际构造中 `source_sites` 应等价于所有 `target_neighbors_by_source_site[i]` 非空的 site。

## 构造方式

为了保持 Emery 现有数据流最小改动, `BackflowEpsilonTerm` 构造函数可以继续接收 `source_bonds` 作为输入:

```julia
BackflowEpsilonTerm(
    param_name=:bf_epsilon_d,
    epsilon_bf=bf_epsilon_d,
    source_bonds=d_source_bonds,
)
```

构造函数内部将有向 bond list 转成 site-neighbor adjacency:

```julia
source_bonds -> target_neighbors_by_source_site
source_bonds -> source_sites_by_target_neighbor
source_bonds -> source_sites
```

但构造后的 `BackflowEpsilonTerm` 不再保存 `source_bonds`, 也不再保存 `source_amplitudes`。

`Emery.jl` 中:

- `d_source_bonds = vcat(dd_source_bonds, dp_source_bonds)` 保留。
- `p_source_bonds = vcat(pd_source_bonds, pp_source_bonds)` 保留。
- 不再为 epsilon 构造或传递 `d_source_amplitudes/p_source_amplitudes`。
- 不再传递 `epsilon_mask_terms`。

## 计算路径

新增或替换一个 site-row 判断函数:

```julia
is_backflow_epsilon_site_row_active(state_i, state_j, spin)
```

逻辑为:

```julia
backflow_n_sigma(state_i, spin) != 0.0 &&
backflow_h_sigma(state_j, spin) != 0.0
```

对固定 `(site_i, sigma)`, `xi_{i,sigma}` 的计算方式为:

```julia
xi_{i,sigma} = any(
    is_backflow_epsilon_site_row_active(state_i, state_j, spin)
    for site_j in target_neighbors_by_source_site[site_i]
)
```

所有现有 epsilon 路径都使用这一个判断:

- full rebuild: `compute_backflow_epsilon_row_mask`
- proposal site row: `add_backflow_correction_site_row_after_proposal!`
- proposal site block: `add_backflow_correction_site_block_after_proposal!`
- grouped proposal block 中的 epsilon 部分
- chain-rule row
- epsilon 参数 derivative

## Composite affected site 图

`CompositeBackflowTerm.incoming_source_sites_by_target` 当前会合并 epsilon term 和 source group 的 incoming graph。迁移后:

- eta group 继续使用 `incoming_source_sites_by_target`。
- epsilon term 改为贡献 `source_sites_by_target_neighbor`。

语义不变: 当某个 changed site 作为 target neighbor 发生状态变化时, 需要把相关 source site 加入 affected site 集合。

## 修改顺序

1. 新增 epsilon site-neighbor cache 构造函数。
2. 修改 `BackflowEpsilonTerm` 字段和 constructor, 让其从 `source_bonds` 构造 site-neighbor adjacency。
3. 新增 `is_backflow_epsilon_site_row_active`。
4. 替换所有 `is_backflow_epsilon_row_active(..., epsilon_mask_terms)` 调用。
5. 删除 `BACKFLOW_EPSILON_MASK_TERMS`, `normalize_backflow_epsilon_mask_terms`, `epsilon_mask_terms` 字段和相关文档。
6. 修改 `validate_backflow_correction_source_data!` 为 epsilon neighbor 数据校验。
7. 修改 `CompositeBackflowTerm` 合并 affected site 图时使用 epsilon 的 `source_sites_by_target_neighbor`。
8. 修改 `Emery.jl` 中 epsilon 构造参数。
9. 扩展测试并运行 Emery 相关验证。

## 测试要点

自动化测试需要覆盖:

- `BackflowEpsilonTerm` 不再保存 `source_bonds/source_amplitudes`。
- 从重复 `source_bonds` 构造出的 `target_neighbors_by_source_site` 去重并排序。
- `source_sites` 与非空 `target_neighbors_by_source_site` 一致。
- `source_sites_by_target_neighbor` 能正确反查受 target 变化影响的 source site。
- 旧四个 eta channel 对应的组合都会激活 epsilon:
  - doublon -> hole
  - single-`sigma` -> hole
  - doublon -> single-opposite-spin
  - single-`sigma` -> single-opposite-spin
- source site 没有 `sigma` 电子时不激活。
- target neighbor 有同自旋 `sigma` 电子时不激活。
- full rebuild, proposal row, proposal block, chain-rule, epsilon derivative 的结果一致。
- `julia test\emery_grouped_backflow_cleanup_test.jl` 通过。

## 已知边界

这个设计仍然保留 `bf_epsilon_d/bf_epsilon_p` 两个参数, 因此 epsilon 的参数粒度仍是 source orbital type, 不是 `dd/dp/pd/pp` directed group。这样可以把本轮改动限制在 mask 语义和内部表示上, 避免同时改变参数化。
