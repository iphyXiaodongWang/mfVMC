# Twist Hubbard PH Backflow Design

## 背景

本阶段目标是在 `twist_Hubbard_PH.jl` 中启用物理正确的 PH backflow。前一阶段已经完成 PH/no-backflow determinant, 并用 `bf_epsilon=1, bf_eta*=0` 等价路径作为退化基准。当前阶段只处理 PH backflow 的设计与接入, 不改变 nonPH twist backflow 的默认行为。

参考 `reference/PH_backflow_VMC_notes.md` 的核心结论:

- PH determinant 的 upper block 是 up electron orbital。
- PH determinant 的 lower block 是 down-hole orbital, 不是 down-electron orbital。
- determinant occupied row set 为 `X_up ∪ {j+L | j notin X_down}`。
- backflow 中的 `D_i`, `H_i`, `n_{iσ}`, `h_{iσ}` 必须从原始电子构型定义。
- lower block 的虚跳方向相对 down electron 反向。

## 已确认决策

采用一个显式 flag 表示 backflow 使用 PH 或 nonPH 基。推荐名字为:

```julia
particle_hole_lower_block::Bool
```

该 flag 放在 `CompositeBackflowTerm` 中:

- `false`: 默认 nonPH 行为, lower row 表示 down electron。
- `true`: PH 行为, lower row 表示 down hole。

这个选择的目的是在同一套 backflow 数据结构里共享参数管理、source graph、local rank-k update 和 chain rule 路径, 同时避免把 PH lower block 错当成 down electron。

## 数学定义

令 `row_offset = 1` 表示 upper row, `row_offset = 2` 表示 lower row。

### nonPH 或 PH upper block

对 row `(i, up)` 使用原有电子表象 Eq.(5):

```text
eta1: D_i H_j
eta2: n_iσ h_i,-σ n_j,-σ h_jσ
eta3: D_i n_j,-σ h_jσ
eta4: n_iσ h_i,-σ H_j
```

其中 source row 为 `(j, same row_offset)`。

### PH lower block

对 row `(i, down-hole)` 使用 hole 方向的虚跳。等价实现方式是:

```text
使用 same lower row source: (j, down-hole)
但 eta 因子用 swapped sites 计算: contribution(state_j, state_i, DN)
```

因此最简单 eta1 变为:

```text
nonPH/down electron: D_i H_j
PH/down hole:        H_i D_j
```

eta2/eta3/eta4 也按同一规则从 `reference/PH_backflow_VMC_notes.md` 的 lower block 公式得到。

PH lower row 的 occupation guard 不能继续用 `n_i_down`。因为 lower row 表示 down hole, 它在当前电子构型中被占据的条件是:

```text
h_i_down = 1 - n_i_down
```

所以:

- nonPH row active: `n_i_sigma == 1`
- PH lower row active: `h_i_down == 1`

## 修改范围

### `src/Backflow.jl`

需要修改:

- `CompositeBackflowTerm` 增加 `particle_hole_lower_block::Bool` 字段, 默认 `false`。
- 构造函数增加关键字参数, 保持已有调用不变。
- source-weight 路径在 row-level 判断是否为 PH lower row。
- PH lower row 使用 down-hole occupation guard。
- PH lower row 计算 eta contribution 时交换 `state_i` 与 `state_j`。
- `build_backflow_derivative_orbitals` 使用相同 PH/nonPH 逻辑, 保证 backflow 参数 SR 导数一致。
- 相关 docstring 用中文补充 PH flag 的含义和公式。

不需要修改:

- `NoBackflowTerm`。
- sampler 的 PH row set。
- `vmc_det.jl` 的 rank-k update 公式。它已经通过 occupied rows 和 source weights 工作。

### `twist_Hubbard.jl`

需要最小修改:

- `build_twist_composite_backflow` 增加可选关键字参数 `particle_hole_lower_block=false`, 传给 `CompositeBackflowTerm`。
- nonPH 主程序保持默认 `false`。

### `twist_Hubbard_PH.jl`

需要接入:

- 命令行加入 `--enable_backflow`, `--bf_epsilon`, `--bf_eta1`, `--bf_eta2`, `--bf_eta3`, `--bf_eta4`。
- 构造 `source_bonds, source_amplitudes` 时复用 `build_twist_backflow_source_data`。
- 构造 backflow 时传入 `particle_hole_lower_block=true`。
- 参数拼接顺序扩展为 `wf, projector, backflow`。
- `update_twist_ph_ansatz!` 增加 backflow 参数更新逻辑。
- active/fixed 参数筛选加入 backflow 参数集合。

## 测试策略

### 单元测试

新增 `test/twist_hubbard_ph_backflow_test.jl`:

- PH eta1 upper row: `D_i H_j` 非零时混入 upper source row `(j, up)`。
- PH eta1 lower row: `H_i D_j` 非零时混入 lower source row `(j, down-hole)`。
- PH lower row guard 使用 `h_i_down`, 不使用 `n_i_down`。
- PH eta2/eta3/eta4 lower row 与 swapped-sites 规则一致。
- `bf_epsilon=1, bf_eta*=0` 时 source weights 退化为单个 identity source row。
- PH flag 为 `false` 时现有 nonPH source weights 不变。

### 回归测试

继续运行:

```text
julia test\twist_hubbard_ph_no_backflow_test.jl
julia test\twist_hubbard_ph_nonph_benchmark_test.jl
julia test\emery_grouped_backflow_cleanup_test.jl
julia test\backflow_composite_terms_test.jl
```

### PH backflow smoke test

新增或扩展小尺寸测试:

- `twist_Hubbard_PH.jl --enable_backflow true --bf_epsilon 1.0 --bf_eta1 0.0 ...` 能跑通, 且退化到 no-backflow。
- 非零 `bf_eta1` 时 fast local ratio 与 rebuild ratio 一致。
- SR gradient 的 backflow 参数导数至少对一个小构型通过 finite difference sanity check。

## Benchmark 边界

必须作为硬断言:

- zero-backflow PH 路径与 no-backflow PH 路径一致。
- PH backflow fast ratio 与 rebuild ratio 一致。
- PH backflow 参数导数与 finite difference 一致。
- nonPH backflow 现有测试不回归。

不建议作为硬断言:

- 非零 backflow 后 PH driver 与 nonPH driver 的能量逐点完全相等。

原因是 PH driver 的 determinant lower block 是 down-hole complement 表示, nonPH driver 是 down-electron Slater 表示。非零、构型依赖的 backflow 会改变 ansatz 的具体表示, 二者不应在当前阶段被强行要求数值完全相同。可以把 nonzero backflow PH/nonPH 能量比较作为 sanity benchmark, 但不能作为 correctness criterion。

## 风险与缓解

- 风险: PH flag 泄漏导致 nonPH 行为改变。
  缓解: 默认值为 `false`, 并跑现有 nonPH backflow 测试。
- 风险: 只改 source-weight 路径, 忘记 backflow 参数导数路径。
  缓解: `build_backflow_derivative_orbitals` 使用同一 PH contribution helper, 并加 finite difference 测试。
- 风险: local rank-k affected sites 漏掉 swapped lower block 依赖。
  缓解: source graph 仍然以 `(i,j)` source-target 记录, PH lower row 仍依赖 row `i` 的 outgoing bonds 和 target `j` 的 state, 所以 affected-site 收集逻辑仍覆盖 changed target 对 source row 的影响。用 fast-vs-rebuild 测试确认。
