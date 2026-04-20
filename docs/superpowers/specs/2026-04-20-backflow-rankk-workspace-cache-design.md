# Backflow Rank-k Workspace Cache Design

**背景**

当前 `Eq4BackflowTerm` 的 determinant local update 已经把受影响站点收缩到了较小范围, 但 `measure` 主链路里仍然有两类重复工作:

1. `calc_backflow_ratio_local_update(...)` 在 ratio 阶段已经收集了一次 `changed_electron_ids / new_columns`, 但 `accept_backflow_local_update!(...)` 在 accept 阶段又会对同一个 proposal 再收集一次。
2. 通用 `rank-k` 路径在 ratio 阶段构造过一次 `c_matrix / small_k_matrix`, 但 accept 阶段又会重复构造。

`vmc_pfa` 已经有类似优化经验: ratio 阶段把中间量保存在 workspace 中, accept 阶段直接继续消费这些中间量。因此本设计把 `vmc_pfa` 的这一策略引入到 `vmc_det` 的 backflow `rank-k` 路径。

---

## 目标

在不改变 public API、不改变物理结果、不触碰 `SR` 梯度路径的前提下, 为 `measure` 主链路中的 backflow determinant fast path 增加“当前 proposal 生命周期内”的 workspace cache, 以减少重复收集和重复矩阵构造。

本轮只覆盖:

- `ConfigurationPH + Eq4BackflowTerm + HubbardKernel(conserve_sz=true)` 的 determinant backflow local update
- `calc_backflow_ratio_local_update(...) -> accept_backflow_local_update!(...)` 这条链路

本轮明确不做:

- `compute_grad_log_psi!` / backflow derivative 的优化
- `k <= 4` specialized fast path
- `refresh_backflow_orbitals!` 的局域刷新改造
- 新 public 参数、新 CLI、新用户可见配置项

---

## 设计原则

### 1. 只缓存“当前 proposal”的临时中间量

workspace cache 只对最近一次 `calc_backflow_ratio_local_update(...)` 产生的数据有效。

使用契约为:

1. 先调用 `calc_backflow_ratio_local_update(vwf, proposal)`
2. 如果 proposal 被接受, 立刻调用 `accept_backflow_local_update!(vwf, proposal, ratio)`

这与当前 `mcmc_step!` 的调用顺序一致, 也与 `vmc_pfa` 的 workspace 复用风格一致。

### 2. 不做 proposal 内容匹配检查

本设计不比较 `proposal.site1/site2/...` 等字段来确认 cache 是否属于“当前 proposal”。

原因:

- 当前主链路保证 `accept` 紧跟在 `ratio` 之后
- 额外的 proposal 匹配检查会增加复杂度和额外状态
- `vmc_pfa` 的现有策略也是依赖调用顺序契约

### 3. 保留轻量 cache valid 标记

虽然不做 proposal 匹配检查, 但仍保留一个 workspace 内部的 `has_cached_rankk_update` 标记, 用于防止误用:

- 如果直接调用 `accept_backflow_local_update!(...)`
- 但此前没有先调用 `calc_backflow_ratio_local_update(...)`

则应 fail fast, 抛出明确错误, 而不是静默读取未初始化缓存。

### 4. 最小改动原则

本轮不改变 `calc_ratio(...)`、`accept_move!(...)` 的 public 接口形状, 也不扩散到无 backflow 分支。所有新增状态都局限在 `vwf_det.ws` 内部。

---

## 架构修改

### 一. 扩展 `R1R2WS`

文件: `src/vmc_det.jl`

当前 `R1R2WS` 只覆盖 rank-1 / rank-2 路径的向量 buffer 和梯度 buffer。需要在此基础上增加一组 backflow rank-k cache 字段。

建议新增的字段类别:

- cache 有效标记:
  - `has_cached_rankk_update::Bool`

- proposal 收集结果:
  - `cached_changed_count::Int`
  - `cached_changed_electron_ids::Vector{Int}`
  - `cached_changed_row_indices::Vector{Int}`
  - `cached_new_columns::Matrix{T}`

- rank-k 因子:
  - `cached_c_matrix::Matrix{T}`
  - `cached_small_k_matrix::Matrix{T}`
  - `cached_small_k_inverse::Matrix{T}`

- 可能的辅助 buffer:
  - `cached_basis_columns::Matrix{T}`
  - `cached_s_matrix::Matrix{T}`

说明:

- 上面缓存的都是 `N x k` 或 `k x k` 级别的对象, 不缓存 `N x N` 的中间结果。
- `cached_small_k_inverse` 是否在 ratio 阶段就提前算出, 可以在实现时视 profiling 结果决定; 但接口上应允许 accept 阶段直接复用。

### 二. 让 `ensure_ws!` 负责容量管理

文件: `src/vmc_det.jl`

`ensure_ws!` 需要扩展, 在 `N = size(v.awf_mat_t, 1)` 确定之后, 负责:

- 为 rank-1 / rank-2 旧字段分配空间
- 为 rank-k cache 的向量与矩阵分配默认空间
- 当 `N` 变化时重新初始化所有相关 buffer

这里建议采用“按当前 `N` 分配最大列数为 `N`”的简单策略, 而不是做更复杂的动态增长逻辑。原因是:

- determinant 最多也只会替换 `N` 列
- 简单直接, 边界清楚
- 这轮重点是消掉重复计算, 不是继续做 allocator 微调

---

## 数据流设计

### 三. `calc_backflow_ratio_local_update(...)` 改为“计算并写缓存”

文件: `src/vmc_det.jl`

当前逻辑:

1. `collect_backflow_local_column_updates(...)`
2. `ratio_rankk(...)`
3. 可选 debug verify

修改后逻辑:

1. 调用一次 `collect_backflow_local_column_updates(...)`
2. 把 `changed_electron_ids / changed_row_indices / new_columns` 写入 `ws`
3. 基于这份数据构造 `c_matrix / small_k_matrix`
4. 把这些 rank-k 因子写入 `ws`
5. 计算 `ratio`
6. 设置 `ws.has_cached_rankk_update = true`
7. 可选 debug verify

这样 ratio 阶段的结果会完整留在 workspace 中, 供 accept 阶段直接消费。

### 四. `accept_backflow_local_update!(...)` 改为直接消费缓存

文件: `src/vmc_det.jl`

当前逻辑:

1. 再次 `collect_backflow_local_column_updates(...)`
2. `update_rankk!(...)`
3. `commit_move!(...)`
4. `refresh_backflow_orbitals!(...)`

修改后逻辑:

1. 检查 `ws.has_cached_rankk_update`
2. 若为 `false`, 抛出明确错误
3. 若为 `true`, 直接从 `ws` 读取缓存
4. 调用使用缓存的 `update_rankk!`
5. `commit_move!(...)`
6. `refresh_backflow_orbitals!(...)`
7. 把 `ws.has_cached_rankk_update` 清回 `false`

这样 accept 阶段不再对同一个 proposal 做第二次收集。

### 五. cache 生命周期

cache 生命周期规则如下:

- `init_gswf!` / `rebuild_slater_state!` / `find_stable_config!` 之后:
  - `has_cached_rankk_update = false`

- 成功调用一次 `calc_backflow_ratio_local_update(...)` 之后:
  - `has_cached_rankk_update = true`

- 调用 `accept_backflow_local_update!(...)` 成功消费后:
  - `has_cached_rankk_update = false`

- 若 proposal 被拒绝:
  - 不需要显式清空
  - 下一次 `calc_backflow_ratio_local_update(...)` 直接覆盖旧缓存

这与“cache 只服务最近一次 ratio”的契约一致。

---

## `rank-k` 计算的 cheap 化方向

### 六. 统一保留通用公式, 但避免重复构造

文件: `src/vmc_det.jl`

当前通用路径的瓶颈不在数学公式本身, 而在于:

- ratio 和 accept 各自重复构造 `c_matrix / small_k_matrix`
- 接受阶段重复切片和重复小矩阵求逆

本轮的 cheap 化不改变公式, 只改变执行方式:

- `compute_rankk_update_factors(...)` 改为支持写入 workspace cache
- `ratio_rankk(...)` 直接从缓存的 `small_k_matrix` 计算 determinant
- `update_rankk!(...)` 改为消费 workspace 中已有的:
  - `cached_changed_electron_ids`
  - `cached_new_columns`
  - `cached_c_matrix`
  - `cached_small_k_matrix`
  - 以及必要时的 `cached_small_k_inverse`

### 七. `update_rankk!` 的执行风格尽量向 `vmc_pfa` 靠拢

文件: `src/vmc_det.jl`

本轮不做 `k <= 4` specialized path, 但仍应避免 accept 阶段的明显重复工作。

具体方向:

- 让 `ratio` 阶段尽可能多地准备 accept 所需小矩阵
- `accept` 阶段避免再走一次完整的 `compute_rankk_update_factors(...)`
- 若实现上可行, 优先把 `small_k_inverse` 或 `S = B*K^{-1}` 这类小对象提前算好并缓存

这里仍然保持一个清晰边界:

- 可以缓存 `N x k` 和 `k x k` 中间量
- 不缓存 `N x N` 修正矩阵

---

## 需求依赖与开发顺序

### 阶段 1: workspace 结构扩展

依赖最少, 只涉及 `R1R2WS` 与 `ensure_ws!`。

完成标志:

- `vwf_det` 可以持有 rank-k cache buffer
- 初始化和 rebuild 路径都会把 cache 置为 invalid

### 阶段 2: ratio 阶段写缓存

在不改 accept 的前提下, 先让 `calc_backflow_ratio_local_update(...)` 能把数据写进 workspace。

完成标志:

- ratio 结束后 cache 为 valid
- debug verify 行为不变

### 阶段 3: accept 阶段读缓存

把 `accept_backflow_local_update!(...)` 改成直接消费 workspace cache, 去掉重复收集。

完成标志:

- accepted move 不再二次调用 `collect_backflow_local_column_updates(...)`
- 无 ratio 先行时会 fail fast

### 阶段 4: rank-k 因子复用

让 `update_rankk!` 直接复用 ratio 阶段已有的 `c_matrix / small_k_matrix` 等结果, 去掉重复构造。

完成标志:

- accept 阶段不再重新构造通用 rank-k 因子
- local update 的 determinant、inverse 与 rebuild 仍然一致

---

## 文件范围

### 需要修改

- `src/vmc_det.jl`
  - 扩展 `R1R2WS`
  - 扩展 `ensure_ws!`
  - 增加 rank-k cache 生命周期管理
  - 改造 `calc_backflow_ratio_local_update(...)`
  - 改造 `accept_backflow_local_update!(...)`
  - 改造通用 `rank-k` 的 factor 计算与消费路径

### 本地测试文件

- `test/backflow_eq4_test.jl`
  - 增加 workspace cache 复用测试
  - 增加 invalid cache 的误用测试

- `test/profile_backflow_4x16.jl`
  - 继续作为本地 ignored profiling 工具
  - 验证 `accept_collect_total` 与 `update_rankk_total` 是否下降

### 本轮不改

- `src/Backflow.jl`
- `src/vmc.jl`
- `Hubbard.jl`

---

## 测试与验证要点

### 功能正确性

1. 现有 `backflow_eq4_test.jl` 全部继续通过
2. ratio 后 accept 直接复用 cache 时:
   - `awf_val` 与 rebuild 一致
   - `awf_inv` 与 rebuild 一致
   - `awf_mat_t` 与 rebuild 一致
3. 若在没有 ratio cache 的情况下直接调用 `accept_backflow_local_update!(...)`, 应明确报错

### 性能验证

使用本地 profiling 脚本:

- `julia test/profile_backflow_4x16.jl breakdown withbf`

重点比较这些指标:

- `accept_collect_total`
- `accept_collect_per_accept`
- `update_rankk_total`
- `total_move_total`

### 端到端验证

- `julia Hubbard.jl --Lx 4 --Ly 16 --target_sz 0 --doping 0.0 --nMC 40 --wMC 2 --rMC 50 --dMC 1 --job measure --ansatz AFM --bf_epsilon 0.9 --bf_eta 0.1`

目标:

- 程序行为与当前版本一致
- 端到端采样时间继续下降或至少不回退

---

## 已知限制

1. 本设计依赖当前 `mcmc_step!` 的同步调用顺序契约, 不适用于未来如果引入“先批量算 ratio, 再延迟 accept”的流程。
2. 本设计只优化 determinant backflow fast path, 不解决 `SR` 梯度路径的 full derivative build 开销。
3. 本设计暂不引入 `k <= 4` specialized path, 因此仍然会保留部分通用 `rank-k` 的小矩阵常数项。

---

## 后续候选优化

若本轮完成后仍有明显瓶颈, 下一轮优先考虑:

1. 给 `k <= 4` 加 specialized fast path
2. 把 backflow gradient 改成直接累加, 不构造整块 derivative orbital matrix

