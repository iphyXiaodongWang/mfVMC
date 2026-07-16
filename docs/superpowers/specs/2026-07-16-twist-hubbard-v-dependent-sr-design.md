# Twist Hubbard V-dependent SR 计算设计

## 目标

在 `t'=0`、`U=8`、无 backflow 的条件下，分别重新优化 AFM、Stripe4 和 Stripe8 波函数，研究其能量差随最近邻库仑相互作用 `V` 的变化。计算点为 `V=0.5, 1.0, 1.5, 2.0`，`V=0` 继续复用已有结果。

## 目录结构

新增根目录：

`results/twist_Hubbard/V_dependence/tx1ty1/SR`

目录下按 ansatz 和 `V` 分层：

- `AFM/{V0.50,V1.00,V1.50,V2.00}`
- `Stripe4/{V0.50,V1.00,V1.50,V2.00}`
- `Stripe8/{V0.50,V1.00,V1.50,V2.00}`

共生成 12 个互相独立的任务目录。每个目录包含 `submit.sh`、`logs/min_params.json` 和空的 `slurm_out` 目录。

## 初始化策略

每个 `V` 点均独立从相同 ansatz 在 `V=0`、`t'=0` 时的最优参数初始化：

- AFM：`results/twist_Hubbard/Energy_dependence/tx1ty1/AFM_SR/tp0.00/logs/min_params.json`
- Stripe4：`results/twist_Hubbard/Energy_dependence/tx1ty1/Stripe4_SR/tp0.00/logs/min_params.json`
- Stripe8：`results/twist_Hubbard/Energy_dependence/tx1ty1/Stripe8_SR/tp0.00/logs/min_params.json`

不同 `V` 点之间不继承参数，因此 12 个任务可以并行提交，也不会产生串行优化误差传播。

## 单个任务流程

每个 `submit.sh` 依次执行以下步骤：

1. SR 优化：`job=SR`、`nSR=200`、`dMC=10`、`lr=0.03`、`lr_end=0.01`。
2. 重新测量：读取 SR 更新后的 `logs/min_params.json`，使用 `job=measure`、`dMC=40`。
3. 结果绘图：用 `plot_charge_spin_distribution.py` 读取 `logs/block_binning_mean.json`，生成 charge/spin distribution。

`set -euo pipefail` 保证任一步骤失败后停止，不继续产生可能误导的后续结果。

## 物理与采样参数

- 晶格：`Lx=16`、`Ly=16`
- 跃迁：`tx=1.0`、`ty=1.0`、`t2=0.0`
- 相互作用：`U=8.0`，`V` 取对应目录值
- 边界条件：`bcx=1.0`、`bcy=1.0`
- 掺杂与自旋：`doping=-0.125`、`target_sz=0`
- Monte Carlo：`nMC=96000`、`rMC=50`、`wMC=100`
- Backflow：`enable_backflow=false`
- Stripe 中心：`stripe_center=bond`
- AFM：`ansatz=AFM`、`lambda=4`
- Stripe4：`ansatz=Stripe`、`lambda=4`
- Stripe8：`ansatz=Stripe`、`lambda=8`

## 文件生成与验证

仅新增 `SR` 目录，不修改或覆盖已有 `fixed_params` 数据。生成后执行：

1. 检查 12 个目录、12 个 `submit.sh` 和 12 个初始 `min_params.json` 是否完整。
2. 对所有 `submit.sh` 执行 `bash -n`。
3. 自动解析脚本，核对 ansatz、`lambda`、`V`、SR/measure 顺序以及共同物理参数。
4. 比较复制后的 `min_params.json` 与三个源文件，确认内容一致。

## 后续边界

本步骤只生成本地提交目录与脚本，不自动上传或提交到 Paracloud。任务完成后，再从重新优化的 measure 结果计算 Stripe4/AFM 和 Stripe8/AFM 能量差并绘图。
