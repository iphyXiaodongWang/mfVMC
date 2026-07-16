# Twist Hubbard V-dependent SR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 生成 AFM、Stripe4 和 Stripe8 在 `V=0.5, 1.0, 1.5, 2.0` 下独立执行 `SR → measure → plot` 的 12 个本地提交目录。

**Architecture:** 在现有 `fixed_params` 的同级新建独立 `SR` 结果树，避免覆盖对照数据。每个 case 使用同一 ansatz 的 `V=0` 最优参数初始化，并在单个 Slurm 脚本内串行执行优化、测量和分布绘图；一个独立 Python 检查器验证完整的 3×4 参数矩阵。

**Tech Stack:** Bash/Slurm、Julia MPI (`mpiexecjl`)、Python 3 标准库、`bash -n`。

---

### Task 1: 建立提交配置的失败测试

**Files:**
- Create: `.codex_tmp/check_twist_v_sr_submits.py`

- [ ] **Step 1: 写入配置检查器**

检查器需要定义 `WORKSPACE_ROOT`、`RESULT_ROOT`、`SOURCE_ROOT`、`STATE_SPECS` 和 `V_VALUES`，并在 `check_submit_cases()` 中逐 case 验证：

```python
"""检查 twist Hubbard V-dependent SR 提交目录是否满足已确认配置。"""

import json
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = WORKSPACE_ROOT / "results" / "twist_Hubbard" / "V_dependence" / "tx1ty1" / "SR"
SOURCE_ROOT = WORKSPACE_ROOT / "results" / "twist_Hubbard" / "Energy_dependence" / "tx1ty1"
STATE_SPECS = {
    "AFM": {"ansatz": "AFM", "lambda": 4, "source_state": "AFM_SR"},
    "Stripe4": {"ansatz": "Stripe", "lambda": 4, "source_state": "Stripe4_SR"},
    "Stripe8": {"ansatz": "Stripe", "lambda": 8, "source_state": "Stripe8_SR"},
}
V_VALUES = (0.5, 1.0, 1.5, 2.0)
```

`check_submit_cases()` 必须检查每个脚本含两个 `mpiexecjl` 命令，且 SR 命令位于 measure 命令之前；核对 `--nSR 200`、`--lr 0.03`、`--lr_end 0.01`、SR 的 `--dMC 10`、measure 的 `--dMC 40`、对应的 `--V`、ansatz 和 `lambda`；还要核对绘图命令、`slurm_out` 目录和初始 JSON 内容。

- [ ] **Step 2: 运行检查器并确认失败**

Run:

```powershell
D:/software/anaconda3/envs/quspin/python.exe .codex_tmp/check_twist_v_sr_submits.py
```

Expected: FAIL，首个缺失路径位于 `results/twist_Hubbard/V_dependence/tx1ty1/SR`。

### Task 2: 生成 12 个 SR 提交 case

**Files:**
- Create: `results/twist_Hubbard/V_dependence/tx1ty1/SR/AFM/V0.50/submit.sh`
- Create: `results/twist_Hubbard/V_dependence/tx1ty1/SR/AFM/V1.00/submit.sh`
- Create: `results/twist_Hubbard/V_dependence/tx1ty1/SR/AFM/V1.50/submit.sh`
- Create: `results/twist_Hubbard/V_dependence/tx1ty1/SR/AFM/V2.00/submit.sh`
- Create: `results/twist_Hubbard/V_dependence/tx1ty1/SR/Stripe4/V0.50/submit.sh`
- Create: `results/twist_Hubbard/V_dependence/tx1ty1/SR/Stripe4/V1.00/submit.sh`
- Create: `results/twist_Hubbard/V_dependence/tx1ty1/SR/Stripe4/V1.50/submit.sh`
- Create: `results/twist_Hubbard/V_dependence/tx1ty1/SR/Stripe4/V2.00/submit.sh`
- Create: `results/twist_Hubbard/V_dependence/tx1ty1/SR/Stripe8/V0.50/submit.sh`
- Create: `results/twist_Hubbard/V_dependence/tx1ty1/SR/Stripe8/V1.00/submit.sh`
- Create: `results/twist_Hubbard/V_dependence/tx1ty1/SR/Stripe8/V1.50/submit.sh`
- Create: `results/twist_Hubbard/V_dependence/tx1ty1/SR/Stripe8/V2.00/submit.sh`
- Create: each case `logs/min_params.json`
- Create: each case `slurm_out/`

- [ ] **Step 1: 建立目录和复制初始参数**

按以下固定映射复制源参数，不允许从 `fixed_params` 的测量目录反向复制：

| 目标状态 | 源参数 |
|---|---|
| AFM | `Energy_dependence/tx1ty1/AFM_SR/tp0.00/logs/min_params.json` |
| Stripe4 | `Energy_dependence/tx1ty1/Stripe4_SR/tp0.00/logs/min_params.json` |
| Stripe8 | `Energy_dependence/tx1ty1/Stripe8_SR/tp0.00/logs/min_params.json` |

- [ ] **Step 2: 写入 12 个提交脚本**

所有脚本使用现有 Energy-dependence SR 模板中的 Slurm 资源、线程环境变量和远端源路径。每个脚本的第一条 Julia 命令使用：

```bash
--Lx 16 --Ly 16 --tx 1.0 --ty 1.0 --t2 0.0 --U 8.0
--nMC 96000 --rMC 50 --wMC 100 --dMC 10 --nSR 200 --lr 0.03 --lr_end 0.01
--bcx 1.0 --bcy 1.0 --target_sz 0 --doping -0.125
--job SR --enable_backflow false --stripe_center bond
--init_params_json logs/min_params.json
```

第二条命令保持相同的 Hamiltonian 和 ansatz 参数，但使用：

```bash
--dMC 40 --nSR 200 --lr 0.03 --lr_end 0.01 --job measure
```

case 的精确差异矩阵为：

| 状态 | `--ansatz` | `--lambda` | 初始场 | `--V` |
|---|---:|---:|---|---|
| AFM | AFM | 4 | `--Delta_AF 0.3 --g 2.0` | 0.5, 1.0, 1.5, 2.0 |
| Stripe4 | Stripe | 4 | `--Delta_c 0.3 --Delta_s 0.3 --g 2.0` | 0.5, 1.0, 1.5, 2.0 |
| Stripe8 | Stripe | 8 | `--Delta_c 0.3 --Delta_s 0.3 --g 2.0` | 0.5, 1.0, 1.5, 2.0 |

最后执行：

```bash
"$PYTHON_BIN" "$PLOT_CHARGE_SPIN" logs/block_binning_mean.json
```

- [ ] **Step 3: 运行配置检查器并确认通过**

Run:

```powershell
D:/software/anaconda3/envs/quspin/python.exe .codex_tmp/check_twist_v_sr_submits.py
```

Expected: `V-dependent SR 提交配置检查通过: 12/12 cases。`

### Task 3: 验证 Shell 语法和最终改动边界

**Files:**
- Test: all 12 `results/twist_Hubbard/V_dependence/tx1ty1/SR/*/V*/submit.sh`

- [ ] **Step 1: 对所有脚本执行 Bash 语法检查**

Run:

```powershell
Get-ChildItem results/twist_Hubbard/V_dependence/tx1ty1/SR -Recurse -Filter submit.sh | ForEach-Object { bash -n $_.FullName }
```

Expected: 12 个脚本全部退出码为 0，无标准错误输出。

- [ ] **Step 2: 复跑配置检查器**

Run:

```powershell
D:/software/anaconda3/envs/quspin/python.exe .codex_tmp/check_twist_v_sr_submits.py
```

Expected: `V-dependent SR 提交配置检查通过: 12/12 cases。`

- [ ] **Step 3: 检查未覆盖旧结果**

Run:

```powershell
git status --short
```

Expected: 已有 `twist_Hubbard.jl`、测试和绘图脚本改动保持不变；本任务只新增设计/计划文档及被 `.gitignore` 忽略的 `results/.../SR`、`.codex_tmp` 内容。
