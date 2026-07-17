# Twist Hubbard V-dependent SR Energy Difference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 复用 fixed-parameter 能量差逻辑，整理 V-dependent SR 结果并生成同样式的 CSV、PDF 和 PNG。

**Architecture:** 保留 fixed-parameter 入口与默认图形不变，将扫描目录命名和图标题参数做最小通用化。新增一个只负责配置 SR 输入/输出路径的薄入口，所有科学计算继续由现有模块实现。

**Tech Stack:** Python 3.10、标准库 `csv/math/pathlib`、Matplotlib、`unittest`。

---

### Task 1: 建立 SR 真实数据的失败测试

**Files:**
- Create: `test/plot_twist_hubbard_v_sr_energy_difference_test.py`

- [ ] **Step 1: 写入入口存在性和真实数据测试**

测试通过文件路径加载 SR 入口，并断言：

```python
self.assertTrue(SR_MODULE_PATH.is_file())
rows = module.collect_sr_energy_difference_rows()
self.assertEqual(len(rows), 10)
self.assertEqual({row["state"] for row in rows}, {"Stripe4", "Stripe8"})
self.assertEqual({row["V"] for row in rows}, {0.0, 0.5, 1.0, 1.5, 2.0})
```

另外用真实 `V=0.5` 数据断言 Stripe4-AFM 差值为 `1.2876384786`，并确认误差为正且有限。

- [ ] **Step 2: 运行测试并确认 RED**

Run:

```powershell
D:/software/anaconda3/envs/quspin/python.exe test/plot_twist_hubbard_v_sr_energy_difference_test.py -v
```

Expected: FAIL，原因是 `plot/plot_twist_hubbard_v_sr_energy_difference.py` 尚不存在。

### Task 2: 通用化 fixed 绘图函数并增加 SR 薄入口

**Files:**
- Modify: `plot/plot_twist_hubbard_v_fixed_energy_difference.py`
- Create: `plot/plot_twist_hubbard_v_sr_energy_difference.py`

- [ ] **Step 1: 允许绘图函数接收标题**

将函数签名改为：

```python
def create_energy_difference_figure(
    rows,
    title="Fixed-parameter Stripe-AFM energy difference",
):
```

坐标、线型、颜色、marker、误差棒、零参考线和画布大小均不改变，只将固定标题替换为 `axes.set_title(title)`。

- [ ] **Step 2: 将汇总参数名改为通用扫描目录**

将 `collect_energy_difference_rows(fixed_result_root, baseline_result_root)` 的第一个形参和内部局部变量改为 `scan_result_root`，不改变目录布局和返回数据结构。

- [ ] **Step 3: 新增 SR 入口**

入口定义：

```python
SR_RESULT_ROOT = PROJECT_ROOT / "results" / "twist_Hubbard" / "V_dependence" / "tx1ty1" / "SR"
OUTPUT_DIRECTORY = SR_RESULT_ROOT / "energy_difference_plots"

def collect_sr_energy_difference_rows():
    return collect_energy_difference_rows(SR_RESULT_ROOT, BASELINE_RESULT_ROOT)
```

主程序调用公共 CSV/绘图函数，标题使用 `SR-optimized Stripe-AFM energy difference`，输出文件名与 fixed 阶段一致。

- [ ] **Step 4: 运行新旧测试并确认 GREEN**

Run:

```powershell
D:/software/anaconda3/envs/quspin/python.exe test/plot_twist_hubbard_v_sr_energy_difference_test.py -v
D:/software/anaconda3/envs/quspin/python.exe test/plot_twist_hubbard_v_fixed_energy_difference_test.py -v
```

Expected: SR 测试和原 fixed 测试全部通过。

### Task 3: 生成并核验 SR 输出

**Files:**
- Create: `results/twist_Hubbard/V_dependence/tx1ty1/SR/energy_difference_plots/stripe_afm_energy_difference.csv`
- Create: `results/twist_Hubbard/V_dependence/tx1ty1/SR/energy_difference_plots/stripe_afm_energy_difference.pdf`
- Create: `results/twist_Hubbard/V_dependence/tx1ty1/SR/energy_difference_plots/stripe_afm_energy_difference.png`

- [ ] **Step 1: 执行 SR 绘图入口**

Run:

```powershell
D:/software/anaconda3/envs/quspin/python.exe plot/plot_twist_hubbard_v_sr_energy_difference.py
```

Expected: 程序退出码为 0，并报告三个输出路径。

- [ ] **Step 2: 核对 CSV 数值和文件完整性**

读取 CSV，确认 10 条记录、两种 Stripe 和五个 `V` 点；确认 PDF/PNG 文件大小非零。

- [ ] **Step 3: 视觉检查 PNG**

确认 SR 图与 fixed 图具有相同坐标、两条曲线、误差棒、零参考线、图例和布局，标题为 `SR-optimized Stripe-AFM energy difference`。

- [ ] **Step 4: 最终复跑两组测试和绘图入口**

Run:

```powershell
D:/software/anaconda3/envs/quspin/python.exe test/plot_twist_hubbard_v_sr_energy_difference_test.py -v
D:/software/anaconda3/envs/quspin/python.exe test/plot_twist_hubbard_v_fixed_energy_difference_test.py -v
D:/software/anaconda3/envs/quspin/python.exe plot/plot_twist_hubbard_v_sr_energy_difference.py
```

Expected: 全部退出码为 0，无失败或错误。
