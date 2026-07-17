# Twist Hubbard V-dependent SR 能量差整理设计

## 目标

整理 `results/twist_Hubbard/V_dependence/tx1ty1/SR` 中 AFM、Stripe4 和 Stripe8 的 measure 总能量，绘制与 fixed-parameter 阶段相同的 Stripe-AFM 能量差图，并输出可复核的 CSV。

## 数据定义

- `V=0.5, 1.0, 1.5, 2.0` 读取 SR 目录下各 case 的 `logs/block_binning.txt`。
- `V=0` 继续复用 `Energy_dependence/tx1ty1` 中三种 ansatz 在 `t'=0` 的已有 SR measure 结果。
- 总能量差定义为 `Delta_E = E_Stripe - E_AFM`。
- 假设两次独立测量，误差传播为 `sigma_Delta_E = sqrt(sigma_Stripe^2 + sigma_AFM^2)`。
- CSV 同时保留总能量差和除以 `16*16` 后的每格点能量差。

## 图形要求

SR 图与现有 fixed-parameter 图保持相同：

- 横坐标为 `V/t`，取 `0, 0.5, 1.0, 1.5, 2.0`。
- 纵坐标为 `E_Stripe - E_AFM`。
- 同一张图包含 Stripe4-AFM 和 Stripe8-AFM 两条误差棒曲线。
- 保留 `Delta_E=0` 水平参考线、相同的 marker、linestyle、颜色、图例、网格和画布大小。
- 仅将标题改为 `SR-optimized Stripe-AFM energy difference`。

## 实现结构

对 `plot/plot_twist_hubbard_v_fixed_energy_difference.py` 做最小通用化：保留 fixed-parameter 入口及默认输出不变，将扫描根目录参数命名改为通用含义，并允许绘图函数接收标题。新增 `plot/plot_twist_hubbard_v_sr_energy_difference.py` 作为薄入口，复用既有解析、误差传播、CSV 和绘图函数，不复制科学计算逻辑。

SR 输出目录为：

`results/twist_Hubbard/V_dependence/tx1ty1/SR/energy_difference_plots`

输出文件：

- `stripe_afm_energy_difference.csv`
- `stripe_afm_energy_difference.pdf`
- `stripe_afm_energy_difference.png`

## 验证

1. 先增加失败测试，要求 SR 入口存在并能汇总两种 Stripe 的 10 条记录。
2. 核对一个 SR 实测差值和误差传播，确保符号仍为 Stripe-AFM。
3. 运行新增 SR 测试和原 fixed-parameter 测试，保证旧图逻辑不回归。
4. 执行 SR 绘图入口，确认 CSV、PDF 和 PNG 均生成。
5. 读取 CSV 并视觉检查 PNG，确认曲线、误差棒、零参考线、坐标和标题正确。

## 边界

本步骤不叠加 fixed-parameter 曲线，不修改原始 SR 数据，也不重新运行 Monte Carlo。物理结论只基于当前同步到本地的 measure 结果。
