# twist Emery PBC AFM/Stripe 设计

## 1. 目标

新增 `twist_Emery.jl`, 用 non-particle-hole determinant VMC 计算二维 PBC Emery 三带模型的 AFM 和沿 x 方向调制的 Stripe 态, 并支持两类 ansatz 的能量比较.

本次工作的核心变化为:

1. Emery 几何由当前 `Emery.jl` 的 x-OBC/y-PBC cylinder 改为 x/y 双向 PBC torus.
2. physical Hamiltonian 支持方向分辨的 `tpd_x/tpd_y`, `ep_x/ep_y`, `Vpd_x/Vpd_y`.
3. mean-field ansatz 不再为每个 x 列设置独立参数, 而是只优化固定数量的 uniform 和 Stripe/AFM 振幅.
4. 第一版保留 Gutzwiller 和共享的最近邻 Jastrow projector, 不实现 backflow.
5. measure 输出总能量、方向/作用项分辨的能量、局域密度和自旋、Cu-only `Szz(q)`.

## 2. 非目标

第一版明确不包含:

- complex twist phase `exp(i theta_x/y)`;
- backflow;
- `Lx`, `Ly`, `lambda` 的 commensurability 限制;
- 方向分辨的 Jastrow 参数;
- oxygen orbital 上的 AFM 或 spin-Stripe 磁场;
- 修改或重构现有 OBC `Emery.jl`;
- edge pinning, 因为 PBC torus 没有物理边界.

## 3. 文件与依赖边界

新增文件:

- `twist_Emery.jl`: PBC Emery geometry, physical model, mean-field ansatz, projector, observables, CLI 和主流程.
- `test/twist_emery_test.jl`: PBC geometry, Hamiltonian, ansatz, derivative, projector 和 observable 测试.

`twist_Emery.jl` 使用 `include(joinpath(@__DIR__, "Emery.jl"))` 复用经过现有测试覆盖的通用能力. 因为 `Emery.jl` 的主函数受 `abspath(PROGRAM_FILE) == @__FILE__` 保护, include 时不会启动 OBC 计算.

复用范围包括:

- `EmeryBond`;
- `emery_spin_index`;
- spinful/general-model hopping term 写入;
- Jastrow neighbor table 构造;
- 单 site density/`Sz` observable 辅助函数;
- Cu `Szz(q)` estimator;
- doping 和 bool 解析;
- measure 文件输出;
- `make_column_emery_dense_tensor_shared` 与 MPI shared derivative workspace.

所有 PBC 专用函数使用 `twist_emery_*` 或 `build_twist_emery_*` 前缀, 禁止复用 OBC site count 和 OBC bond builders.

## 4. PBC Emery 几何

### 4.1 Site 数量和编号

每个 Cu unit cell 包含:

- `d`, orbital id `1`;
- `p_y`, orbital id `2`;
- `p_x`, orbital id `3`.

总 site 数为:

```text
Nsite = 3 * Lx * Ly
```

坐标在 site mapping 内通过 `mod1` wrap:

```text
cell(x, y) = (mod1(x, Lx) - 1) * Ly + mod1(y, Ly)
site(x, y, orbital) = 3 * (cell(x, y) - 1) + orbital
```

实际 orbital 坐标定义为:

```text
r_d(x, y)  = (x,       y)
r_px(x, y) = (x + 1/2, y)
r_py(x, y) = (x,       y + 1/2)
```

PBC 下不存在独立的 `p_x(0,y)`. `p_x(Lx,y)` 位于 `d(Lx,y)` 和跨边界的 `d(1,y)` 之间.

### 4.2 Cu-O bond 分组

每个 unit cell 生成四条代表 bond.

x 方向:

```text
d(x,y)  -> px(x,y):   -tpd_x
px(x,y) -> d(x+1,y):  +tpd_x
```

y 方向:

```text
d(x,y)  -> py(x,y):   +tpd_y
py(x,y) -> d(x,y+1):  -tpd_y
```

physical hopping 和 `Vpd` interaction 使用相同的方向分类.

### 4.3 O-O bonds

`tpp` 和 `Vpp` 保持各向同性. 每个 unit cell 使用四类现有 Emery 符号约定:

```text
py(x,y) -> px(x,y):       +tpp
py(x,y) -> px(x,y+1):     -tpp
px(x,y) -> py(x+1,y):     -tpp
px(x,y) -> py(x+1,y-1):   +tpp
```

### 4.4 Cu-Cu bonds

Cu-Cu x/y 最近邻 bond 仅用于:

- auxiliary mean-field hopping `chi1_dd`;
- projector 参数 `vj_cucu`.

physical Emery Hamiltonian 不加入 Cu-Cu hopping.

### 4.5 Boundary factor

physical Hamiltonian 始终使用严格 PBC, 不乘 `bcx/bcy`.

mean-field Hamiltonian 中:

- 跨 `x=Lx -> 1` 的 bond 乘实数 `bcx`;
- 跨 y 边界的 bond 乘实数 `bcy`;
- 同时跨 x/y 边界的 O-O bond 乘 `bcx * bcy`.

第一版不使用 complex phase. `bcx/bcy` 默认值为 `1.0`.

## 5. Physical Hamiltonian

CLI physical 参数为:

```text
--tpd_x
--tpd_y
--tpp
--ep_x
--ep_y
--Udd
--Up
--Vpd_x
--Vpd_y
--Vpp
```

Hamiltonian 为:

```text
H = H_tpd_x + H_tpd_y + H_tpp
  + H_ep_x + H_ep_y
  + H_Udd + H_Up
  + H_Vpd_x + H_Vpd_y + H_Vpp
```

其中:

```text
H_ep_x = ep_x * sum_{i in px} n_i
H_ep_y = ep_y * sum_{i in py} n_i
H_Vpd_x = Vpd_x * sum_{<d,px>} n_d n_px
H_Vpd_y = Vpd_y * sum_{<d,py>} n_d n_py
```

term builder 返回以下 NamedTuple 字段:

```text
tpd_x_terms
tpd_y_terms
tpp_terms
ep_x_terms
ep_y_terms
udd_terms
up_terms
vpd_x_terms
vpd_y_terms
vpp_terms
all_terms
```

即使某个 coefficient 为零, 对应 term group 仍然构造并保留. 这让所有计算的 observable schema 一致, 也使能量分解恒满足固定字段关系.

## 6. Mean-field ansatz

### 6.1 Gauge

固定:

```text
chi1_dp_x = 1
mu_d = 0
```

理由:

- 整体缩放 one-body Hamiltonian 不改变 occupied eigenvectors, 因此固定 `chi1_dp_x` 去除尺度零模.
- 同时平移 `mu_d`, `mu_px`, `mu_py` 只给 Hamiltonian 加 identity, 因此固定 `mu_d=0` 去除 uniform onsite 零模.

要求 `tpd_x != 0`, 以支持默认归一化初值.

### 6.2 公共 mean-field 参数

AFM 和 Stripe 共用:

```text
chi1_dp_y
chi1_pp
chi1_dd
mu_px
mu_py
```

默认初值在对应 CLI 值为 `NaN` 时由 physical 参数生成:

```text
chi1_dp_y = tpd_y / tpd_x
chi1_pp   = tpp / tpd_x
mu_px     = ep_x / tpd_x
mu_py     = ep_y / tpd_x
chi1_dd   = 0
```

CLI 允许显式覆盖这些初值.

### 6.3 AFM

AFM 额外优化:

```text
Delta_AF_d
```

onsite field:

```text
H_d_up(x,y)   += (+1) * (-1)^(x+y) * Delta_AF_d
H_d_down(x,y) += (-1) * (-1)^(x+y) * Delta_AF_d
H_px_sigma    += mu_px
H_py_sigma    += mu_py
```

oxygen orbital 不加入 AFM field.

### 6.4 Stripe

Stripe 额外优化:

```text
Delta_c_d
Delta_c_px
Delta_c_py
Delta_s_d
```

`lambda::Int` 且只要求 `lambda > 0`:

```text
Q = 2pi / lambda
```

`stripe_center`:

```text
site -> x0 = 0.0
bond -> x0 = 0.5
```

onsite field:

```text
mu_d(x)  = -Delta_c_d  * cos(Q * (x       - x0))
mu_px(x) = mu_px - Delta_c_px * cos(Q * (x + 1/2 - x0))
mu_py(x) = mu_py - Delta_c_py * cos(Q * (x       - x0))

m_d(x) = Delta_s_d * sin((Q / 2) * (x - x0))

H_d_up(x,y)   += mu_d(x) + (-1)^(x+y) * m_d(x)
H_d_down(x,y) += mu_d(x) - (-1)^(x+y) * m_d(x)
```

`p_x/p_y` 不加入 spin field.

不要求 Stripe field 在 PBC 接缝处连续. `p_x(Lx,y)` 的代表坐标使用 `Lx + 1/2`, 不强制映射为 `1/2`.

### 6.5 参数顺序

AFM:

```text
chi1_dp_y, chi1_pp, chi1_dd, mu_px, mu_py, Delta_AF_d
```

Stripe:

```text
chi1_dp_y, chi1_pp, chi1_dd, mu_px, mu_py,
Delta_c_d, Delta_c_px, Delta_c_py, Delta_s_d
```

projector 参数统一追加在 mean-field 参数之后.

## 7. Mean-field derivative 和 determinant 更新

新增 `TwistEmeryNonPHParams` 保存 geometry, boundary factor, hopping 和 onsite amplitude.

所有优化参数都线性进入 mean-field Hamiltonian. `dH/dp` 通过构造一个只把目标参数设为 `1`, 其它优化参数和固定 hopping 设为 `0` 的参数对象获得.

occupied orbitals 和 `dU/dp` 继续调用:

```text
make_column_emery_dense_tensor_shared
```

该函数只依赖传入的 `build_hamiltonian` 和 `build_dh_dparam`, 不依赖 column-resolved 参数结构, 因此无需修改 `src/Utils.jl`.

wavefunction 更新顺序:

1. 将 active mean-field 参数合并回完整参数向量.
2. 构造 `TwistEmeryNonPHParams`.
3. 只为 active mean-field 参数构造 derivative tensor.
4. 更新 `base_gs_U`, `gs_U`, `gs_U_t`.
5. 更新 active mean-field 参数名和 derivative tensor.
6. 更新完整 projector 数值.
7. 调用 `init_gswf!`.

## 8. Projector

第一版 projector 参数保持:

```text
g_d
g_p
vj_oo
vj_cuo
vj_cucu
```

`enable_orbital_gutzwiller=false` 时只保留三个 Jastrow 参数.

PBC projector 使用 PBC bond groups:

- `vj_oo`: 所有 O-O bonds;
- `vj_cuo`: x/y 两类 Cu-O bonds共用;
- `vj_cucu`: x/y 两类 Cu-Cu bonds共用.

Jastrow pair 规范化为 `(min(site_i,site_j), max(...))` 并去重, 然后构造对称 neighbor table.

## 9. Fixed/active 参数控制

移植 `twist_Hubbard.jl` 的用户接口, 但使用 `twist_emery_*` 前缀实现:

```text
--init_params_json
--fixed_params
--active_params
```

规则:

1. JSON 中存在的参数覆盖默认值.
2. JSON 缺失的当前参数使用 CLI/default 值.
3. `fixed_params` 再覆盖 JSON/default.
4. `active_params` 为空时, 所有非 fixed 参数参与 SR.
5. 显式 active 参数不能同时 fixed.
6. 未知、重复或不属于当前 ansatz 的参数立即报错.
7. `chi1_dp_x` 和 `mu_d` 是 gauge, 不出现在参数列表中, 因而不能 active/fixed.

第一版没有 backflow. 为支持 inactive projector 参数, `twist_Emery.jl` 为当前进程重定义 determinant `compute_grad_log_psi!`, 保留:

- active mean-field dense tensor gradient;
- active projector log derivative;

不包含 backflow gradient 分支.

SR 完成后, `min_params.json` 补写所有 inactive/fixed 参数, 保证文件包含完整 ansatz.

## 10. Observable 和输出

### 10.1 能量

measure 输出:

```text
E
E_tpd_x
E_tpd_y
E_tpp
E_ep_x
E_ep_y
E_Udd
E_Up
E_Vpd_x
E_Vpd_y
E_Vpp
```

必须满足:

```text
E = E_tpd_x + E_tpd_y + E_tpp
  + E_ep_x + E_ep_y
  + E_Udd + E_Up
  + E_Vpd_x + E_Vpd_y + E_Vpp
```

所有字段固定存在, 零 coupling 时对应 estimator 返回零.

### 10.2 空间 observable

每个 PBC unit cell 输出:

```text
n_d_x_y,  Sz_d_x_y
n_px_x_y, Sz_px_x_y
n_py_x_y, Sz_py_x_y
```

不输出 OBC 专用的 `px_0_y`.

继续输出 Cu-only:

```text
Szzq_nx_ny
```

其 momentum index 为 `nx=0:Lx-1`, `ny=0:Ly-1`.

### 10.3 输出目录

新增:

```text
--output_dir
```

默认 `logs`. SR history, minimum parameter JSON, measure JSON/text 和 timing 文件全部写入该目录.

## 11. CLI 与主流程

保留现有 Emery 的主要运行参数:

```text
--Lx, --Ly
--bcx, --bcy
--target_sz
--nMC, --wMC, --rMC, --dMC, --seed
--nSR, --lr, --lr_end, --eigen_cutoff
--job SR|measure
--doping
--ansatz AFM|Stripe
--lambda
--stripe_center
--enable_orbital_gutzwiller
--g_d, --g_p, --vj_oo, --vj_cuo, --vj_cucu
--enable_timing
--output_dir
```

不提供 `--enable_backflow` 和所有 `bf_*` 参数.

粒子数继续使用当前 Emery 约定:

```text
N_e = Lx * Ly * (1 + doping)
N_up = (N_e + target_sz) / 2
N_down = N_e - N_up
```

`doping` 继续支持小数和分数字符串.

## 12. 错误处理

以下情况立即报错:

- `Lx <= 0`, `Ly <= 0`;
- `lambda <= 0`;
- `tpd_x` 近似为零, 无法固定 hopping gauge;
- 未知 `ansatz` 或 `stripe_center`;
- 粒子数不是整数;
- `N_e` 与 `target_sz` parity 不一致;
- `N_up/N_down` 超出合法范围;
- fixed/active/JSON 参数名非法;
- 参数名和值数量不一致;
- derivative 参数不属于当前 ansatz.

不检查 `Lx/Ly/lambda` commensurability.

## 13. 测试设计

### 13.1 Geometry

- `Nsite == 3 * Lx * Ly`.
- 所有 `(x,y,o)` 映射唯一覆盖 `1:Nsite`.
- `x/y` 超界坐标正确 wrap.
- `p_x/p_y` 实际坐标分别包含对应的半格 shift.

### 13.2 Bonds

- Cu-O x/y bond 数量、方向和符号正确.
- x/y boundary bond存在且 site 正确.
- O-O 四类符号和跨单/双边界 factor 正确.
- mean-field Cu-Cu x/y PBC bond正确.

### 13.3 Physical Hamiltonian

- `tpd_x/tpd_y`, `ep_x/ep_y`, `Vpd_x/Vpd_y` 只进入对应方向/轨道 group.
- onsite `Udd/Up` 只作用于正确 orbital.
- term group 合并后与 `all_terms` 一致.
- 方向能量 estimator 的和等于总能量 estimator.

### 13.4 Mean-field

- AFM 只有 d orbital 含 staggered field.
- Stripe `p_x` charge phase使用 `x+1/2`.
- Stripe `p_y` charge phase使用 x, 但 coordinate helper 保存 `y+1/2`.
- `site/bond` center offset 正确.
- 非相互作用、无序参量、`bcx=bcy=1` 时, mean-field one-body matrix 等于 physical one-body matrix 除以 `tpd_x`.

### 13.5 Derivative

- 每个 AFM/Stripe mean-field 参数的解析 `dH/dp` 与中心有限差分一致.
- dense occupied-orbital derivative tensor 维度和参数顺序正确.
- active 参数子集只生成对应 derivative.

### 13.6 Projector 和参数控制

- PBC orbital group vector 长度和 d/p 分类正确.
- projector 参数名保持 `g_d,g_p,vj_oo,vj_cuo,vj_cucu`.
- fixed/active 冲突、未知参数和重复参数报错.
- JSON 缺失参数使用 default.
- inactive/fixed 参数正确合并回完整参数.

### 13.7 CLI/流程

- `lambda` 的 ArgParse 类型为 `Int`.
- `output_dir` 默认和显式值正确.
- 第一版 CLI 不包含 backflow 参数.
- 小尺寸 `update_twist_emery_ansatz!` 可以初始化 determinant.
- 现有 `test/emery_obc_geometry_test.jl` 继续通过, 证明没有破坏 OBC 工作流.

## 14. 实现依赖顺序

1. PBC site mapping、coordinate 和 bond groups.
2. physical term groups 与方向能量 observables.
3. AFM/Stripe mean-field 参数对象、Hamiltonian 和 `dH/dp`.
4. PBC projector.
5. fixed/active/JSON 参数控制和 determinant update.
6. CLI、SR/measure 主流程和 `output_dir`.
7. 单元测试、OBC regression 和小尺寸 smoke verification.

每一步只依赖前面的稳定接口, 不需要先实现 backflow.
