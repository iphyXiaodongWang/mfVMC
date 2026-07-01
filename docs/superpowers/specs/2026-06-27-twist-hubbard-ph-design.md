# twist Hubbard PH determinant no-backflow design

## 目标

新增 `twist_Hubbard_PH.jl`, 第一阶段只实现 PH 表象下带 singlet pairing 的 twist Hubbard determinant, 不提供 backflow 参数, 不复用当前 nonPH backflow 规则. 完成后先与 `twist_Hubbard.jl` 的 nonPH/no-backflow 路线做 benchmark. 确认 PH determinant, row mapping 和 local energy 符号正确后, 第二阶段再实现物理正确的 PH-aware backflow.

## 关键约束

- 所有开发在 worktree `D:/study/研究生/科研/VMC/HKJ_s/mfVMC/.worktrees/twist-hubbard-ph` 中进行.
- 第一阶段不修改 `src/Backflow.jl`, 也不提供 `--enable_backflow` 和 `bf_*` 参数.
- 新 PH 文件参数接口贴近 `twist_Hubbard.jl`: 保留 `chi1y`, `chi2`, `Delta_AF`, `Delta_c`, `Delta_s`, 并新增 pairing 参数 `etax`, `etay`.
- pairing 的物理定义参考 `Hubbard_restricted.jl` 中 `PartonSquare.RestrictedHubbardParams`: AFM 下可视为 uniform pairing, Stripe 下 `etax` 和 `etay` 必须随 stripe envelope 做 column modulation.
- sampler 使用 `config_Hubbard(...; ifPH=true)`, determinant 列数使用 `N_sites + target_sz`.
- PH determinant 的 row set 依赖现有 `ConfigurationPH`: occupied rows 为 up electron rows 加 down-hole rows.

## 第一阶段修改思路

1. 从 `twist_Hubbard.jl` 复用最小必要结构, 新建 `twist_Hubbard_PH.jl`.
2. 新增 PH mean-field 参数类型, 例如 `TwistHubbardPHParams`, 字段包含 lattice, boundary, hopping, pairing 和 AFM/Stripe 参数.
3. 新增 PH mean-field Hamiltonian 构造函数:
   - hopping 使用 `add_term_ij_PH(H, i, j, -chi, eta)`.
   - x/y pairing 参考 `Hubbard_restricted.jl` 的定义. AFM 下使用 uniform `etax`, `etay`; Stripe 下使用
     `etax0 = etax * abs(cos(Q / 2 * (x + 0.5 - x0)))`,
     `etay0 = -etay * abs(cos(Q / 2 * (x - x0)))`,
     其中 `Q = 2π / lambda`, `x0` 来自 `stripe_center`.
   - onsite field 采用 PH 约定, lower block 是 down-hole sector.
4. 新增 ansatz 生成与导数函数:
   - 对角化 PH Hamiltonian.
   - 取最低 `N_sites + target_sz` 个 occupied orbitals.
   - 导数张量沿用 determinant 的 `dUt_matrix` 约定.
5. 主流程:
   - physical Hamiltonian 仍用 `GeneralModel` 的 electron hopping + onsite Hubbard `U`.
   - projector 沿用 twist PBC finite-distance Jastrow + Gutzwiller.
   - `job=SR/measure`, `fixed_params`, `active_params`, `init_params_json` 尽量保持与 `twist_Hubbard.jl` 一致.

## 依赖拆解与顺序

1. 建立 PH mean-field Hamiltonian 和参数解析.
2. 接入 `vwf_det` 与 `ConfigurationPH`, 完成 `update_twist_ph_ansatz!`.
3. 接入 measure/SR 主流程, 保持 no-backflow.
4. 增加 U=0 精确检验.
5. 增加 PH/no-backflow 与 nonPH/no-backflow benchmark.
6. Review 通过后, 第二阶段再设计 PH-aware backflow.

## 文件计划

- 新增 `twist_Hubbard_PH.jl`.
- 新增 `test/twist_hubbard_ph_no_backflow_test.jl`.
- 新增 `bench_twist_hubbard_ph_nonph.jl` 或测试型 benchmark 文件.
- 暂不修改 `src/Backflow.jl`.

## 测试与 benchmark 要点

- U=0, `etax=etay=0`, no projector 时, PH local energy 方差应接近 0.
- U=0 精确能量应与 single-particle occupied energies 或 nonPH free-fermion 参考一致.
- 同一短 MC 设置下, PH/no-backflow 与 nonPH/no-backflow 输出 `E`, `E_hop`, `E_int` 的差异应在统计误差内.
- 检查 `target_sz` 与 `doping` parity, 并验证 `nup`, `ndn`, `N_sites + target_sz` 维度一致.
- 检查 `fixed_params`, `active_params`, `init_params_json` 对 PH 新增参数的行为.

## 已知风险

- PH lower block 是 down-hole orbital, 不能复用当前 nonPH backflow 逻辑. 第一阶段通过删除 backflow 参数避免误用.
- pairing convention 的整体符号可能影响与 nonPH/Pfaffian 路线的比较, 需要通过 U=0 与短 MC benchmark 先锁定.
- `Hubbard.jl` 中旧 PH backflow 构造看起来与当前 `src/Backflow.jl` 新接口不完全一致, 第一阶段不依赖它.

## 后续阶段

第二阶段将新增 PH-aware backflow. 重点是 lower block 的 down-hole 运动方向反转, 例如 `eta1` 从 upper block 的 `D_i H_j` 变为 lower block 的 `H_i D_j`, 并同步处理 eta2/eta3/eta4 与 backflow 参数导数.
