# Twist Emery PBC AFM/Stripe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增支持双向 PBC、方向分辨 `tpd/ep/Vpd`、固定振幅 AFM/Stripe ansatz、无 backflow SR/measure 的 `twist_Emery.jl`.

**Architecture:** `twist_Emery.jl` include 现有 `Emery.jl` 复用几何无关辅助函数, 所有 PBC API 使用 `twist_emery_*` 前缀. Physical Hamiltonian 先构造 term groups 再生成 `GeneralModel` 和能量 observable; mean-field 使用固定参数对象和线性 `dH/dp`, 继续复用现有 shared dense derivative 实现.

**Tech Stack:** Julia, mfVMC, ArgParse, MPI, JSON, Julia `Test`.

---

## 文件结构

- Create: `twist_Emery.jl`
  - PBC site/coordinate/bond API.
  - Physical term groups.
  - AFM/Stripe mean-field Hamiltonian 和 derivative.
  - PBC projector 和 observables.
  - fixed/active/JSON 参数控制.
  - SR/measure CLI 主流程.
- Create: `test/twist_emery_test.jl`
  - 上述行为的单元、集成和回归测试.
- Verify only: `test/emery_obc_geometry_test.jl`
  - 证明 include/reuse 没有破坏现有 OBC 工作流.

### Task 1: PBC site mapping、实际坐标和 bond groups

**Files:**
- Create: `test/twist_emery_test.jl`
- Create: `twist_Emery.jl`

- [ ] **Step 1: 写 geometry 和 Cu-O bond 的失败测试**

在 `test/twist_emery_test.jl` 写入:

```julia
using Test
using LinearAlgebra

include(joinpath(@__DIR__, "..", "twist_Emery.jl"))

@testset "twist Emery PBC site mapping and coordinates" begin
    lx, ly = 3, 2
    @test twist_emery_n_sites(lx, ly) == 18
    sites = [
        twist_emery_xyo_to_site_index(x, y, orbital, lx, ly)
        for x in 1:lx, y in 1:ly, orbital in (EMERY_ORB_D, EMERY_ORB_PY, EMERY_ORB_PX)
    ]
    @test sort(vec(sites)) == collect(1:18)
    @test twist_emery_xyo_to_site_index(lx + 1, 1, EMERY_ORB_D, lx, ly) ==
          twist_emery_xyo_to_site_index(1, 1, EMERY_ORB_D, lx, ly)
    @test twist_emery_orbital_coordinate(2, 1, EMERY_ORB_D) == (2.0, 1.0)
    @test twist_emery_orbital_coordinate(2, 1, EMERY_ORB_PX) == (2.5, 1.0)
    @test twist_emery_orbital_coordinate(2, 1, EMERY_ORB_PY) == (2.0, 1.5)
end

@testset "twist Emery directional Cu-O PBC bonds" begin
    lx, ly = 3, 2
    groups = build_twist_emery_pd_bond_groups(
        lx,
        ly;
        amplitude_x=2.0,
        amplitude_y=3.0,
        bcx=-1.0,
        bcy=0.5,
    )
    @test length(groups.x_bonds) == 2 * lx * ly
    @test length(groups.y_bonds) == 2 * lx * ly
    px_boundary = twist_emery_xyo_to_site_index(lx, 1, EMERY_ORB_PX, lx, ly)
    d_first = twist_emery_xyo_to_site_index(1, 1, EMERY_ORB_D, lx, ly)
    py_boundary = twist_emery_xyo_to_site_index(1, ly, EMERY_ORB_PY, lx, ly)
    @test any(b -> b.i == px_boundary && b.j == d_first && b.coef == -2.0, groups.x_bonds)
    @test any(b -> b.i == py_boundary && b.j == d_first && b.coef == -1.5, groups.y_bonds)
end
```

- [ ] **Step 2: 运行测试并确认因 API 缺失而失败**

Run:

```powershell
julia test\twist_emery_test.jl
```

Expected: `UndefVarError: twist_emery_n_sites not defined`.

- [ ] **Step 3: 实现最小 PBC geometry 和 Cu-O bonds**

在 `twist_Emery.jl` 写入 imports、include 和以下 API:

```julia
include(joinpath(@__DIR__, "Emery.jl"))

function twist_emery_n_sites(lx::Int, ly::Int)::Int
    lx > 0 || error("lx must be positive, got $(lx).")
    ly > 0 || error("ly must be positive, got $(ly).")
    return 3 * lx * ly
end

function twist_emery_xyo_to_site_index(
    x::Int,
    y::Int,
    orbital::Int,
    lx::Int,
    ly::Int,
)::Int
    orbital in (EMERY_ORB_D, EMERY_ORB_PY, EMERY_ORB_PX) ||
        error("Unknown Emery orbital $(orbital).")
    cell_index = (mod1(x, lx) - 1) * ly + mod1(y, ly)
    return 3 * (cell_index - 1) + orbital
end

function twist_emery_orbital_coordinate(
    x::Int,
    y::Int,
    orbital::Int,
)::Tuple{Float64,Float64}
    orbital == EMERY_ORB_D && return (Float64(x), Float64(y))
    orbital == EMERY_ORB_PX && return (Float64(x) + 0.5, Float64(y))
    orbital == EMERY_ORB_PY && return (Float64(x), Float64(y) + 0.5)
    error("Unknown Emery orbital $(orbital).")
end
```

`build_twist_emery_pd_bond_groups` 对每个 cell 按设计文档的四条 bond 写入 `x_bonds/y_bonds`, 只给跨边界 bond 乘 `bcx/bcy`.

- [ ] **Step 4: 补写 O-O 和 Cu-Cu bond 失败测试**

新增断言:

```julia
@testset "twist Emery PBC O-O and Cu-Cu bonds" begin
    lx, ly = 3, 2
    pp_bonds = build_twist_emery_pp_bonds(lx, ly; amplitude=4.0, bcx=-1.0, bcy=0.5)
    dd_groups = build_twist_emery_dd_bond_groups(lx, ly; amplitude=0.7, bcx=-1.0, bcy=0.5)
    @test length(pp_bonds) == 4 * lx * ly
    @test length(dd_groups.x_bonds) == lx * ly
    @test length(dd_groups.y_bonds) == lx * ly
    px_corner = twist_emery_xyo_to_site_index(lx, 1, EMERY_ORB_PX, lx, ly)
    py_corner = twist_emery_xyo_to_site_index(1, ly, EMERY_ORB_PY, lx, ly)
    @test any(b -> b.i == px_corner && b.j == py_corner && b.coef == -2.0, pp_bonds)
end
```

- [ ] **Step 5: 运行并确认缺少 O-O API 的预期失败**

Run: `julia test\twist_emery_test.jl`

Expected: 前两个 testset 通过, 新 testset 因 `build_twist_emery_pp_bonds` 未定义而失败.

- [ ] **Step 6: 实现 O-O、Cu-Cu bond groups 并运行测试**

实现设计文档中的四类 O-O bond 和 x/y Cu-Cu bond. Run:

```powershell
julia test\twist_emery_test.jl
```

Expected: geometry/bond testsets 全部 PASS.

- [ ] **Step 7: 提交**

```powershell
git add twist_Emery.jl test/twist_emery_test.jl
git commit -m "feat: add twist Emery PBC geometry"
```

### Task 2: Physical term groups 和方向能量 observables

**Files:**
- Modify: `twist_Emery.jl`
- Modify: `test/twist_emery_test.jl`

- [ ] **Step 1: 写 physical term group 失败测试**

```julia
@testset "twist Emery anisotropic physical terms" begin
    lx, ly = 3, 2
    setup = build_twist_emery_physical_term_groups(
        lx,
        ly;
        tpd_x=1.2,
        tpd_y=0.8,
        tpp=0.4,
        ep_x=3.1,
        ep_y=2.9,
        Udd=8.0,
        Up=3.0,
        Vpd_x=1.1,
        Vpd_y=0.7,
        Vpp=0.5,
    )
    ncells = lx * ly
    @test length(setup.tpd_x_terms) == 8 * ncells
    @test length(setup.tpd_y_terms) == 8 * ncells
    @test length(setup.tpp_terms) == 16 * ncells
    @test length(setup.ep_x_terms) == ncells
    @test length(setup.ep_y_terms) == ncells
    @test length(setup.udd_terms) == ncells
    @test length(setup.up_terms) == 2 * ncells
    @test length(setup.vpd_x_terms) == 2 * ncells
    @test length(setup.vpd_y_terms) == 2 * ncells
    @test length(setup.vpp_terms) == 4 * ncells
    @test length(setup.all_terms) == sum(length(getproperty(setup, name)) for name in (
        :tpd_x_terms, :tpd_y_terms, :tpp_terms, :ep_x_terms, :ep_y_terms,
        :udd_terms, :up_terms, :vpd_x_terms, :vpd_y_terms, :vpp_terms,
    ))
end
```

- [ ] **Step 2: 运行并确认 API 缺失失败**

Run: `julia test\twist_emery_test.jl`

Expected: `UndefVarError: build_twist_emery_physical_term_groups not defined`.

- [ ] **Step 3: 实现 physical term groups**

实现:

```julia
function build_twist_emery_physical_term_groups(
    lx::Int,
    ly::Int;
    tpd_x::Real,
    tpd_y::Real,
    tpp::Real,
    ep_x::Real,
    ep_y::Real,
    Udd::Real,
    Up::Real,
    Vpd_x::Real,
    Vpd_y::Real,
    Vpp::Real,
)
```

所有 group 即使 coefficient 为零也创建. hopping 每条代表 bond展开 up/down 和 Hermitian conjugate; density interaction 每条代表 bond写一个 `[:n,:n]` term.

- [ ] **Step 4: 写方向能量分解失败测试**

构造小尺寸随机 determinant, 使用:

```julia
observables = build_twist_emery_observables(lx, ly, setup)
@test all(haskey(observables, key) for key in (
    :E, :E_tpd_x, :E_tpd_y, :E_tpp, :E_ep_x, :E_ep_y,
    :E_Udd, :E_Up, :E_Vpd_x, :E_Vpd_y, :E_Vpp,
))
@test sum(observables[key](ham, vwf) for key in (
    :E_tpd_x, :E_tpd_y, :E_tpp, :E_ep_x, :E_ep_y,
    :E_Udd, :E_Up, :E_Vpd_x, :E_Vpd_y, :E_Vpp,
)) ≈ observables[:E](ham, vwf)
```

- [ ] **Step 5: 运行并确认 observable API 缺失失败**

Run: `julia test\twist_emery_test.jl`

Expected: physical group test通过, observable test因函数未定义失败.

- [ ] **Step 6: 实现 model、term energy sum、空间 observable**

实现:

```julia
build_twist_emery_general_model(lx, ly, setup)::GeneralModel
measure_twist_emery_term_energy_sum(terms, model, vwf)
build_twist_emery_cu_site_coordinates(lx, ly)
build_twist_emery_observables(lx, ly, setup)::Dict{Symbol,Function}
build_twist_emery_history_observables()::Vector{Symbol}
```

空间 key 只使用 `d/px/py` 的 `x=1:Lx`, 不创建 `px_0_y`.

- [ ] **Step 7: 运行测试并提交**

Run: `julia test\twist_emery_test.jl`

Expected: 所有现有 testsets PASS.

```powershell
git add twist_Emery.jl test/twist_emery_test.jl
git commit -m "feat: add anisotropic twist Emery model"
```

### Task 3: AFM/Stripe amplitude mean-field 和 derivative

**Files:**
- Modify: `twist_Emery.jl`
- Modify: `test/twist_emery_test.jl`

- [ ] **Step 1: 写 AFM/Stripe onsite 和 hopping 失败测试**

测试构造 `TwistEmeryNonPHParams` 并验证:

```julia
stripe_params = TwistEmeryNonPHParams(
    lx=4,
    ly=2,
    chi1_dp_x=1.0,
    chi1_dp_y=0.8,
    chi1_pp=0.4,
    mu_px=2.0,
    mu_py=3.0,
    delta_c_d=0.5,
    delta_c_px=0.6,
    delta_c_py=0.7,
    delta_s_d=0.9,
    stripe_wavevector=pi / 2,
    stripe_center_offset=0.0,
)
h = Matrix(build_twist_emery_nonph_hamiltonian(stripe_params))
px = twist_emery_xyo_to_site_index(1, 1, EMERY_ORB_PX, 4, 2)
py = twist_emery_xyo_to_site_index(1, 1, EMERY_ORB_PY, 4, 2)
@test h[emery_spin_index(px, 1), emery_spin_index(px, 1)] ≈
      2.0 - 0.6 * cos((pi / 2) * 1.5)
@test h[emery_spin_index(py, 1), emery_spin_index(py, 1)] ≈
      3.0 - 0.7 * cos((pi / 2) * 1.0)
```

另建 AFM 参数并断言只有 d orbital 的 up/down diagonal 含相反 staggered field.

- [ ] **Step 2: 运行并确认类型缺失失败**

Run: `julia test\twist_emery_test.jl`

Expected: `UndefVarError: TwistEmeryNonPHParams not defined`.

- [ ] **Step 3: 实现参数对象和 Hamiltonian**

实现包含所有固定/可优化字段的 keyword constructor. `build_twist_emery_nonph_hamiltonian`:

- 使用 PBC bond builders和 `bcx/bcy`;
- 将 AFM/Stripe field统一表示为参数字段;
- `mu_d` 固定为零;
- oxygen 不加入 spin field.

- [ ] **Step 4: 写每个参数 `dH/dp` 的有限差分失败测试**

对 AFM 和 Stripe 参数列表逐一执行:

```julia
analytic = build_twist_emery_nonph_dh_dparam(params, param_name)
plus = replace_twist_emery_param(params, param_name, current_value + step)
minus = replace_twist_emery_param(params, param_name, current_value - step)
numeric = (
    Matrix(build_twist_emery_nonph_hamiltonian(plus)) -
    Matrix(build_twist_emery_nonph_hamiltonian(minus))
) / (2step)
@test analytic ≈ numeric atol=1.0e-9 rtol=1.0e-9
```

- [ ] **Step 5: 运行并确认 derivative API 缺失失败**

Run: `julia test\twist_emery_test.jl`

Expected: onsite test通过, derivative test因函数未定义失败.

- [ ] **Step 6: 实现线性 derivative 和参数 setup**

实现:

```julia
build_twist_emery_nonph_dh_dparam(params, param_name)
build_twist_emery_mean_field_parameter_setup(args)
```

AFM/Stripe 参数名顺序严格采用设计文档. `NaN` hopping/onsite 初值按 physical 参数除以 `tpd_x` 生成; `tpd_x≈0` 报错.

- [ ] **Step 7: 写非相互作用 one-body parity 测试并实现到通过**

设置 `bcx=bcy=1`, order amplitude 和 `chi1_dd` 为零. 提取 physical up-spin one-body matrix, 断言:

```julia
@test mean_field_up ≈ physical_up / tpd_x
```

Run: `julia test\twist_emery_test.jl`

Expected: 全部 PASS.

- [ ] **Step 8: 提交**

```powershell
git add twist_Emery.jl test/twist_emery_test.jl
git commit -m "feat: add twist Emery AFM and stripe ansatz"
```

### Task 4: PBC projector、fixed/active 参数和 determinant update

**Files:**
- Modify: `twist_Emery.jl`
- Modify: `test/twist_emery_test.jl`

- [ ] **Step 1: 写 PBC projector 失败测试**

```julia
projector = build_twist_emery_density_jastrow_projector(
    4,
    2;
    enable_orbital_gutzwiller=true,
    g_d=0.7,
    g_p=0.3,
    vj_oo=0.1,
    vj_cuo=0.2,
    vj_cucu=0.3,
)
@test projector_param_names(projector) == [:g_d, :g_p, :vj_oo, :vj_cuo, :vj_cucu]
@test projector_param_values(projector) == [0.7, 0.3, 0.1, 0.2, 0.3]
```

- [ ] **Step 2: 运行确认函数缺失失败, 再实现 projector**

Run: `julia test\twist_emery_test.jl`

Expected: projector 函数未定义.

实现 PBC orbital group vector、规范化无向 pair 和三个共享 Jastrow terms.

- [ ] **Step 3: 写 fixed/active/JSON 失败测试**

覆盖:

```julia
@test parse_twist_emery_fixed_param_string("chi1_dd=0.0,g_d=0.5") ==
      Dict(:chi1_dd => 0.0, :g_d => 0.5)
@test build_twist_emery_active_param_indices(
    [:chi1_dd, :Delta_AF_d, :g_d],
    Dict(:chi1_dd => 0.0),
    [:Delta_AF_d],
) == [2]
@test_throws ErrorException build_twist_emery_active_param_indices(
    [:chi1_dd],
    Dict(:chi1_dd => 0.0),
    [:chi1_dd],
)
```

临时 JSON 只写部分参数, 断言缺失项使用 default.

- [ ] **Step 4: 运行确认 API 缺失失败, 再实现参数工具**

实现带 `twist_emery_*` 前缀的 parse/apply/active-index/merge/append JSON 辅助函数.

- [ ] **Step 5: 写 determinant update 失败测试**

构造小尺寸 sampler/projector/vwf, 调用:

```julia
update_twist_emery_ansatz!(
    vwf,
    param_names,
    init_params,
    setup,
    nelec;
    nparams_proj=length(projector_param_names(projector)),
    active_wf_param_names=setup.wf_param_names[1:2],
)
@test vwf.param_keys == setup.wf_param_names[1:2]
@test size(vwf.dUt_matrix, 3) == 2
```

- [ ] **Step 6: 运行确认 update API 缺失失败, 再实现 update 和 active projector gradient**

实现:

```julia
update_twist_emery_ansatz!
set_active_twist_emery_projector_derivative_param_names!
mfVMC.VMC.compute_grad_log_psi!(vwf::mfVMC.VMC.vwf_det{T}) where T
```

override 只包含 dense mean-field 和 selected projector gradient, 并显式拒绝启用 backflow 的 vwf.

- [ ] **Step 7: 运行测试并提交**

Run: `julia test\twist_emery_test.jl`

Expected: 全部 PASS.

```powershell
git add twist_Emery.jl test/twist_emery_test.jl
git commit -m "feat: add twist Emery projector parameter control"
```

### Task 5: CLI、SR/measure 主流程和输出目录

**Files:**
- Modify: `twist_Emery.jl`
- Modify: `test/twist_emery_test.jl`

- [ ] **Step 1: 写 CLI 失败测试**

用保存/恢复 `ARGS` 的 helper 验证:

```julia
args = parse_with_temporary_args(parse_twist_emery_commandline, String[])
@test args["lambda"] isa Int
@test args["output_dir"] == "logs"
@test haskey(args, "tpd_x")
@test haskey(args, "tpd_y")
@test haskey(args, "ep_x")
@test haskey(args, "ep_y")
@test haskey(args, "Vpd_x")
@test haskey(args, "Vpd_y")
@test !haskey(args, "enable_backflow")
```

- [ ] **Step 2: 运行确认 parser 缺失失败, 再实现 CLI**

Run: `julia test\twist_emery_test.jl`

Expected: `parse_twist_emery_commandline` 未定义.

实现设计文档列出的全部参数和 help text.

- [ ] **Step 3: 写 main wiring source test**

读取 `twist_Emery.jl` source 并断言:

```julia
@test occursin("output_dir = args[\"output_dir\"]", source)
@test occursin("history_observables=build_twist_emery_history_observables()", source)
@test occursin("build_twist_emery_physical_term_groups(", source)
@test occursin("update_twist_emery_ansatz!(", source)
```

- [ ] **Step 4: 实现 `main_twist_emery`**

主流程严格按:

1. parse args/timing/MPI;
2. mean-field setup;
3. PBC projector;
4. default/JSON/fixed/active 参数;
5. physical term groups和 `GeneralModel`;
6. particle number/sampler/vwf;
7. shared derivative workspace;
8. initial ansatz update;
9. SR 或 measure;
10. 所有输出写入 `output_dir`.

文件结尾:

```julia
if abspath(PROGRAM_FILE) == @__FILE__
    main_twist_emery()
end
```

- [ ] **Step 5: 运行 feature test 和 OBC regression**

Run:

```powershell
julia test\twist_emery_test.jl
julia test\emery_obc_geometry_test.jl
```

Expected: 两个命令全部 PASS.

- [ ] **Step 6: 小尺寸 CLI smoke**

Run:

```powershell
julia twist_Emery.jl --Lx 2 --Ly 2 --ansatz AFM --job measure --nMC 8 --wMC 2 --rMC 4 --dMC 1 --enable_orbital_gutzwiller false --output_dir logs/twist_emery_smoke
```

Expected:

- exit code `0`;
- `logs/twist_emery_smoke/block_binning_mean.json` 存在;
- JSON 包含 `E`, `E_tpd_x`, `E_tpd_y`, `E_ep_x`, `E_ep_y`.

- [ ] **Step 7: 提交**

```powershell
git add twist_Emery.jl test/twist_emery_test.jl
git commit -m "feat: add twist Emery SR and measure workflow"
```

### Task 6: 完整验收

**Files:**
- Verify: `twist_Emery.jl`
- Verify: `test/twist_emery_test.jl`
- Verify: `test/emery_obc_geometry_test.jl`

- [ ] **Step 1: 文档和格式检查**

Run:

```powershell
git diff --check
rg -n "TODO|TBD|COLUMN_NONPH|emery_n_sites\\(" twist_Emery.jl test/twist_emery_test.jl
```

Expected:

- `git diff --check` 无输出;
- 新文件无 placeholder;
- PBC 逻辑不调用 OBC `emery_n_sites`.

- [ ] **Step 2: 运行完整相关 Julia tests**

```powershell
julia test\twist_emery_test.jl
julia test\emery_obc_geometry_test.jl
julia test\dense_tensor_gradient_test.jl
julia test\projector_site_group_gutzwiller_test.jl
```

Expected: 全部 PASS, 零失败.

- [ ] **Step 3: 运行 AFM 和 Stripe 双 smoke**

```powershell
julia twist_Emery.jl --Lx 2 --Ly 2 --ansatz AFM --job measure --nMC 8 --wMC 2 --rMC 4 --dMC 1 --enable_orbital_gutzwiller false --output_dir logs/twist_emery_afm_smoke
julia twist_Emery.jl --Lx 2 --Ly 2 --ansatz Stripe --lambda 3 --job measure --nMC 8 --wMC 2 --rMC 4 --dMC 1 --enable_orbital_gutzwiller false --output_dir logs/twist_emery_stripe_smoke
```

Expected: 两个命令 exit `0`, 各自输出完整能量字段.

- [ ] **Step 4: 检查提交范围**

```powershell
git status --short
git diff HEAD~4 --stat
git log -6 --oneline
```

Expected: 只有计划内的 `twist_Emery.jl`, `test/twist_emery_test.jl` 和已提交文档; 无意外修改.

- [ ] **Step 5: 请求代码审查并处理 Critical/Important 反馈**

审查范围从 design commit `6e15e88` 到实现 HEAD, 对照:

```text
docs/superpowers/specs/2026-07-23-twist-emery-pbc-stripe-design.md
docs/superpowers/plans/2026-07-23-twist-emery-pbc-stripe.md
```

- [ ] **Step 6: 最终 fresh verification 后提交必要修复**

重新运行 Step 2 和 Step 3. 只有所有命令 exit `0` 后才能报告完成.
