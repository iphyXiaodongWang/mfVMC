# Emery Remove Full Backflow Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** In Emery backflow mode, stop caching the full configuration-dependent orbital matrix `U_b(x)` and instead materialize only determinant-needed selected rows.

**Architecture:** Keep `base_gs_U`, `gs_U`, and `gs_U_t` for the non-backflow determinant path. In backflow mode, never treat `gs_U` / `gs_U_t` / `backflow_u` as a full cached `U_b(x)`; rebuild, proposal ratio, accept, debug verification, and stable-config search should construct only occupied basis rows or proposal-changed occupied rows. Remove `backflow_u` from `vwf_det` after all call sites stop depending on it.

**Tech Stack:** Julia, existing `src/vmc_det.jl` determinant code, existing `src/Backflow.jl` source-weight row materialization helpers, Emery tests under `test/`.

---

## Scope And Non-Goals

This plan targets the Emery grouped backflow path. Do not migrate Hubbard backflow behavior in this plan.

Keep the non-backflow determinant path unchanged where possible:

- non-backflow still uses `vwf.gs_U_t[:, row_index]` in `ratio_rank1`, `ratio_rank2`, `update_rank1!`, and `update_rank2!`;
- backflow moves must use `calc_backflow_ratio_local_update` and `accept_backflow_local_update!`;
- `Backflow.build_backflow_orbitals` may remain as a debug/test utility if still useful, but `vwf_det` should not cache its full result.

The central invariant after this refactor is:

```julia
# non-backflow
Slater column elec = vwf.gs_U_t[:, row_index]

# backflow
Slater column elec = materialize U_b(row_index, state) directly into awf_mat_t[:, elec]
```

## File Structure

- Modify: `src/vmc_det.jl`
  - remove `backflow_u` from `vwf_det`;
  - add selected-row Slater materialization helpers for backflow;
  - update rebuild/init/debug/stable-config paths to use selected rows;
  - remove accept-time full `U_b(x)` cache refresh.
- Modify: `Emery.jl`
  - remove writes to `vwf.backflow_u` in parameter update code.
- Modify: `test/emery_grouped_backflow_cleanup_test.jl`
  - add regression tests proving backflow rebuild and local update no longer require a full `backflow_u` cache.
- Optional Modify: `src/Backflow.jl`
  - only add a small current-state row wrapper if the implementation would otherwise duplicate source-weight materialization logic. Do not rewrite source-weight internals.

---

### Task 1: Add Tests For Selected-Row Backflow Rebuild

**Files:**
- Modify: `test/emery_grouped_backflow_cleanup_test.jl`
- Read: `src/vmc_det.jl`
- Read: `src/Backflow.jl`

- [ ] **Step 1: Add a deterministic Emery vwf fixture**

Add this helper near the existing test helpers in `test/emery_grouped_backflow_cleanup_test.jl`:

```julia
"""
用途: 构造一个小尺寸 Emery grouped backflow determinant 测试对象。

参数:
- 无。

返回:
- `vwf_det`: 已设置 grouped Emery backflow 的 determinant 波函数对象。
"""
function build_grouped_emery_backflow_vwf_fixture()
    lx = 2
    ly = 2
    n_sites = emery_n_sites(lx, ly)
    state_chars = join(("Dhud"[mod1(site_index, 4)] for site_index in 1:n_sites))
    sampler = mfVMC.Sampler.init_config_Hubbard_by_state_char(state_chars)

    dd_source_bonds,
    dd_source_amplitudes,
    dp_source_bonds,
    dp_source_amplitudes,
    pd_source_bonds,
    pd_source_amplitudes,
    pp_source_bonds,
    pp_source_amplitudes = build_emery_backflow_source_data_by_directed_orbital_type(
        lx,
        ly;
        tpd=1.0,
        tpp=0.45,
        bcy=1.0,
    )
    backflow = build_column_directed_emery_backflow(
        dd_source_bonds,
        dd_source_amplitudes,
        dp_source_bonds,
        dp_source_amplitudes,
        pd_source_bonds,
        pd_source_amplitudes,
        pp_source_bonds,
        pp_source_amplitudes,
        0.91,
        1.08,
        0.11,
        -0.07,
        0.05,
        -0.03,
        -0.13,
        0.17,
        -0.19,
        0.23,
        0.29,
        -0.31,
        0.37,
        -0.41,
        -0.43,
        0.47,
        -0.53,
        0.59,
    )

    n_rows = 2 * n_sites
    n_electrons = length(sampler.electron_locs)
    base_orbitals = [
        0.2 * row_index + 0.07 * orbital_index + 0.003 * row_index * orbital_index
        for row_index in 1:n_rows, orbital_index in 1:n_electrons
    ]
    return mfVMC.vwf_det(base_orbitals, sampler; backflow=backflow)
end
```

- [ ] **Step 2: Add a test that compares selected-row rebuild against full `U_b(x)` truth**

Add this testset near the existing grouped backflow tests:

```julia
@testset "Backflow rebuild materializes only occupied rows" begin
    vwf = build_grouped_emery_backflow_vwf_fixture()
    init_gswf!(vwf)

    full_orbitals = Backflow.build_backflow_orbitals(
        vwf.base_gs_U,
        vwf.sampler.state,
        vwf.backflow,
    )
    expected_slater = build_slater_matrix_from_orbitals(
        full_orbitals,
        vwf.sampler.electron_locs,
    )

    @test vwf.awf_mat_t ≈ transpose(expected_slater)
    @test vwf.awf_val ≈ det(expected_slater)
    @test vwf.awf_inv ≈ inv(expected_slater)
end
```

- [ ] **Step 3: Run the new test before implementation**

Run:

```powershell
julia test\emery_grouped_backflow_cleanup_test.jl
```

Expected before implementation: the test may pass because the current full-cache rebuild is still mathematically correct. That is acceptable; this is a characterization test, not necessarily a failing test.

- [ ] **Step 4: Add a structural test that `vwf_det` no longer exposes `backflow_u` after implementation**

Add this test now, but expect it to fail until Task 5:

```julia
@testset "vwf_det does not store full backflow cache" begin
    vwf = build_grouped_emery_backflow_vwf_fixture()
    @test !hasproperty(vwf, :backflow_u)
end
```

- [ ] **Step 5: Commit the tests after they are added**

Run:

```powershell
git add test\emery_grouped_backflow_cleanup_test.jl
git commit -m "test: characterize selected-row Emery backflow rebuild"
```

---

### Task 2: Add Backflow Selected-Row Slater Materialization Helpers

**Files:**
- Modify: `src/vmc_det.jl`
- Optional Modify: `src/Backflow.jl`
- Test: `test/emery_grouped_backflow_cleanup_test.jl`

- [ ] **Step 1: Add a helper that fills one current-state backflow row**

Prefer placing this helper in `src/vmc_det.jl` first, because it is determinant-specific and can reuse existing `Backflow.fill_backflow_chain_rule_source_weights!`.

Add a function with a Chinese docstring:

```julia
"""
用途: 根据当前构型按需写入一条 backflow orbital row, 不构造完整 `U_b(x)` 矩阵。

数学公式:
- `U_b(row,:) = sum_s w_s(row,x) * U_0(s,:)`。
- `row_index` 是 PH 基底内部行编号, 即 `2 * (site_index - 1) + row_offset`。

参数:
- `row_buffer::AbstractVector{T}`: 输出 row buffer, 长度等于轨道数。
- `vwf::vwf_det{T}`: determinant 波函数对象。
- `row_index::Int`: 需要 materialize 的 PH 基底行编号。

返回:
- `nothing`。
"""
function fill_current_backflow_orbital_row!(
    row_buffer::AbstractVector{T},
    vwf::vwf_det{T},
    row_index::Int,
) where {T}
    ws = ensure_ws!(vwf)
    source_count = Backflow.fill_backflow_chain_rule_source_weights!(
        ws.backflow_chain_rule_source_rows,
        ws.backflow_chain_rule_source_weights,
        vwf.sampler.state,
        vwf.backflow,
        row_index,
    )
    Backflow.fill_backflow_row_from_source_weights!(
        row_buffer,
        vwf.base_gs_U,
        ws.backflow_chain_rule_source_rows,
        ws.backflow_chain_rule_source_weights,
        source_count,
    )
    return nothing
end
```

If `Backflow.fill_backflow_row_from_source_weights!` is not exported, either export it from `src/Backflow.jl` or add a small public wrapper in `Backflow.jl`. Prefer the wrapper/export with minimal changes over duplicating the materialization loop in `vmc_det.jl`.

- [ ] **Step 2: Add a helper that fills the full occupied Slater matrix from selected rows**

Add this function in `src/vmc_det.jl`:

```julia
"""
用途: 在 backflow 模式下只对当前 occupied basis rows 构造 Slater 矩阵。

数学公式:
- 第 `elec` 个 determinant column 为 `U_b(row_index,x)`。
- `row_index = vwf.sampler.electron_locs[elec]`。

参数:
- `vwf::vwf_det{T}`: determinant 波函数对象。

返回:
- `nothing`。结果写入 `vwf.awf_mat_t`。
"""
function fill_backflow_slater_matrix_from_occupied_rows!(vwf::vwf_det{T}) where {T}
    ss = vwf.sampler
    total_elec_count = total_elec(ss)
    if size(vwf.awf_mat_t, 1) != total_elec_count
        vwf.awf_mat_t = zeros(T, total_elec_count, total_elec_count)
    end

    for elec in 1:total_elec_count
        row_index = ss.electron_locs[elec]
        fill_current_backflow_orbital_row!(
            @view(vwf.awf_mat_t[:, elec]),
            vwf,
            row_index,
        )
    end
    return nothing
end
```

- [ ] **Step 3: Run the focused test**

Run:

```powershell
julia test\emery_grouped_backflow_cleanup_test.jl
```

Expected: existing tests still pass except the structural `!hasproperty(vwf, :backflow_u)` test still fails until Task 5.

- [ ] **Step 4: Commit the helper**

Run:

```powershell
git add src\vmc_det.jl src\Backflow.jl
git commit -m "refactor: add selected-row backflow slater materialization"
```

---

### Task 3: Stop Full `U_b(x)` Rebuild In Determinant Rebuild Paths

**Files:**
- Modify: `src/vmc_det.jl`
- Test: `test/emery_grouped_backflow_cleanup_test.jl`

- [ ] **Step 1: Change `rebuild_slater_state!`**

Replace the unconditional `refresh_backflow_orbitals!(vwf)` + `gs_U_t` copy logic with a branch:

```julia
function rebuild_slater_state!(vwf::vwf_det{T,S}) where {T,S}
    ss = vwf.sampler
    total_elec_count = total_elec(ss)

    if size(vwf.awf_mat_t, 1) != total_elec_count
        vwf.awf_mat_t = zeros(T, total_elec_count, total_elec_count)
    end

    if Backflow.uses_backflow(vwf.backflow)
        fill_backflow_slater_matrix_from_occupied_rows!(vwf)
    else
        refresh_orbitals_without_backflow!(vwf)
        for i in 1:total_elec_count
            row_in_U = ss.electron_locs[i]
            copyto!(@view(vwf.awf_mat_t[:, i]), @view(vwf.gs_U_t[:, row_in_U]))
        end
    end

    A_physical = transpose(vwf.awf_mat_t)
    F = lu(A_physical)

    vwf.awf_val = det(F)
    vwf.awf_inv = inv(F)
    vwf.current_ratio = one(T)

    ensure_ws!(vwf)
    reset_cached_rankk_update!(vwf.ws)
    return nothing
end
```

Do not leave `refresh_backflow_orbitals!(vwf)` in the backflow branch.

- [ ] **Step 2: Split `refresh_backflow_orbitals!` or replace it with a non-backflow helper**

The old function name is misleading once full `U_b(x)` cache is removed. Replace call sites in non-backflow contexts with:

```julia
"""
用途: 刷新非 backflow determinant 使用的裸轨道矩阵缓存。

参数:
- `vwf::vwf_det{T}`: determinant 波函数对象。

返回:
- `nothing`。
"""
function refresh_orbitals_without_backflow!(vwf::vwf_det{T}) where {T}
    copyto!(vwf.gs_U, vwf.base_gs_U)
    permutedims!(vwf.gs_U_t, vwf.gs_U, (2, 1))
    return nothing
end
```

If keeping `refresh_backflow_orbitals!` temporarily reduces diff size, make it throw in active backflow mode:

```julia
function refresh_backflow_orbitals!(vwf::vwf_det{T}) where {T}
    if Backflow.uses_backflow(vwf.backflow)
        error("Full U_b(x) cache is removed. Use selected-row backflow materialization instead.")
    end
    refresh_orbitals_without_backflow!(vwf)
    return nothing
end
```

- [ ] **Step 3: Update `set_backflow!` and backflow parameter updates**

Remove eager full-cache refresh from:

- `set_backflow!`;
- `update_vwf_backflow_params!(vwf, param_names, param_values)`;
- `update_vwf_backflow_params!(vwf, param_values)`.

The replacement should be:

```julia
vwf.backflow = backflow
ensure_ws!(vwf)
return nothing
```

and for parameter updates:

```julia
Backflow.update_backflow_params!(vwf.backflow, param_names, param_values)
ensure_ws!(vwf)
return nothing
```

Do not silently rebuild the determinant there; callers that need a consistent determinant must call `init_gswf!(vwf)` or `rebuild_slater_state!(vwf)`, which is already the pattern in `Emery.jl`.

- [ ] **Step 4: Update `find_stable_config!`**

Inside the attempt loop, replace:

```julia
refresh_backflow_orbitals!(vwf)
...
copyto!(@view(vwf.awf_mat_t[:, i]), @view(vwf.gs_U_t[:, basis_idx]))
```

with:

```julia
if Backflow.uses_backflow(vwf.backflow)
    fill_backflow_slater_matrix_from_occupied_rows!(vwf)
else
    refresh_orbitals_without_backflow!(vwf)
    total_elec_count = total_elec(ss)
    for i in 1:total_elec_count
        basis_idx = ss.electron_locs[i]
        copyto!(@view(vwf.awf_mat_t[:, i]), @view(vwf.gs_U_t[:, basis_idx]))
    end
end
```

- [ ] **Step 5: Run focused tests**

Run:

```powershell
julia test\emery_grouped_backflow_cleanup_test.jl
julia -e 'include("Emery.jl"); println("ok")'
```

Expected: behavioral tests pass; structural `backflow_u` test may still fail until Task 5.

- [ ] **Step 6: Commit rebuild-path changes**

Run:

```powershell
git add src\vmc_det.jl test\emery_grouped_backflow_cleanup_test.jl
git commit -m "refactor: rebuild Emery backflow determinant from selected rows"
```

---

### Task 4: Remove Accept-Time Full Cache Refresh

**Files:**
- Modify: `src/vmc_det.jl`
- Test: `test/emery_grouped_backflow_cleanup_test.jl`

- [ ] **Step 1: Remove full-cache row writeback from local accept**

In `accept_backflow_local_update!`, remove this line:

```julia
@timed "backflow_refresh_accepted_rows" refresh_backflow_affected_orbital_rows_after_accept!(vwf, proposal)
```

After `commit_move!`, the determinant state is already updated by `update_rankk_from_cache!`. Future proposals should compute proposal rows from `base_gs_U`, `sampler.state`, `backflow`, and `proposal`, not from a cached full `U_b(x)`.

- [ ] **Step 2: Delete or isolate obsolete full-cache accept helpers**

Remove these functions if no call sites remain:

```julia
refresh_backflow_affected_orbital_rows_after_accept!
apply_cached_backflow_orbital_rows!
```

If a debug-only call site remains, replace it with selected-row rebuild or selected-row verification before deleting.

- [ ] **Step 3: Make `verify_backflow_local_accept` selected-row based**

Replace the full orbital rebuild:

```julia
orbitals_check = Backflow.build_backflow_orbitals(
    vwf.base_gs_U,
    vwf.sampler.state,
    vwf.backflow,
)
slater_check = build_slater_matrix_from_orbitals(
    orbitals_check,
    vwf.sampler.electron_locs,
)
```

with a selected-row truth matrix:

```julia
slater_t_check = similar(vwf.awf_mat_t)
saved_awf_mat_t = vwf.awf_mat_t
try
    vwf.awf_mat_t = slater_t_check
    fill_backflow_slater_matrix_from_occupied_rows!(vwf)
finally
    vwf.awf_mat_t = saved_awf_mat_t
end
slater_check = transpose(slater_t_check)
```

If mutating `vwf.awf_mat_t` temporarily feels too risky, instead add a helper that accepts an explicit output matrix:

```julia
fill_backflow_slater_matrix_from_occupied_rows!(output_matrix_t, vwf)
```

This explicit-output overload is the cleaner option if the diff remains small.

- [ ] **Step 4: Make `calc_ratio_rebuild` selected-row based**

Replace full `build_backflow_orbitals` truth construction with:

```julia
new_sampler = copy_config(vwf.sampler)
commit_move!(new_sampler, proposal)

slater_t_new = similar(vwf.awf_mat_t)
fill_backflow_slater_matrix_from_sampler_state!(
    slater_t_new,
    vwf.base_gs_U,
    new_sampler,
    vwf.backflow,
    ensure_ws!(vwf),
)
new_slater = transpose(slater_t_new)
return det(new_slater) / vwf.awf_val
```

This requires adding an explicit helper:

```julia
"""
用途: 使用给定 sampler 的当前构型, 只对 occupied basis rows 构造 backflow Slater 矩阵。

参数:
- `output_matrix_t::AbstractMatrix{T}`: 输出转置 Slater 矩阵, 形状为 `N_elec x N_elec`。
- `base_orbitals::AbstractMatrix{T}`: 裸轨道矩阵 `U_0`。
- `sampler`: 已包含 `state` 与 `electron_locs` 的采样器。
- `backflow::Backflow.AbstractBackflowTerm`: backflow 项。
- `ws::R1R2WS{T}`: 工作缓冲区。

返回:
- `nothing`。
"""
function fill_backflow_slater_matrix_from_sampler_state!(
    output_matrix_t::AbstractMatrix{T},
    base_orbitals::AbstractMatrix{T},
    sampler,
    backflow::Backflow.AbstractBackflowTerm,
    ws::R1R2WS{T},
) where {T}
    for elec in 1:total_elec(sampler)
        row_index = sampler.electron_locs[elec]
        source_count = Backflow.fill_backflow_chain_rule_source_weights!(
            ws.backflow_chain_rule_source_rows,
            ws.backflow_chain_rule_source_weights,
            sampler.state,
            backflow,
            row_index,
        )
        Backflow.fill_backflow_row_from_source_weights!(
            @view(output_matrix_t[:, elec]),
            base_orbitals,
            ws.backflow_chain_rule_source_rows,
            ws.backflow_chain_rule_source_weights,
            source_count,
        )
    end
    return nothing
end
```

- [ ] **Step 5: Run debug verification test**

Add or update a test that enables:

```julia
vwf.backflow_debug_verify = true
```

Then perform one accepted local backflow move using the existing move/proposal helper in the test file. The test should pass without calling full `Backflow.build_backflow_orbitals` from `vmc_det.jl`.

Run:

```powershell
julia test\emery_grouped_backflow_cleanup_test.jl
```

- [ ] **Step 6: Commit accept/debug changes**

Run:

```powershell
git add src\vmc_det.jl test\emery_grouped_backflow_cleanup_test.jl
git commit -m "refactor: remove accepted-row backflow cache refresh"
```

---

### Task 5: Remove `backflow_u` Field And External Writes

**Files:**
- Modify: `src/vmc_det.jl`
- Modify: `Emery.jl`
- Test: `test/emery_grouped_backflow_cleanup_test.jl`

- [ ] **Step 1: Remove `backflow_u` from `vwf_det`**

In `mutable struct vwf_det{T,S}`, remove:

```julia
backflow_u::Matrix{T}
```

In the constructor, remove:

```julia
backflow_u = copy(U)
...
backflow_u,     # backflow_u
```

Keep:

```julia
base_gs_u = copy(U)
gs_u = copy(U)
gs_u_t = permutedims(U)
```

because they remain useful for non-backflow.

- [ ] **Step 2: Remove external writes in `Emery.jl`**

In `Emery.jl`, remove this line:

```julia
copyto!(vwf.backflow_u, gs_u)
```

Keep:

```julia
copyto!(vwf.base_gs_U, gs_u)
copyto!(vwf.gs_U, gs_u)
copyto!(vwf.gs_U_t, permutedims(gs_u))
```

These lines update the non-backflow orbital cache and the base orbitals used by backflow selected-row materialization.

- [ ] **Step 3: Search for remaining references**

Run:

```powershell
rg -n "backflow_u|refresh_backflow_affected_orbital_rows_after_accept!|apply_cached_backflow_orbital_rows!" src Emery.jl test
```

Expected: no results.

- [ ] **Step 4: Run focused tests**

Run:

```powershell
julia test\emery_grouped_backflow_cleanup_test.jl
julia test\simplify_emery_backflow_api_test.jl
julia -e 'include("Emery.jl"); println("ok")'
```

Expected: all commands pass, including the structural `!hasproperty(vwf, :backflow_u)` test.

- [ ] **Step 5: Commit cache-field removal**

Run:

```powershell
git add src\vmc_det.jl Emery.jl test\emery_grouped_backflow_cleanup_test.jl
git commit -m "refactor: remove full Emery backflow orbital cache"
```

---

### Task 6: Guard Backflow Against Accidental `gs_U_t` Rank1/Rank2 Use

**Files:**
- Modify: `src/vmc_det.jl`
- Test: `test/emery_grouped_backflow_cleanup_test.jl`

- [ ] **Step 1: Add explicit guards to non-backflow rank1/rank2 helpers**

At the top of these functions:

- `ratio_rank1`;
- `update_rank1!`;
- `ratio_rank2`;
- `update_rank2!`;

add:

```julia
if Backflow.uses_backflow(vwf.backflow)
    error("Backflow mode must use selected-row local rank-k update, not gs_U_t rank1/rank2 update.")
end
```

This prevents accidental future code from reading stale non-backflow `gs_U_t` as if it were `U_b(x)`.

- [ ] **Step 2: Add a regression test for the guard**

Add:

```julia
@testset "Backflow rejects gs_U_t rank1 path" begin
    vwf = build_grouped_emery_backflow_vwf_fixture()
    init_gswf!(vwf)
    @test_throws ErrorException ratio_rank1(vwf, 1, vwf.sampler.electron_locs[1])
end
```

- [ ] **Step 3: Run focused tests**

Run:

```powershell
julia test\emery_grouped_backflow_cleanup_test.jl
```

Expected: pass.

- [ ] **Step 4: Commit guard changes**

Run:

```powershell
git add src\vmc_det.jl test\emery_grouped_backflow_cleanup_test.jl
git commit -m "fix: guard backflow against gs_U_t rank update path"
```

---

### Task 7: Final Verification

**Files:**
- Read: `src/vmc_det.jl`
- Read: `src/Backflow.jl`
- Read: `Emery.jl`
- Read: `test/emery_grouped_backflow_cleanup_test.jl`

- [ ] **Step 1: Search for removed full-cache concepts**

Run:

```powershell
rg -n "backflow_u|Full U_b|full U_b|refresh_backflow_orbitals!|refresh_backflow_affected_orbital_rows_after_accept!|apply_cached_backflow_orbital_rows!" src Emery.jl test
```

Expected:

- no `backflow_u`;
- no accept-time full cache refresh helpers;
- `refresh_backflow_orbitals!` should be absent or non-backflow-only with a clear error in backflow mode.

- [ ] **Step 2: Run Emery backflow tests**

Run:

```powershell
julia test\emery_grouped_backflow_cleanup_test.jl
julia test\simplify_emery_backflow_api_test.jl
julia test\tmp_emery_directed_backflow_gradient_correctness.jl
```

Expected: all pass.

- [ ] **Step 3: Run non-backflow smoke test**

Run:

```powershell
julia -e 'include("Emery.jl"); println("ok")'
```

Expected: prints `ok`.

- [ ] **Step 4: Check formatting and whitespace**

Run:

```powershell
git diff --check
git status --short
```

Expected:

- `git diff --check` reports no whitespace errors;
- `git status --short` only shows intentional files before the final commit, or is clean after all commits.

---

## Review Checklist For Codex

When reviewing the other agent's implementation, check these points first:

- In active backflow mode, no determinant code reads proposal/current backflow rows from `vwf.gs_U_t`.
- `vwf_det` no longer has a `backflow_u` field.
- `accept_backflow_local_update!` does not refresh affected rows into a global orbital cache after `commit_move!`.
- `rebuild_slater_state!`, `find_stable_config!`, `calc_ratio_rebuild`, and `verify_backflow_local_accept` all use selected-row materialization.
- Non-backflow behavior still uses `gs_U_t` and rank1/rank2 update paths.
- Tests compare selected-row results against full `Backflow.build_backflow_orbitals` truth at least once, so the refactor changes storage strategy without changing math.
