# Emery Backflow Source-Weight Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Emery grouped backflow row construction use one shared source-row weight representation before the future determinant-only optimization.

**Architecture:** A backflow row should first be represented as `(source_row_index, source_weight)` pairs. Full row rebuild, proposal row rebuild, and chain-rule row propagation should then materialize from that same pair list, so eta activation and epsilon activation have one source of truth.

**Tech Stack:** Julia, existing `src/Backflow.jl` grouped Emery backflow code, existing tests in `test/emery_grouped_backflow_cleanup_test.jl` and `test/simplify_emery_backflow_api_test.jl`.

---

## Scope And Non-Goals

This plan covers only the next 1/2/3 refactor steps:

1. Introduce a state-getter based source-weight helper.
2. Make chain-rule row materialization consume that helper.
3. Make full/proposal row materialization consume that helper.

Do not change determinant storage, sampler update strategy, Hubbard paths, parameter names, Emery backflow physics, or epsilon activation semantics in this plan.

Keep the current rule exactly: epsilon contributes only when at least one owned source group has a nonzero actual eta coefficient for the same row and configuration.

## Current Code Anchors

Modify only:

- `src/Backflow.jl`
- `test/emery_grouped_backflow_cleanup_test.jl`

Important existing functions:

- `fill_backflow_site_row_from_state_getter!`
- `add_source_group_eta_contributions_and_track_activation!`
- `add_epsilon_contributions_from_active_groups!`
- `fill_backflow_chain_rule_row!`
- `fill_backflow_chain_rule_source_weights!`
- `add_source_group_chain_rule_source_weights_and_track!`
- `add_backflow_chain_rule_source_weight!`
- `initialize_backflow_chain_rule_source_weights!`

Existing test helper from the previous commit:

- `build_minimal_dd_eta4_backflow`
- `compute_backflow_logabsdet`
- `compute_backflow_param_log_derivative`
- `compute_central_finite_difference_log_derivative`

---

### Task 1: Add The Shared Source-Weight Helper

**Files:**

- Modify: `src/Backflow.jl`
- Modify: `test/emery_grouped_backflow_cleanup_test.jl`

- [ ] **Step 1: Write the failing tests**

Add this testset near the existing eta-driven tests in `test/emery_grouped_backflow_cleanup_test.jl`:

```julia
@testset "Backflow source weights use eta-driven epsilon activation" begin
    source_row_indices = zeros(Int, 4)
    source_row_weights = zeros(Float64, 4)

    active_backflow = build_minimal_dd_eta4_backflow(
        bf_epsilon_d=1.5,
        bf_eta4_dd=0.5,
        dd_amplitude=1.0,
    )
    active_state = Int8[1, 0]
    active_count = mfVMC.Backflow.fill_backflow_row_source_weights_from_state_getter!(
        source_row_indices,
        source_row_weights,
        active_state,
        active_backflow,
        1,
        site_index -> active_state[site_index],
    )
    @test active_count == 2
    @test source_row_indices[1:active_count] == [1, 3]
    @test source_row_weights[1:active_count] ≈ [1.5, 0.5]

    fill!(source_row_indices, 0)
    fill!(source_row_weights, 0.0)
    zero_eta_backflow = build_minimal_dd_eta4_backflow(
        bf_epsilon_d=1.5,
        bf_eta4_dd=0.0,
        dd_amplitude=1.0,
    )
    zero_eta_count = mfVMC.Backflow.fill_backflow_row_source_weights_from_state_getter!(
        source_row_indices,
        source_row_weights,
        active_state,
        zero_eta_backflow,
        1,
        site_index -> active_state[site_index],
    )
    @test zero_eta_count == 1
    @test source_row_indices[1:zero_eta_count] == [1]
    @test source_row_weights[1:zero_eta_count] ≈ [1.0]

    fill!(source_row_indices, 0)
    fill!(source_row_weights, 0.0)
    no_up_state = Int8[2, 0]
    no_up_count = mfVMC.Backflow.fill_backflow_row_source_weights_from_state_getter!(
        source_row_indices,
        source_row_weights,
        no_up_state,
        active_backflow,
        1,
        site_index -> no_up_state[site_index],
    )
    @test no_up_count == 1
    @test source_row_indices[1:no_up_count] == [1]
    @test source_row_weights[1:no_up_count] ≈ [1.0]
end
```

- [ ] **Step 2: Run the focused test and verify it fails because the helper is missing**

Run:

```bash
julia test\emery_grouped_backflow_cleanup_test.jl
```

Expected: failure mentioning `fill_backflow_row_source_weights_from_state_getter!` is not defined.

- [ ] **Step 3: Implement the helper**

In `src/Backflow.jl`, add this function near `fill_backflow_chain_rule_source_weights!`, reusing existing source-weight helpers:

```julia
"""
用途: 根据给定 state getter 将一个 backflow output row 展开为 source row 权重列表。

参数:
- `source_row_indices::AbstractVector{Int}`: 输出 source row 编号 buffer, 长度至少为 `2 * length(state_vector)`。
- `source_row_weights::AbstractVector{T}`: 输出 source row 权重 buffer, 与 `source_row_indices` 对齐。
- `state_vector::Vector{Int8}`: 当前 Monte Carlo 构型, 用于确定总 site 数和 row 边界。
- `backflow_term::CompositeBackflowTerm`: Emery grouped backflow 对象。
- `row_index::Int`: 需要展开的 spin-resolved 全局行号。
- `get_state::Function`: `(site_index) -> Int8`, 用于读取当前或 proposal 后的 site 状态。

返回:
- `Int`: 有效 source row 数量。

公式:
- `U_b(row,:) = sum_s w_s * U_0(s,:)`。
- 初始恒等项为 `w_row = 1`。
- eta 项对 target row 增加 `t_ij * eta_k * f_k(x)`。
- epsilon 项只在对应 group 存在非零 eta contribution 时对本 row 增加 `epsilon_bf - 1`。
"""
function fill_backflow_row_source_weights_from_state_getter!(
    source_row_indices::AbstractVector{Int},
    source_row_weights::AbstractVector{T},
    state_vector::Vector{Int8},
    backflow_term::CompositeBackflowTerm,
    row_index::Int,
    get_state::Function,
)::Int where {T}
    site_index, row_offset, source_count = initialize_backflow_chain_rule_source_weights!(
        source_row_indices,
        source_row_weights,
        state_vector,
        row_index,
    )
    state_i = get_state(site_index)
    spin = backflow_spin_from_row_offset(row_offset)

    if backflow_n_sigma(state_i, spin) == 0.0
        return source_count
    end

    active_group_names = Set{Symbol}()
    for source_group in backflow_term.source_groups
        source_count = add_source_group_chain_rule_source_weights_and_track!(
            source_row_indices,
            source_row_weights,
            source_count,
            state_i,
            get_state,
            source_group,
            site_index,
            row_offset,
            active_group_names,
        )
    end

    for epsilon_term in backflow_term.epsilon_terms
        epsilon_shift = T(epsilon_term.epsilon_bf - 1.0)
        if epsilon_shift == zero(T)
            continue
        end
        for group_name in epsilon_term.group_names
            if group_name in active_group_names
                source_count = add_backflow_chain_rule_source_weight!(
                    source_row_indices,
                    source_row_weights,
                    source_count,
                    row_index,
                    epsilon_shift,
                )
                break
            end
        end
    end

    return source_count
end
```

Change `add_source_group_chain_rule_source_weights_and_track!` so it accepts a state getter instead of hard-coding `state_vector[target_site]`:

```julia
function add_source_group_chain_rule_source_weights_and_track!(
    source_row_indices::AbstractVector{Int},
    source_row_weights::AbstractVector{T},
    source_count::Int,
    state_i::Int8,
    get_state_j::Function,
    source_group::DirectedBackflowSourceGroup,
    site_index::Int,
    row_offset::Int,
    active_group_names::Set{Symbol},
)::Int where {T}
    if site_index > length(source_group.outgoing_bond_indices_by_source)
        return source_count
    end

    has_eta = false
    spin = backflow_spin_from_row_offset(row_offset)
    eta1_value = T(source_group.eta1_term.eta1_bf)
    eta2_value = T(source_group.eta2_term.eta2_bf)
    eta3_value = T(source_group.eta3_term.eta3_bf)
    eta4_value = T(source_group.eta4_term.eta4_bf)

    for bond_index in source_group.outgoing_bond_indices_by_source[site_index]
        (_, target_site) = source_group.source_bonds[bond_index]
        state_j = get_state_j(target_site)
        target_row = 2 * (target_site - 1) + row_offset
        bond_amplitude = T(source_group.source_amplitudes[bond_index])
        eta_contribution = compute_backflow_eta_contribution(
            state_i,
            state_j,
            spin,
            bond_amplitude,
            eta1_value,
            eta2_value,
            eta3_value,
            eta4_value,
        )
        coefficient = eta_contribution.coefficient
        if coefficient == zero(T)
            continue
        end

        source_count = add_backflow_chain_rule_source_weight!(
            source_row_indices,
            source_row_weights,
            source_count,
            target_row,
            coefficient,
        )
        has_eta = true
    end

    if has_eta
        push!(active_group_names, source_group.group_name)
    end

    return source_count
end
```

- [ ] **Step 4: Update the current-state wrapper**

Replace the body of `fill_backflow_chain_rule_source_weights!(..., backflow_term::CompositeBackflowTerm, row_index)` with:

```julia
return fill_backflow_row_source_weights_from_state_getter!(
    source_row_indices,
    source_row_weights,
    state_vector,
    backflow_term,
    row_index,
    site_index -> state_vector[site_index],
)
```

- [ ] **Step 5: Run tests**

Run:

```bash
julia test\emery_grouped_backflow_cleanup_test.jl
julia test\simplify_emery_backflow_api_test.jl
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src\Backflow.jl test\emery_grouped_backflow_cleanup_test.jl
git commit -m "refactor: add shared backflow source weight helper"
```

---

### Task 2: Materialize Chain-Rule Rows From Source Weights

**Files:**

- Modify: `src/Backflow.jl`
- Modify: `test/emery_grouped_backflow_cleanup_test.jl`

- [ ] **Step 1: Write the failing equivalence test**

Add this testset:

```julia
@testset "Backflow chain-rule row materializes from source weights" begin
    input_derivative_orbitals = [
        2.0 0.2
        3.0 0.3
        5.0 0.5
        7.0 0.7
    ]
    state_up_hole = Int8[1, 0]
    backflow = build_minimal_dd_eta4_backflow(
        bf_epsilon_d=1.5,
        bf_eta4_dd=0.5,
        dd_amplitude=1.0,
    )

    source_row_indices = zeros(Int, 4)
    source_row_weights = zeros(Float64, 4)
    source_count = mfVMC.Backflow.fill_backflow_row_source_weights_from_state_getter!(
        source_row_indices,
        source_row_weights,
        state_up_hole,
        backflow,
        1,
        site_index -> state_up_hole[site_index],
    )

    expected_row = zeros(Float64, size(input_derivative_orbitals, 2))
    mfVMC.Backflow.fill_backflow_row_from_source_weights!(
        expected_row,
        input_derivative_orbitals,
        source_row_indices,
        source_row_weights,
        source_count,
    )

    actual_row = zeros(Float64, size(input_derivative_orbitals, 2))
    mfVMC.Backflow.fill_backflow_chain_rule_row!(
        actual_row,
        input_derivative_orbitals,
        state_up_hole,
        backflow,
        1,
    )

    @test actual_row ≈ expected_row
    @test actual_row ≈ 1.5 .* input_derivative_orbitals[1, :] .+ 0.5 .* input_derivative_orbitals[3, :]
end
```

- [ ] **Step 2: Run the focused test and verify it fails because `fill_backflow_row_from_source_weights!` is missing**

Run:

```bash
julia test\emery_grouped_backflow_cleanup_test.jl
```

Expected: failure mentioning `fill_backflow_row_from_source_weights!` is not defined.

- [ ] **Step 3: Implement row materialization**

Add this helper near the source-weight helper:

```julia
"""
用途: 根据 source row 权重列表 materialize 一个 backflow row。

参数:
- `output_row::AbstractVector{T}`: 输出 buffer, 长度等于 orbital 列数。
- `input_orbitals::AbstractMatrix{T}`: 被线性组合的输入轨道矩阵, 可以是 `U_0` 或 `dU_0/dp`。
- `source_row_indices::AbstractVector{Int}`: source row 编号 buffer。
- `source_row_weights::AbstractVector{T}`: source row 权重 buffer。
- `source_count::Int`: 有效 source row 数量。

返回:
- `nothing`。

公式:
- `output_row[:] = sum_{k=1}^{source_count} source_row_weights[k] * input_orbitals[source_row_indices[k], :]`。
"""
function fill_backflow_row_from_source_weights!(
    output_row::AbstractVector{T},
    input_orbitals::AbstractMatrix{T},
    source_row_indices::AbstractVector{Int},
    source_row_weights::AbstractVector{T},
    source_count::Int,
) where {T}
    fill!(output_row, zero(T))
    for source_offset in 1:source_count
        source_row_index = source_row_indices[source_offset]
        source_weight = source_row_weights[source_offset]
        @views output_row .+= source_weight .* input_orbitals[source_row_index, :]
    end
    return nothing
end
```

- [ ] **Step 4: Rewrite `fill_backflow_chain_rule_row!` for `CompositeBackflowTerm`**

Replace the body of `fill_backflow_chain_rule_row!(output_row, input_derivative_orbitals, state_vector, backflow_term::CompositeBackflowTerm, row_index)` with source weights:

```julia
row_count = 2 * length(state_vector)
source_row_indices = Vector{Int}(undef, row_count)
source_row_weights = Vector{T}(undef, row_count)
source_count = fill_backflow_row_source_weights_from_state_getter!(
    source_row_indices,
    source_row_weights,
    state_vector,
    backflow_term,
    row_index,
    site_index -> state_vector[site_index],
)
fill_backflow_row_from_source_weights!(
    output_row,
    input_derivative_orbitals,
    source_row_indices,
    source_row_weights,
    source_count,
)
return nothing
```

Keep the existing `NoBackflowTerm` method unchanged.

- [ ] **Step 5: Remove now-unused chain-rule duplication if it becomes unused**

After the rewrite, run:

```bash
rg "add_source_group_eta_contributions_and_track_activation!|add_epsilon_contributions_from_active_groups!" src\Backflow.jl
```

If these functions are still used by full/proposal row construction, keep them for Task 3. If one is only used by deleted chain-rule logic but still needed elsewhere, keep it.

- [ ] **Step 6: Run tests**

Run:

```bash
julia test\emery_grouped_backflow_cleanup_test.jl
julia test\simplify_emery_backflow_api_test.jl
julia test\backflow_chain_rule_selected_rows_test.jl
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src\Backflow.jl test\emery_grouped_backflow_cleanup_test.jl
git commit -m "refactor: materialize backflow chain rule from source weights"
```

---

### Task 3: Materialize Full And Proposal Rows From Source Weights

**Files:**

- Modify: `src/Backflow.jl`
- Modify: `test/emery_grouped_backflow_cleanup_test.jl`

- [ ] **Step 1: Write the failing proposal/full equivalence test**

Add this testset:

```julia
@testset "Backflow full and proposal rows materialize from source weights" begin
    base_orbitals = [
        2.0 0.2
        3.0 0.3
        5.0 0.5
        7.0 0.7
    ]
    state_vector = Int8[1, 0]
    backflow = build_minimal_dd_eta4_backflow(
        bf_epsilon_d=1.5,
        bf_eta4_dd=0.5,
        dd_amplitude=1.0,
    )

    source_row_indices = zeros(Int, 4)
    source_row_weights = zeros(Float64, 4)
    source_count = mfVMC.Backflow.fill_backflow_row_source_weights_from_state_getter!(
        source_row_indices,
        source_row_weights,
        state_vector,
        backflow,
        1,
        site_index -> state_vector[site_index],
    )
    expected_row = zeros(Float64, size(base_orbitals, 2))
    mfVMC.Backflow.fill_backflow_row_from_source_weights!(
        expected_row,
        base_orbitals,
        source_row_indices,
        source_row_weights,
        source_count,
    )

    full_orbitals = mfVMC.build_backflow_orbitals(base_orbitals, state_vector, backflow)
    @test full_orbitals[1, :] ≈ expected_row

    proposal = MoveProposal(
        1, 2,
        Int8(1), Int8(1),
        Int8(0), Int8(0),
        0, 0, 0, 0, 0, 0, 0,
    )
    proposal_row = zeros(Float64, size(base_orbitals, 2))
    mfVMC.Backflow.fill_grouped_source_composite_site_row_after_proposal!(
        proposal_row,
        base_orbitals,
        state_vector,
        backflow,
        proposal,
        1,
        1,
    )
    @test proposal_row ≈ expected_row
end
```

This test may pass before implementation because behavior already agrees. Keep it anyway because it protects the refactor.

- [ ] **Step 2: Rewrite `fill_backflow_site_row_from_state_getter!`**

Replace the eta/epsilon-specific body with the source-weight helper and materializer:

```julia
row_index = 2 * (site_index - 1) + row_offset
row_count = size(base_orbitals, 1)
source_row_indices = Vector{Int}(undef, row_count)
source_row_weights = Vector{T}(undef, row_count)
state_vector_for_bounds = Vector{Int8}(undef, row_count ÷ 2)
source_count = fill_backflow_row_source_weights_from_state_getter!(
    source_row_indices,
    source_row_weights,
    state_vector_for_bounds,
    backflow_term,
    row_index,
    get_state,
)
fill_backflow_row_from_source_weights!(
    output_row,
    base_orbitals,
    source_row_indices,
    source_row_weights,
    source_count,
)
return row_index
```

The temporary `state_vector_for_bounds` is acceptable in this step because `initialize_backflow_chain_rule_source_weights!` only needs the state-vector length for bounds checks. Do not change public signatures in this task.

- [ ] **Step 3: Remove obsolete full/proposal eta materializers if unused**

Run:

```bash
rg "add_source_group_eta_contributions_and_track_activation!|add_epsilon_contributions_from_active_groups!" src\Backflow.jl
```

If both helpers have no call sites after Step 2, delete both functions and remove them from cleanup tests that assert they exist. Add cleanup tests asserting they are no longer defined:

```julia
@test !isdefined(mfVMC.Backflow, Symbol("add_source_group_eta_contributions_and_track_activation!"))
@test !isdefined(mfVMC.Backflow, Symbol("add_epsilon_contributions_from_active_groups!"))
```

Keep `compute_backflow_eta_contribution` because it remains the shared scalar eta coefficient helper.

- [ ] **Step 4: Run tests**

Run:

```bash
julia test\emery_grouped_backflow_cleanup_test.jl
julia test\simplify_emery_backflow_api_test.jl
julia test\backflow_chain_rule_selected_rows_test.jl
julia test\tmp_emery_directed_backflow_gradient_correctness.jl
julia -e 'include("Emery.jl"); println("ok")'
git diff --check
```

Expected: all Julia commands exit with code 0 and `git diff --check` reports no whitespace errors.

- [ ] **Step 5: Commit**

```bash
git add src\Backflow.jl test\emery_grouped_backflow_cleanup_test.jl
git commit -m "refactor: materialize Emery backflow rows from source weights"
```

---

## Review Checklist For Codex After The Other Agent Finishes

- Confirm `fill_backflow_row_source_weights_from_state_getter!` is the only place that decides eta-driven epsilon activation for row construction.
- Confirm `fill_backflow_chain_rule_source_weights!`, `fill_backflow_chain_rule_row!`, and `fill_backflow_site_row_from_state_getter!` agree on the same source rows and weights.
- Confirm no Hubbard backflow code was migrated or rewritten.
- Confirm determinant gradient tests still pass, especially the default point `epsilon=1, eta=0`.
- Look for avoidable allocations introduced in hot paths; accept them during this refactor only if behavior is correct and the next optimization step will address buffers.
