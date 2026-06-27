# Twist Hubbard PH Backflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add physically correct PH-aware lattice backflow to `twist_Hubbard_PH.jl` using an explicit `particle_hole_lower_block` flag.

**Architecture:** Keep the existing `CompositeBackflowTerm` as the shared backflow type, and add a boolean flag that changes only the interpretation of the lower block. The core source-weight and derivative paths must use the same PH/nonPH helper logic so determinant ratios and SR derivatives stay consistent.

**Tech Stack:** Julia, existing `mfVMC` modules, `ArgParse`, `Test`, git worktree `D:/study/研究生/科研/VMC/HKJ_s/mfVMC/.worktrees/twist-hubbard-ph`.

---

## File Structure

- Modify `src/Backflow.jl`: add the PH flag to `CompositeBackflowTerm`; add helper functions for row occupation guard and eta contribution under PH lower block; update source-weight and derivative-orbital paths.
- Modify `twist_Hubbard.jl`: allow `build_twist_composite_backflow` and `build_twist_optional_backflow` to forward `particle_hole_lower_block=false` by default.
- Modify `twist_Hubbard_PH.jl`: expose backflow command-line parameters, construct PH-mode backflow, include backflow parameters in SR parameter vectors, and update backflow parameters in `update_twist_ph_ansatz!`.
- Create `test/twist_hubbard_ph_backflow_test.jl`: focused PH backflow unit tests and small PH driver smoke tests.
- Modify existing tests only when parameter lists or constructor signatures require explicit updates.

---

### Task 1: Backflow PH Flag and Source-Weight Rules

**Files:**
- Modify: `src/Backflow.jl`
- Create: `test/twist_hubbard_ph_backflow_test.jl`

- [ ] **Step 1: Write the failing PH eta1 source-weight tests**

Create `test/twist_hubbard_ph_backflow_test.jl` with this initial content:

```julia
using Test

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using mfVMC
using mfVMC.Sampler
using mfVMC.Backflow

function build_test_backflow(; particle_hole_lower_block::Bool)
    source_bonds = [(1, 2), (2, 1)]
    source_amplitudes = [1.0, 1.0]
    group = mfVMC.Backflow.build_directed_backflow_source_group(
        :hubbard,
        source_bonds,
        source_amplitudes,
        BackflowEta1DoublonHoleTerm(param_name=:bf_eta1, eta1_bf=0.7),
        BackflowEta2SpinExchangeTerm(param_name=:bf_eta2, eta2_bf=0.0),
        BackflowEta3DoublonSingleTerm(param_name=:bf_eta3, eta3_bf=0.0),
        BackflowEta4SingleHoleTerm(param_name=:bf_eta4, eta4_bf=0.0),
    )
    return CompositeBackflowTerm(
        [BackflowEpsilonTerm(param_name=:bf_epsilon, epsilon_bf=1.0, group_names=Symbol[:hubbard])],
        [group];
        particle_hole_lower_block=particle_hole_lower_block,
    )
end

function collect_source_weight_map(state_vector, backflow_term, row_index)
    source_rows = Vector{Int}(undef, 2 * length(state_vector))
    source_weights = Vector{Float64}(undef, 2 * length(state_vector))
    source_count = mfVMC.Backflow.fill_backflow_chain_rule_source_weights!(
        source_rows,
        source_weights,
        state_vector,
        backflow_term,
        row_index,
    )
    return Dict(source_rows[index] => source_weights[index] for index in 1:source_count)
end

@testset "PH backflow eta1 uses down-hole lower block" begin
    state_vector = Int8[HOLE, DB]
    ph_backflow = build_test_backflow(particle_hole_lower_block=true)

    lower_row_site1 = 2
    lower_row_site2 = 4
    weights = collect_source_weight_map(state_vector, ph_backflow, lower_row_site1)

    @test weights[lower_row_site1] ≈ 1.0
    @test weights[lower_row_site2] ≈ 0.7
end

@testset "nonPH backflow keeps down-electron eta1 direction" begin
    state_vector = Int8[DB, HOLE]
    nonph_backflow = build_test_backflow(particle_hole_lower_block=false)

    lower_row_site1 = 2
    lower_row_site2 = 4
    weights = collect_source_weight_map(state_vector, nonph_backflow, lower_row_site1)

    @test weights[lower_row_site1] ≈ 1.0
    @test weights[lower_row_site2] ≈ 0.7
end
```

- [ ] **Step 2: Run the new test and verify it fails because the keyword flag is missing**

Run:

```powershell
julia test\twist_hubbard_ph_backflow_test.jl
```

Expected result:

```text
ERROR: MethodError: no method matching CompositeBackflowTerm(...; particle_hole_lower_block=true)
```

- [ ] **Step 3: Add `particle_hole_lower_block` to `CompositeBackflowTerm`**

In `src/Backflow.jl`, update the struct and inner constructor to this shape:

```julia
struct CompositeBackflowTerm <: AbstractBackflowTerm
    epsilon_terms::Vector{BackflowEpsilonTerm}
    source_groups::Vector{DirectedBackflowSourceGroup}
    terms::Vector{AbstractBackflowCorrectionTerm}
    incoming_source_sites_by_target::Vector{Vector{Int}}
    particle_hole_lower_block::Bool

    function CompositeBackflowTerm(
        epsilon_terms::Vector{BackflowEpsilonTerm},
        source_groups::Vector{DirectedBackflowSourceGroup};
        particle_hole_lower_block::Bool=false,
    )
        term_list = AbstractBackflowCorrectionTerm[]
        for epsilon_term in epsilon_terms
            push!(term_list, epsilon_term)
        end
        for source_group in source_groups
            push!(term_list, source_group.eta1_term)
            push!(term_list, source_group.eta2_term)
            push!(term_list, source_group.eta3_term)
            push!(term_list, source_group.eta4_term)
        end
        return new(
            epsilon_terms,
            source_groups,
            term_list,
            build_composite_incoming_from_groups(source_groups),
            particle_hole_lower_block,
        )
    end
end
```

Add or update the Chinese docstring for `CompositeBackflowTerm` fields:

```julia
- `particle_hole_lower_block::Bool`: 若为 `true`, lower row 表示 PH 表象的 down hole;
  若为 `false`, lower row 表示 nonPH 表象的 down electron。该 flag 只改变 lower block
  的 occupation guard 与 eta 因子方向, 不改变 source graph 和参数顺序。
```

- [ ] **Step 4: Add PH helper functions for row interpretation**

In `src/Backflow.jl`, near `backflow_n_sigma`, add:

```julia
"""
用途: 判断当前 row 是否是 PH 表象中的 lower block down-hole row。

参数:
- `backflow_term::CompositeBackflowTerm`: backflow 对象, 其中包含 PH lower-block flag。
- `row_offset::Int`: site 内部行偏移, `1` 为 upper row, `2` 为 lower row。

返回:
- `Bool`: 当 backflow 使用 PH lower block 且 row_offset 为 `2` 时返回 `true`。
"""
function is_particle_hole_lower_row(
    backflow_term::CompositeBackflowTerm,
    row_offset::Int,
)::Bool
    return backflow_term.particle_hole_lower_block && row_offset == 2
end

"""
用途: 计算 backflow output row 在当前物理电子构型中是否被占据。

数学公式:
- nonPH 或 upper row: `n_{i,sigma}`。
- PH lower row: `h_{i,down} = 1 - n_{i,down}`。

参数:
- `state_code::Int8`: site 的原始电子构型编码。
- `spin::Int8`: 当前 row 对应的自旋标签, lower row 仍传入 `DN`。
- `is_ph_lower_row::Bool`: 是否按 PH down-hole 解释 lower row。

返回:
- `Float64`: 取值为 `0.0` 或 `1.0`。
"""
function backflow_row_occupation_factor(
    state_code::Int8,
    spin::Int8,
    is_ph_lower_row::Bool,
)::Float64
    if is_ph_lower_row
        return backflow_h_sigma(state_code, DN)
    end
    return backflow_n_sigma(state_code, spin)
end

"""
用途: 根据 PH/nonPH row 解释计算单条 source bond 的 eta contribution。

数学公式:
- nonPH 或 upper row 使用 `f(state_i, state_j, sigma)`。
- PH lower row 使用 swapped-sites 规则 `f(state_j, state_i, DN)`, 例如 eta1 从
  `D_i H_j` 变为 `H_i D_j`。

参数:
- `state_i, state_j::Int8`: source site 与 target site 的原始电子构型。
- `spin::Int8`: 当前 row 的自旋标签。
- `bond_amplitude::T`: source bond 振幅。
- `eta1_value, eta2_value, eta3_value, eta4_value::T`: backflow 参数值。
- `is_ph_lower_row::Bool`: 是否按 PH lower block 解释。

返回:
- `NamedTuple`: 与 `compute_backflow_eta_contribution` 相同。
"""
function compute_backflow_eta_contribution_for_row(
    state_i::Int8,
    state_j::Int8,
    spin::Int8,
    bond_amplitude::T,
    eta1_value::T,
    eta2_value::T,
    eta3_value::T,
    eta4_value::T,
    is_ph_lower_row::Bool,
) where {T}
    if is_ph_lower_row
        return compute_backflow_eta_contribution(
            state_j,
            state_i,
            DN,
            bond_amplitude,
            eta1_value,
            eta2_value,
            eta3_value,
            eta4_value,
        )
    end
    return compute_backflow_eta_contribution(
        state_i,
        state_j,
        spin,
        bond_amplitude,
        eta1_value,
        eta2_value,
        eta3_value,
        eta4_value,
    )
end
```

- [ ] **Step 5: Use the helpers in source-weight construction**

Change `add_source_group_chain_rule_source_weights_and_track!` signature from:

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
```

to:

```julia
function add_source_group_chain_rule_source_weights_and_track!(
    source_row_indices::AbstractVector{Int},
    source_row_weights::AbstractVector{T},
    source_count::Int,
    state_i::Int8,
    get_state_j::Function,
    backflow_term::CompositeBackflowTerm,
    source_group::DirectedBackflowSourceGroup,
    site_index::Int,
    row_offset::Int,
    active_group_names::Set{Symbol},
)::Int where {T}
```

Inside this function, after `spin = backflow_spin_from_row_offset(row_offset)`, add:

```julia
is_ph_lower_row = is_particle_hole_lower_row(backflow_term, row_offset)
```

Then replace the eta calculation with:

```julia
eta_contribution = compute_backflow_eta_contribution_for_row(
    state_i,
    state_j,
    spin,
    bond_amplitude,
    eta1_value,
    eta2_value,
    eta3_value,
    eta4_value,
    is_ph_lower_row,
)
```

In `fill_backflow_row_source_weights_from_state_getter!`, replace:

```julia
if backflow_n_sigma(state_i, spin) == 0.0
    return source_count
end
```

with:

```julia
is_ph_lower_row = is_particle_hole_lower_row(backflow_term, row_offset)
if backflow_row_occupation_factor(state_i, spin, is_ph_lower_row) == 0.0
    return source_count
end
```

In the same function, update the call to `add_source_group_chain_rule_source_weights_and_track!` so it passes `backflow_term`:

```julia
source_count = add_source_group_chain_rule_source_weights_and_track!(
    source_row_indices,
    source_row_weights,
    source_count,
    state_i,
    get_state,
    backflow_term,
    source_group,
    site_index,
    row_offset,
    active_group_names,
)
```

- [ ] **Step 6: Run the new test and verify Task 1 passes**

Run:

```powershell
julia test\twist_hubbard_ph_backflow_test.jl
```

Expected result:

```text
Test Summary: ... PH backflow eta1 ...
Test Summary: ... nonPH backflow ...
```

- [ ] **Step 7: Commit Task 1**

Run:

```powershell
git add src\Backflow.jl test\twist_hubbard_ph_backflow_test.jl
git commit -m "feat: add PH lower-block backflow flag"
```

---

### Task 2: PH Backflow Derivative Orbitals

**Files:**
- Modify: `src/Backflow.jl`
- Modify: `test/twist_hubbard_ph_backflow_test.jl`

- [ ] **Step 1: Add failing derivative-orbital tests**

Append to `test/twist_hubbard_ph_backflow_test.jl`:

```julia
@testset "PH backflow eta1 derivative uses down-hole direction" begin
    state_vector = Int8[HOLE, DB]
    ph_backflow = build_test_backflow(particle_hole_lower_block=true)
    base_orbitals = reshape(collect(1.0:16.0), 4, 4)

    derivative_pairs = mfVMC.Backflow.build_backflow_derivative_orbitals(
        base_orbitals,
        state_vector,
        ph_backflow,
    )
    derivative_map = Dict(first(pair) => last(pair) for pair in derivative_pairs)

    lower_row_site1 = 2
    lower_row_site2 = 4
    @test derivative_map[:bf_eta1][lower_row_site1, :] ≈ base_orbitals[lower_row_site2, :]
end
```

- [ ] **Step 2: Run the test and verify it fails on derivative direction**

Run:

```powershell
julia test\twist_hubbard_ph_backflow_test.jl
```

Expected failure:

```text
Test Failed
Expression: derivative_map[:bf_eta1][lower_row_site1, :] ≈ base_orbitals[lower_row_site2, :]
```

- [ ] **Step 3: Update `build_backflow_derivative_orbitals` to use PH helper logic**

Inside the loop:

```julia
for row_offset in 1:2
    spin = backflow_spin_from_row_offset(row_offset)
    row_i = 2 * (site_i - 1) + row_offset
    row_j = 2 * (site_j - 1) + row_offset
```

add:

```julia
is_ph_lower_row = is_particle_hole_lower_row(backflow_term, row_offset)
if backflow_row_occupation_factor(state_i, spin, is_ph_lower_row) == zero(T)
    continue
end
```

Replace `compute_backflow_eta_contribution(...)` with `compute_backflow_eta_contribution_for_row(...)`.

Keep all derivative writes to `row_i` and `row_j` unchanged:

```julia
@views eta1_deriv[row_i, :] .+= bond_amplitude .* base_orbitals[row_j, :]
```

This is intentional: PH lower block changes the factor direction, not the lower-block row index layout.

- [ ] **Step 4: Run the focused test**

Run:

```powershell
julia test\twist_hubbard_ph_backflow_test.jl
```

Expected: all tests in the new file pass.

- [ ] **Step 5: Run existing nonPH backflow regression tests**

Run:

```powershell
julia test\backflow_composite_terms_test.jl
julia test\emery_grouped_backflow_cleanup_test.jl
```

Expected: both pass, proving default `particle_hole_lower_block=false` did not change existing behavior.

- [ ] **Step 6: Commit Task 2**

Run:

```powershell
git add src\Backflow.jl test\twist_hubbard_ph_backflow_test.jl
git commit -m "fix: align PH backflow parameter derivatives"
```

---

### Task 3: Connect PH Backflow to `twist_Hubbard_PH.jl`

**Files:**
- Modify: `twist_Hubbard.jl`
- Modify: `twist_Hubbard_PH.jl`
- Modify: `test/twist_hubbard_ph_backflow_test.jl`

- [ ] **Step 1: Add failing PH driver setup test**

Append to `test/twist_hubbard_ph_backflow_test.jl`:

```julia
include(joinpath(@__DIR__, "..", "twist_Hubbard_PH.jl"))

@testset "twist Hubbard PH constructs PH-mode backflow" begin
    hopping_bonds = build_twist_nearest_neighbor_bonds(2, 2)
    source_bonds, source_amplitudes = build_twist_backflow_source_data(hopping_bonds, 1.0, 1.0, 0.0)
    backflow = build_twist_optional_backflow(
        true,
        source_bonds,
        source_amplitudes,
        1.0,
        0.2,
        0.0,
        0.0,
        0.0;
        particle_hole_lower_block=true,
    )

    @test mfVMC.Backflow.uses_backflow(backflow)
    @test backflow.particle_hole_lower_block
    @test mfVMC.Backflow.backflow_param_names(backflow) == [:bf_epsilon, :bf_eta1, :bf_eta2, :bf_eta3, :bf_eta4]
end
```

- [ ] **Step 2: Run the test and verify keyword forwarding fails**

Run:

```powershell
julia test\twist_hubbard_ph_backflow_test.jl
```

Expected:

```text
ERROR: MethodError: no method matching build_twist_optional_backflow(...; particle_hole_lower_block=true)
```

- [ ] **Step 3: Add keyword forwarding in `twist_Hubbard.jl`**

Modify `build_twist_composite_backflow` signature:

```julia
function build_twist_composite_backflow(
    source_bonds::Vector{Tuple{Int,Int}},
    source_amplitudes::Vector{Float64},
    bf_epsilon::Float64,
    bf_eta1::Float64,
    bf_eta2::Float64,
    bf_eta3::Float64,
    bf_eta4::Float64;
    particle_hole_lower_block::Bool=false,
)
```

Modify its return:

```julia
return CompositeBackflowTerm(
    epsilon_terms,
    [hubbard_group];
    particle_hole_lower_block=particle_hole_lower_block,
)
```

Modify `build_twist_optional_backflow` signature similarly and forward the keyword to `build_twist_composite_backflow`.

Add Chinese docstring lines explaining:

```julia
- `particle_hole_lower_block::Bool`: 是否将 lower block 解释为 PH down-hole row。
```

- [ ] **Step 4: Add PH command-line and parameter plumbing**

In `twist_Hubbard_PH.jl`, add command-line options:

```julia
"--enable_backflow"
help = "Enable twist Hubbard PH backflow"
arg_type = String
default = "true"
"--bf_epsilon"
help = "Backflow epsilon initial value"
arg_type = Float64
default = 1.0
"--bf_eta1"
help = "Backflow eta1 initial value"
arg_type = Float64
default = 0.0
"--bf_eta2"
help = "Backflow eta2 initial value"
arg_type = Float64
default = 0.0
"--bf_eta3"
help = "Backflow eta3 initial value"
arg_type = Float64
default = 0.0
"--bf_eta4"
help = "Backflow eta4 initial value"
arg_type = Float64
default = 0.0
```

In `main_twist_ph`, replace:

```julia
backflow = NoBackflowTerm()
```

with:

```julia
hopping_bonds = build_twist_nearest_neighbor_bonds(lx, ly)
source_bonds, source_amplitudes = build_twist_backflow_source_data(hopping_bonds, tx, ty, t2)
backflow = build_twist_optional_backflow(
    parse_twist_bool_flag(args["enable_backflow"], "--enable_backflow"),
    source_bonds,
    source_amplitudes,
    args["bf_epsilon"],
    args["bf_eta1"],
    args["bf_eta2"],
    args["bf_eta3"],
    args["bf_eta4"];
    particle_hole_lower_block=true,
)
```

Add:

```julia
backflow_param_name_list = backflow_param_names(backflow)
backflow_init_params = backflow_param_values(backflow)
nparams_backflow = length(backflow_param_name_list)
init_params = vcat(wf_init_params, proj_init_params, backflow_init_params)
param_names = vcat(wf_param_names, proj_param_names, backflow_param_name_list)
```

Change `update_twist_ph_ansatz!`:

- add keyword `nparams_backflow::Int=0`;
- split `backflow_param_names` and `backflow_param_values` after projector params;
- call `update_vwf_backflow_params!` when nonempty;
- update the docstring to explain parameter order `wf, projector, backflow`.

In `main_twist_ph`, compute active backflow params:

```julia
backflow_param_name_set = Set(backflow_param_name_list)
active_backflow_param_names = [name for name in sr_param_names if name in backflow_param_name_set]
set_active_twist_backflow_derivative_param_names!(
    backflow_param_name_list;
    active_backflow_param_names=uses_param_subset ? active_backflow_param_names : nothing,
)
```

Forward `nparams_backflow=nparams_backflow` in both `update_twist_ph_ansatz!` calls.

Print:

```julia
println("Backflow enabled: $(mfVMC.Backflow.uses_backflow(backflow))")
```

- [ ] **Step 5: Run the focused PH backflow setup test**

Run:

```powershell
julia test\twist_hubbard_ph_backflow_test.jl
```

Expected: all focused PH backflow tests pass.

- [ ] **Step 6: Commit Task 3**

Run:

```powershell
git add twist_Hubbard.jl twist_Hubbard_PH.jl test\twist_hubbard_ph_backflow_test.jl
git commit -m "feat: enable twist Hubbard PH backflow"
```

---

### Task 4: PH Backflow Degeneration and Ratio Verification

**Files:**
- Modify: `test/twist_hubbard_ph_backflow_test.jl`

- [ ] **Step 1: Add zero-backflow degeneration test**

Append:

```julia
@testset "PH backflow zero parameters degenerates to identity rows" begin
    hopping_bonds = build_twist_nearest_neighbor_bonds(2, 2)
    source_bonds, source_amplitudes = build_twist_backflow_source_data(hopping_bonds, 1.0, 1.0, 0.0)
    backflow = build_twist_optional_backflow(
        true,
        source_bonds,
        source_amplitudes,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0;
        particle_hole_lower_block=true,
    )
    state_vector = Int8[DB, HOLE, UP, DN]

    for row_index in 1:(2 * length(state_vector))
        weights = collect_source_weight_map(state_vector, backflow, row_index)
        @test length(weights) == 1
        @test weights[row_index] ≈ 1.0
    end
end
```

- [ ] **Step 2: Run and verify the degeneration test**

Run:

```powershell
julia test\twist_hubbard_ph_backflow_test.jl
```

Expected: pass.

- [ ] **Step 3: Add PH local ratio fast-vs-rebuild smoke test**

Append a small test that constructs a `vwf_det` with PH sampler, PH backflow, `backflow_debug_verify=true`, and one legal hop. Use existing exported APIs:

```julia
@testset "PH backflow local ratio matches rebuild" begin
    lx = 2
    ly = 2
    n_sites = lx * ly
    sampler = config_Hubbard(n_sites, 2, 2; ifPH=true)
    sampler.state .= Int8[DB, HOLE, UP, DN]
    initialize_lists!(sampler)

    hopping_bonds = build_twist_nearest_neighbor_bonds(lx, ly)
    source_bonds, source_amplitudes = build_twist_backflow_source_data(hopping_bonds, 1.0, 1.0, 0.0)
    backflow = build_twist_optional_backflow(
        true,
        source_bonds,
        source_amplitudes,
        1.0,
        0.2,
        0.0,
        0.0,
        0.0;
        particle_hole_lower_block=true,
    )

    base_orbitals = [
        1.0 1.0 1.0 1.0
        1.0 2.0 4.0 8.0
        1.0 3.0 9.0 27.0
        1.0 4.0 16.0 64.0
        1.0 5.0 25.0 125.0
        1.0 6.0 36.0 216.0
        1.0 7.0 49.0 343.0
        1.0 8.0 64.0 512.0
    ]
    vwf = mfVMC.VMC.vwf_det(base_orbitals, sampler; backflow=backflow)
    vwf.backflow_debug_verify = true
    mfVMC.VMC.init_gswf!(vwf)

    proposal = build_single_hop(sampler, 1, 2, UP)
    fast_ratio = mfVMC.VMC.calc_backflow_ratio_local_update(vwf, proposal)
    rebuild_ratio = mfVMC.VMC.calc_ratio_rebuild(vwf, proposal)

    @test fast_ratio ≈ rebuild_ratio
end
```

- [ ] **Step 4: Run focused test**

Run:

```powershell
julia test\twist_hubbard_ph_backflow_test.jl
```

Expected: all PH backflow tests pass.

- [ ] **Step 5: Commit Task 4**

Run:

```powershell
git add test\twist_hubbard_ph_backflow_test.jl
git commit -m "test: verify PH backflow degeneration and ratio"
```

---

### Task 5: End-to-End Verification and Benchmark

**Files:**
- No code changes unless a previous task revealed a bug.

- [ ] **Step 1: Run focused PH and no-backflow tests**

Run:

```powershell
julia test\twist_hubbard_ph_backflow_test.jl
julia test\twist_hubbard_ph_no_backflow_test.jl
```

Expected: both pass.

- [ ] **Step 2: Run existing nonPH/backflow regression tests**

Run:

```powershell
julia test\backflow_composite_terms_test.jl
julia test\emery_grouped_backflow_cleanup_test.jl
```

Expected: both pass.

- [ ] **Step 3: Re-run PH/nonPH zero-backflow benchmark**

Run:

```powershell
julia test\twist_hubbard_ph_nonph_benchmark_test.jl
julia bench_twist_hubbard_ph_nonph.jl
```

Expected:

```text
twist Hubbard PH/nonPH U=0 benchmark | Pass
E_PH - E_nonPH approximately 0 within floating point roundoff
```

- [ ] **Step 4: Run PH driver smoke tests**

Run zero-backflow through backflow-enabled code path:

```powershell
julia twist_Hubbard_PH.jl --job measure --Lx 2 --Ly 2 --doping 0.0 --target_sz 0 --U 0.0 --nMC 4 --wMC 1 --rMC 2 --dMC 1 --ansatz AFM --Delta_AF 0.0 --etax 0.0 --etay 0.0 --mu 0.0 --enable_backflow true --bf_epsilon 1.0 --bf_eta1 0.0 --bf_eta2 0.0 --bf_eta3 0.0 --bf_eta4 0.0 --fixed_params chi1y=1.0,chi2=0.0,etax=0.0,etay=0.0,Delta_AF=0.0,mu=0.0,g=0.0,bf_epsilon=1.0,bf_eta1=0.0,bf_eta2=0.0,bf_eta3=0.0,bf_eta4=0.0
```

Run nonzero eta1 smoke:

```powershell
julia twist_Hubbard_PH.jl --job measure --Lx 2 --Ly 2 --doping 0.0 --target_sz 0 --U 0.0 --nMC 4 --wMC 1 --rMC 2 --dMC 1 --ansatz AFM --Delta_AF 0.0 --etax 0.0 --etay 0.0 --mu 0.0 --enable_backflow true --bf_epsilon 1.0 --bf_eta1 0.1 --bf_eta2 0.0 --bf_eta3 0.0 --bf_eta4 0.0 --fixed_params chi1y=1.0,chi2=0.0,etax=0.0,etay=0.0,Delta_AF=0.0,mu=0.0,g=0.0,bf_epsilon=1.0,bf_eta1=0.1,bf_eta2=0.0,bf_eta3=0.0,bf_eta4=0.0
```

Expected: both commands finish and print `Backflow enabled: true`.

- [ ] **Step 5: Check git status**

Run:

```powershell
git status --short
```

Expected: empty output.

- [ ] **Step 6: Report benchmark limits clearly**

Final report must say:

```text
已验证 zero-backflow PH/nonPH benchmark 保持一致。
已验证 PH backflow fast ratio 与 rebuild ratio 一致。
非零 backflow PH/nonPH 能量比较只作为 sanity benchmark, 不作为硬断言。
```
