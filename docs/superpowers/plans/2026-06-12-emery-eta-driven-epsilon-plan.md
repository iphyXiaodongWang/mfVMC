# Emery Eta-Contribution-Driven Epsilon Implementation Plan

> **For agentic workers:** implement this plan task-by-task. Keep the scope limited to the Emery directed backflow path unless the human explicitly asks to migrate Hubbard paths.

## Goal

Change Emery backflow so each `epsilon` correction is activated only when the corresponding row has at least one **actual nonzero eta contribution**. Occupation-only epsilon masks are not sufficient: if all relevant eta parameters or hopping amplitudes make the eta contribution zero, epsilon must not activate.

## Current Baseline

Baseline commit before this plan:

```text
ed39493 refactor: use site-based Emery epsilon mask
```

The current code has already moved `BackflowEpsilonTerm` away from bond/amplitude storage and into site-neighbor adjacency:

```julia
mutable struct BackflowEpsilonTerm <: AbstractBackflowCorrectionTerm
    param_name::Symbol
    epsilon_bf::Float64
    target_neighbors_by_source_site::Vector{Vector{Int}}
    source_sites_by_target_neighbor::Vector{Vector{Int}}
    source_sites::Vector{Int}
    neighbor_data_signature::UInt
end
```

The current epsilon activation condition is:

```julia
n_{i,sigma} == 1 && exists target j with h_{j,sigma} == 1
```

That is equivalent to the union of the four eta occupation channels, but it does **not** check whether the eta coefficient is numerically nonzero.

## Desired Semantics

For a row `(site_i, sigma)`, epsilon should activate only if at least one source group attached to the same epsilon term produces a nonzero eta contribution for that row.

For a directed bond `(i, j)` in a source group:

```julia
eta_coefficient =
    t_ij * (
        eta1_bf * eta1_factor(i, j) +
        eta2_bf * eta2_factor(i, j, sigma) +
        eta3_bf * eta3_factor(i, j, sigma) +
        eta4_bf * eta4_factor(i, j, sigma)
    )
```

`epsilon` activates for `(i, sigma)` if:

```julia
exists source_group, exists bond (i, j) in that group:
    eta_coefficient != 0
```

If occupation factors are nonzero but all relevant eta parameters are zero, epsilon must not activate.

If eta parameters are nonzero but the bond amplitude is zero, epsilon must not activate.

## Non-Goals

- Do not remove the full `backflow_u` / `gs_U` cache in this step.
- Do not rewrite determinant rank-k update logic in this step.
- Do not migrate Hubbard backflow in this step.
- Do not change the Emery parameter list or command line names.
- Do not split `bf_epsilon_d` / `bf_epsilon_p` into four `dd/dp/pd/pp` epsilon parameters.

## Design Direction

The current independent epsilon neighbor mask should be removed or made derived from eta groups. Epsilon should be computed in the same row/group traversal that computes eta contributions.

The key structural idea:

- `bf_epsilon_d` is shared by the source groups `:dd` and `:dp`.
- `bf_epsilon_p` is shared by the source groups `:pd` and `:pp`.
- When computing a row for source site `i`, first compute eta contributions from all groups whose source bonds start at `i`.
- Track whether any eta coefficient is nonzero.
- If at least one eta coefficient is nonzero, add the corresponding epsilon correction:

```julia
backflow_row += (epsilon_bf - 1.0) * base_row
```

This means `BackflowEpsilonTerm` no longer needs its own `target_neighbors_by_source_site` mask. It needs only enough metadata to know which source groups it controls.

## Suggested Data Model

Prefer adding a lightweight group ownership field to epsilon:

```julia
mutable struct BackflowEpsilonTerm <: AbstractBackflowCorrectionTerm
    param_name::Symbol
    epsilon_bf::Float64
    group_names::Vector{Symbol}
end
```

For Emery:

```julia
BackflowEpsilonTerm(
    param_name=:bf_epsilon_d,
    epsilon_bf=bf_epsilon_d,
    group_names=Symbol[:dd, :dp],
)

BackflowEpsilonTerm(
    param_name=:bf_epsilon_p,
    epsilon_bf=bf_epsilon_p,
    group_names=Symbol[:pd, :pp],
)
```

If this feels too large for the first pass, keep the existing `BackflowEpsilonTerm` fields temporarily but stop using its mask during orbital construction. However, the final version should delete the independent epsilon neighbor cache because it duplicates eta source-group topology.

## Core Helper Functions

Introduce a row-level helper that computes eta contributions and reports whether epsilon should activate.

Recommended shape:

```julia
function add_backflow_source_group_row_contribution!(
    output_row::AbstractVector{T},
    base_orbitals::AbstractMatrix{T},
    state_vector::Vector{Int8},
    source_group::DirectedBackflowSourceGroup,
    site_index::Int,
    row_offset::Int,
    row_index::Int,
)::Bool where {T}
```

Behavior:

- Return `true` if at least one eta coefficient is nonzero.
- Add all eta contributions for this group and row to `output_row`.
- Do not add epsilon inside this helper.

Pseudo-code:

```julia
has_eta_contribution = false
state_i = state_vector[site_index]
spin = backflow_spin_from_row_offset(row_offset)

for bond_index in source_group.outgoing_bond_indices_by_source[site_index]
    (_, target_site) = source_group.source_bonds[bond_index]
    state_j = state_vector[target_site]
    bond_amplitude = T(source_group.source_amplitudes[bond_index])

    eta1_factor = (state_i == DB && state_j == HOLE) ? one(T) : zero(T)
    eta2_factor = T(compute_eta2_virtual_hopping_factor(state_i, state_j, spin))
    eta3_factor = T(compute_eta3_doublon_single_factor(state_i, state_j, spin))
    eta4_factor = T(compute_eta4_single_hole_factor(state_i, state_j, spin))

    coefficient = bond_amplitude * (
        T(source_group.eta1_term.eta1_bf) * eta1_factor +
        T(source_group.eta2_term.eta2_bf) * eta2_factor +
        T(source_group.eta3_term.eta3_bf) * eta3_factor +
        T(source_group.eta4_term.eta4_bf) * eta4_factor
    )

    if coefficient == zero(T)
        continue
    end

    target_row = 2 * (target_site - 1) + row_offset
    @views output_row .+= coefficient .* base_orbitals[target_row, :]
    has_eta_contribution = true
end

return has_eta_contribution
```

Then a composite row helper should:

1. Start from the base row.
2. For all source groups touching this source site, add eta contributions.
3. Track `has_eta_by_epsilon_term`.
4. Add each epsilon term only if one of its owned groups produced eta contribution.

## Functions To Modify

### `src/Backflow.jl`

Modify:

- `BackflowEpsilonTerm`
- `BackflowEpsilonTerm(...)` constructor
- `build_composite_incoming_from_groups`
- `add_backflow_correction_orbitals!` for epsilon, or delete it if no longer needed
- `compute_backflow_epsilon_row_mask`, or delete it if no longer needed
- `build_backflow_orbitals(::CompositeBackflowTerm)`
- `fill_grouped_source_composite_site_row_after_proposal!`
- `fill_grouped_source_composite_site_block_after_proposal!`
- `fill_backflow_chain_rule_row!`
- `fill_backflow_chain_rule_source_weights!`
- `build_backflow_derivative_orbitals`

Important: proposal paths must use proposal-after states, not current states:

```julia
state_i = get_site_state_after_proposal(state_vector, proposal, site_index)
state_j = get_site_state_after_proposal(state_vector, proposal, target_site)
```

### `Emery.jl`

Modify `build_column_directed_emery_backflow` so epsilon terms are built with group ownership:

```julia
epsilon_terms = mfVMC.BackflowEpsilonTerm[
    mfVMC.Backflow.BackflowEpsilonTerm(
        param_name=:bf_epsilon_d,
        epsilon_bf=bf_epsilon_d,
        group_names=Symbol[:dd, :dp],
    ),
    mfVMC.Backflow.BackflowEpsilonTerm(
        param_name=:bf_epsilon_p,
        epsilon_bf=bf_epsilon_p,
        group_names=Symbol[:pd, :pp],
    ),
]
```

Construct `dd_group`, `dp_group`, `pd_group`, `pp_group` exactly as now.

### Tests

Modify or add tests in:

```text
test/emery_grouped_backflow_cleanup_test.jl
```

Recommended new tests:

1. `BackflowEpsilonTerm` no longer stores site-neighbor mask fields:

```julia
@test !(:target_neighbors_by_source_site in fieldnames(typeof(epsilon_term)))
@test !(:source_sites_by_target_neighbor in fieldnames(typeof(epsilon_term)))
@test epsilon_term.group_names == Symbol[:dd, :dp]
```

2. Epsilon is inactive when occupation allows hopping but eta parameter is zero:

Use a simple two-site bond `(1, 2)`, source state `UP`, target state `HOLE`, nonzero `epsilon_bf`, but all eta values zero. Expected row is unchanged from `U_0`.

3. Epsilon is active when eta contribution is nonzero:

Same state, set `eta4_bf != 0` and nonzero bond amplitude. Expected row includes:

```julia
eta4_bf * t_12 * U0(target_row, :) +
(epsilon_bf - 1.0) * U0(source_row, :)
```

4. Epsilon is inactive when eta parameter is nonzero but bond amplitude is zero.

5. Epsilon sharing:

- `bf_epsilon_d` should respond to `:dd` and `:dp`.
- `bf_epsilon_p` should respond to `:pd` and `:pp`.
- `bf_epsilon_d` must not respond to `:pd` or `:pp`.

6. Proposal row/block consistency:

Compare `fill_backflow_site_block_after_proposal!` against full rebuild after proposal for at least one case where epsilon should activate and one case where it should not.

## Implementation Order

1. Add failing tests for eta-zero and eta-nonzero epsilon activation.
2. Add group ownership to `BackflowEpsilonTerm`.
3. Update Emery epsilon construction to use group names.
4. Introduce row-level eta contribution helper returning `has_eta_contribution`.
5. Refactor full `build_backflow_orbitals` to use the new row helper and add epsilon based on actual eta contribution.
6. Refactor proposal row/block paths to use the same eta-driven rule with proposal-after states.
7. Refactor chain-rule paths so epsilon chain-rule contribution is included only when the corresponding eta contribution exists for that row.
8. Refactor epsilon parameter derivative so:

```julia
partial U_b(row) / partial epsilon_bf = U0(row)
```

only when the corresponding eta contribution is nonzero.

9. Delete obsolete independent epsilon mask/cache helpers.
10. Run verification commands.

## Verification Commands

Run:

```powershell
julia test\emery_grouped_backflow_cleanup_test.jl
julia test\simplify_emery_backflow_api_test.jl
julia -e 'include("Emery.jl"); println("ok")'
git diff --check
```

If the repository has additional Emery gradient correctness tests in the current branch, also run them.

## Review Checklist

Before marking the implementation complete, verify:

- Epsilon activation depends on actual eta coefficient, not only occupation.
- Zero eta parameters do not trigger epsilon.
- Zero bond amplitudes do not trigger epsilon.
- Full rebuild and proposal block paths agree.
- Chain-rule and epsilon parameter derivative paths use the same activation rule.
- `bf_epsilon_d` only follows `dd/dp`; `bf_epsilon_p` only follows `pd/pp`.
- No Hubbard migration was accidentally included.
- No independent epsilon neighbor mask remains unless explicitly justified as a temporary compatibility layer.
