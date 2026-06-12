# Eta-Driven Emery Epsilon Implementation Summary

> Implements plan: `docs/superpowers/plans/2026-06-12-emery-eta-driven-epsilon-plan.md`

## Goal

Changed Emery backflow so each `epsilon` correction is activated only when the corresponding row has at least one **actual nonzero eta contribution**. Occupation-only epsilon masks are no longer sufficient: if all relevant eta parameters or hopping amplitudes make the eta contribution zero, epsilon does not activate.

## Changes

### `src/Backflow.jl` — Core refactoring

**Data model change:**
- `BackflowEpsilonTerm` simplified from 6 fields to 3: `param_name`, `epsilon_bf`, `group_names`
  - Removed: `target_neighbors_by_source_site`, `source_sites_by_target_neighbor`, `source_sites`, `neighbor_data_signature`
  - Added: `group_names::Vector{Symbol}` — which source group names trigger this epsilon (e.g., `[:dd, :dp]`)

**Deleted obsolete functions (11 total):**
- `compute_backflow_epsilon_neighbor_data_signature`
- `build_backflow_epsilon_neighbor_cache`
- `add_backflow_correction_site_block_after_proposal!` (epsilon-specific)
- `add_backflow_correction_site_row_after_proposal!` (epsilon-specific)
- `compute_backflow_epsilon_row_mask`
- `add_backflow_correction_orbitals!` (epsilon-specific)
- `add_backflow_correction_chain_rule_row!` (epsilon-specific)
- `add_backflow_correction_chain_rule_source_weights!` (epsilon-specific)
- `add_backflow_correction_derivative_orbitals!` (epsilon-specific)
- `validate_backflow_correction_source_data!` (BackflowEpsilonTerm-specific)
- `is_backflow_epsilon_site_row_active`

**New helper functions (3):**
- `add_source_group_eta_contributions_and_track_activation!` — computes eta contributions for one source group and tracks which group names produce nonzero eta
- `add_epsilon_contributions_from_active_groups!` — adds epsilon correction only for epsilon terms whose `group_names` intersect the active set
- `add_source_group_chain_rule_source_weights_and_track!` — chain-rule source-weight variant

**Refactored functions (7 total, now eta-driven):**
- `build_backflow_orbitals` — full orbital rebuild
- `fill_backflow_chain_rule_orbitals!` — chain rule full matrix
- `fill_backflow_chain_rule_row!` — chain rule single row
- `fill_backflow_chain_rule_source_weights!` — chain rule source-weight expansion
- `fill_grouped_source_composite_site_row_after_proposal!` — proposal single row
- `fill_grouped_source_composite_site_block_after_proposal!` — proposal site block
- `build_backflow_derivative_orbitals` — backflow parameter derivative matrices

### `Emery.jl` — Construction change

In `build_column_directed_emery_backflow`:
```julia
# Before: epsilon terms took source_bonds for site-neighbor adjacency
BackflowEpsilonTerm(param_name=:bf_epsilon_d, epsilon_bf=bf_epsilon_d, source_bonds=d_source_bonds)

# After: epsilon terms take group_names for eta-driven activation
BackflowEpsilonTerm(param_name=:bf_epsilon_d, epsilon_bf=bf_epsilon_d, group_names=[:dd, :dp])
BackflowEpsilonTerm(param_name=:bf_epsilon_p, epsilon_bf=bf_epsilon_p, group_names=[:pd, :pp])
```

### `test/emery_grouped_backflow_cleanup_test.jl` — Test update

Rewrote all tests for new API. Added 3 new test sets (21 tests):
1. **Eta-driven epsilon term structure** (11 tests) — verifies new field layout, old keyword constructors reject
2. **Eta-driven epsilon activation** (6 tests) — eta=0 → no epsilon; eta≠0 → epsilon; zero amplitude → no epsilon; group isolation (`bf_epsilon_d` only responds to `:dd`/`:dp`)
3. **Proposal row-block consistency** (4 tests) — proposal path matches full rebuild

## Non-Goals (not changed)

- Did NOT remove `backflow_u` / `gs_U` cache
- Did NOT rewrite determinant rank-k update logic
- Did NOT migrate Hubbard backflow paths
- Did NOT change Emery parameter list or CLI names
- Did NOT split `bf_epsilon_d` / `bf_epsilon_p` into four orbital-specific epsilon parameters

## Design Pattern

Each `BackflowEpsilonTerm` owns a list of `group_names`. During orbital construction, all eta source groups are traversed. For each row `(site_i, sigma)`:

1. Compute eta coefficients for every bond from that site in every source group
2. Collect group names that produce at least one nonzero eta coefficient
3. For each epsilon term, if any of its `group_names` is in the active set, add `(epsilon_bf - 1.0) * U_0(row_i, :)`

This ensures epsilon activation is driven by actual eta coefficient values, not just occupation state.

## Verification

| Test suite | Result |
|------------|--------|
| `test/emery_grouped_backflow_cleanup_test.jl` | 34/34 passed |
| `test/simplify_emery_backflow_api_test.jl` | 10/10 passed |
| `Emery.jl` load | OK |
| `git diff --check` | clean |
