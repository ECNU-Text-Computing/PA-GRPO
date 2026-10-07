# PA-GRPO Paper–Code Fidelity Audit

## Upstream

- Repository: `ECNU-Text-Computing/PA-GRPO`
- Pinned commit: `0ee9abd903cb4ac4945f1176e943d20436470096`
- Date checked: 2026-10-04
- Audit runtime: Windows 11, Python 3.12.15, PyTorch 2.6.0+cpu
- Scope: CPU synthetic tests and source-equation characterization. Importing the
  official `verl` module on this Windows environment stops at the absent `ray`
  dependency. Direct execution of the Ray trainer remains a Linux smoke item.

## A. Sigma Gate

### Paper states

The paper-form definition sets a whole group's advantage to zero when its
reward standard deviation is strictly below a threshold `delta`. This is a
signal-quality gate, not merely division-by-zero protection.

### Official code does

The pinned `verl/trainer/ppo/ray_trainer.py` computes population variance,
floors it at `1e-6`, standardizes the collapsed returns, and clamps the result
to `[-5, 5]`. No explicit `sigma < delta -> advantage = 0` branch appears in
that path.

### Our test observes

`tests_permstudy/test_pa_sigma_gate.py` passes all three synthetic cases:

- equal rewards produce zero numerator and therefore zero advantage either way;
- nearly equal rewards have population sigma `4.3308468e-05` and yield nonzero
  ungated advantages, while a synthetic `delta=1e-3` gates all four to zero;
- normal-variance rewards have sigma `1.0`, so gated and ungated results match.

The `1e-3` value is only an audit fixture. It is not represented as a paper or
author default.

### Project decision

Keep `sigma_gate_threshold=None` as the default until a credible source for
`delta` is recorded. The controlled paper implementation supports the strict
gate, while the official path remains separately characterizable.

## B. Response-Length Dependence

### Paper states

The group statistics operate on response-level scalar rewards. Holding rewards
and groups fixed while changing response token counts should not change the
paper-form advantage.

### Official code does

The pinned path first calls `compute_grpo_outcome_advantage(...,
norm_adv_by_std_in_grpo=False)`, which group-centers the scalar score and
broadcasts it through `response_mask`. `apply_group_baseline_from_returns` then
uses an unmasked mean across the full padded token dimension before a second
group standardization. The resulting collapsed return therefore contains the
valid-token fraction.

### Synthetic setup

- Scalar rewards: `[1, 1, -1, -1]`
- One group containing all four samples
- Equal response lengths: `[8, 8, 8, 8]`
- Unequal response lengths: `[2, 8, 4, 10]`
- Standard deviation convention in the paper helper: population standard
  deviation

### Equal-length results

- Paper: approximately `[1, 1, -1, -1]`
- Official-compatible source-equation path: `[1, 1, -1, -1]`
- Mean absolute difference: `9.5367432e-07` (the paper helper's `+eps`)
- Sign disagreement rate: `0.0`

### Unequal-length results

- Paper: approximately `[1, 1, -1, -1]`
- Official-compatible source-equation path:
  `[0.4472136, 1.3416407, -0.4472136, -1.3416407]`
- Mean absolute difference: `0.44721356`
- Maximum absolute difference: `0.55278546`
- Sign disagreement rate: `0.0`

### Our test observes

The source-equation characterization of the current official path changes the
advantage magnitudes when only response lengths change. The response-level
paper implementation remains unchanged. This is an observed implementation-
level response-length dependence; this report does not label it an official
bug.

### Project decision

Use response-level `global_advantage_paper` for controlled EIS/PA/ALC
comparisons. Preserve an official-compatible mode for sanity reproduction.
Confirm the same numbers by direct execution inside the Linux Ray/verl stack
before treating the characterization as an end-to-end trainer result.

## C. Consistency Pairing

### Paper states

For each rollout slot `t`, the two permutations should compare the same
semantic trial: `(AB, t) <-> (BA, t)`.

### Official code does

`my_reward/judge_llama.py` derives `pair_id` from `index // 2`, appends samples
to per-permutation lists in batch appearance order, and pairs `idxs0[t]` with
`idxs1[t]`. The rollout slot is implicit in relative list position.

### Explicit-ID behavior

`permstudy.grouping.pair_consistency_rollouts` pairs using
`(pair_id, rollout_slot)`, validates permutation IDs, reports missing partners,
and raises `DuplicateRolloutKeyError` instead of silently overwriting duplicate
identities.

### Reorder tests

The synthetic fixture deliberately assigns semantic answers that make cross-
slot mismatches observable.

| Scenario | Pairing match | Reward-vector match | Mean reward diff | Mean absolute reward diff |
|---|---:|---:|---:|---:|
| Original order | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| Random shuffle, seed 20261004 | 0.3333 | 0.6667 | 0.0000 | 0.6667 |
| Response-length sort | 0.0000 | 0.3333 | 0.0000 | 1.3333 |
| Simulated balance reorder | 0.3333 | 0.6667 | 0.0000 | 0.6667 |

The zero mean difference does not imply sample-level equivalence: each pair
adds the same signed value to two samples, so errors can redistribute reward
while preserving its batch mean. The reward-vector match rate exposes that
redistribution.

### Our test observes

The official order convention and explicit-ID convention agree in the original
layout. They can differ after order changes that do not preserve the same
within-permutation slot order. Explicit IDs remain invariant. Separate tests
also confirm that a missing slot is reported without cross-pairing and that a
duplicate identity raises an error.

### Project decision

Controlled experiments will carry `pair_id`, `permutation_id`, and
`rollout_slot`, and pair consistency by `(pair_id, rollout_slot)`. The legacy
order behavior remains available only for official sanity reproduction. No
vendored reward or trainer code was changed in this audit.

## D. Controlled-Experiment Implementation

- Advantage implementation: response-level scalar `global_advantage_paper`
- Sigma gate: disabled by default; no guessed paper threshold
- Grouping identity: explicit dataclasses in `permstudy/ids.py`
- Consistency pairing: explicit `(pair_id, rollout_slot)`
- Missing partner: reported in `PairingResult.unpaired`
- Duplicate identity: hard error
- Legacy fallback: official source behavior remains untouched; future trainer
  dispatch must be config-controlled

## Open ambiguities

1. A trustworthy numerical value and provenance for the paper's `delta` have
   not been established.
2. Direct execution of the official Ray trainer functions is pending the Linux
   environment; Windows currently fails import at `ModuleNotFoundError: ray`.
3. The simulated balance reorder is a controlled permutation, not evidence
   that every concrete verl balancing configuration reorders samples this way.
4. End-to-end propagation of `rollout_slot` through DataProto/repeat/reorder has
   not yet been integrated; this audit only establishes and tests the pure
   identity and pairing contract.

## Reproduction

From the repository root, using the documented Windows smoke environment:

```powershell
python -m pytest tests_permstudy -q
python scripts_permstudy/audit/audit_official_advantage.py
python scripts_permstudy/audit/audit_consistency_pairing.py
```

Machine-readable results are stored beside this report in
`sigma_gate_results.json`, `advantage_equal_length.json`,
`advantage_unequal_length.json`, and `consistency_pairing_reorder.json`.
