# Linux Fidelity Confirmation Report

## Scope and provenance

- Repository: `heziss666/PA-GRPO`
- Harness commit: `f4f801ebed5056ab6cd72ce8d6fabb7f1743759d`
- Task 2B integration commit: `82049c42edcc74393a09ed02141a7e663094054e`
- Pinned PA upstream commit: `0ee9abd903cb4ac4945f1176e943d20436470096`
- Date checked: 2026-10-06
- Runtime: Ubuntu 24.04.5 LTS on WSL2, Python 3.12.3
- PyTorch: 2.6.0+cpu
- Ray: 2.49.0
- GPU visibility: NVIDIA GeForce RTX 5070 Ti Laptop GPU, driver 616.92, 12227 MiB
- Execution device: CPU tensors; GPU visibility was checked independently
- vLLM: not installed, because this synthetic advantage confirmation does not use rollout inference

This report covers Task 2A and the CPU/WSL integration harness for Task 2B. It
does not claim that a real vLLM process, FSDP, LoRA, full GRPO training, or an
8B model has run.

## Path-execution proof

The tests call the real `verl.trainer.ppo.ray_trainer.compute_advantage()` with
a real `DataProto`. A test-only wrapper around
`apply_group_baseline_from_returns()` records a sentinel without replacing the
function's behavior.

All three required checks passed:

1. The test sentinel confirms `apply_group_baseline_from_returns()` was called.
2. Captured stdout/stderr does not contain `fallback to original GRPO`.
3. `data.meta_info` contains `pair_baseline_metrics`.

The unequal-length result also differs from the original pre-pair-baseline GRPO
advantages, so the final values cannot be explained by the fallback branch.

## Paper states

The paper-form global advantage operates on one scalar reward per response:

```text
(reward - group_mean) / (group_std + epsilon)
```

Changing only response token length should not change this scalar definition.
The paper also specifies a low-standard-deviation threshold gate. No numerical
threshold with reliable provenance has been established, so the controlled
implementation leaves that gate disabled by setting its threshold to `None`.

## Official code does

The pinned official path first calls
`compute_grpo_outcome_advantage(..., norm_adv_by_std_in_grpo=False)`, broadcasting
the group-centered scalar value through `response_mask`. It then collapses
`returns` using an unmasked mean across the padded response width inside
`apply_group_baseline_from_returns()`, applies a population group baseline, and
clips the resulting scalar advantage.

The function has a broad exception handler that can fall back to original GRPO.
The path-execution checks above ensure that fallback was not used in this run.

## Official Linux run observes

### Equal-length case

Setup:

```text
rewards = [1, 1, -1, -1]
lengths = [8, 8, 8, 8]
group_ids = [0, 0, 0, 0]
```

Results:

```text
official-real       = [ 1.0,  1.0, -1.0, -1.0]
official-compatible = [ 1.0,  1.0, -1.0, -1.0]
paper               ≈ [ 1.0,  1.0, -1.0, -1.0]
```

- Official-real vs compatible L-infinity difference: `0.0`
- Official-real vs paper L-infinity difference: `9.536743e-07`
- Sign disagreement rate: `0.0`

### Unequal-length case

Setup:

```text
rewards = [1, 1, -1, -1]
lengths = [2, 8, 4, 10]
group_ids = [0, 0, 0, 0]
```

Results:

```text
official-real       = [ 0.44721359,  1.34164059, -0.44721359, -1.34164059]
official-compatible = [ 0.44721359,  1.34164071, -0.44721359, -1.34164071]
paper               ≈ [ 0.99999905,  0.99999905, -0.99999905, -0.99999905]
original GRPO       = [ 1.0,         1.0,        -1.0,        -1.0       ]
```

- Official-real vs compatible L-infinity difference: `1.192093e-07`
- Official-real vs paper mean absolute difference: `0.44721350`
- Official-real vs original GRPO mean absolute difference: `0.44721350`
- Sign disagreement rate: `0.0`

The real Linux execution therefore confirms implementation-level response-length
dependence in the pinned official PA path. This dependence is absent from the
paper's response-level scalar formulation.

One official diagnostic,
`pair_baseline/std_of_pair_stds`, is undefined for this one-group fixture and is
stored as JSON `null`; it is not used to compute advantages.

## Project controlled implementation does

`permstudy.advantages.global_advantage_paper()` computes statistics directly
from response-level scalar rewards and is independent of response masks and
token counts. It preserves the optional strict sigma gate but does not invent a
default threshold. The official-compatible helper remains available only for
characterization and reproduction comparisons.

For Task 2B, `permstudy.rollout_identity` now derives controlled pair identity
from dataset metadata at the DataProto boundary. It accepts `pair_id` or
`original_question_id` and `permutation_id` or `permutation`; it deliberately
does not use `index // 2` as a controlled fallback.

The real Trainer path now performs the following sequence when
`grouping.identity_mode=explicit`:

```text
dataset extra_info
-> pair_id / permutation_id in DataProto.non_tensor_batch
-> repeat(n, interleave=True)
-> rollout_slot = [0, ..., n-1] for every permutation input
-> snapshot identity before generate_sequences()
-> compare vLLM RequestOutput.prompt_token_ids with input prompt order
-> assert returned DataProto preserved identity metadata order
-> repeat the reward-side batch with the same identity
-> DataProto.union()
-> balance/reorder (all non-tensor identity arrays reorder together)
-> BatchRewardManager
-> consistency pairing by (pair_id, rollout_slot)
```

For diagnostic calls with `return_dict=True`, an incomplete controlled key is
exposed through the per-sample `consistency_unpaired` field. The production
reward path records it through the `reward/consistency_unpaired_rate`
TensorBoard metric and a `PAIR_UNPAIRED` log record. A duplicate
`(pair_id, permutation_id, rollout_slot)` raises
`DuplicateRolloutKeyError` instead of being overwritten.

`grouping.identity_mode=legacy_index` remains the default. In that mode the
identity helpers are no-ops, the original repeat behavior is used, and the
Judge reward retains the official index/appearance-order pairing behavior.
Controlled training must opt in explicitly, for example:

```bash
bash scripts/run_judge_llama.sh grouping.identity_mode=explicit
```

The explicit path currently supports synchronous vLLM rollout and the GRPO
path used by this project. Async rollout, REMAX, and non-vLLM backends such as
SGLang fail before generation with a clear `NotImplementedError`; their
auxiliary output/baseline paths have not yet been integrated with the
three-part identity or equivalent backend-order validation. Conflicting
Trainer/reward identity-mode settings—including custom reward kwargs—and
non-batch reward managers also fail fast.

The WSL tests execute the real DataProto, Trainer `_get_gen_batch`,
`DataProto.repeat`, `DataProto.reorder`, `BatchRewardManager`, and both Judge
reward modules. They also exercise the production vLLM prompt-order validator
with aligned and misordered RequestOutput-shaped objects. Because vLLM is
intentionally not installed in this WSL environment, a real GPU/vLLM
generation-process smoke is still required on AutoDL; the production vLLM path
will fail closed if actual output prompt order or identity metadata disagrees.

## Project decision

1. Treat response-length dependence as confirmed for the pinned official path.
2. Keep PA-Official behavior unchanged for sanity reproduction.
3. Use the response-level paper implementation for controlled PA/EIS/ALC
   comparisons.
4. Keep the vendored `verl` change limited to the Task 2B dispatch points:
   DataProto identity attachment/repeat, vLLM output-order validation, config
   routing, and reward-manager metadata transfer.
5. Treat Task 2B code integration as complete only for the CPU/WSL harness;
   run a real vLLM smoke on AutoDL before formal GRPO training.
6. Do not begin EIS/ALC or large-scale training before the controlled evaluator
   work and the AutoDL vLLM smoke are reviewed.

## Reproduction

```bash
python -m pytest tests_permstudy/test_linux_official_advantage.py -q
python scripts_permstudy/linux_fidelity/run_official_advantage_fidelity.py \
  --git-commit f4f801ebed5056ab6cd72ce8d6fabb7f1743759d

# Task 2B Linux integration and configuration checks
python -m pytest tests_permstudy \
  tests/special_sanity/test_config_docs.py \
  tests/trainer/config/test_legacy_config_on_cpu.py -q
```

Task 2B verification observed `40 passed` on WSL/Linux. The Windows smoke
environment observed `17 passed, 19 skipped`; the skipped cases are the
documented Linux/verl integration tests.

Machine-readable results:

- `advantage_equal_length.json`
- `advantage_unequal_length.json`
- `environment.txt`
