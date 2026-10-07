"""Emit trainer-side evidence in the real formats, from the Task 1 manifest.

Stdlib only, so it runs under the WSL python3 that the launcher's shim delegates to.
Formats are taken from the real emitters:
  - reward log lines: my_reward/judge_qwen.py:308-312 / :363-378 / :325-330, prefix at :142
  - rollout dumps:    verl/trainer/ppo/ray_trainer.py:518-542 ({global_steps}.jsonl, "step" field)
  - console metrics:  verl/utils/logger/aggregate_logger.py:26-32
  - eval summary:     evaluation/evaluate_models.py:311-318 + :1136-1141
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

RUN = Path(os.environ["TASK4_RUN_DIR"])
# The reward log path is authoritative in the window the launcher wrote before training:
# the real reward derives its own LOG_PATH from its module location and reads no environment
# variable, and the launcher deliberately no longer sets or exports a REWARD_LOG override.
REWARD_LOG = Path(json.loads((RUN / "reward_log_window.json").read_text(encoding="utf-8"))["reward_log_path"])
ADAPTER_SRC = Path(os.environ["TASK4_ADAPTER_SRC"])
MANIFEST = Path(os.environ["TASK4_MANIFEST"])
STEPS = (1, 2)
ROWS_PER_STEP = 32

manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
pairs = manifest["pairs"]
row_index = {
    (pair["pair_id"], int(row["permutation"])): int(row["row_index"])
    for pair in pairs
    for row in pair["rows"]
}

# canonical pre-balance batch order: repeat_for_rollout(2, interleave=True)
canonical = []
for pair in pairs:
    for permutation in (0, 1):
        canonical.extend([(pair["pair_id"], permutation)] * 2)


def base_line(i: int, pair_id: str, permutation: int) -> str:
    return (
        f"2026-01-01 00:00:00.000000 | idx={row_index[(pair_id, permutation)]} | pid=1 | cuda=cuda:0 | "
        f"BASE | i={i} | pair_id={pair_id} | perm={permutation} | ans=A | gt=A | confidence=1.00 | "
        f"method=boxed | base_score=0.500 | corr=0.500 | fmt=0.000 | len=0.000"
    )


def check_line(pair_id: str, slot: int) -> str:
    return (
        f"2026-01-01 00:00:00.000000 | idx={row_index[(pair_id, 0)]},{row_index[(pair_id, 1)]} | pid=1 | "
        f"cuda=cuda:0 | PAIR_CHECK | pair_id={pair_id} | rollout_slot={slot} | "
        f"i0=0,idx0={row_index[(pair_id, 0)]},ans0=A,conf0=1.00 | "
        f"i1=1,idx1={row_index[(pair_id, 1)]},ans1=A,conf1=1.00 | mapped(ans0)=A | is_consistent=True | "
        f"pair_bonus=1.000 | final_score0=1.000 | final_score1=1.000"
    )


# ---- reward log: two invocations, the first one genuinely reordered by balance_batch ----
lines: list[str] = []
for step_index in range(len(STEPS)):
    order = list(reversed(canonical)) if step_index == 0 else list(canonical)
    for i, (pair_id, permutation) in enumerate(order):
        lines.append(base_line(i, pair_id, permutation))
    for pair in pairs:
        for slot in (0, 1):
            lines.append(check_line(pair["pair_id"], slot))
with REWARD_LOG.open("a", encoding="utf-8") as handle:
    handle.write("\n".join(lines) + "\n")

# ---- rollout dumps ----
rollout_dir = RUN / "rollouts"
rollout_dir.mkdir(parents=True, exist_ok=True)
for step in STEPS:
    with (rollout_dir / f"{step}.jsonl").open("w", encoding="utf-8") as handle:
        for _ in range(ROWS_PER_STEP):
            handle.write(json.dumps({"input": "p", "output": "o", "gts": "A", "score": 1.0, "step": step}) + "\n")

# ---- console log in the real logger format ----
metric_lines = [
    "INFO 01-01 00:00:00 engine.py:120] Initializing an LLM engine (v0.8.5) with config:",
    "INFO 01-01 00:00:00 model_runner.py:907] Loading model weights took 15.1234 GB",
]
for step in STEPS:
    metrics = {
        "training/global_step": step,
        "actor/pg_loss": 0.5,
        "actor/grad_norm": 0.75,
        "actor/kl_loss": 0.01,
        "reward/consistency_unpaired_rate": 0.0,
        "pair_baseline/mean_of_pair_means": 0.1,
        "pair_baseline/mean_of_pair_stds": 0.2,
        "pair_baseline/std_of_pair_stds": 0.3,
        "critic/advantages": 0.42,
    }
    parts = [f"step:{step}"] + [f"{key}:{value}" for key, value in metrics.items()]
    metric_lines.append(" - ".join(parts))
(RUN / "train.log").write_text("\n".join(metric_lines) + "\n", encoding="utf-8")

# ---- checkpoints and the merged LoRA adapter ----
for step in STEPS:
    (RUN / "checkpoints" / f"global_step_{step}").mkdir(parents=True, exist_ok=True)
(RUN / "checkpoints" / "latest_checkpointed_iteration.txt").write_text(f"{STEPS[-1]}\n", encoding="utf-8")
adapter_dir = RUN / "checkpoints" / f"global_step_{STEPS[-1]}" / "actor" / "hf" / "lora_adapter"
adapter_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2(ADAPTER_SRC, adapter_dir / "adapter_model.safetensors")
# the normalized config the Task 5 step must produce before the controlled evaluation
adapter_config_path = adapter_dir / "adapter_config.json"
adapter_config_path.write_text(
    json.dumps({"r": 32, "lora_alpha": 64, "target_modules": ["q_proj"], "peft_type": "LORA"}),
    encoding="utf-8",
)
# the Task 5 correction step's evidence, digest-bound to the config above
import hashlib

(adapter_dir / "adapter_normalization.json").write_text(
    json.dumps(
        {
            "schema_version": "task4_adapter_normalization_v1",
            "observed_rank": 32,
            "observed_alpha_before": 0,
            "observed_alpha_after": 64,
            "corrected": True,
            "after_sha256": hashlib.sha256(adapter_config_path.read_bytes()).hexdigest(),
        }
    ),
    encoding="utf-8",
)

# ---- controlled evaluation output tree ----
eval_dir = RUN / "controlled_eval" / "task4_tiny_8pairs"
eval_dir.mkdir(parents=True, exist_ok=True)
(eval_dir / "summary.json").write_text(
    json.dumps(
        {
            "task4_tiny_8pairs": {
                "evaluated_samples": 4,
                "correct": 1,
                "total_with_answer": 4,
                "accuracy": 0.25,
                "evaluation_contract": "controlled",
                "num_options": 2,
            }
        }
    ),
    encoding="utf-8",
)

print("trainer emulation wrote evidence into", RUN, file=sys.stderr)
