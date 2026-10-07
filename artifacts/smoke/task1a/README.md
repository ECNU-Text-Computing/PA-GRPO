# Task 1A — Windows inference and scorer smoke

## Scope

This artifact proves the approved Windows-only path:

```text
synthetic pairwise parquet
  -> Transformers CPU generation
  -> official response parser
  -> official Judge scorer
  -> Acc / Consistency / Consistent Accuracy
```

It does not validate Ray, verl imports, vLLM, FSDP, LoRA training, or GRPO.
Those checks remain assigned to Linux/WSL and the AutoDL GPU environment.

## Provenance

- PA-GRPO upstream commit: `0ee9abd903cb4ac4945f1176e943d20436470096`
- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Model revision: `12fd25f77366fa6b3b4b768ec3050bf629380bac`
- Backend: Transformers on CPU
- Mode: `direct`
- Samples: one AB/BA synthetic pair
- Maximum new tokens: 4

Exact package versions are in `env.txt`; direct dependencies are pinned in
`requirements-windows-smoke.txt`; commands are in `command.txt`.

## Result

The real model output was parsed and the official scorer produced:

```text
Accuracy = 0.50
Consistency = 0.00
Consistent Accuracy = 0.00
```

The values are deliberately treated only as smoke evidence. A 135M model with
four generated tokens is not a research baseline.

## Files

- `smoke_judge_direct.parquet`: two-row AB/BA input fixture.
- `eval_output.json`: verbatim result JSON from the successful inference run.
- `metrics.txt`: scorer output summary.
- `command.txt`: reproducible command sequence.
- `env.txt`: environment and model provenance.

## Known Windows limitations

- The environment intentionally does not install the full Ray/verl/vLLM stack.
- `torch` is CPU-only in this smoke environment.
- The official direct parser remains permissive: text such as `invalid` can be
  interpreted as `A`. Controlled experiments use
  `--evaluation_contract controlled`, whose direct parser accepts only a bare
  legal answer letter.
- The official evaluator recorded `num_options=4` for this A/B parquet because
  it checks only the first prompt and does not handle parquet `numpy.ndarray`
  prompts. Controlled evaluation scans every row and requires a consistent
  explicit option contract, or an explicit `--num_options 2|4` override.
