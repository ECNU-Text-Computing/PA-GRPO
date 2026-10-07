# Task 3 Controlled Evaluator Evidence

## Purpose

This artifact records reproducible test evidence for the strict controlled evaluator contract. It does not change evaluator, reward, trainer, advantage, rollout identity, EIS, PA, or ALC semantics.

## Code under test

```text
commit: 8b9cfff57d4668d1fcaa7f95039dce6782dcd621
subject: feat: add strict controlled evaluation contract
```

The evidence files added after that commit are documentation-only; the tested evaluator code is unchanged.

## Commands

Windows:

```powershell
C:\Study\Anaconda\envs\pagrpo-smoke\python.exe -m pytest tests_permstudy\test_controlled_evaluator.py -q
```

WSL2 Ubuntu 24.04:

```bash
/home/djl/.venvs/pagrpo-fidelity/bin/python \
  -m pytest tests_permstudy/test_controlled_evaluator.py -q --disable-warnings
```

## Result

```text
Windows: 42 passed in 3.76s
WSL2:    42 passed in 3.48s
```

Task 3 status: **PASS**.

## Scope and limitations

- This is evaluator-contract unit/integration evidence, not a real vLLM or GRPO training smoke.
- The official evaluator remains the default reproduction path.
- Controlled experiments must explicitly select `--evaluation_contract controlled`.
- `answer_probability` remains diagnostic and cannot override controlled parsing.
- The WSL launcher printed a host-network warning before pytest; pytest exited successfully with code 0.
