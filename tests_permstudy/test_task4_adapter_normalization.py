"""Tests for the controlled merged-adapter config normalization step.

`verl/model_merger/base_model_merger.py:266-269` writes ``"lora_alpha": 0`` for a merged LoRA
adapter while training uses ``lora_alpha=64, r=32``, so PEFT's ``alpha / r`` scaling would be
``0`` instead of ``2`` and the controlled evaluation would not see the trained adapter.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts_permstudy" / "task4" / "normalize_merged_adapter.py"


def load_normalizer():
    assert SCRIPT.is_file(), f"missing adapter normalizer: {SCRIPT}"
    spec = importlib.util.spec_from_file_location("task4_normalize_merged_adapter", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_config(directory: Path, **overrides) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "r": 32,
        "lora_alpha": 0,
        "target_modules": ["q_proj", "k_proj"],
        "task_type": "CAUSAL_LM",
        "peft_type": "LORA",
    }
    payload.update(overrides)
    path = directory / "adapter_config.json"
    path.write_text(json.dumps(payload, indent=4), encoding="utf-8")
    return path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_merger_zero_alpha_is_corrected_to_the_training_value(tmp_path):
    normalizer = load_normalizer()
    directory = tmp_path / "lora_adapter"
    config = write_config(directory, lora_alpha=0)
    before = sha256(config)

    evidence = normalizer.normalize_merged_adapter(directory, 32, 64)

    assert evidence["corrected"] is True
    assert evidence["observed_alpha_before"] == 0
    assert evidence["observed_alpha_after"] == 64
    assert evidence["scaling_before"] == 0.0
    assert evidence["scaling_after"] == 2.0
    assert evidence["before_sha256"] == before
    assert evidence["after_sha256"] == sha256(config)
    assert evidence["before_sha256"] != evidence["after_sha256"]

    written = json.loads(config.read_text(encoding="utf-8"))
    assert written["lora_alpha"] == 64
    assert written["r"] == 32
    assert written["target_modules"] == ["q_proj", "k_proj"]

    recorded = json.loads((directory / "adapter_normalization.json").read_text(encoding="utf-8"))
    assert recorded == evidence


def test_already_correct_config_is_left_untouched(tmp_path):
    normalizer = load_normalizer()
    directory = tmp_path / "lora_adapter"
    config = write_config(directory, lora_alpha=64)
    before = sha256(config)

    evidence = normalizer.normalize_merged_adapter(directory, 32, 64)

    assert evidence["corrected"] is False
    assert evidence["before_sha256"] == evidence["after_sha256"] == before
    assert sha256(config) == before
    assert json.loads(config.read_text(encoding="utf-8"))["lora_alpha"] == 64


def test_rank_mismatch_is_refused_without_rewriting(tmp_path):
    normalizer = load_normalizer()
    directory = tmp_path / "lora_adapter"
    config = write_config(directory, r=16, lora_alpha=0)
    before = sha256(config)

    with pytest.raises(normalizer.AdapterConfigError, match="rank r is 16"):
        normalizer.normalize_merged_adapter(directory, 32, 64)

    assert sha256(config) == before, "a rank mismatch must not leave the config modified"
    assert not (directory / "adapter_normalization.json").exists()


def test_missing_or_malformed_config_is_refused(tmp_path):
    normalizer = load_normalizer()
    with pytest.raises(normalizer.AdapterConfigError, match="missing merged adapter config"):
        normalizer.normalize_merged_adapter(tmp_path / "absent", 32, 64)

    directory = tmp_path / "broken"
    directory.mkdir()
    (directory / "adapter_config.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(normalizer.AdapterConfigError, match="not valid JSON"):
        normalizer.normalize_merged_adapter(directory, 32, 64)


def test_missing_keys_are_refused(tmp_path):
    normalizer = load_normalizer()
    directory = tmp_path / "lora_adapter"
    directory.mkdir()
    (directory / "adapter_config.json").write_text(json.dumps({"r": 32}), encoding="utf-8")
    with pytest.raises(normalizer.AdapterConfigError, match="missing lora_alpha"):
        normalizer.normalize_merged_adapter(directory, 32, 64)


def test_cli_reports_and_exits(tmp_path, capsys):
    normalizer = load_normalizer()
    directory = tmp_path / "lora_adapter"
    write_config(directory, lora_alpha=0)

    assert normalizer.main(["--adapter-dir", str(directory)]) == 0
    out = capsys.readouterr().out
    assert "alpha 0->64" in out
    assert "corrected=True" in out

    bad = tmp_path / "bad"
    write_config(bad, r=8)
    assert normalizer.main(["--adapter-dir", str(bad)]) == 2
    assert "adapter config error" in capsys.readouterr().err
