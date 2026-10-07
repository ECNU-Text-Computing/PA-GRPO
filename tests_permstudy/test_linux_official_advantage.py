import sys
import pytest
import torch


pytestmark = [
    pytest.mark.skipif(sys.platform != "linux", reason="requires the Linux verl/Ray import path"),
    pytest.mark.filterwarnings("ignore:Ray state API is no longer experimental:DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::UserWarning:verl\\.trainer\\.ppo\\.ray_trainer"),
]


def test_json_safe_metric_maps_non_finite_official_diagnostics_to_null():
    from permstudy.linux_fidelity import json_safe_metric

    assert json_safe_metric(1.25) == 1.25
    assert json_safe_metric(float("nan")) is None
    assert json_safe_metric(float("inf")) is None


def _run_with_pair_baseline_probe(monkeypatch, lengths):
    from permstudy.linux_fidelity import run_official_advantage_case
    from verl.trainer.ppo import ray_trainer

    called = {"value": False}
    real_apply_group_baseline = ray_trainer.apply_group_baseline_from_returns

    def wrapped_apply_group_baseline(*args, **kwargs):
        called["value"] = True
        return real_apply_group_baseline(*args, **kwargs)

    monkeypatch.setattr(
        ray_trainer,
        "apply_group_baseline_from_returns",
        wrapped_apply_group_baseline,
    )
    result = run_official_advantage_case(
        rewards=[1.0, 1.0, -1.0, -1.0],
        lengths=lengths,
        group_ids=[0, 0, 0, 0],
    )
    return called["value"], result


def test_equal_length_executes_real_pair_baseline_without_fallback(monkeypatch):
    called, result = _run_with_pair_baseline_probe(monkeypatch, [8, 8, 8, 8])

    assert called is True
    assert "fallback to original GRPO" not in result.trainer_output
    assert set(result.pair_baseline_metrics) == {
        "pair_baseline/mean_of_pair_means",
        "pair_baseline/mean_of_pair_stds",
        "pair_baseline/std_of_pair_stds",
    }
    expected = torch.tensor([1.0, 1.0, -1.0, -1.0])
    torch.testing.assert_close(result.official_real, expected, atol=1e-6, rtol=0)
    torch.testing.assert_close(result.official_compatible, expected, atol=1e-6, rtol=0)
    torch.testing.assert_close(result.paper, expected, atol=2e-6, rtol=0)


def test_unequal_length_real_path_matches_source_equation_and_differs_from_original_grpo(monkeypatch):
    called, result = _run_with_pair_baseline_probe(monkeypatch, [2, 8, 4, 10])

    assert called is True
    assert "fallback to original GRPO" not in result.trainer_output
    expected_real = torch.tensor([0.4472136, 1.3416408, -0.4472136, -1.3416408])
    expected_original_grpo = torch.tensor([1.0, 1.0, -1.0, -1.0])
    torch.testing.assert_close(result.official_real, expected_real, atol=2e-6, rtol=0)
    torch.testing.assert_close(result.official_compatible, expected_real, atol=2e-6, rtol=0)
    torch.testing.assert_close(result.original_grpo, expected_original_grpo, atol=1e-6, rtol=0)
    assert not torch.allclose(result.official_real, result.original_grpo)
