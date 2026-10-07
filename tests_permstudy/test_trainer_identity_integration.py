import importlib.util

import numpy as np
import pytest
import torch


pytestmark = pytest.mark.skipif(importlib.util.find_spec("ray") is None, reason="requires verl DataProto dependencies")


def test_grouping_config_defaults_to_official_and_allows_explicit_override():
    """Catches making controlled grouping the implicit official behavior."""
    import os

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    config_dir = os.path.abspath("verl/trainer/config")
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            official = compose(config_name="ppo_trainer")
            controlled = compose(config_name="ppo_trainer", overrides=["grouping.identity_mode=explicit"])
        assert official.grouping.identity_mode == "legacy_index"
        assert controlled.grouping.identity_mode == "explicit"
    finally:
        GlobalHydra.instance().clear()


def test_reward_manager_receives_explicit_mode_only_when_selected():
    """Catches config selecting explicit grouping only inside the trainer."""
    import os

    from omegaconf import OmegaConf

    from verl.trainer.ppo.reward import load_reward_manager

    def config(identity_mode):
        return OmegaConf.create(
            {
                "grouping": {"identity_mode": identity_mode},
                "reward_model": {"reward_manager": "batch"},
                "data": {"reward_fn_key": "data_source"},
                "custom_reward_function": {
                    "path": os.path.abspath("my_reward/judge_llama.py"),
                    "name": "compute_score",
                },
            }
        )

    official = load_reward_manager(config("legacy_index"), tokenizer=object(), num_examine=0)
    controlled = load_reward_manager(config("explicit"), tokenizer=object(), num_examine=0)

    assert official.reward_kwargs == {}
    assert controlled.reward_kwargs == {"identity_mode": "explicit"}


def test_explicit_mode_rejects_unsupported_async_and_remax_paths():
    """Catches entering rollout paths that do not propagate the three-part identity."""
    from permstudy.rollout_identity import ensure_supported_trainer_identity_mode

    with pytest.raises(NotImplementedError, match="async rollout"):
        ensure_supported_trainer_identity_mode(
            "explicit", async_rollout_mode=True, advantage_estimator="grpo", rollout_backend="vllm"
        )
    with pytest.raises(NotImplementedError, match="REMAX"):
        ensure_supported_trainer_identity_mode(
            "explicit", async_rollout_mode=False, advantage_estimator="remax", rollout_backend="vllm"
        )
    with pytest.raises(NotImplementedError, match="synchronous vLLM"):
        ensure_supported_trainer_identity_mode(
            "explicit", async_rollout_mode=False, advantage_estimator="grpo", rollout_backend="sglang"
        )
    ensure_supported_trainer_identity_mode(
        "legacy_index", async_rollout_mode=True, advantage_estimator="remax", rollout_backend="sglang"
    )


def test_reward_manager_rejects_identity_mode_disagreement():
    """Catches an explicit Trainer silently dispatching legacy reward pairing."""
    import os

    from omegaconf import OmegaConf

    from verl.trainer.ppo.reward import load_reward_manager

    config = OmegaConf.create(
        {
            "grouping": {"identity_mode": "explicit"},
            "reward_model": {"reward_manager": "batch"},
            "data": {"reward_fn_key": "data_source"},
            "custom_reward_function": {
                "path": os.path.abspath("my_reward/judge_llama.py"),
                "name": "compute_score",
            },
        }
    )

    with pytest.raises(ValueError, match="conflicts with grouping.identity_mode"):
        load_reward_manager(config, tokenizer=object(), num_examine=0, identity_mode="legacy_index")


def test_custom_reward_kwargs_cannot_override_grouping_identity_mode():
    """Catches custom-function kwargs taking precedence over explicit dispatch."""
    import os

    from omegaconf import OmegaConf

    from verl.trainer.ppo.reward import load_reward_manager

    config = OmegaConf.create(
        {
            "grouping": {"identity_mode": "explicit"},
            "reward_model": {"reward_manager": "batch"},
            "data": {"reward_fn_key": "data_source"},
            "custom_reward_function": {
                "path": os.path.abspath("my_reward/judge_llama.py"),
                "name": "compute_score",
                "reward_kwargs": {"identity_mode": "legacy_index"},
            },
        }
    )

    with pytest.raises(ValueError, match="custom_reward_function.reward_kwargs.identity_mode"):
        load_reward_manager(config, tokenizer=object(), num_examine=0)


def test_explicit_mode_requires_batch_reward_manager():
    """Catches selecting a manager that cannot pass batch-level pairing identity."""
    from omegaconf import OmegaConf

    from verl.trainer.ppo.reward import load_reward_manager

    config = OmegaConf.create(
        {
            "grouping": {"identity_mode": "explicit"},
            "reward_model": {"reward_manager": "naive"},
            "data": {"reward_fn_key": "data_source"},
            "custom_reward_function": {"path": None, "name": "compute_score"},
        }
    )

    with pytest.raises(ValueError, match="requires reward_model.reward_manager=batch"):
        load_reward_manager(config, tokenizer=object(), num_examine=0)


def _trainer_batch():
    from verl import DataProto

    return DataProto.from_single_dict(
        {
            "input_ids": torch.tensor([[1, 2], [3, 4]]),
            "attention_mask": torch.ones((2, 2), dtype=torch.long),
            "position_ids": torch.tensor([[0, 1], [0, 1]]),
            "data_source": np.array(["judge", "judge"], dtype=object),
            "reward_model": np.array(
                [{"ground_truth": "A"}, {"ground_truth": "B"}], dtype=object
            ),
            "extra_info": np.array(
                [
                    {"original_question_id": "pair-1", "permutation": 0},
                    {"original_question_id": "pair-1", "permutation": 1},
                ],
                dtype=object,
            ),
        }
    )


def test_real_trainer_generation_batch_keeps_explicit_identity_keys():
    """Catches _get_gen_batch popping identity before vLLM receives it."""
    from types import SimpleNamespace

    from permstudy.rollout_identity import attach_permutation_identity, repeat_for_rollout
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    batch = _trainer_batch()
    attach_permutation_identity(batch, identity_mode="explicit")
    trainer = SimpleNamespace(async_rollout_mode=False)

    gen_batch = RayPPOTrainer._get_gen_batch(trainer, batch)
    repeated = repeat_for_rollout(gen_batch, repeat_times=2, identity_mode="explicit")

    assert batch.non_tensor_batch["pair_id"].tolist() == ["pair-1", "pair-1"]
    assert batch.non_tensor_batch["permutation_id"].tolist() == [0, 1]
    assert repeated.non_tensor_batch["pair_id"].tolist() == ["pair-1"] * 4
    assert repeated.non_tensor_batch["permutation_id"].tolist() == [0, 0, 1, 1]
    assert repeated.non_tensor_batch["rollout_slot"].tolist() == [0, 1, 0, 1]


def test_batch_reward_manager_passes_reordered_top_level_identity_to_reward():
    """Catches the reward boundary reading stale nested dataset metadata."""
    from verl.workers.reward_manager.batch import BatchRewardManager

    captured = {}

    def compute_score(**kwargs):
        captured.update(kwargs)
        return torch.zeros(2)

    class Tokenizer:
        def decode(self, token_ids, skip_special_tokens=True):
            return "decoded"

    data = _trainer_batch()
    data.batch["prompts"] = data.batch.pop("input_ids")
    data.batch["responses"] = torch.tensor([[5], [6]])
    data.batch["attention_mask"] = torch.ones((2, 3), dtype=torch.long)
    data.non_tensor_batch["pair_id"] = np.array(["pair-1", "pair-1"], dtype=object)
    data.non_tensor_batch["permutation_id"] = np.array([1, 0], dtype=np.int64)
    data.non_tensor_batch["rollout_slot"] = np.array([3, 3], dtype=np.int64)

    manager = BatchRewardManager(
        tokenizer=Tokenizer(),
        num_examine=0,
        compute_score=compute_score,
        identity_mode="explicit",
    )
    manager.verify(data)

    assert captured["identity_mode"] == "explicit"
    assert list(captured["extra_infos"]) == [
        {
            "original_question_id": "pair-1",
            "permutation": 0,
            "pair_id": "pair-1",
            "permutation_id": 1,
            "rollout_slot": 3,
            "rollout_reward_scores": {},
        },
        {
            "original_question_id": "pair-1",
            "permutation": 1,
            "pair_id": "pair-1",
            "permutation_id": 0,
            "rollout_slot": 3,
            "rollout_reward_scores": {},
        },
    ]


@pytest.mark.parametrize("module_name", ["my_reward.judge_llama", "my_reward.judge_qwen"])
def test_dataproto_repeat_reorder_reward_flow_uses_explicit_slot_pairing(monkeypatch, module_name):
    """Catches any disconnect across DataProto repeat, reorder, manager, and reward."""
    import importlib
    from types import SimpleNamespace

    from permstudy.rollout_identity import attach_permutation_identity, repeat_for_rollout
    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from verl.workers.reward_manager.batch import BatchRewardManager

    reward_module = importlib.import_module(module_name)
    monkeypatch.setattr(reward_module, "get_tb_writer", lambda: type("Writer", (), {"add_scalar": lambda *a: None})())
    monkeypatch.setattr(reward_module, "safe_log", lambda *_args, **_kwargs: None)
    if hasattr(reward_module, "safe_log_output"):
        monkeypatch.setattr(reward_module, "safe_log_output", lambda *_args, **_kwargs: None)

    data = _trainer_batch()
    attach_permutation_identity(data, identity_mode="explicit")
    gen_batch = RayPPOTrainer._get_gen_batch(SimpleNamespace(async_rollout_mode=False), data)
    gen_batch = repeat_for_rollout(gen_batch, repeat_times=2, identity_mode="explicit")
    data = repeat_for_rollout(data, repeat_times=2, identity_mode="explicit")
    generated = DataProto.from_single_dict(
        {
            "prompts": gen_batch.batch["input_ids"],
            "responses": torch.tensor([[8], [8], [8], [8]]),
            "attention_mask": torch.ones((4, 3), dtype=torch.long),
            "position_ids": torch.tensor([[0, 1, 2]] * 4),
            **gen_batch.non_tensor_batch,
        }
    )
    data = data.union(generated)

    # BA slot 1, AB slot 0, BA slot 0, AB slot 1.
    data.reorder(torch.tensor([3, 0, 2, 1]))
    decoded = iter(
        [
            "<answer>A</answer>",
            "<answer>A</answer>",
            "<answer>B</answer>",
            "<answer>A</answer>",
        ]
    )

    class Tokenizer:
        def decode(self, _token_ids, skip_special_tokens=True):
            return next(decoded)

    manager = BatchRewardManager(
        tokenizer=Tokenizer(),
        num_examine=0,
        compute_score=reward_module.compute_score,
        identity_mode="explicit",
    )
    reward_tensor = manager(data)

    torch.testing.assert_close(
        reward_tensor[:, -1],
        torch.tensor([-1.8, 2.2, 2.2, 0.2]),
        atol=1e-6,
        rtol=0,
    )
