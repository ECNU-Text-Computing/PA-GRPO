import torch


def test_equal_rewards_produce_zero_advantage_with_or_without_gate():
    from permstudy.advantages import global_advantage_paper

    rewards = torch.tensor([1.0, 1.0, 1.0, 1.0])
    groups = ["pair-1"] * 4

    ungated = global_advantage_paper(rewards, groups)
    gated = global_advantage_paper(rewards, groups, sigma_gate_threshold=0.01)

    torch.testing.assert_close(ungated.advantage, torch.zeros(4))
    torch.testing.assert_close(gated.advantage, torch.zeros(4))
    assert gated.gate_mask.tolist() == [True, True, True, True]


def test_near_zero_variance_is_nonzero_without_gate_and_zero_with_gate():
    from permstudy.advantages import global_advantage_paper

    rewards = torch.tensor([1.0, 1.0, 1.0001, 1.0])
    groups = ["pair-1"] * 4

    ungated = global_advantage_paper(rewards, groups)
    gated = global_advantage_paper(rewards, groups, sigma_gate_threshold=0.001)

    assert torch.count_nonzero(ungated.advantage).item() == 4
    torch.testing.assert_close(gated.advantage, torch.zeros(4))
    assert gated.gate_mask.tolist() == [True, True, True, True]


def test_gate_preserves_standardized_advantage_above_threshold():
    from permstudy.advantages import global_advantage_paper

    rewards = torch.tensor([1.0, 1.0, -1.0, -1.0])
    groups = ["pair-1"] * 4

    ungated = global_advantage_paper(rewards, groups)
    gated = global_advantage_paper(rewards, groups, sigma_gate_threshold=0.1)

    torch.testing.assert_close(gated.advantage, ungated.advantage)
    assert gated.gate_mask.tolist() == [False, False, False, False]
