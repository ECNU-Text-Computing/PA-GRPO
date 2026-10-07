import torch


def test_equal_lengths_match_hand_derived_global_direction():
    from permstudy.advantages import global_advantage_official_compatible, global_advantage_paper

    rewards = torch.tensor([1.0, 1.0, -1.0, -1.0])
    groups = ["pair-1"] * 4
    response_mask = torch.ones((4, 8))

    paper = global_advantage_paper(rewards, groups)
    official = global_advantage_official_compatible(rewards, groups, response_mask)

    expected = torch.tensor([1.0, 1.0, -1.0, -1.0])
    torch.testing.assert_close(paper.advantage, expected, atol=2e-6, rtol=0)
    torch.testing.assert_close(official.advantage, expected, atol=1e-6, rtol=0)


def test_only_changing_lengths_changes_official_but_not_paper_advantage():
    from permstudy.advantages import global_advantage_official_compatible, global_advantage_paper

    rewards = torch.tensor([1.0, 1.0, -1.0, -1.0])
    groups = ["pair-1"] * 4
    unequal_mask = torch.tensor(
        [
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.float32,
    )

    paper = global_advantage_paper(rewards, groups)
    official = global_advantage_official_compatible(rewards, groups, unequal_mask)

    expected_paper = torch.tensor([1.0, 1.0, -1.0, -1.0])
    expected_official = torch.tensor([0.4472136, 1.3416408, -0.4472136, -1.3416408])
    torch.testing.assert_close(paper.advantage, expected_paper, atol=2e-6, rtol=0)
    torch.testing.assert_close(official.advantage, expected_official, atol=2e-6, rtol=0)
    assert not torch.allclose(official.advantage, paper.advantage)
