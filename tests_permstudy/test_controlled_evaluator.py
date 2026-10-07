import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _judge_prompt(container_type=list):
    messages = [
        {"role": "system", "content": "Reply with only A or B."},
        {
            "role": "user",
            "content": "Which response is better?\nA: Candidate C is concise.\nB: Candidate D is detailed.\nAnswer with A or B only.",
        },
    ]
    return container_type(messages)


def _mcq_prompt(container_type=list):
    messages = [
        {
            "role": "user",
            "content": (
                "Choose one option.\n"
                "A. Alpha\n"
                "B. Beta\n"
                "C. Gamma\n"
                "D. Delta\n"
                "Answer with A, B, C, or D only."
            ),
        }
    ]
    return container_type(messages)


@pytest.mark.parametrize("container_type", [list, tuple, lambda value: np.asarray(value, dtype=object)])
def test_controlled_prompt_detector_supports_parquet_container_shapes(container_type):
    """Catches treating ndarray prompts as unknown and silently defaulting to four options."""
    from permstudy.controlled_evaluator import infer_prompt_num_options

    assert infer_prompt_num_options(_judge_prompt(container_type)) == 2
    assert infer_prompt_num_options(_mcq_prompt(container_type)) == 4


def test_controlled_prompt_detector_ignores_incidental_c_and_d_letters():
    """Catches ordinary candidate text being mistaken for a four-option structure."""
    from permstudy.controlled_evaluator import infer_prompt_num_options

    assert infer_prompt_num_options(_judge_prompt()) == 2


def test_controlled_prompt_detector_rejects_conflicting_evidence_within_one_prompt():
    """Catches a four-option structure silently overriding an explicit two-option directive."""
    from permstudy.controlled_evaluator import ControlledEvaluationContractError, infer_prompt_num_options

    conflicting = [
        {"role": "system", "content": "Reply with only A or B."},
        _mcq_prompt()[0],
    ]

    with pytest.raises(ControlledEvaluationContractError, match="conflicting option contracts"):
        infer_prompt_num_options(conflicting)


def test_controlled_prompt_detector_reads_complete_or_chained_option_directive():
    """Catches the A-or-B prefix of a four-option directive being misclassified as Judge."""
    from permstudy.controlled_evaluator import infer_prompt_num_options

    prompt = [{"role": "user", "content": "Choose A or B or C or D."}]

    assert infer_prompt_num_options(prompt) == 4


def test_controlled_prompt_detector_reads_comma_separated_or_chained_four_option_directive():
    """Catches a punctuated four-option list being accepted as an A/B prefix."""
    from permstudy.controlled_evaluator import infer_prompt_num_options

    prompt = [{"role": "user", "content": "Choose A or B, or C or D."}]

    assert infer_prompt_num_options(prompt) == 4


@pytest.mark.parametrize(
    "content",
    [
        "Choose A or B or D.",
        "Choose A or B or C or D or E.",
        "Choose A, B, C, D, or E.",
    ],
)
def test_controlled_prompt_detector_rejects_extended_or_unsupported_option_lists(content):
    """Catches a supported option-list prefix hiding extra or unsupported choices."""
    from permstudy.controlled_evaluator import (
        ControlledEvaluationContractError,
        infer_prompt_num_options,
    )

    with pytest.raises(ControlledEvaluationContractError, match="cannot infer"):
        infer_prompt_num_options([{"role": "user", "content": content}])


def test_controlled_prompt_detector_rejects_option_structure_extended_beyond_d():
    """Catches a five-option structure being accepted merely because it contains A-D."""
    from permstudy.controlled_evaluator import (
        ControlledEvaluationContractError,
        infer_prompt_num_options,
    )

    prompt = [
        {
            "role": "user",
            "content": "Choose one option.\nA. Alpha\nB. Beta\nC. Gamma\nD. Delta\nE. Epsilon",
        }
    ]

    with pytest.raises(ControlledEvaluationContractError, match="unsupported option labels"):
        infer_prompt_num_options(prompt)


def test_controlled_prompt_detector_validates_structure_even_with_four_option_directive():
    """Catches an explicit A-D directive short-circuiting validation of A-E choices."""
    from permstudy.controlled_evaluator import (
        ControlledEvaluationContractError,
        infer_prompt_num_options,
    )

    prompt = [
        {"role": "system", "content": "Answer with A, B, C, or D only."},
        {
            "role": "user",
            "content": "Choose one option.\nA. Alpha\nB. Beta\nC. Gamma\nD. Delta\nE. Epsilon",
        },
    ]

    with pytest.raises(ControlledEvaluationContractError, match="unsupported option labels"):
        infer_prompt_num_options(prompt)


def test_controlled_prompt_detector_scopes_extra_labels_to_the_option_block():
    """Catches transcript speaker labels outside the choices being treated as options."""
    from permstudy.controlled_evaluator import infer_prompt_num_options

    prompt = [
        {
            "role": "user",
            "content": (
                "M: Ask a question. W: Give an answer.\n\n"
                "Options:\nA. Alpha\nB. Beta\nC. Gamma\nD. Delta"
            ),
        }
    ]

    assert infer_prompt_num_options(prompt) == 4


def test_controlled_prompt_detector_ignores_choice_lists_inside_option_text():
    """Catches an MCQ option such as 'A. either A or B' being treated as a Judge directive."""
    from permstudy.controlled_evaluator import infer_prompt_num_options

    prompt = [
        {
            "role": "user",
            "content": (
                "Choose the best answer.\n"
                "A. either A or B\n"
                "B. both values\n"
                "C. neither value\n"
                "D. cannot determine"
            ),
        }
    ]

    assert infer_prompt_num_options(prompt) == 4


@pytest.mark.parametrize(
    "content",
    [
        "Reply with only A or B.\nChoose a better response.",
        "Reply with A or B; candidate C and candidate D are names, not choices.",
    ],
)
def test_controlled_prompt_detector_ignores_prose_outside_explicit_choice_list(content):
    """Catches incidental A-D words after a valid directive changing its contract."""
    from permstudy.controlled_evaluator import infer_prompt_num_options

    assert infer_prompt_num_options([{"role": "system", "content": content}]) == 2


def test_controlled_prompt_detector_accepts_bundled_pairwise_judge_prompts_with_nested_options():
    """Catches candidate prose in the real Judge schema contaminating option detection."""
    from permstudy.controlled_evaluator import infer_prompt_num_options

    repository_root = Path(__file__).resolve().parents[1]
    dataset_path = repository_root / "dataset" / "train" / "chatbot_arena_filter_2perm_think.parquet"
    prompts = pd.read_parquet(dataset_path, columns=["prompt"]).iloc[[0, 40]]["prompt"]

    assert [infer_prompt_num_options(prompt) for prompt in prompts] == [2, 2]


def test_controlled_file_detector_scans_every_row_and_rejects_mixed_contract(tmp_path):
    """Catches selecting a dataset contract from only the first parquet row."""
    from permstudy.controlled_evaluator import ControlledEvaluationContractError, detect_dataset_num_options

    path = tmp_path / "mixed.parquet"
    pd.DataFrame({"prompt": [_judge_prompt(), _mcq_prompt()]}).to_parquet(path, index=False)

    with pytest.raises(ControlledEvaluationContractError, match="mixed option contracts"):
        detect_dataset_num_options(path)


def test_controlled_file_detector_rejects_any_ambiguous_row(tmp_path):
    """Catches ambiguous rows being hidden by otherwise consistent rows."""
    from permstudy.controlled_evaluator import ControlledEvaluationContractError, detect_dataset_num_options

    ambiguous = [{"role": "user", "content": "Which response is preferable? Explain briefly."}]
    path = tmp_path / "ambiguous.parquet"
    pd.DataFrame({"prompt": [_judge_prompt(), ambiguous]}).to_parquet(path, index=False)

    with pytest.raises(ControlledEvaluationContractError, match="row 1.*cannot infer"):
        detect_dataset_num_options(path)


def test_explicit_num_options_override_bypasses_auto_detection(tmp_path):
    """Catches explicit benchmark configuration accidentally re-entering heuristic detection."""
    from permstudy.controlled_evaluator import resolve_num_options

    path = tmp_path / "ambiguous.parquet"
    pd.DataFrame({"prompt": [[{"role": "user", "content": "Choose carefully."}]]}).to_parquet(path, index=False)

    assert resolve_num_options(path, "2") == 2
    assert resolve_num_options(path, "4") == 4


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("A", "A"),
        (" A \n", "A"),
        ("b", "B"),
        ("A.", None),
        ("Answer: A", None),
        ("I think the answer might be A", None),
        ("invalid", None),
    ],
)
def test_controlled_direct_parser_accepts_only_one_bare_legal_letter(response, expected):
    """Catches prose or punctuation being promoted to a controlled answer."""
    from permstudy.controlled_evaluator import extract_controlled_answer

    assert extract_controlled_answer(response, mode="direct", num_options=2) == expected


@pytest.mark.parametrize(
    ("response", "num_options", "expected"),
    [
        ("<think>reasoning</think><answer>A</answer>", 2, "A"),
        ("<thinking>reasoning</thinking><answer>d</answer>", 4, "D"),
        ("reasoning without a final tag", 2, None),
        ("<answer>A</answer><answer>A</answer>", 2, None),
        ("<answer>A</answer> then <answer>B</answer>", 2, None),
        ("<answer>A</answer><answer>B", 2, None),
        ("<answer>A</answer></answer>", 2, None),
        ("<answer>C</answer>", 2, None),
        ("<answer>A.</answer>", 2, None),
    ],
)
def test_controlled_thinking_parser_requires_exactly_one_legal_answer_tag(response, num_options, expected):
    """Catches missing, duplicate, conflicting, or out-of-contract final tags."""
    from permstudy.controlled_evaluator import extract_controlled_answer

    assert extract_controlled_answer(response, mode="thinking", num_options=num_options) == expected


def test_official_contract_keeps_existing_detector_and_parser_behavior(tmp_path):
    """Catches controlled hardening silently changing official reproduction defaults."""
    from evaluation.evaluate_models import (
        detect_num_options_from_file,
        extract_answer_for_contract,
        extract_answer_from_response,
        resolve_evaluation_num_options,
    )

    path = tmp_path / "judge.parquet"
    pd.DataFrame({"prompt": [_judge_prompt()]}).to_parquet(path, index=False)

    assert detect_num_options_from_file(path) == 4
    assert resolve_evaluation_num_options(path, evaluation_contract="official", num_options_override="auto") == 4
    assert resolve_evaluation_num_options(path, evaluation_contract="controlled", num_options_override="auto") == 2
    assert extract_answer_from_response("invalid", mode="direct", num_options=2) == "A"
    assert extract_answer_for_contract("invalid", mode="direct", num_options=2, evaluation_contract="official") == "A"
    assert extract_answer_for_contract("invalid", mode="direct", num_options=2, evaluation_contract="controlled") is None


def test_evaluator_cli_defaults_to_official_and_auto_options():
    """Catches old commands changing contract merely because new flags exist."""
    from evaluation.evaluate_models import build_argument_parser

    parser = build_argument_parser()
    default_args = parser.parse_args(["--model_path", "model", "--mode", "direct"])
    controlled_args = parser.parse_args(
        [
            "--model_path",
            "model",
            "--mode",
            "direct",
            "--evaluation_contract",
            "controlled",
            "--num_options",
            "2",
        ]
    )

    assert default_args.evaluation_contract == "official"
    assert default_args.num_options == "auto"
    assert controlled_args.evaluation_contract == "controlled"
    assert controlled_args.num_options == "2"


def test_evaluator_script_help_runs_from_repository_root():
    """Catches direct-script imports losing access to the repository-local permstudy package."""
    repository_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "evaluation/evaluate_models.py", "--help"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--evaluation_contract" in completed.stdout
    assert "--num_options" in completed.stdout


class _SequentialDiagnosticModel:
    def generate(self, messages, **kwargs):
        return {
            "response": "I think the answer might be A",
            "thinking_content": "",
            "answer_probability": {"A": 0.99, "B": 0.01},
        }


class _BatchDiagnosticModel:
    def generate_batch(self, prompts, **kwargs):
        return [
            {
                "response": "I think the answer might be A",
                "thinking_content": "",
                "answer_probability": {"A": 0.99, "B": 0.01},
                "answer_debug_info": {"located_token": "A"},
            }
            for _ in prompts
        ]


def _write_evaluator_fixture(path):
    pd.DataFrame(
        {
            "prompt": [_judge_prompt()],
            "reward_model": [{"ground_truth": "A"}],
            "data_source": ["judge"],
            "ability": ["reasoning"],
            "extra_info": [{"pair_id": "pair-1", "permutation_id": 0}],
        }
    ).to_parquet(path, index=False)


@pytest.mark.parametrize(
    ("evaluator_name", "model"),
    [
        ("evaluate_dataset_sequential", _SequentialDiagnosticModel()),
        ("evaluate_dataset_batch", _BatchDiagnosticModel()),
    ],
)
def test_controlled_evaluators_ignore_answer_probability_for_answer_and_accuracy(
    tmp_path, evaluator_name, model
):
    """Catches token-probability diagnostics overriding strict controlled parsing."""
    from evaluation import evaluate_models

    dataset_path = tmp_path / "judge.parquet"
    output_path = tmp_path / f"{evaluator_name}.json"
    _write_evaluator_fixture(dataset_path)

    evaluator = getattr(evaluate_models, evaluator_name)
    summary = evaluator(
        model,
        dataset_path,
        mode="direct",
        num_options=2,
        output_path=output_path,
        evaluation_contract="controlled",
    )
    output = json.loads(output_path.read_text(encoding="utf-8"))

    assert output["evaluation_contract"] == "controlled"
    assert output["results"][0]["extracted_answer"] is None
    assert output["results"][0]["is_correct"] is False
    assert output["results"][0]["answer_probability"]["A"] == 0.99
    assert output["accuracy"] == 0.0
    assert summary["evaluation_contract"] == "controlled"
    assert summary["accuracy"] == 0.0


def test_default_sequential_evaluator_still_uses_official_parser(tmp_path):
    """Catches the new controlled parser becoming the implicit default."""
    from evaluation.evaluate_models import evaluate_dataset_sequential

    dataset_path = tmp_path / "judge.parquet"
    output_path = tmp_path / "official.json"
    _write_evaluator_fixture(dataset_path)

    summary = evaluate_dataset_sequential(
        _SequentialDiagnosticModel(),
        dataset_path,
        mode="direct",
        num_options=2,
        output_path=output_path,
    )
    output = json.loads(output_path.read_text(encoding="utf-8"))

    assert output["evaluation_contract"] == "official"
    assert output["results"][0]["extracted_answer"] == "A"
    assert output["results"][0]["is_correct"] is True
    assert summary["evaluation_contract"] == "official"
    assert summary["accuracy"] == 1.0
