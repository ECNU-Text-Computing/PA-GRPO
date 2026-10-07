"""Strict evaluation contracts for controlled permutation experiments."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class ControlledEvaluationContractError(ValueError):
    """Raised when controlled evaluation input is ambiguous or inconsistent."""


_DIRECTIVE_CUE = re.compile(
    r"\b(?:answer|reply|respond|choose|select|single\s+letter)\b|\bX\s+is\b",
    re.IGNORECASE,
)
_NO_FURTHER_OPTION = r"(?!\s*(?:(?:,\s*)?(?:or|and)\s+|[,/|]\s*)[A-Z]\b)"
_TWO_OPTION_LIST = re.compile(r"\bA\s+or\s+B\b" + _NO_FURTHER_OPTION, re.IGNORECASE)
_FOUR_OPTION_LISTS = (
    re.compile(
        r"\bA\s+or\s+B\s+or\s+C\s+or\s+D\b" + _NO_FURTHER_OPTION,
        re.IGNORECASE,
    ),
    re.compile(
        r"\bA\s+or\s+B\s*,\s*or\s+C\s+or\s+D\b" + _NO_FURTHER_OPTION,
        re.IGNORECASE,
    ),
    re.compile(
        r"\bA\s*,\s*B\s*,\s*C\s*,?\s*(?:or|and)\s*D\b" + _NO_FURTHER_OPTION,
        re.IGNORECASE,
    ),
)
_PAIRWISE_A_ANSWER = re.compile(r"###\s*A\s+Answer\s*:", re.IGNORECASE)
_PAIRWISE_B_ANSWER = re.compile(r"###\s*B\s+Answer\s*:", re.IGNORECASE)
_PAIRWISE_QUERY = re.compile(r"###\s*Query\s*:", re.IGNORECASE)


def _prompt_messages(prompt: Any) -> list[Mapping[str, Any]]:
    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()
    if not isinstance(prompt, (list, tuple)):
        raise ControlledEvaluationContractError(
            f"prompt must be list, tuple, or numpy.ndarray, got {type(prompt).__name__}"
        )
    messages = list(prompt)
    if not messages or not all(isinstance(message, Mapping) for message in messages):
        raise ControlledEvaluationContractError("prompt must contain message mappings")
    return messages


def _has_full_four_option_structure(content: str) -> bool:
    for block in re.split(r"\n\s*\n", content):
        labels = set(
            re.findall(r"(?im)^\s*(?:option\s+)?([A-Z])\s*[\).:\]]", block)
        )
        if not set("ABCD").issubset(labels):
            continue
        unsupported_labels = labels - set("ABCD")
        if unsupported_labels:
            raise ControlledEvaluationContractError(
                f"unsupported option labels in structured choices: {sorted(unsupported_labels)}"
            )
        return True
    return False


def _contract_region(content: str) -> str:
    """Exclude candidate bodies inside the repository's pairwise Judge wrapper."""

    a_answer = _PAIRWISE_A_ANSWER.search(content)
    b_answer = _PAIRWISE_B_ANSWER.search(content)
    if not (a_answer and b_answer):
        return content
    query = _PAIRWISE_QUERY.search(content)
    boundary = query.start() if query else min(a_answer.start(), b_answer.start())
    return content[:boundary]


def _inside_labeled_option(content: str, position: int) -> bool:
    line_start = content.rfind("\n", 0, position) + 1
    escaped_line_start = content.rfind(r"\n", 0, position)
    if escaped_line_start >= line_start:
        line_start = escaped_line_start + 2
    line_prefix = content[line_start:position]
    return bool(re.match(r"^\s*(?:option\s+)?[A-D]\s*[\).:\]]", line_prefix, re.IGNORECASE))


def _directive_option_contracts(content: str) -> set[int]:
    contracts: set[int] = set()
    for option_count, patterns in ((2, (_TWO_OPTION_LIST,)), (4, _FOUR_OPTION_LISTS)):
        for pattern in patterns:
            for match in pattern.finditer(content):
                if _inside_labeled_option(content, match.start()):
                    continue
                cue_window = content[max(0, match.start() - 120) : match.start()]
                if _DIRECTIVE_CUE.search(cue_window):
                    contracts.add(option_count)
    return contracts


def infer_prompt_num_options(prompt: Any) -> int:
    """Infer a strict 2- or 4-option contract from one prompt."""

    messages = _prompt_messages(prompt)
    content = "\n".join(
        _contract_region(str(message.get("content", ""))) for message in messages
    )
    directive_contracts = _directive_option_contracts(content)
    has_two_option_contract = 2 in directive_contracts
    has_full_four_option_structure = _has_full_four_option_structure(content)
    has_four_option_contract = 4 in directive_contracts or has_full_four_option_structure
    if has_two_option_contract and has_four_option_contract:
        raise ControlledEvaluationContractError("conflicting option contracts within one prompt")
    if has_four_option_contract:
        return 4
    if has_two_option_contract:
        return 2
    raise ControlledEvaluationContractError("cannot infer an explicit 2- or 4-option contract")


def detect_dataset_num_options(file_path: str | Path) -> int:
    """Require every parquet row to declare the same option contract."""

    dataframe = pd.read_parquet(file_path)
    if "prompt" not in dataframe.columns:
        raise ControlledEvaluationContractError("dataset is missing the prompt column")
    if dataframe.empty:
        raise ControlledEvaluationContractError("cannot infer options from an empty dataset")

    inferred: set[int] = set()
    for row_position, prompt in enumerate(dataframe["prompt"]):
        try:
            inferred.add(infer_prompt_num_options(prompt))
        except ControlledEvaluationContractError as error:
            raise ControlledEvaluationContractError(f"row {row_position}: {error}") from error

    if len(inferred) != 1:
        raise ControlledEvaluationContractError(
            f"mixed option contracts across dataset rows: {sorted(inferred)}"
        )
    return inferred.pop()


def resolve_num_options(file_path: str | Path, override: str | int = "auto") -> int:
    """Resolve an explicit override or run strict whole-dataset detection."""

    if str(override) in {"2", "4"}:
        return int(override)
    if override != "auto":
        raise ControlledEvaluationContractError("num_options must be 'auto', '2', or '4'")
    return detect_dataset_num_options(file_path)


def extract_controlled_answer(response: Any, mode: str, num_options: int) -> str | None:
    """Extract only the final-answer formats allowed by the controlled contract."""

    if num_options not in {2, 4}:
        raise ControlledEvaluationContractError("num_options must be 2 or 4")
    if mode not in {"direct", "think", "thinking"}:
        raise ControlledEvaluationContractError(f"unsupported evaluation mode: {mode!r}")
    if not isinstance(response, str):
        return None

    valid_answers = set("AB" if num_options == 2 else "ABCD")
    if mode == "direct":
        answer = response.strip().upper()
        return answer if answer in valid_answers else None

    opening_tags = re.findall(r"<\s*answer\s*>", response, flags=re.IGNORECASE)
    closing_tags = re.findall(r"<\s*/\s*answer\s*>", response, flags=re.IGNORECASE)
    if len(opening_tags) != 1 or len(closing_tags) != 1:
        return None
    tagged_answers = re.findall(
        r"<answer>\s*(.*?)\s*</answer>", response, flags=re.IGNORECASE | re.DOTALL
    )
    if len(tagged_answers) != 1:
        return None
    answer = tagged_answers[0].strip().upper()
    return answer if answer in valid_answers else None
