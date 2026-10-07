"""Controlled rollout identity propagation for the real verl trainer path."""

from __future__ import annotations

from typing import Any

import numpy as np

from permstudy.ids import RolloutId


LEGACY_IDENTITY_MODE = "legacy_index"
EXPLICIT_IDENTITY_MODE = "explicit"
IDENTITY_KEYS = ("pair_id", "permutation_id", "rollout_slot")


class IdentityMetadataError(ValueError):
    """Raised when controlled-mode source metadata cannot form an identity."""


class RolloutIdentityAlignmentError(RuntimeError):
    """Raised when a generation backend loses or reorders rollout identity."""


def validate_identity_mode(identity_mode: str) -> str:
    if identity_mode not in {LEGACY_IDENTITY_MODE, EXPLICIT_IDENTITY_MODE}:
        raise ValueError(
            f"identity_mode must be {LEGACY_IDENTITY_MODE!r} or {EXPLICIT_IDENTITY_MODE!r}, got {identity_mode!r}"
        )
    return identity_mode


def ensure_supported_trainer_identity_mode(
    identity_mode: str,
    *,
    async_rollout_mode: bool,
    advantage_estimator: Any,
    rollout_backend: str,
) -> None:
    """Fail before rollout when an explicit identity path is not integrated."""

    validate_identity_mode(identity_mode)
    if identity_mode == LEGACY_IDENTITY_MODE:
        return
    if async_rollout_mode:
        raise NotImplementedError(
            "explicit rollout identity does not yet support async rollout; use synchronous rollout or legacy_index"
        )
    if rollout_backend != "vllm":
        raise NotImplementedError(
            "explicit rollout identity currently requires synchronous vLLM so generated prompt order can be "
            "validated; use rollout.name=vllm or legacy_index"
        )
    estimator_name = getattr(advantage_estimator, "value", advantage_estimator)
    if str(estimator_name).lower() == "remax":
        raise NotImplementedError(
            "explicit rollout identity does not yet support the REMAX baseline generation path"
        )


def _metadata_value(extra: Any, *keys: str) -> Any:
    for key in keys:
        if isinstance(extra, dict) and key in extra:
            return extra[key]
        if hasattr(extra, "get"):
            value = extra.get(key, None)
            if value is not None:
                return value
        if hasattr(extra, key):
            return getattr(extra, key)
    return None


def attach_permutation_identity(batch: Any, identity_mode: str) -> Any:
    """Attach explicit pair/permutation fields at the dataset-to-DataProto boundary.

    Controlled mode accepts a dataset-provided ``pair_id`` or
    ``original_question_id``. It deliberately has no ``index // 2`` fallback.
    Legacy mode is a no-op so the official reproduction batch stays unchanged.
    """

    validate_identity_mode(identity_mode)
    if identity_mode == LEGACY_IDENTITY_MODE:
        return batch

    extras = batch.non_tensor_batch.get("extra_info")
    if extras is None:
        raise IdentityMetadataError("explicit identity requires extra_info metadata")

    pair_ids: list[str] = []
    permutation_ids: list[int] = []
    for position, extra in enumerate(extras):
        pair_id = _metadata_value(extra, "pair_id", "original_question_id")
        if pair_id is None:
            raise IdentityMetadataError(
                f"explicit identity at position {position} requires pair_id or original_question_id"
            )
        permutation_id = _metadata_value(extra, "permutation_id", "permutation")
        if permutation_id is None:
            raise IdentityMetadataError(
                f"explicit identity at position {position} requires permutation_id or permutation"
            )
        try:
            permutation_id = int(permutation_id)
        except (TypeError, ValueError) as exc:
            raise IdentityMetadataError(
                f"permutation_id at position {position} must be integer-like, got {permutation_id!r}"
            ) from exc
        if permutation_id not in (0, 1):
            raise IdentityMetadataError(
                f"permutation_id at position {position} must be 0 or 1, got {permutation_id}"
            )
        pair_ids.append(str(pair_id))
        permutation_ids.append(permutation_id)

    batch.non_tensor_batch["pair_id"] = np.asarray(pair_ids, dtype=object)
    batch.non_tensor_batch["permutation_id"] = np.asarray(permutation_ids, dtype=np.int64)
    return batch


def repeat_for_rollout(batch: Any, repeat_times: int, identity_mode: str) -> Any:
    """Repeat prompt rows and assign ``0..n-1`` slots inside each prompt row."""

    validate_identity_mode(identity_mode)
    if repeat_times <= 0:
        raise ValueError(f"repeat_times must be positive, got {repeat_times}")

    repeated = batch.repeat(repeat_times=repeat_times, interleave=True)
    if identity_mode == LEGACY_IDENTITY_MODE:
        return repeated

    for key in ("pair_id", "permutation_id"):
        if key not in batch.non_tensor_batch:
            raise IdentityMetadataError(f"explicit identity requires {key} before rollout repeat")

    original_batch_size = len(batch)
    repeated.non_tensor_batch["rollout_slot"] = np.tile(
        np.arange(repeat_times, dtype=np.int64), original_batch_size
    )
    return repeated


def snapshot_rollout_identity(batch: Any, identity_mode: str) -> dict[str, np.ndarray] | None:
    """Copy expected identity before handing a batch to a generation backend."""

    validate_identity_mode(identity_mode)
    if identity_mode == LEGACY_IDENTITY_MODE:
        return None
    missing = [key for key in IDENTITY_KEYS if key not in batch.non_tensor_batch]
    if missing:
        raise IdentityMetadataError(f"missing rollout identity fields: {', '.join(missing)}")
    return {key: batch.non_tensor_batch[key].copy() for key in IDENTITY_KEYS}


def validate_generation_identity(generation_input: Any, generation_output: Any, identity_mode: str) -> None:
    """Prove that the actual generation result retained input identity and order."""

    validate_identity_mode(identity_mode)
    if identity_mode == LEGACY_IDENTITY_MODE:
        return
    if isinstance(generation_input, dict):
        expected_identity = generation_input
        input_length = len(next(iter(expected_identity.values())))
    else:
        expected_identity = generation_input.non_tensor_batch
        input_length = len(generation_input)
    if input_length != len(generation_output):
        raise RolloutIdentityAlignmentError(
            "generation output changed rollout batch size: "
            f"input={input_length}, output={len(generation_output)}"
        )

    for key in IDENTITY_KEYS:
        expected = expected_identity.get(key)
        actual = generation_output.non_tensor_batch.get(key)
        if expected is None or actual is None:
            raise RolloutIdentityAlignmentError(f"generation output did not preserve {key}")
        if not np.array_equal(expected, actual):
            raise RolloutIdentityAlignmentError(
                f"generation output reordered rollout identity for {key}: "
                f"expected={expected.tolist()!r}, actual={actual.tolist()!r}"
            )


def validate_backend_prompt_order(inputs: list[dict[str, Any]], outputs: list[Any]) -> None:
    """Bind backend response objects to input rows using their echoed prompt IDs."""

    if len(inputs) != len(outputs):
        raise RolloutIdentityAlignmentError(
            f"generation backend changed request count: input={len(inputs)}, output={len(outputs)}"
        )
    for position, (input_data, output) in enumerate(zip(inputs, outputs, strict=True)):
        expected_prompt_ids = list(input_data["prompt_token_ids"])
        actual_prompt_ids = getattr(output, "prompt_token_ids", None)
        if actual_prompt_ids is None or list(actual_prompt_ids) != expected_prompt_ids:
            raise RolloutIdentityAlignmentError(
                "generation backend output prompt order does not match rollout identity order "
                f"at position {position}"
            )


def rollout_ids_from_batch(batch: Any) -> list[RolloutId]:
    """Read explicit rollout identities after any DataProto reorder."""

    missing = [key for key in IDENTITY_KEYS if key not in batch.non_tensor_batch]
    if missing:
        raise IdentityMetadataError(f"missing rollout identity fields: {', '.join(missing)}")
    return [
        RolloutId(str(pair_id), int(permutation_id), int(rollout_slot))
        for pair_id, permutation_id, rollout_slot in zip(
            batch.non_tensor_batch["pair_id"],
            batch.non_tensor_batch["permutation_id"],
            batch.non_tensor_batch["rollout_slot"],
            strict=True,
        )
    ]


def merge_identity_into_extra_infos(batch: Any) -> list[dict[str, Any]]:
    """Copy top-level DataProto identity fields into reward-function metadata."""

    rollout_ids = rollout_ids_from_batch(batch)
    source_extras = batch.non_tensor_batch.get("extra_info", np.array([{}] * len(batch), dtype=object))
    merged: list[dict[str, Any]] = []
    for extra, rollout_id in zip(source_extras, rollout_ids, strict=True):
        copied = dict(extra) if isinstance(extra, dict) else {}
        copied.update(
            pair_id=rollout_id.pair_id,
            permutation_id=rollout_id.permutation_id,
            rollout_slot=rollout_id.rollout_slot,
        )
        merged.append(copied)
    return merged
