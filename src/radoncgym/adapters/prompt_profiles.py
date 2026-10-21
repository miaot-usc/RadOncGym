

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from radoncgym.core.errors import ContractError


PROMPT_PROFILE_SCHEMA_VERSION = "language-prompt-profile-0.1.0"


def system_prompt_sha256(system_prompt: str) -> str:
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()


def load_prompt_profile(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError("prompt profile must be a JSON object")
    validate_prompt_profile(value)
    return value


def validate_prompt_profile(profile: Mapping[str, Any]) -> None:
    if profile.get("schema_version") != PROMPT_PROFILE_SCHEMA_VERSION:
        raise ContractError("unsupported prompt profile schema")
    if not str(profile.get("profile_id", "")):
        raise ContractError("prompt profile ID is required")
    digest = str(profile.get("base_system_prompt_sha256", ""))
    if len(digest) != 64:
        raise ContractError("prompt profile base hash must be SHA-256")
    appendix = profile.get("instruction_appendix")
    if not isinstance(appendix, str) or not appendix.strip():
        raise ContractError("prompt profile instruction appendix is required")
    if len(appendix) > 6000:
        raise ContractError("prompt profile instruction appendix is too long")
    forbidden = ("reference_dose", "hidden_m1", "ground_truth", "oracle_score")
    lowered = appendix.lower()
    if any(token in lowered for token in forbidden):
        raise ContractError("prompt profile requests forbidden held-out information")
    search = profile.get("search_provenance")
    if not isinstance(search, Mapping):
        raise ContractError("prompt profile search provenance is required")
    if search.get("frozen_before_test") is not True:
        raise ContractError("prompt profile must be frozen before held-out test")
    splits = search.get("optimization_splits")
    if not isinstance(splits, list) or not splits:
        raise ContractError("prompt profile optimization splits are required")
    if not set(str(value) for value in splits).issubset({"train", "validation"}):
        raise ContractError("prompt profile optimization used an invalid split")


def apply_prompt_profile(
    base_system_prompt: str,
    profile: Mapping[str, Any] | None,
) -> str:
    if profile is None:
        return base_system_prompt
    validate_prompt_profile(profile)
    expected = str(profile["base_system_prompt_sha256"])
    actual = system_prompt_sha256(base_system_prompt)
    if expected != actual:
        raise ContractError("prompt profile was compiled for a different base prompt")
    return (
        base_system_prompt.rstrip()
        + "\n\nVersioned planning instruction appendix:\n"
        + str(profile["instruction_appendix"]).strip()
        + "\n"
    )
