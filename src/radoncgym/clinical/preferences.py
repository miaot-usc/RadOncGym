

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from radoncgym.core.constants import OBJECTIVE_NAMES
from radoncgym.core.errors import ContractError


CLINICAL_PREFERENCE_SCHEMA_VERSION = "clinical-preference-profile-0.1.0"
DEFAULT_CLINICAL_PROFILE_ID = "openkbp_hn_v1"


@dataclass(frozen=True, slots=True)
class ClinicalPreferenceProfile:
    pass

    profile_id: str
    label: str
    objective_emphasis: tuple[float, ...]
    priority_tiers: tuple[tuple[str, ...], ...]
    description: str
    hard_constraint_policy_id: str = "openkbp_hn_protocol_v1"
    source_method: str = "preference_conditioned_policy_state"
    schema_version: str = CLINICAL_PREFERENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.profile_id or not self.label or not self.description:
            raise ContractError("clinical preference identity and description are required")
        if self.schema_version != CLINICAL_PREFERENCE_SCHEMA_VERSION:
            raise ContractError("unsupported clinical preference schema")
        if len(self.objective_emphasis) != len(OBJECTIVE_NAMES):
            raise ContractError("clinical preference vector has wrong dimension")
        if any(
            not math.isfinite(value) or value < 0.0 or value > 1.0
            for value in self.objective_emphasis
        ):
            raise ContractError("clinical preference emphasis must lie in [0, 1]")
        flattened = tuple(name for tier in self.priority_tiers for name in tier)
        if len(flattened) != len(set(flattened)):
            raise ContractError("clinical preference priority tiers contain duplicates")
        if set(flattened) != set(OBJECTIVE_NAMES):
            raise ContractError("clinical preference priority tiers must cover all objectives")

    def emphasis_dict(self) -> dict[str, float]:
        return dict(zip(OBJECTIVE_NAMES, self.objective_emphasis, strict=True))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "label": self.label,
            "description": self.description,
            "hard_constraint_policy_id": self.hard_constraint_policy_id,
            "hard_constraint_thresholds_unchanged": True,
            "source_method": self.source_method,
            "objective_emphasis": self.emphasis_dict(),
            "priority_tiers": [list(tier) for tier in self.priority_tiers],
            "semantics": (
                "Public relative task preference in [0,1], not an optimizer "
                "weight, dose prescription, or permission to violate protocol."
            ),
        }


def _vector(**values: float) -> tuple[float, ...]:
    missing = sorted(set(OBJECTIVE_NAMES) - set(values))
    extra = sorted(set(values) - set(OBJECTIVE_NAMES))
    if missing or extra:
        raise RuntimeError(f"invalid built-in preference vector: missing={missing}, extra={extra}")
    return tuple(float(values[name]) for name in OBJECTIVE_NAMES)


_TARGETS = ("PTV70_coverage", "PTV63_coverage", "PTV56_coverage")
_SERIAL = ("SpinalCord_sparing", "Brainstem_sparing", "Mandible_sparing")
_PAROTIDS = ("RightParotid_sparing", "LeftParotid_sparing")
_OTHER_SOFT = (
    "Esophagus_sparing",
    "Larynx_sparing",
    "ring_falloff",
    "posterior_neck_sparing",
)


_PROFILES = (
    ClinicalPreferenceProfile(
        profile_id=DEFAULT_CLINICAL_PROFILE_ID,
        label="Balanced head-and-neck planning",
        description=(
            "Maintain target coverage and serial-OAR limits, then balance the "
            "remaining OAR and falloff goals without a side-specific preference."
        ),
        objective_emphasis=_vector(**{name: 0.5 for name in OBJECTIVE_NAMES}),
        priority_tiers=(_TARGETS, _SERIAL, _PAROTIDS + _OTHER_SOFT),
    ),
    ClinicalPreferenceProfile(
        profile_id="openkbp_hn_target_first_v1",
        label="Target-coverage emphasis",
        description=(
            "Prefer robust PTV coverage after preserving all fixed protocol "
            "requirements; discretionary OAR gains must not trade away coverage."
        ),
        objective_emphasis=_vector(
            PTV56_coverage=1.0,
            PTV63_coverage=1.0,
            PTV70_coverage=1.0,
            Brainstem_sparing=0.75,
            SpinalCord_sparing=0.75,
            RightParotid_sparing=0.35,
            LeftParotid_sparing=0.35,
            Esophagus_sparing=0.35,
            Larynx_sparing=0.35,
            Mandible_sparing=0.65,
            ring_falloff=0.5,
            posterior_neck_sparing=0.35,
        ),
        priority_tiers=(_TARGETS, _SERIAL, _PAROTIDS + _OTHER_SOFT),
    ),
    ClinicalPreferenceProfile(
        profile_id="openkbp_hn_bilateral_parotid_v1",
        label="Bilateral parotid-sparing emphasis",
        description=(
            "After fixed target and serial-OAR requirements, prefer reducing "
            "mean dose to both parotids before other discretionary OAR goals."
        ),
        objective_emphasis=_vector(
            PTV56_coverage=1.0,
            PTV63_coverage=1.0,
            PTV70_coverage=1.0,
            Brainstem_sparing=0.85,
            SpinalCord_sparing=0.85,
            RightParotid_sparing=1.0,
            LeftParotid_sparing=1.0,
            Esophagus_sparing=0.4,
            Larynx_sparing=0.4,
            Mandible_sparing=0.7,
            ring_falloff=0.5,
            posterior_neck_sparing=0.4,
        ),
        priority_tiers=(_TARGETS, _SERIAL, _PAROTIDS, _OTHER_SOFT),
    ),
    ClinicalPreferenceProfile(
        profile_id="openkbp_hn_left_parotid_v1",
        label="Left-parotid-sparing emphasis",
        description=(
            "After fixed target and serial-OAR requirements, prefer reducing "
            "left-parotid mean dose before other discretionary OAR goals."
        ),
        objective_emphasis=_vector(
            PTV56_coverage=1.0,
            PTV63_coverage=1.0,
            PTV70_coverage=1.0,
            Brainstem_sparing=0.85,
            SpinalCord_sparing=0.85,
            RightParotid_sparing=0.4,
            LeftParotid_sparing=1.0,
            Esophagus_sparing=0.4,
            Larynx_sparing=0.4,
            Mandible_sparing=0.7,
            ring_falloff=0.5,
            posterior_neck_sparing=0.4,
        ),
        priority_tiers=(_TARGETS, _SERIAL, ("LeftParotid_sparing",), ("RightParotid_sparing",) + _OTHER_SOFT),
    ),
    ClinicalPreferenceProfile(
        profile_id="openkbp_hn_right_parotid_v1",
        label="Right-parotid-sparing emphasis",
        description=(
            "After fixed target and serial-OAR requirements, prefer reducing "
            "right-parotid mean dose before other discretionary OAR goals."
        ),
        objective_emphasis=_vector(
            PTV56_coverage=1.0,
            PTV63_coverage=1.0,
            PTV70_coverage=1.0,
            Brainstem_sparing=0.85,
            SpinalCord_sparing=0.85,
            RightParotid_sparing=1.0,
            LeftParotid_sparing=0.4,
            Esophagus_sparing=0.4,
            Larynx_sparing=0.4,
            Mandible_sparing=0.7,
            ring_falloff=0.5,
            posterior_neck_sparing=0.4,
        ),
        priority_tiers=(_TARGETS, _SERIAL, ("RightParotid_sparing",), ("LeftParotid_sparing",) + _OTHER_SOFT),
    ),
)


def clinical_preference_profiles() -> tuple[ClinicalPreferenceProfile, ...]:
    pass

    return _PROFILES


def clinical_preference_profile(profile_id: str) -> ClinicalPreferenceProfile:
    pass

    for profile in _PROFILES:
        if profile.profile_id == profile_id:
            return profile
    raise ContractError(f"unknown clinical preference profile: {profile_id!r}")
