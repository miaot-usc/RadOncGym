

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from radoncgym.core.errors import ContractError
from radoncgym.core.models import PlanningContext


class PredictionTier(str, Enum):
    pass

    CANONICAL = "canonical"
    OUTLIER_STRESS = "outlier_stress"


@dataclass(frozen=True, slots=True)
class PredictionSetSpec:
    pass

    prediction_set_id: str
    tier: PredictionTier
    official_rank: int
    included_in_openkbp_opt_analysis: bool
    note: str

    def __post_init__(self) -> None:
        if not self.prediction_set_id.startswith("set_"):
            raise ContractError(
                f"Invalid prediction set ID: {self.prediction_set_id!r}"
            )
        if self.official_rank <= 0:
            raise ContractError("official_rank must be positive")
        if not self.note:
            raise ContractError("prediction-set note is required")


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    pass

    name: str
    version: str
    source_repository: str
    source_commit: str
    root_hint: str
    patients: tuple[str, ...]
    splits: tuple[tuple[str, tuple[str, ...]], ...]
    prediction_sets: tuple[PredictionSetSpec, ...]
    required_reference_files: tuple[str, ...]
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        required_strings = {
            "name": self.name,
            "version": self.version,
            "source_repository": self.source_repository,
            "source_commit": self.source_commit,
            "root_hint": self.root_hint,
        }
        for name, value in required_strings.items():
            if not value:
                raise ContractError(f"{name} must be non-empty")

        if not self.patients or len(self.patients) != len(set(self.patients)):
            raise ContractError("patients must be non-empty and unique")
        if any(not patient.startswith("pt_") for patient in self.patients):
            raise ContractError("all patient IDs must start with 'pt_'")

        split_names = [name for name, _ in self.splits]
        if not split_names or len(split_names) != len(set(split_names)):
            raise ContractError("split names must be non-empty and unique")
        split_patients = [
            patient for _, patients in self.splits for patient in patients
        ]
        if len(split_patients) != len(set(split_patients)):
            raise ContractError("patient splits must be disjoint")
        if set(split_patients) != set(self.patients):
            missing = sorted(set(self.patients) - set(split_patients))
            extra = sorted(set(split_patients) - set(self.patients))
            raise ContractError(
                f"split coverage mismatch: missing={missing}, extra={extra}"
            )

        prediction_ids = [spec.prediction_set_id for spec in self.prediction_sets]
        prediction_ranks = [spec.official_rank for spec in self.prediction_sets]
        if not prediction_ids or len(prediction_ids) != len(set(prediction_ids)):
            raise ContractError("prediction set IDs must be non-empty and unique")
        if len(prediction_ranks) != len(set(prediction_ranks)):
            raise ContractError("prediction-set ranks must be unique")
        canonical = self.predictions_for_tier(PredictionTier.CANONICAL)
        expected_canonical = tuple(f"set_{index}" for index in range(1, 20))
        if canonical != expected_canonical:
            raise ContractError(
                "canonical prediction tier must be set_1 through set_19 in order"
            )
        outliers = self.predictions_for_tier(PredictionTier.OUTLIER_STRESS)
        if outliers != ("set_20", "set_21"):
            raise ContractError("outlier_stress tier must contain set_20 and set_21")

        if not self.required_reference_files:
            raise ContractError("required_reference_files cannot be empty")

    def split_dict(self) -> dict[str, tuple[str, ...]]:
        return dict(self.splits)

    def predictions_for_tier(
        self, tier: PredictionTier | str
    ) -> tuple[str, ...]:
        normalized = PredictionTier(tier)
        return tuple(
            spec.prediction_set_id
            for spec in self.prediction_sets
            if spec.tier is normalized
        )

    def iter_contexts(
        self,
        *,
        split_names: Iterable[str] | None = None,
        prediction_tiers: Iterable[PredictionTier | str] = (
            PredictionTier.CANONICAL,
        ),
        planning_formulation: str = "absolute_mean",
        clinical_profile_id: str = "openkbp_hn_v1",
        environment_version: str = "1.0.0-draft",
    ) -> Iterable[PlanningContext]:
        splits = self.split_dict()
        selected_split_names = tuple(split_names) if split_names is not None else tuple(splits)
        unknown_splits = sorted(set(selected_split_names) - set(splits))
        if unknown_splits:
            raise ContractError(f"unknown split names: {unknown_splits}")

        prediction_ids = tuple(
            prediction_id
            for tier in prediction_tiers
            for prediction_id in self.predictions_for_tier(tier)
        )
        if len(prediction_ids) != len(set(prediction_ids)):
            raise ContractError("prediction_tiers select duplicate prediction sets")

        for split_name in selected_split_names:
            for patient_id in splits[split_name]:
                for prediction_set_id in prediction_ids:
                    case_id = ":".join(
                        (patient_id, prediction_set_id, planning_formulation)
                    )
                    yield PlanningContext(
                        case_id=case_id,
                        patient_id=patient_id,
                        prediction_set_id=prediction_set_id,
                        split_id=split_name,
                        clinical_profile_id=clinical_profile_id,
                        planning_formulation=planning_formulation,
                        environment_version=environment_version,
                    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "source_repository": self.source_repository,
            "source_commit": self.source_commit,
            "root_hint": self.root_hint,
            "patients": list(self.patients),
            "splits": {name: list(patients) for name, patients in self.splits},
            "prediction_sets": [
                {
                    "prediction_set_id": spec.prediction_set_id,
                    "tier": spec.tier.value,
                    "official_rank": spec.official_rank,
                    "included_in_openkbp_opt_analysis": (
                        spec.included_in_openkbp_opt_analysis
                    ),
                    "note": spec.note,
                }
                for spec in self.prediction_sets
            ],
            "required_reference_files": list(self.required_reference_files),
            "notes": list(self.notes),
        }

    def sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class DatasetAudit:
    pass

    manifest_sha256: str
    data_root: str
    patient_directories_found: int
    prediction_directories_found: int
    prediction_files_found: int
    missing_reference_files: tuple[str, ...]
    missing_prediction_files: tuple[str, ...]
    unexpected_patient_directories: tuple[str, ...]
    unexpected_prediction_directories: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not (
            self.missing_reference_files
            or self.missing_prediction_files
            or self.unexpected_patient_directories
            or self.unexpected_prediction_directories
        )


def load_dataset_manifest(path: str | Path) -> DatasetManifest:
    pass

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ContractError("dataset manifest must be a JSON object")
    splits_data = payload.get("splits")
    if not isinstance(splits_data, Mapping):
        raise ContractError("dataset manifest splits must be a mapping")
    prediction_data = payload.get("prediction_sets")
    if not isinstance(prediction_data, list):
        raise ContractError("dataset manifest prediction_sets must be a list")

    return DatasetManifest(
        name=str(payload["name"]),
        version=str(payload["version"]),
        source_repository=str(payload["source_repository"]),
        source_commit=str(payload["source_commit"]),
        root_hint=str(payload["root_hint"]),
        patients=tuple(str(patient) for patient in payload["patients"]),
        splits=tuple(
            (str(name), tuple(str(patient) for patient in patients))
            for name, patients in splits_data.items()
        ),
        prediction_sets=tuple(
            PredictionSetSpec(
                prediction_set_id=str(item["prediction_set_id"]),
                tier=PredictionTier(str(item["tier"])),
                official_rank=int(item["official_rank"]),
                included_in_openkbp_opt_analysis=bool(
                    item["included_in_openkbp_opt_analysis"]
                ),
                note=str(item["note"]),
            )
            for item in prediction_data
        ),
        required_reference_files=tuple(
            str(filename) for filename in payload["required_reference_files"]
        ),
        notes=tuple(str(note) for note in payload.get("notes", [])),
    )


def audit_dataset(manifest: DatasetManifest, data_root: str | Path) -> DatasetAudit:
    pass

    root = Path(data_root)
    reference_root = root / "reference-plans"
    prediction_root = root / "paper-predictions"

    expected_patients = set(manifest.patients)
    found_patients = {
        path.name for path in reference_root.iterdir() if path.is_dir()
    } if reference_root.is_dir() else set()
    expected_predictions = {
        spec.prediction_set_id for spec in manifest.prediction_sets
    }
    found_predictions = {
        path.name for path in prediction_root.iterdir() if path.is_dir()
    } if prediction_root.is_dir() else set()

    missing_reference_files: list[str] = []
    for patient_id in manifest.patients:
        patient_root = reference_root / patient_id
        for filename in manifest.required_reference_files:
            path = patient_root / filename
            if not path.is_file():
                missing_reference_files.append(str(path.relative_to(root)))

    missing_prediction_files: list[str] = []
    prediction_files_found = 0
    for spec in manifest.prediction_sets:
        for patient_id in manifest.patients:
            path = prediction_root / spec.prediction_set_id / f"{patient_id}.csv"
            if path.is_file():
                prediction_files_found += 1
            else:
                missing_prediction_files.append(str(path.relative_to(root)))

    return DatasetAudit(
        manifest_sha256=manifest.sha256(),
        data_root=str(root.resolve()),
        patient_directories_found=len(found_patients),
        prediction_directories_found=len(found_predictions),
        prediction_files_found=prediction_files_found,
        missing_reference_files=tuple(missing_reference_files),
        missing_prediction_files=tuple(missing_prediction_files),
        unexpected_patient_directories=tuple(sorted(found_patients - expected_patients)),
        unexpected_prediction_directories=tuple(
            sorted(found_predictions - expected_predictions)
        ),
    )
