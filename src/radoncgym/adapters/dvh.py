

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from radoncgym.core.errors import ContractError
from radoncgym.core.models import FeedbackSource, PlanningContext
from radoncgym.data.openkbp import (
    OPENKBP_GRID_SHAPE,
    ROI_NAMES,
    OpenKBPOptDataLayout,
)
from radoncgym.data.voxel_io import (
    load_dense_sparse_values,
    load_optional_mask_indices,
)


DVH_SCHEMA_VERSION = "dvh-observation-0.1.0"
DVH_DOSE_BINS_GY = tuple(float(value) for value in np.arange(0.0, 85.0 + 0.5, 0.5))
_ALLOWED_SOURCES = {
    FeedbackSource.CONTEXT,
    FeedbackSource.CHEAP_LEARNED_SURROGATE,
    FeedbackSource.FULL_LIVE_OPTIMIZER,
    FeedbackSource.FULL_EXACT_CACHE,
    FeedbackSource.APPROXIMATE_CACHE_SURROGATE,
}


@dataclass(frozen=True, slots=True)
class DVHObservation:
    pass

    patient_id: str
    prediction_set_id: str
    source: FeedbackSource
    dose_bins_gy: tuple[float, ...]
    roi_names: tuple[str, ...]
    present_mask: tuple[bool, ...]
    volume_fractions: tuple[tuple[float, ...], ...]
    schema_version: str = DVH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DVH_SCHEMA_VERSION:
            raise ContractError("unsupported DVH observation schema")
        if not self.patient_id or not self.prediction_set_id:
            raise ContractError("DVH context identity is required")
        if self.source not in _ALLOWED_SOURCES:
            raise ContractError("unsupported DVH feedback source")
        if self.roi_names != ROI_NAMES:
            raise ContractError("DVH ROI order must match the canonical OpenKBP order")
        if len(self.present_mask) != len(self.roi_names):
            raise ContractError("DVH presence mask must align with ROI names")
        if len(self.volume_fractions) != len(self.roi_names):
            raise ContractError("DVH curves must align with ROI names")
        if len(self.dose_bins_gy) < 2 or any(
            not math.isfinite(value) for value in self.dose_bins_gy
        ):
            raise ContractError("DVH dose bins must be finite and non-empty")
        if any(
            current >= following
            for current, following in zip(
                self.dose_bins_gy, self.dose_bins_gy[1:]
            )
        ):
            raise ContractError("DVH dose bins must be strictly increasing")
        for present, curve in zip(
            self.present_mask, self.volume_fractions, strict=True
        ):
            if len(curve) != len(self.dose_bins_gy):
                raise ContractError("every DVH curve must cover every dose bin")
            if any(
                not math.isfinite(value) or value < 0.0 or value > 1.0
                for value in curve
            ):
                raise ContractError("DVH volume fractions must lie in [0, 1]")
            if any(
                current + 1e-12 < following
                for current, following in zip(curve, curve[1:])
            ):
                raise ContractError("cumulative DVH curves must be non-increasing")
            if not present and any(value != 0.0 for value in curve):
                raise ContractError("missing ROI curves must be zero-masked")

    def as_array(self, *, dtype: np.dtype = np.dtype(np.float32)) -> np.ndarray:
        return np.asarray(self.volume_fractions, dtype=dtype)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "patient_id": self.patient_id,
            "prediction_set_id": self.prediction_set_id,
            "source": self.source.value,
            "dose_bins_gy": list(self.dose_bins_gy),
            "roi_names": list(self.roi_names),
            "present_mask": list(self.present_mask),
            "volume_fractions": [list(curve) for curve in self.volume_fractions],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DVHObservation":
        return cls(
            patient_id=str(value["patient_id"]),
            prediction_set_id=str(value["prediction_set_id"]),
            source=FeedbackSource(str(value["source"])),
            dose_bins_gy=tuple(float(item) for item in value["dose_bins_gy"]),
            roi_names=tuple(str(item) for item in value["roi_names"]),
            present_mask=tuple(bool(item) for item in value["present_mask"]),
            volume_fractions=tuple(
                tuple(float(item) for item in curve)
                for curve in value["volume_fractions"]
            ),
            schema_version=str(value["schema_version"]),
        )


def calculate_dvh_observation(
    *,
    context: PlanningContext,
    dose: np.ndarray,
    roi_indices: Mapping[str, np.ndarray],
    source: FeedbackSource,
    dose_bins_gy: tuple[float, ...] = DVH_DOSE_BINS_GY,
) -> DVHObservation:
    pass

    flattened = np.asarray(dose, dtype=np.float64).reshape(-1)
    curves: list[tuple[float, ...]] = []
    present: list[bool] = []
    bins = np.asarray(dose_bins_gy, dtype=np.float64)
    for roi_name in ROI_NAMES:
        indices = np.asarray(roi_indices.get(roi_name, ()), dtype=np.int64)
        if indices.size == 0:
            present.append(False)
            curves.append(tuple(0.0 for _ in dose_bins_gy))
            continue
        if indices.min() < 0 or indices.max() >= flattened.size:
            raise ContractError(f"DVH ROI index outside dose grid: {roi_name}")
        ordered = np.sort(flattened[indices])
        counts_below = np.searchsorted(ordered, bins, side="left")
        fractions = 1.0 - counts_below.astype(np.float64) / ordered.size
        present.append(True)
        curves.append(tuple(float(value) for value in fractions))
    return DVHObservation(
        patient_id=context.patient_id,
        prediction_set_id=context.prediction_set_id,
        source=source,
        dose_bins_gy=dose_bins_gy,
        roi_names=ROI_NAMES,
        present_mask=tuple(present),
        volume_fractions=tuple(curves),
    )


class OpenKBPDVHLoader:
    pass

    def __init__(
        self,
        repo_root: str | Path = "datasets/open-kbp-opt",
        *,
        grid_shape: tuple[int, int, int] = OPENKBP_GRID_SHAPE,
    ) -> None:
        self.layout = OpenKBPOptDataLayout.from_repo_root(repo_root)
        self.grid_shape = grid_shape

    def load_prediction(self, context: PlanningContext) -> DVHObservation:
        patient_directory = self.layout.patient_directory(context.patient_id)
        dose = load_dense_sparse_values(
            self.layout.prediction_file(
                context.prediction_set_id, context.patient_id
            ),
            self.grid_shape,
        )
        return calculate_dvh_observation(
            context=context,
            dose=dose,
            roi_indices=self._roi_indices(patient_directory),
            source=FeedbackSource.CONTEXT,
        )

    def load_generated(
        self,
        context: PlanningContext,
        generated_dose_path: str | Path,
        *,
        source: FeedbackSource,
    ) -> DVHObservation:
        if source not in {
            FeedbackSource.FULL_LIVE_OPTIMIZER,
            FeedbackSource.FULL_EXACT_CACHE,
        }:
            raise ContractError("generated DVH must have an authoritative full source")
        patient_directory = self.layout.patient_directory(context.patient_id)
        resolved = Path(generated_dose_path).resolve()
        if self.layout.patient_root.resolve() in resolved.parents:
            raise ContractError(
                "generated DVH path cannot reside under the reference-plan tree"
            )
        dose = load_dense_sparse_values(resolved, self.grid_shape)
        return calculate_dvh_observation(
            context=context,
            dose=dose,
            roi_indices=self._roi_indices(patient_directory),
            source=source,
        )

    @staticmethod
    def _roi_indices(patient_directory: Path) -> dict[str, np.ndarray]:
        return {
            roi_name: load_optional_mask_indices(patient_directory / f"{roi_name}.csv")
            for roi_name in ROI_NAMES
        }
