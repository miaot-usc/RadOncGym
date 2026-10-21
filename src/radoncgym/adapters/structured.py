

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from radoncgym.core.errors import ContractError
from radoncgym.core.models import PublicContextFeatures
from radoncgym.data.openkbp import PublicCaseFeatures


_SUMMARY_FIELDS = ("min", "mean", "max", "p01", "p05", "p50", "p95", "p99")
_AXES = ("x", "y", "z")


@dataclass(frozen=True, slots=True)
class StructuredFeatureVector:
    pass

    names: tuple[str, ...]
    values: tuple[float, ...]
    schema_version: str = "structured-public-case-0.1.0"

    def __post_init__(self) -> None:
        if len(self.names) != len(self.values):
            raise ContractError("structured feature names and values must align")
        if len(self.names) != len(set(self.names)):
            raise ContractError("structured feature names must be unique")
        if any(not name for name in self.names):
            raise ContractError("structured feature names must be non-empty")
        if any(not math.isfinite(value) for value in self.values):
            raise ContractError("structured feature values must be finite")

    def as_array(self, *, dtype: np.dtype = np.dtype(np.float32)) -> np.ndarray:
        return np.asarray(self.values, dtype=dtype)

    def to_public_context(self) -> PublicContextFeatures:
        pass

        return PublicContextFeatures(
            schema_version=self.schema_version,
            items=tuple(zip(self.names, self.values, strict=True)),
        )


def vectorize_public_case(
    case: PublicCaseFeatures,
    *,
    include_prediction_identity: bool = False,
    prediction_set_count: int = 21,
) -> StructuredFeatureVector:
    pass

    names: list[str] = []
    values: list[float] = []

    def add(name: str, value: float) -> None:
        names.append(name)
        values.append(float(value))

    for axis, value in zip(_AXES, case.voxel_dimensions_mm, strict=True):
        add(f"anatomy:voxel_dimensions_mm:{axis}", value)
    add("anatomy:voxel_volume_cc", case.voxel_volume_cc)
    add("anatomy:possible_dose_voxel_count", case.possible_dose_voxel_count)
    add("anatomy:possible_dose_volume_cc", case.possible_dose_volume_cc)
    for field, value in case.ct.to_dict().items():
        add(f"anatomy:ct_stored_grid:{field}", value)
    for field, value in case.predicted_dose_in_possible_mask.to_dict().items():
        add(f"prediction:possible_dose_mask:{field}", value)

    for roi in case.rois:
        prefix = f"roi:{roi.name}"
        add(f"{prefix}:present", float(roi.present))
        add(f"{prefix}:voxel_count", roi.voxel_count)
        add(f"{prefix}:volume_cc", roi.volume_cc)
        for axis, value in zip(_AXES, roi.centroid_fraction, strict=True):
            add(f"{prefix}:centroid_fraction:{axis}", value)
        for axis, value in zip(_AXES, roi.bounding_box_mm, strict=True):
            add(f"{prefix}:bounding_box_mm:{axis}", value)
        for field, value in roi.predicted_dose.to_dict().items():
            add(f"{prefix}:predicted_dose:{field}", value)

    for overlap in case.oar_target_overlaps:
        prefix = f"overlap:{overlap.first_roi}:{overlap.second_roi}"
        add(f"{prefix}:intersection_cc", overlap.intersection_cc)
        add(f"{prefix}:fraction_of_first", overlap.fraction_of_first)
        add(f"{prefix}:fraction_of_second", overlap.fraction_of_second)
        add(f"{prefix}:dice", overlap.dice)

    if include_prediction_identity:
        try:
            prediction_index = int(case.prediction_set_id.removeprefix("set_"))
        except ValueError as error:
            raise ContractError(
                f"cannot encode prediction identity: {case.prediction_set_id!r}"
            ) from error
        if prediction_index < 1 or prediction_index > prediction_set_count:
            raise ContractError(
                f"prediction index {prediction_index} outside 1..{prediction_set_count}"
            )
        for index in range(1, prediction_set_count + 1):
            add(f"context:prediction_set:set_{index}", float(index == prediction_index))

    return StructuredFeatureVector(tuple(names), tuple(values))
