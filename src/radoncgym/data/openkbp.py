

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np

from radoncgym.core.errors import ContractError
from radoncgym.core.models import PlanningContext
from radoncgym.data.voxel_io import (
    load_dense_sparse_values,
    load_mask_indices,
    load_optional_mask_indices,
)


OPENKBP_GRID_SHAPE = (128, 128, 128)
OAR_NAMES = (
    "Brainstem",
    "SpinalCord",
    "RightParotid",
    "LeftParotid",
    "Esophagus",
    "Larynx",
    "Mandible",
)
TARGET_NAMES = ("PTV56", "PTV63", "PTV70")
ROI_NAMES = OAR_NAMES + TARGET_NAMES

_PATIENT_PATTERN = re.compile(r"pt_[0-9]+\Z")
_PREDICTION_PATTERN = re.compile(r"set_[0-9]+\Z")


@dataclass(frozen=True, slots=True)
class ArraySummary:
    minimum: float
    mean: float
    maximum: float
    p01: float
    p05: float
    p50: float
    p95: float
    p99: float

    @classmethod
    def from_array(cls, values: np.ndarray) -> "ArraySummary":
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0:
            return cls(*(0.0 for _ in range(8)))
        percentiles = np.percentile(array, (1, 5, 50, 95, 99))
        return cls(
            minimum=float(np.min(array)),
            mean=float(np.mean(array)),
            maximum=float(np.max(array)),
            p01=float(percentiles[0]),
            p05=float(percentiles[1]),
            p50=float(percentiles[2]),
            p95=float(percentiles[3]),
            p99=float(percentiles[4]),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "min": self.minimum,
            "mean": self.mean,
            "max": self.maximum,
            "p01": self.p01,
            "p05": self.p05,
            "p50": self.p50,
            "p95": self.p95,
            "p99": self.p99,
        }


@dataclass(frozen=True, slots=True)
class ROIPublicFeatures:
    name: str
    roi_type: str
    present: bool
    voxel_count: int
    volume_cc: float
    centroid_fraction: tuple[float, float, float]
    bounding_box_mm: tuple[float, float, float]
    predicted_dose: ArraySummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "roi_type": self.roi_type,
            "present": self.present,
            "voxel_count": self.voxel_count,
            "volume_cc": self.volume_cc,
            "centroid_fraction": list(self.centroid_fraction),
            "bounding_box_mm": list(self.bounding_box_mm),
            "predicted_dose": self.predicted_dose.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ROIOverlapFeatures:
    first_roi: str
    second_roi: str
    intersection_cc: float
    fraction_of_first: float
    fraction_of_second: float
    dice: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "first_roi": self.first_roi,
            "second_roi": self.second_roi,
            "intersection_cc": self.intersection_cc,
            "fraction_of_first": self.fraction_of_first,
            "fraction_of_second": self.fraction_of_second,
            "dice": self.dice,
        }


@dataclass(frozen=True, slots=True)
class PublicCaseFeatures:
    pass

    patient_id: str
    prediction_set_id: str
    grid_shape: tuple[int, int, int]
    voxel_dimensions_mm: tuple[float, float, float]
    voxel_volume_cc: float
    possible_dose_voxel_count: int
    possible_dose_volume_cc: float
    ct: ArraySummary
    predicted_dose_in_possible_mask: ArraySummary
    rois: tuple[ROIPublicFeatures, ...]
    oar_target_overlaps: tuple[ROIOverlapFeatures, ...]
    source: str = (
        "CT, structure masks, possible-dose mask, and selected model prediction; "
        "patient reference dose is not loaded"
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "prediction_set_id": self.prediction_set_id,
            "grid_shape": list(self.grid_shape),
            "voxel_dimensions_mm": list(self.voxel_dimensions_mm),
            "voxel_volume_cc": self.voxel_volume_cc,
            "possible_dose_voxel_count": self.possible_dose_voxel_count,
            "possible_dose_volume_cc": self.possible_dose_volume_cc,
            "ct": self.ct.to_dict(),
            "predicted_dose_in_possible_mask": (
                self.predicted_dose_in_possible_mask.to_dict()
            ),
            "rois": [roi.to_dict() for roi in self.rois],
            "oar_target_overlaps": [
                overlap.to_dict() for overlap in self.oar_target_overlaps
            ],
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class OpenKBPOptDataLayout:
    repo_root: Path
    data_root: Path
    patient_root: Path
    prediction_root: Path

    @classmethod
    def from_repo_root(cls, repo_root: str | Path) -> "OpenKBPOptDataLayout":
        resolved_repo = Path(repo_root).resolve()
        data_root = resolved_repo / "open-kbp-opt-data"
        layout = cls(
            repo_root=resolved_repo,
            data_root=data_root,
            patient_root=data_root / "reference-plans",
            prediction_root=data_root / "paper-predictions",
        )
        missing = [
            path
            for path in (layout.patient_root, layout.prediction_root)
            if not path.is_dir()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing OpenKBP-Opt data directories: "
                + ", ".join(str(path) for path in missing)
            )
        return layout

    def patient_directory(self, patient_id: str) -> Path:
        _validate_identifier(patient_id, _PATIENT_PATTERN, "patient")
        path = self.patient_root / patient_id
        if not path.is_dir():
            raise FileNotFoundError(f"OpenKBP patient not found: {path}")
        return path

    def prediction_file(self, prediction_set_id: str, patient_id: str) -> Path:
        _validate_identifier(prediction_set_id, _PREDICTION_PATTERN, "prediction set")
        _validate_identifier(patient_id, _PATIENT_PATTERN, "patient")
        path = self.prediction_root / prediction_set_id / f"{patient_id}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"OpenKBP prediction not found: {path}")
        return path


class OpenKBPPublicCaseLoader:
    pass

    def __init__(
        self,
        repo_root: str | Path = "datasets/open-kbp-opt",
        *,
        grid_shape: tuple[int, int, int] = OPENKBP_GRID_SHAPE,
    ) -> None:
        self.layout = OpenKBPOptDataLayout.from_repo_root(repo_root)
        self.grid_shape = tuple(int(value) for value in grid_shape)
        if len(self.grid_shape) != 3 or any(value <= 0 for value in self.grid_shape):
            raise ContractError("grid_shape must contain three positive dimensions")
        self._cache: dict[tuple[str, str], PublicCaseFeatures] = {}

    def load(self, context: PlanningContext) -> PublicCaseFeatures:
        key = (context.patient_id, context.prediction_set_id)
        if key not in self._cache:
            self._cache[key] = self._load_uncached(*key)
        return self._cache[key]

    def _load_uncached(
        self, patient_id: str, prediction_set_id: str
    ) -> PublicCaseFeatures:
        patient_directory = self.layout.patient_directory(patient_id)
        prediction_file = self.layout.prediction_file(
            prediction_set_id, patient_id
        )
        voxel_dimensions = tuple(
            float(value)
            for value in np.atleast_1d(
                np.loadtxt(patient_directory / "voxel_dimensions.csv")
            )
        )
        if len(voxel_dimensions) != 3:
            raise ContractError("voxel_dimensions.csv must contain three values")
        voxel_volume_cc = float(np.prod(voxel_dimensions) / 1000.0)

        possible_indices = load_mask_indices(
            patient_directory / "possible_dose_mask.csv"
        )
        ct = load_dense_sparse_values(patient_directory / "ct.csv", self.grid_shape)
        prediction = load_dense_sparse_values(prediction_file, self.grid_shape)
        roi_indices = {
            roi_name: load_optional_mask_indices(patient_directory / f"{roi_name}.csv")
            for roi_name in ROI_NAMES
        }

        rois = tuple(
            _roi_features(
                roi_name,
                roi_indices[roi_name],
                prediction,
                self.grid_shape,
                voxel_dimensions,
                voxel_volume_cc,
            )
            for roi_name in ROI_NAMES
        )
        overlaps = tuple(
            _overlap_features(
                oar,
                target,
                roi_indices[oar],
                roi_indices[target],
                voxel_volume_cc,
            )
            for oar in OAR_NAMES
            for target in TARGET_NAMES
        )
        possible_prediction = prediction.reshape(-1)[possible_indices]
        return PublicCaseFeatures(
            patient_id=patient_id,
            prediction_set_id=prediction_set_id,
            grid_shape=self.grid_shape,
            voxel_dimensions_mm=voxel_dimensions,
            voxel_volume_cc=voxel_volume_cc,
            possible_dose_voxel_count=int(possible_indices.size),
            possible_dose_volume_cc=float(possible_indices.size * voxel_volume_cc),
            ct=ArraySummary.from_array(ct),
            predicted_dose_in_possible_mask=ArraySummary.from_array(
                possible_prediction
            ),
            rois=rois,
            oar_target_overlaps=overlaps,
        )


def _validate_identifier(value: str, pattern: re.Pattern[str], name: str) -> None:
    if not pattern.fullmatch(value):
        raise ContractError(f"invalid {name} identifier: {value!r}")


def _roi_features(
    roi_name: str,
    indices: np.ndarray,
    prediction: np.ndarray,
    grid_shape: tuple[int, int, int],
    voxel_dimensions: tuple[float, float, float],
    voxel_volume_cc: float,
) -> ROIPublicFeatures:
    if indices.size == 0:
        centroid = (0.0, 0.0, 0.0)
        bounding_box = (0.0, 0.0, 0.0)
        predicted_dose = ArraySummary.from_array(np.empty((0,)))
    else:
        coordinates = np.asarray(np.unravel_index(indices, grid_shape)).T
        denominator = np.maximum(np.asarray(grid_shape, dtype=float) - 1.0, 1.0)
        centroid = tuple(float(value) for value in coordinates.mean(axis=0) / denominator)
        extents = coordinates.max(axis=0) - coordinates.min(axis=0) + 1
        bounding_box = tuple(
            float(value)
            for value in extents * np.asarray(voxel_dimensions, dtype=float)
        )
        predicted_dose = ArraySummary.from_array(prediction.reshape(-1)[indices])
    return ROIPublicFeatures(
        name=roi_name,
        roi_type="target" if roi_name in TARGET_NAMES else "oar",
        present=bool(indices.size),
        voxel_count=int(indices.size),
        volume_cc=float(indices.size * voxel_volume_cc),
        centroid_fraction=centroid,
        bounding_box_mm=bounding_box,
        predicted_dose=predicted_dose,
    )


def _overlap_features(
    first_roi: str,
    second_roi: str,
    first_indices: np.ndarray,
    second_indices: np.ndarray,
    voxel_volume_cc: float,
) -> ROIOverlapFeatures:
    intersection_count = int(
        np.intersect1d(first_indices, second_indices, assume_unique=True).size
    )
    first_count = int(first_indices.size)
    second_count = int(second_indices.size)
    denominator = first_count + second_count
    return ROIOverlapFeatures(
        first_roi=first_roi,
        second_roi=second_roi,
        intersection_cc=float(intersection_count * voxel_volume_cc),
        fraction_of_first=(intersection_count / first_count if first_count else 0.0),
        fraction_of_second=(
            intersection_count / second_count if second_count else 0.0
        ),
        dice=(2.0 * intersection_count / denominator if denominator else 0.0),
    )
