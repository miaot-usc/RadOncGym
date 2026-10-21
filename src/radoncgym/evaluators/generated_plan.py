

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from radoncgym.clinical.protocol import (
    CLINICAL_CRITERIA,
    TARGET_PRESCRIPTIONS_GY,
)
from radoncgym.core.models import MetricVector
from radoncgym.data.openkbp import (
    OAR_NAMES,
    OPENKBP_GRID_SHAPE,
    ROI_NAMES,
    TARGET_NAMES,
    OpenKBPOptDataLayout,
)
from radoncgym.data.voxel_io import (
    load_dense_sparse_values,
    load_optional_mask_indices,
)


FALLOFF_SHELLS_MM = ((0.0, 5.0), (5.0, 10.0), (10.0, 20.0))


@dataclass(frozen=True, slots=True)
class GeneratedPlanEvaluation:
    pass

    patient_id: str
    dvh_metrics: MetricVector
    protocol_metrics: MetricVector
    spatial_quality_metrics: MetricVector
    constraint_violations_gy: MetricVector
    mean_violation_gy: float
    criteria_evaluated: int


class GeneratedPlanEvaluator:
    pass

    def __init__(
        self,
        repo_root: str | Path = "datasets/open-kbp-opt",
        *,
        grid_shape: tuple[int, int, int] = OPENKBP_GRID_SHAPE,
    ) -> None:
        self.layout = OpenKBPOptDataLayout.from_repo_root(repo_root)
        self.grid_shape = grid_shape

    def evaluate(
        self, patient_id: str, generated_dose_path: str | Path
    ) -> GeneratedPlanEvaluation:
        patient_directory = self.layout.patient_directory(patient_id)
        dose = load_dense_sparse_values(generated_dose_path, self.grid_shape)
        voxel_dimensions = np.atleast_1d(
            np.loadtxt(patient_directory / "voxel_dimensions.csv")
        )
        voxel_volume_mm3 = float(np.prod(voxel_dimensions))
        roi_indices = {
            roi_name: load_optional_mask_indices(
                patient_directory / f"{roi_name}.csv"
            )
            for roi_name in ROI_NAMES
        }
        possible_dose_indices = load_optional_mask_indices(
            patient_directory / "possible_dose_mask.csv"
        )
        dvh_metrics = calculate_dvh_metrics(
            dose,
            roi_indices,
            voxel_volume_mm3=voxel_volume_mm3,
            merge_targets=False,
        )
        protocol_metrics = calculate_dvh_metrics(
            dose,
            roi_indices,
            voxel_volume_mm3=voxel_volume_mm3,
            merge_targets=True,
        )
        spatial_quality_metrics = calculate_spatial_quality_metrics(
            dose,
            roi_indices,
            possible_dose_indices=possible_dose_indices,
            voxel_dimensions_mm=voxel_dimensions,
        )
        violations: dict[str, float] = {}
        protocol_mapping = protocol_metrics.to_dict()
        for criterion in CLINICAL_CRITERIA:
            value = protocol_mapping.get(criterion.key)
            if value is None:
                continue
            if criterion.relation == "at_least":
                violation = max(0.0, criterion.threshold_gy - value)
            else:
                violation = max(0.0, value - criterion.threshold_gy)
            violations[criterion.key] = float(violation)
        mean_violation = float(np.mean(tuple(violations.values()))) if violations else 0.0
        return GeneratedPlanEvaluation(
            patient_id=patient_id,
            dvh_metrics=dvh_metrics,
            protocol_metrics=protocol_metrics,
            spatial_quality_metrics=spatial_quality_metrics,
            constraint_violations_gy=MetricVector.from_mapping(violations),
            mean_violation_gy=mean_violation,
            criteria_evaluated=len(violations),
        )


def calculate_dvh_metrics(
    dose: np.ndarray,
    roi_indices: dict[str, np.ndarray],
    *,
    voxel_volume_mm3: float,
    merge_targets: bool,
) -> MetricVector:
    pass

    flattened_dose = np.asarray(dose).reshape(-1)
    metrics: dict[str, float] = {}
    voxels_in_tenth_cc = max(1.0, float(np.round(100.0 / voxel_volume_mm3)))
    for roi_name in OAR_NAMES:
        indices = roi_indices[roi_name]
        if indices.size == 0:
            continue
        roi_dose = flattened_dose[indices]
        percentile = 100.0 - voxels_in_tenth_cc / roi_dose.size * 100.0


        percentile = max(0.0, percentile)
        metrics[f"D_0.1_cc:{roi_name}"] = float(np.percentile(roi_dose, percentile))
        metrics[f"mean:{roi_name}"] = float(np.mean(roi_dose))

    for target_index, roi_name in enumerate(TARGET_NAMES):
        indices = roi_indices[roi_name]


        if indices.size == 0:
            continue
        if merge_targets:
            for higher_target in TARGET_NAMES[target_index + 1 :]:
                indices = np.union1d(indices, roi_indices[higher_target])
        roi_dose = flattened_dose[indices]
        metrics[f"D_99:{roi_name}"] = float(np.percentile(roi_dose, 1.0))
        metrics[f"D_95:{roi_name}"] = float(np.percentile(roi_dose, 5.0))
        metrics[f"D_1:{roi_name}"] = float(np.percentile(roi_dose, 99.0))
    return MetricVector.from_mapping(metrics)


def calculate_spatial_quality_metrics(
    dose: np.ndarray,
    roi_indices: dict[str, np.ndarray],
    *,
    possible_dose_indices: np.ndarray,
    voxel_dimensions_mm: np.ndarray,
) -> MetricVector:
    pass

    dose_array = np.asarray(dose, dtype=np.float64)
    flattened_dose = dose_array.reshape(-1)
    possible = np.asarray(possible_dose_indices, dtype=np.int64)
    if possible.size == 0:
        possible = np.flatnonzero(flattened_dose != 0.0)
    metrics: dict[str, float] = {}
    if possible.size:
        plan_max = float(np.max(flattened_dose[possible]))
        metrics["plan_max_dose_gy"] = plan_max
        metrics["plan_max_dose_percent_of_PTV70_prescription"] = (
            plan_max / TARGET_PRESCRIPTIONS_GY["PTV70"] * 100.0
        )

    for target_index, target_name in enumerate(TARGET_NAMES):
        target_indices = roi_indices[target_name]
        if target_indices.size == 0:
            continue
        for higher_target in TARGET_NAMES[target_index + 1 :]:
            target_indices = np.union1d(
                target_indices, roi_indices[higher_target]
            )
        target_dose = flattened_dose[target_indices]
        d2 = float(np.percentile(target_dose, 98.0))
        d98 = float(np.percentile(target_dose, 2.0))
        prescription = TARGET_PRESCRIPTIONS_GY[target_name]
        metrics[f"D_2:{target_name}"] = d2
        metrics[f"D_98:{target_name}"] = d98
        metrics[f"homogeneity_index:{target_name}"] = (
            (d2 - d98) / prescription
        )
        if possible.size:
            prescription_volume_voxels = int(
                np.count_nonzero(flattened_dose[possible] >= prescription)
            )
            metrics[f"conformity_ratio:{target_name}"] = (
                prescription_volume_voxels / float(target_indices.size)
            )

    target_union_indices = np.unique(
        np.concatenate(
            [
                np.asarray(roi_indices[target], dtype=np.int64)
                for target in TARGET_NAMES
                if roi_indices[target].size
            ]
        )
    ) if any(roi_indices[target].size for target in TARGET_NAMES) else np.empty(
        (0,), dtype=np.int64
    )
    if target_union_indices.size == 0 or possible.size == 0:
        return MetricVector.from_mapping(metrics)

    ptv70_indices = np.asarray(roi_indices["PTV70"], dtype=np.int64)
    outside_ptv70 = np.ones(flattened_dose.size, dtype=bool)
    outside_ptv70[ptv70_indices] = False
    high_dose_spill = (
        outside_ptv70[possible]
        & (flattened_dose[possible] > 73.5)
    )
    voxel_volume_cc = float(np.prod(voxel_dimensions_mm)) / 1000.0
    metrics["high_dose_spill_cc_above_73_5_outside_PTV70"] = (
        float(np.count_nonzero(high_dose_spill)) * voxel_volume_cc
    )

    target_union = np.zeros(flattened_dose.size, dtype=bool)
    target_union[target_union_indices] = True
    target_union = target_union.reshape(dose_array.shape)
    distance_mm = distance_transform_edt(
        ~target_union,
        sampling=tuple(float(value) for value in voxel_dimensions_mm),
    ).reshape(-1)
    possible_mask = np.zeros(flattened_dose.size, dtype=bool)
    possible_mask[possible] = True
    outside_union = possible_mask & ~target_union.reshape(-1)
    for lower_mm, upper_mm in FALLOFF_SHELLS_MM:
        shell = (
            outside_union
            & (distance_mm > lower_mm)
            & (distance_mm <= upper_mm)
        )
        if not np.any(shell):
            continue
        shell_name = f"{int(lower_mm)}_{int(upper_mm)}_mm"
        shell_dose = flattened_dose[shell]
        metrics[f"falloff_mean_gy:{shell_name}"] = float(np.mean(shell_dose))
        metrics[f"falloff_p95_gy:{shell_name}"] = float(
            np.percentile(shell_dose, 95.0)
        )
    return MetricVector.from_mapping(metrics)
