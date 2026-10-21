

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from radoncgym.audit.reference import HiddenReferenceDoseLoader
from radoncgym.core.models import MetricVector
from radoncgym.data.openkbp import (
    OPENKBP_GRID_SHAPE,
    ROI_NAMES,
    OpenKBPOptDataLayout,
)
from radoncgym.data.voxel_io import (
    load_dense_sparse_values,
    load_mask_indices,
    load_optional_mask_indices,
)
from radoncgym.evaluators.generated_plan import (
    GeneratedPlanEvaluation,
    GeneratedPlanEvaluator,
    calculate_dvh_metrics,
)


@dataclass(frozen=True, slots=True)
class HiddenAuditResult:
    pass

    patient_id: str
    reference_dvh_error: float
    reference_voxel_dose_error: float
    generated_plan: GeneratedPlanEvaluation
    reference_dvh_metrics: MetricVector


class HiddenReferenceAuditor:
    pass

    def __init__(
        self,
        repo_root: str | Path = "datasets/open-kbp-opt",
        *,
        grid_shape: tuple[int, int, int] = OPENKBP_GRID_SHAPE,
    ) -> None:
        self.layout = OpenKBPOptDataLayout.from_repo_root(repo_root)
        self.grid_shape = grid_shape
        self.generated_evaluator = GeneratedPlanEvaluator(
            repo_root, grid_shape=grid_shape
        )
        self.reference_loader = HiddenReferenceDoseLoader(
            repo_root, grid_shape=grid_shape
        )

    def evaluate(
        self, patient_id: str, generated_dose_path: str | Path
    ) -> HiddenAuditResult:
        generated = self.generated_evaluator.evaluate(
            patient_id, generated_dose_path
        )
        reference = self.reference_loader.load(patient_id).values
        patient_directory = self.layout.patient_directory(patient_id)
        generated_dose = load_dense_sparse_values(
            generated_dose_path, self.grid_shape
        )
        voxel_dimensions = np.atleast_1d(
            np.loadtxt(patient_directory / "voxel_dimensions.csv")
        )
        roi_indices = {
            roi_name: load_optional_mask_indices(
                patient_directory / f"{roi_name}.csv"
            )
            for roi_name in ROI_NAMES
        }
        reference_metrics = calculate_dvh_metrics(
            reference,
            roi_indices,
            voxel_volume_mm3=float(np.prod(voxel_dimensions)),
            merge_targets=False,
        )
        generated_mapping = generated.dvh_metrics.to_dict()
        metric_errors = [
            abs(reference_value - generated_mapping[name])
            for name, reference_value in reference_metrics.items
            if name in generated_mapping
        ]
        possible_indices = load_mask_indices(
            patient_directory / "possible_dose_mask.csv"
        )
        reference_flat = reference.reshape(-1)
        generated_flat = generated_dose.reshape(-1)
        voxel_error = np.mean(
            np.abs(
                reference_flat[possible_indices]
                - generated_flat[possible_indices]
            )
        )
        return HiddenAuditResult(
            patient_id=patient_id,
            reference_dvh_error=float(np.mean(metric_errors)),
            reference_voxel_dose_error=float(voxel_error),
            generated_plan=generated,
            reference_dvh_metrics=reference_metrics,
        )
