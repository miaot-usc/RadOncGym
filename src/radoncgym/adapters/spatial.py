

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping, Protocol

import numpy as np

from radoncgym.adapters.dvh import DVHObservation, OpenKBPDVHLoader
from radoncgym.backends.cache import ExactFullCacheRecord
from radoncgym.core.errors import ContractError
from radoncgym.core.models import FeedbackSource, Observation, PlanningContext, SolverStatus
from radoncgym.data.openkbp import OPENKBP_GRID_SHAPE, ROI_NAMES, OpenKBPOptDataLayout
from radoncgym.data.voxel_io import load_dense_sparse_values, load_optional_mask_indices


SPATIAL_PANEL_SCHEMA_VERSION = "generated-dose-spatial-panel-0.1.0"

_ROI_COLORS = {
    "PTV70": "#ff2d2d",
    "PTV63": "#ff9900",
    "PTV56": "#ffd000",
    "Brainstem": "#00a6ff",
    "SpinalCord": "#58d68d",
    "LeftParotid": "#b56cff",
    "RightParotid": "#ff66b3",
    "Mandible": "#9b7653",
    "Esophagus": "#17a589",
    "Larynx": "#5d6d7e",
}


@dataclass(frozen=True, slots=True)
class SpatialPanelObservation:
    pass

    patient_id: str
    prediction_set_id: str
    source: FeedbackSource
    image_path: str
    image_sha256: str
    slice_indices: tuple[int, ...]
    includes_generated_dose: bool
    panel_kind: str = "spatial_dose"
    schema_version: str = SPATIAL_PANEL_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "patient_id": self.patient_id,
            "prediction_set_id": self.prediction_set_id,
            "source": self.source.value,
            "image_path": self.image_path,
            "image_sha256": self.image_sha256,
            "slice_indices": list(self.slice_indices),
            "includes_generated_dose": self.includes_generated_dose,
            "panel_kind": self.panel_kind,
            "hidden_reference_included": False,
        }


class FullRecordProvider(Protocol):
    pass

    last_record: ExactFullCacheRecord | None


class PanelRenderer(Protocol):
    pass

    def render(
        self,
        *,
        context: PlanningContext,
        output_path: str | Path,
        source: FeedbackSource = FeedbackSource.CONTEXT,
        generated_dose_path: str | Path | None = None,
    ) -> SpatialPanelObservation: ...


class OpenKBPSpatialPanelRenderer:
    pass

    def __init__(
        self,
        repo_root: str | Path = "datasets/open-kbp-opt",
        *,
        grid_shape: tuple[int, int, int] = OPENKBP_GRID_SHAPE,
    ) -> None:
        self.layout = OpenKBPOptDataLayout.from_repo_root(repo_root)
        self.grid_shape = grid_shape

    def render(
        self,
        *,
        context: PlanningContext,
        output_path: str | Path,
        source: FeedbackSource = FeedbackSource.CONTEXT,
        generated_dose_path: str | Path | None = None,
    ) -> SpatialPanelObservation:
        patient_directory = self.layout.patient_directory(context.patient_id)
        ct = load_dense_sparse_values(patient_directory / "ct.csv", self.grid_shape)
        prediction = load_dense_sparse_values(
            self.layout.prediction_file(context.prediction_set_id, context.patient_id),
            self.grid_shape,
        )
        masks = {
            name: _indices_to_mask(
                load_optional_mask_indices(patient_directory / f"{name}.csv"),
                self.grid_shape,
            )
            for name in ROI_NAMES
        }
        generated: np.ndarray | None = None
        if generated_dose_path is not None:
            if source not in {
                FeedbackSource.FULL_LIVE_OPTIMIZER,
                FeedbackSource.FULL_EXACT_CACHE,
            }:
                raise ContractError("generated spatial panel requires a full source")
            resolved = Path(generated_dose_path).resolve()
            if self.layout.patient_root.resolve() in resolved.parents:
                raise ContractError(
                    "generated spatial panel cannot read from the reference-plan tree"
                )
            generated = load_dense_sparse_values(resolved, self.grid_shape)
        elif source is not FeedbackSource.CONTEXT:
            raise ContractError("prediction-only spatial panel must use context source")

        slices = select_representative_axial_slices(masks)
        path = Path(output_path).resolve()
        render_spatial_panel_arrays(
            ct=ct,
            prediction=prediction,
            generated=generated,
            masks=masks,
            slice_indices=slices,
            output_path=path,
        )
        return SpatialPanelObservation(
            patient_id=context.patient_id,
            prediction_set_id=context.prediction_set_id,
            source=source,
            image_path=str(path),
            image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            slice_indices=slices,
            includes_generated_dose=generated is not None,
        )


class OpenKBPDVHPanelRenderer:
    pass

    def __init__(
        self,
        repo_root: str | Path = "datasets/open-kbp-opt",
        *,
        grid_shape: tuple[int, int, int] = OPENKBP_GRID_SHAPE,
    ) -> None:
        self.loader = OpenKBPDVHLoader(repo_root, grid_shape=grid_shape)

    def render(
        self,
        *,
        context: PlanningContext,
        output_path: str | Path,
        source: FeedbackSource = FeedbackSource.CONTEXT,
        generated_dose_path: str | Path | None = None,
    ) -> SpatialPanelObservation:
        prediction = self.loader.load_prediction(context)
        generated = None
        if generated_dose_path is not None:
            generated = self.loader.load_generated(
                context,
                generated_dose_path,
                source=source,
            )
        elif source is not FeedbackSource.CONTEXT:
            raise ContractError("prediction-only DVH panel must use context source")
        path = Path(output_path).resolve()
        render_dvh_panel_observations(
            prediction=prediction,
            generated=generated,
            output_path=path,
        )
        return SpatialPanelObservation(
            patient_id=context.patient_id,
            prediction_set_id=context.prediction_set_id,
            source=source,
            image_path=str(path),
            image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            slice_indices=(),
            includes_generated_dose=generated is not None,
            panel_kind="dvh_curves",
        )


class BackendSpatialPanelResolver:
    pass

    def __init__(
        self,
        *,
        renderer: PanelRenderer,
        backend: FullRecordProvider,
        output_root: str | Path,
        module_id: str = "generated_dose_multimodal_panel_v1",
    ) -> None:
        self.renderer = renderer
        self.backend = backend
        self.output_root = Path(output_root)
        self.module_id = module_id
        self._cache: dict[str, SpatialPanelObservation] = {}

    def resolve(self, observation: Observation) -> SpatialPanelObservation | None:
        if not observation.history:
            return None
        latest = observation.history[-1]
        if (
            not latest.accepted
            or latest.outcome is None
            or latest.outcome.full is None
            or latest.outcome.full.status is not SolverStatus.SOLVED
        ):
            return None
        solved_record = self.backend.last_record
        if solved_record is None or solved_record.status is not SolverStatus.SOLVED:
            return None
        if solved_record.context != observation.context:
            raise ContractError("spatial backend record belongs to another context")
        if latest.outcome.executed_weights != solved_record.weights:
            raise ContractError("spatial backend record and visible weights disagree")
        if solved_record.dose_artifact is None:
            raise ContractError("spatial panel requires a solved dose artifact")
        key = solved_record.cache_key
        source = latest.outcome.full.source
        dose_path = solved_record.dose_artifact
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        panel = self.renderer.render(
            context=observation.context,
            output_path=self.output_root / f"{key}.png",
            source=source,
            generated_dose_path=dose_path,
        )
        self._cache[key] = panel
        return panel


def select_representative_axial_slices(
    masks: Mapping[str, np.ndarray],
) -> tuple[int, ...]:
    pass

    first = next(iter(masks.values()), None)
    if first is None or first.ndim != 3:
        raise ContractError("spatial panel requires three-dimensional masks")
    groups = (
        ("PTV56", "PTV63", "PTV70"),
        ("Brainstem", "SpinalCord", "Mandible"),
        ("LeftParotid", "RightParotid", "Larynx", "Esophagus"),
    )
    selected: list[int] = []
    for group in groups:
        signal = np.zeros(first.shape[2], dtype=np.float64)
        for name in group:
            mask = np.asarray(masks.get(name, np.zeros_like(first)), dtype=bool)
            signal += mask.sum(axis=(0, 1))
        if np.any(signal):
            candidate = int(np.argmax(signal))
            if candidate not in selected:
                selected.append(candidate)
    if not selected:
        selected.append(first.shape[2] // 2)
    return tuple(selected)


def render_spatial_panel_arrays(
    *,
    ct: np.ndarray,
    prediction: np.ndarray,
    generated: np.ndarray | None,
    masks: Mapping[str, np.ndarray],
    slice_indices: tuple[int, ...],
    output_path: str | Path,
) -> None:
    pass

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as error:
        raise ContractError(
            "spatial panels require the optional multimodal dependencies"
        ) from error
    arrays = [np.asarray(ct), np.asarray(prediction)]
    if generated is not None:
        arrays.append(np.asarray(generated))
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ContractError("spatial panel arrays must share one grid shape")
    if arrays[0].ndim != 3:
        raise ContractError("spatial panel arrays must be three-dimensional")
    if not slice_indices or any(
        index < 0 or index >= arrays[0].shape[2] for index in slice_indices
    ):
        raise ContractError("spatial panel slice index is outside the dose grid")

    columns = 3 if generated is not None else 2
    figure, axes = plt.subplots(
        len(slice_indices),
        columns,
        figsize=(4.3 * columns, 4.0 * len(slice_indices)),
        squeeze=False,
    )
    ct_finite = arrays[0][np.isfinite(arrays[0])]
    ct_low, ct_high = np.percentile(ct_finite, (1.0, 99.0))
    if ct_high <= ct_low:
        ct_high = ct_low + 1.0
    dose_image = None
    for row, axial in enumerate(slice_indices):
        ct_slice = _display_slice(arrays[0], axial)
        for column, title in enumerate(("CT + structures", "Selected prediction")):
            axis = axes[row, column]
            axis.imshow(ct_slice, cmap="gray", vmin=ct_low, vmax=ct_high)
            if column == 1:
                dose_slice = np.ma.masked_less_equal(
                    _display_slice(arrays[1], axial), 0.5
                )
                dose_image = axis.imshow(
                    dose_slice, cmap="turbo", alpha=0.64, vmin=0.0, vmax=75.0
                )
            _draw_contours(axis, masks, axial)
            axis.set_title(f"{title} | axial {axial}")
            axis.set_axis_off()
        if generated is not None:
            axis = axes[row, 2]
            axis.imshow(ct_slice, cmap="gray", vmin=ct_low, vmax=ct_high)
            dose_slice = np.ma.masked_less_equal(_display_slice(arrays[2], axial), 0.5)
            dose_image = axis.imshow(
                dose_slice, cmap="turbo", alpha=0.64, vmin=0.0, vmax=75.0
            )
            _draw_contours(axis, masks, axial)
            axis.set_title(f"Generated full plan | axial {axial}")
            axis.set_axis_off()
    handles = [
        Line2D([0], [0], color=color, lw=2.0, label=name)
        for name, color in _ROI_COLORS.items()
        if name in masks and np.any(masks[name])
    ]
    if handles:
        figure.legend(handles=handles, loc="lower center", ncol=4, fontsize=8)
    if dose_image is not None:
        figure.colorbar(
            dose_image,
            ax=axes[:, 1:].ravel().tolist(),
            fraction=0.02,
            pad=0.02,
            label="Dose (Gy)",
        )
    figure.suptitle(
        "Agent-visible anatomy, selected prediction, and generated-plan dose",
        fontsize=13,
        y=0.995,
    )
    figure.text(
        0.5,
        0.965,
        "No expert reference dose is shown",
        ha="center",
        fontsize=9,
    )
    figure.subplots_adjust(bottom=0.09, top=0.93, wspace=0.03, hspace=0.12)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, facecolor="white")
    plt.close(figure)


def render_dvh_panel_observations(
    *,
    prediction: DVHObservation,
    generated: DVHObservation | None,
    output_path: str | Path,
) -> None:
    pass

    if generated is not None and (
        generated.patient_id != prediction.patient_id
        or generated.prediction_set_id != prediction.prediction_set_id
        or generated.roi_names != prediction.roi_names
        or generated.dose_bins_gy != prediction.dose_bins_gy
    ):
        raise ContractError("prediction and generated DVHs must share one context")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ContractError(
            "DVH panels require the optional multimodal dependencies"
        ) from error

    targets = {"PTV56", "PTV63", "PTV70"}
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.5), sharex=True, sharey=True)
    observations = [(prediction, "Selected prediction", "--", 0.65)]
    if generated is not None:
        observations.append((generated, "Generated full plan", "-", 1.0))
    bins = np.asarray(prediction.dose_bins_gy, dtype=np.float64)
    for axis, is_target, title in (
        (axes[0], True, "Targets"),
        (axes[1], False, "Organs at risk"),
    ):
        for observation, source_label, line_style, alpha in observations:
            for roi_name, present, curve in zip(
                observation.roi_names,
                observation.present_mask,
                observation.volume_fractions,
                strict=True,
            ):
                if not present or (roi_name in targets) != is_target:
                    continue
                axis.plot(
                    bins,
                    np.asarray(curve, dtype=np.float64),
                    color=_ROI_COLORS.get(roi_name),
                    linestyle=line_style,
                    linewidth=2.2 if generated is not None else 1.8,
                    alpha=alpha,
                    label=f"{roi_name} — {source_label}",
                )
        axis.set_title(title)
        axis.set_xlim(float(bins[0]), float(bins[-1]))
        axis.set_ylim(0.0, 1.02)
        axis.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.75)
        axis.set_xlabel("Dose (Gy)")
        axis.legend(loc="best", fontsize=7)
    axes[0].set_ylabel("Cumulative volume fraction receiving at least dose")
    figure.suptitle(
        "Agent-visible cumulative DVH curves",
        fontsize=14,
        y=0.98,
    )
    figure.text(
        0.5,
        0.925,
        "Dashed: selected public prediction  |  Solid: latest generated full plan  |  No expert reference dose",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, facecolor="white")
    plt.close(figure)


def _indices_to_mask(
    indices: np.ndarray, grid_shape: tuple[int, int, int]
) -> np.ndarray:
    mask = np.zeros(int(np.prod(grid_shape)), dtype=bool)
    mask[np.asarray(indices, dtype=np.int64)] = True
    return mask.reshape(grid_shape)


def _display_slice(volume: np.ndarray, index: int) -> np.ndarray:
    return np.rot90(np.asarray(volume)[:, :, index])


def _draw_contours(axis, masks: Mapping[str, np.ndarray], index: int) -> None:
    for name, color in _ROI_COLORS.items():
        mask = masks.get(name)
        if mask is None:
            continue
        visible = _display_slice(np.asarray(mask, dtype=bool), index)
        if np.any(visible):
            axis.contour(
                visible.astype(np.float32),
                levels=[0.5],
                colors=[color],
                linewidths=1.0,
            )
