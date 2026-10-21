

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
import uuid

import numpy as np

from radoncgym.backends.cache import (
    EXACT_CACHE_SCHEMA_VERSION,
    PORTFOLIO_EXACT_CACHE_SCHEMA_VERSION,
    ExactFullCacheRecord,
    PortfolioExactCacheIdentity,
    canonical_solver_parameters_sha256,
    make_exact_cache_key,
    make_portfolio_exact_cache_key,
)
from radoncgym.clinical.preferences import clinical_preference_profile
from radoncgym.clinical.scoring import calculate_preference_compliance
from radoncgym.core.errors import ContractError
from radoncgym.core.models import (
    ComputeCost,
    EvaluationResult,
    FeedbackSource,
    MetricVector,
    ObjectiveWeights,
    PlanningContext,
    SolverStatus,
)
from radoncgym.data.openkbp import OpenKBPOptDataLayout
from radoncgym.evaluators import GeneratedPlanEvaluator
from radoncgym.planning import (
    interpret_absolute_mean_duals,
    objective_artifacts_from_dose,
    objective_name_to_control,
    reconstruct_absolute_mean_losses,
)


@dataclass(frozen=True, slots=True)
class ReadOnlyExactCacheSource:
    pass

    path: Path
    expected_sha256: str
    expected_size_bytes: int

    def __post_init__(self) -> None:
        if len(self.expected_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.expected_sha256
        ):
            raise ContractError(
                "read-only exact-cache source requires a lowercase SHA-256"
            )
        if self.expected_size_bytes < 0:
            raise ContractError(
                "read-only exact-cache source size must be nonnegative"
            )


@dataclass(frozen=True, slots=True)
class LiveBackendConfig:
    backend_profile_id: str
    solver_id: str = "GUROBI_LP"
    output_root: Path = Path("artifacts/live_plans")
    cache_sink: Path | None = Path("artifacts/exact_cache/live_records.jsonl")
    quick_test: bool = False
    time_limit_seconds: float | None = None
    solver_parameters: tuple[str, ...] = ()
    max_voxels_per_structure: int | None = None
    downsample_seed: int = 17
    supported_formulations: tuple[str, ...] = ("absolute_mean",)
    include_inverse_dual_metrics: bool = False
    include_objective_loss_metrics: bool = False
    require_cache_hit: bool = False
    portfolio_cache_identity: PortfolioExactCacheIdentity | None = None
    read_only_cache_sources: tuple[ReadOnlyExactCacheSource, ...] = ()
    required_cache_keys: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.backend_profile_id or not self.solver_id:
            raise ContractError("backend profile and solver IDs are required")
        if self.time_limit_seconds is not None and self.time_limit_seconds <= 0:
            raise ContractError("time_limit_seconds must be positive")
        if (
            self.max_voxels_per_structure is not None
            and self.max_voxels_per_structure <= 0
        ):
            raise ContractError("max_voxels_per_structure must be positive")
        if not self.supported_formulations:
            raise ContractError("at least one supported formulation is required")
        source_paths = {
            source.path.resolve() for source in self.read_only_cache_sources
        }
        if len(source_paths) != len(self.read_only_cache_sources):
            raise ContractError("read-only exact-cache source paths must be unique")
        if any(
            len(key) != 64
            or any(character not in "0123456789abcdef" for character in key)
            for key in self.required_cache_keys
        ):
            raise ContractError("required exact-cache keys must be SHA-256 values")
        if self.portfolio_cache_identity is not None:
            if (
                self.portfolio_cache_identity.solver_profile_id
                != self.backend_profile_id
            ):
                raise ContractError(
                    "portfolio cache identity solver profile does not match "
                    "live backend profile"
                )
            expected_hash = canonical_solver_parameters_sha256(
                self.solver_parameter_identity()
            )
            if (
                self.portfolio_cache_identity.solver_parameters_sha256
                != expected_hash
            ):
                raise ContractError(
                    "portfolio cache identity solver parameters do not match "
                    "live backend configuration"
                )

    @classmethod
    def paper_full_resolution(
        cls,
        *,
        output_root: str | Path,
        cache_sink: str | Path | None = None,
        backend_profile_id: str = "openkbp-opt-gurobi-11.0.3-absolute-mean-weighted-deviations-v2",
    ) -> "LiveBackendConfig":
        return cls(
            backend_profile_id=backend_profile_id,
            solver_id="GUROBI_LP",
            output_root=Path(output_root),
            cache_sink=None if cache_sink is None else Path(cache_sink),
            quick_test=False,
            time_limit_seconds=1800.0,
            solver_parameters=("Threads 4\nBarHomogeneous 1",),
            max_voxels_per_structure=None,
            supported_formulations=("absolute_mean",),
            include_inverse_dual_metrics=False,
            include_objective_loss_metrics=False,
            require_cache_hit=False,
        )

    def solver_parameter_identity(self) -> dict[str, Any]:
        pass

        return {
            "solver_id": self.solver_id,
            "quick_test": self.quick_test,
            "time_limit_seconds": self.time_limit_seconds,
            "solver_parameters": list(self.solver_parameters),
            "max_voxels_per_structure": self.max_voxels_per_structure,
            "downsample_seed": self.downsample_seed,
            "supported_formulations": list(self.supported_formulations),
            "include_inverse_dual_metrics": self.include_inverse_dual_metrics,
            "include_objective_loss_metrics": (
                self.include_objective_loss_metrics
            ),
        }


class OpenKBPOptLiveBackend:
    pass

    def __init__(
        self,
        repo_root: str | Path = "datasets/open-kbp-opt",
        *,
        config: LiveBackendConfig,
    ) -> None:
        self.layout = OpenKBPOptDataLayout.from_repo_root(repo_root)
        self.config = config
        self.generated_evaluator = GeneratedPlanEvaluator(repo_root)
        self.last_record: ExactFullCacheRecord | None = None
        self.exact_cache_hits = 0
        self.physical_solve_attempts = 0
        self._solved_records = _merge_solved_record_sources(
            (
                _load_solved_records(
                    config.cache_sink,
                    backend_profile_id=config.backend_profile_id,
                    portfolio_identity=config.portfolio_cache_identity,
                ),
                *(
                    _load_declared_solved_records(
                        source,
                        backend_profile_id=config.backend_profile_id,
                        portfolio_identity=config.portfolio_cache_identity,
                    )
                    for source in config.read_only_cache_sources
                ),
            )
        )

    def evaluate(
        self, context: PlanningContext, weights: ObjectiveWeights
    ) -> EvaluationResult:
        if context.planning_formulation not in self.config.supported_formulations:
            raise ContractError(
                f"formulation {context.planning_formulation!r} is not enabled for "
                f"backend profile {self.config.backend_profile_id!r}"
            )
        started = time.perf_counter()
        cache_key = (
            make_exact_cache_key(
                context,
                weights,
                backend_profile_id=self.config.backend_profile_id,
            )
            if self.config.portfolio_cache_identity is None
            else make_portfolio_exact_cache_key(
                context,
                weights,
                identity=self.config.portfolio_cache_identity,
            )
        )
        cached = self._solved_records.get(cache_key)
        if cached is None and self.config.cache_sink is not None:
            self._solved_records = _merge_solved_record_sources(
                (
                    self._solved_records,
                    _load_solved_records(
                        self.config.cache_sink,
                        backend_profile_id=self.config.backend_profile_id,
                        portfolio_identity=self.config.portfolio_cache_identity,
                    ),
                )
            )
            cached = self._solved_records.get(cache_key)
        if cached is not None:
            self.last_record = cached
            self.exact_cache_hits = int(
                getattr(self, "exact_cache_hits", 0)
            ) + 1
            return EvaluationResult(
                source=FeedbackSource.FULL_EXACT_CACHE,
                status=cached.status,
                metrics=cached.metrics,
                constraints=cached.constraints,
                cost=ComputeCost(
                    wall_seconds=time.perf_counter() - started,
                    solver_seconds=cached.original_solver_seconds,
                    full_calls=1,
                ),
                uncertainty=0.0,
                message=f"live_resume_exact_cache:{cached.source_record_id}",
            )
        if (
            self.config.require_cache_hit
            or cache_key in self.config.required_cache_keys
        ):
            raise ContractError(
                "required exact cache record is unavailable: "
                f"cache_key={cache_key}, "
                f"backend_profile_id={self.config.backend_profile_id}"
            )
        run_id = f"{cache_key[:16]}-{uuid.uuid4().hex[:12]}"
        run_root = (
            self.config.output_root.resolve()
            / self.config.backend_profile_id
            / run_id
        )
        run_root.mkdir(parents=True, exist_ok=False)
        self.physical_solve_attempts = int(
            getattr(self, "physical_solve_attempts", 0)
        ) + 1

        solve_seconds = 0.0
        dose_path: Path | None = None
        status = SolverStatus.ERROR
        message: str | None = None
        metrics = MetricVector()
        constraints = MetricVector()
        try:
            dose_path, solve_status, solve_seconds = self._solve(
                context, weights, run_root
            )
            status = _solver_status(solve_status)
            if status is SolverStatus.SOLVED:
                generated = self.generated_evaluator.evaluate(
                    context.patient_id, dose_path
                )
                preference = calculate_preference_compliance(
                    generated.protocol_metrics,
                    clinical_preference_profile(context.clinical_profile_id),
                )
                metrics = MetricVector.from_mapping(
                    {
                        "objective": generated.mean_violation_gy,
                        "mean_protocol_violation_gy": generated.mean_violation_gy,
                        "preference_compliance_score": preference.score,
                        "preference_components_evaluated": float(
                            preference.components_evaluated
                        ),
                        **generated.protocol_metrics.to_dict(),
                        **generated.spatial_quality_metrics.to_dict(),
                        **(
                            _inverse_dual_metrics(dose_path, weights)
                            if self.config.include_inverse_dual_metrics
                            else {}
                        ),
                        **(
                            _objective_loss_metrics(dose_path, weights)
                            if self.config.include_objective_loss_metrics
                            else {}
                        ),
                    }
                )
                constraints = generated.constraint_violations_gy
            else:
                message = f"solver finished with status {solve_status}"
        except Exception as error:
            status = SolverStatus.ERROR
            message = f"{type(error).__name__}: {error}"

        wall_seconds = time.perf_counter() - started
        source_record_id = run_id
        record = ExactFullCacheRecord(
            cache_key=cache_key,
            context=context,
            weights=weights,
            backend_profile_id=self.config.backend_profile_id,
            status=status,
            metrics=metrics,
            constraints=constraints,
            original_wall_seconds=wall_seconds,
            original_solver_seconds=solve_seconds,
            solver_id=self.config.solver_id,
            source_record_id=source_record_id,
            dose_artifact=None if dose_path is None else str(dose_path),
            portfolio_identity=self.config.portfolio_cache_identity,
            schema_version=(
                PORTFOLIO_EXACT_CACHE_SCHEMA_VERSION
                if self.config.portfolio_cache_identity is not None
                else EXACT_CACHE_SCHEMA_VERSION
            ),
        )
        self.last_record = record
        if self.config.cache_sink is not None:
            _append_record(self.config.cache_sink.resolve(), record)
        if (
            record.status is SolverStatus.SOLVED
            and record.dose_artifact is not None
            and Path(record.dose_artifact).is_file()
        ):
            self._solved_records[record.cache_key] = record
        return EvaluationResult(
            source=FeedbackSource.FULL_LIVE_OPTIMIZER,
            status=status,
            metrics=metrics,
            constraints=constraints,
            cost=ComputeCost(
                wall_seconds=wall_seconds,
                solver_seconds=solve_seconds,
                full_calls=1,
            ),
            uncertainty=0.0,
            message=message or f"live_solve:{source_record_id}",
        )

    def _solve(
        self,
        context: PlanningContext,
        weights: ObjectiveWeights,
        run_root: Path,
    ) -> tuple[Path, int, float]:
        for path in (
            str(self.layout.repo_root),
            str(self.layout.repo_root / "provided_code"),
        ):
            if path not in sys.path:
                sys.path.insert(0, path)

        from provided_code.constants_class import ModelParameters
        from provided_code.data_loader import DataLoader
        from provided_code.optimizer import PlanningModel
        from provided_code.resources import Patient

        constants = ModelParameters()
        _configure_output_paths(constants, run_root, context.prediction_set_id)
        patient_directory = self.layout.patient_directory(context.patient_id)
        prediction_file = self.layout.prediction_file(
            context.prediction_set_id, context.patient_id
        )
        anatomy = DataLoader(
            [str(patient_directory)], mode_name="optimization"
        ).get_batch(0)
        prediction = DataLoader(
            [str(prediction_file)], mode_name="predicted_dose"
        ).get_batch(patient_list=[context.patient_id])
        patient = Patient(
            cs=constants,
            identifier=context.patient_id,
            patient_path=str(patient_directory),
            dose=prediction["predicted_dose"][0].squeeze(),
            structure_masks=anatomy["structure_masks"][0],
            dij=anatomy["dij"][0].tocsr(),
            voxel_dimensions=anatomy["voxel_dimensions"][0],
            beamlet_indices=anatomy["beamlet_indices"][0].values,
        )
        _downsample_patient(
            patient,
            max_voxels_per_structure=self.config.max_voxels_per_structure,
            seed=self.config.downsample_seed,
        )
        relative_or_absolute, mean_or_max = context.planning_formulation.split("_")
        weighted_model = _weighted_planning_model_class(PlanningModel)
        model_class = (
            PlanningModel
            if weights == ObjectiveWeights.all_ones()
            else weighted_model
        )
        kwargs: dict[str, Any] = {}
        if model_class is weighted_model:
            kwargs["objective_weights"] = weights

        previous_solver = os.environ.get("OPENKBP_OPT_SOLVER")
        os.environ["OPENKBP_OPT_SOLVER"] = self.config.solver_id
        try:
            model = model_class(
                patient,
                constants,
                relative_or_absolute=relative_or_absolute,
                mean_or_max=mean_or_max,
                inverse_plan=False,
                **kwargs,
            )
            if self.config.time_limit_seconds is not None:
                model._solver.SetTimeLimit(
                    int(self.config.time_limit_seconds * 1000)
                )
            for parameter in self.config.solver_parameters:
                model._solver.SetSolverSpecificParametersAsString(parameter)
            model.solve(quick_test=self.config.quick_test)
            solve_status = int(model.solve_status)
            solve_seconds = float(model.solve_time)
            if _solver_status(solve_status) is SolverStatus.SOLVED:
                model.save_fluence_and_dose()
            return Path(patient.get_dose_path()).resolve(), solve_status, solve_seconds
        finally:
            if previous_solver is None:
                os.environ.pop("OPENKBP_OPT_SOLVER", None)
            else:
                os.environ["OPENKBP_OPT_SOLVER"] = previous_solver


def _inverse_dual_metrics(
    dose_path: Path, weights: ObjectiveWeights
) -> dict[str, float]:
    objective_path, _gap_path = objective_artifacts_from_dose(dose_path)
    summary = interpret_absolute_mean_duals(objective_path, weights)
    rows_by_control: dict[str, list[Any]] = {}
    for row in summary.rows:
        rows_by_control.setdefault(row.control_name, []).append(row)
    metrics = {
        "inverse_dual:materially_active_controls": float(
            summary.materially_active_controls
        ),
        "inverse_dual:materially_active_rows": float(summary.materially_active_rows),
        "inverse_dual:entropy": summary.entropy,
        "inverse_dual:max_bound_violation_fraction": (
            summary.maximum_bound_violation_fraction
        ),
    }
    for control, rows in rows_by_control.items():
        metrics[f"inverse_dual:active_rows:{control}"] = float(
            sum(row.materially_active for row in rows)
        )
        metrics[f"inverse_dual:max_upper_fraction:{control}"] = max(
            row.upper_bound_fraction for row in rows
        )
    return metrics


def _objective_loss_metrics(
    dose_path: Path, weights: ObjectiveWeights
) -> dict[str, float]:
    objective_path, gap_path = objective_artifacts_from_dose(dose_path)
    summary = reconstruct_absolute_mean_losses(
        objective_path,
        weights,
        gap_artifact=gap_path,
    )
    positives = dict(summary.positive_excess_by_control)
    contributions = dict(summary.weighted_contribution_by_control)
    metrics = {
        "objective_activity:reconstructed_total": summary.reconstructed_total,
        "objective_activity:reconstructed_mean": summary.reconstructed_mean,
        "objective_activity:active_controls": float(
            sum(value > 0.0 for value in positives.values())
        ),
    }
    if summary.saved_gap_value is not None:
        metrics["objective_activity:saved_gap_value"] = summary.saved_gap_value
    if summary.absolute_gap_error is not None:
        metrics["objective_activity:absolute_gap_error"] = (
            summary.absolute_gap_error
        )
    for control, value in positives.items():
        metrics[f"objective_activity:positive_excess:{control}"] = value
        metrics[f"objective_activity:weighted_contribution:{control}"] = (
            contributions[control]
        )
    return metrics


def _weighted_planning_model_class(base_class: type) -> type:
    from ortools.linear_solver import pywraplp

    class ObjectiveWeightedPlanningModel(base_class):
        def __init__(
            self,
            *args: Any,
            objective_weights: ObjectiveWeights,
            **kwargs: Any,
        ) -> None:
            self.radoncgym_objective_weights = objective_weights
            super().__init__(*args, **kwargs)

        def obj_norm(
            self, input_dose_obj: Any, opt_dose_obj: Any, objective_name: str
        ) -> None:
            if self.relative_or_absolute == "relative" and input_dose_obj <= 1e-5:
                self.constraint_names.remove(objective_name)
                self._solver.Constraint(0, input_dose_obj).SetCoefficient(
                    opt_dose_obj, 1
                )
                return
            control = objective_name_to_control(objective_name)
            multiplier = (
                1.0
                if control is None
                else self.radoncgym_objective_weights.to_dict()[control]
            )
            if self.mean_or_max == "max":
                if self.relative_or_absolute == "relative":
                    lhs = pywraplp.ProductCst(
                        self.obj_variable, input_dose_obj / multiplier
                    )
                    self._solver.Add(
                        input_dose_obj + lhs >= opt_dose_obj,
                        name=objective_name,
                    )
                else:
                    weighted = pywraplp.ProductCst(
                        self.obj_variable, 1.0 / multiplier
                    )
                    self._solver.Add(
                        pywraplp.SumCst(weighted, input_dose_obj) >= opt_dose_obj,
                        name=objective_name,
                    )
            else:
                sigma = self.add_dummy_variable(lb=0)
                psi = self.add_dummy_variable(lb=0)
                if self.relative_or_absolute == "relative":
                    lhs = pywraplp.ProductCst(sigma, input_dose_obj)
                    psi_term = pywraplp.ProductCst(psi, input_dose_obj)
                    self._solver.Add(
                        input_dose_obj + lhs - psi_term == opt_dose_obj,
                        name=objective_name,
                    )
                else:
                    self._solver.Add(
                        pywraplp.SumCst(sigma, input_dose_obj) - psi
                        == opt_dose_obj,
                        name=objective_name,
                    )
                self._objective.SetCoefficient(sigma, multiplier)
                self._objective.SetCoefficient(psi, -1e-5 * multiplier)
            self.input_objective_values.append(input_dose_obj)

    return ObjectiveWeightedPlanningModel


def _configure_output_paths(
    constants: Any, run_root: Path, prediction_set_id: str
) -> None:
    constants.optimized_plans_dir = str(run_root)
    constants.plan_dose_dir = str(run_root / "plan-dose")
    constants.plan_fluence_dir = str(run_root / "plan-fluence")
    constants.plan_gap_dir = str(run_root / "plan-gap")
    constants.plan_weights_dir = str(run_root / "plan-weights")
    constants.plan_dose_from_pred_dir = str(
        run_root / "plan-dose" / prediction_set_id
    )
    constants.plan_fluence_from_pred_dir = str(
        run_root / "plan-fluence" / prediction_set_id
    )
    constants.plan_gap_from_pred_dir = str(
        run_root / "plan-gap" / prediction_set_id
    )
    constants.plan_weights_from_pred_dir = str(
        run_root / "plan-weights" / prediction_set_id
    )
    for path in (
        constants.plan_dose_from_pred_dir,
        constants.plan_fluence_from_pred_dir,
        constants.plan_gap_from_pred_dir,
        constants.plan_weights_from_pred_dir,
    ):
        Path(path).mkdir(parents=True, exist_ok=True)


def _downsample_patient(
    patient: Any, *, max_voxels_per_structure: int | None, seed: int
) -> None:
    if max_voxels_per_structure is None:
        return
    rng = np.random.default_rng(seed)
    positions: list[int] = []
    for sample_positions in patient.sampled_structure_masks.values():
        selected = np.asarray(sample_positions, dtype=int)
        if selected.size > max_voxels_per_structure:
            selected = rng.choice(
                selected, size=max_voxels_per_structure, replace=False
            )
        positions.extend(int(value) for value in selected)
    selected_positions = np.asarray(sorted(set(positions)), dtype=int)
    original_voxels = patient.sampled_voxels
    new_voxels = original_voxels[selected_positions]
    old_to_new = {
        int(old_position): new_position
        for new_position, old_position in enumerate(selected_positions)
    }
    new_masks: dict[str, np.ndarray] = {}
    for structure, sample_positions in patient.sampled_structure_masks.items():
        mapped = [
            old_to_new[int(position)]
            for position in sample_positions
            if int(position) in old_to_new
        ]
        if mapped:
            new_masks[structure] = np.asarray(mapped, dtype=int)
    patient.sampled_voxels = new_voxels.astype(int)
    patient.sampled_dose = patient.sample_tensor(patient.dose)
    patient.sampled_dij = patient.sample_tensor(patient.dij)
    patient.sampled_structure_masks = new_masks


def _solver_status(status: int) -> SolverStatus:
    return {
        0: SolverStatus.SOLVED,
        1: SolverStatus.SOLVED,
        2: SolverStatus.INFEASIBLE,
        3: SolverStatus.UNBOUNDED,
        4: SolverStatus.NUMERICAL_ERROR,
        6: SolverStatus.TIMEOUT,
    }.get(status, SolverStatus.ERROR)


def _append_record(path: Path, record: ExactFullCacheRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(
            json.dumps(
                record.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _load_solved_records(
    path: Path | None,
    *,
    backend_profile_id: str,
    portfolio_identity: PortfolioExactCacheIdentity | None = None,
) -> dict[str, ExactFullCacheRecord]:
    pass

    if path is None:
        return {}
    resolved = path.resolve()
    if not resolved.is_file():
        return {}
    records: dict[str, ExactFullCacheRecord] = {}
    with resolved.open(encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        try:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = ExactFullCacheRecord.from_dict(json.loads(line))
                except (
                    ContractError,
                    json.JSONDecodeError,
                    KeyError,
                    TypeError,
                    ValueError,
                ):
                    continue
                if (
                    record.backend_profile_id == backend_profile_id
                    and record.portfolio_identity == portfolio_identity
                    and record.status is SolverStatus.SOLVED
                    and record.dose_artifact is not None
                    and Path(record.dose_artifact).is_file()
                ):
                    records.setdefault(record.cache_key, record)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return records


def _load_declared_solved_records(
    source: ReadOnlyExactCacheSource,
    *,
    backend_profile_id: str,
    portfolio_identity: PortfolioExactCacheIdentity | None = None,
) -> dict[str, ExactFullCacheRecord]:
    resolved = source.path.resolve()
    if not resolved.is_file():
        raise ContractError(
            f"declared read-only exact-cache source is missing: {resolved}"
        )
    if resolved.stat().st_size != source.expected_size_bytes:
        raise ContractError(
            "declared read-only exact-cache source size mismatch"
        )
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != source.expected_sha256:
        raise ContractError(
            "declared read-only exact-cache source checksum mismatch"
        )
    return _load_solved_records(
        resolved,
        backend_profile_id=backend_profile_id,
        portfolio_identity=portfolio_identity,
    )


def _merge_solved_record_sources(
    sources: tuple[dict[str, ExactFullCacheRecord], ...],
) -> dict[str, ExactFullCacheRecord]:
    merged: dict[str, ExactFullCacheRecord] = {}
    for source in sources:
        for cache_key, record in source.items():
            prior = merged.get(cache_key)
            if prior is not None and prior.to_dict() != record.to_dict():
                raise ContractError(
                    "conflicting exact-cache records share cache key "
                    f"{cache_key}"
                )
            merged.setdefault(cache_key, record)
    return merged
