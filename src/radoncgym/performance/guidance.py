

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

from radoncgym.core.errors import ContractError
from radoncgym.core.constants import (
    FULL_CLINICAL_IMPROVEMENT_METRIC,
    PREDICTED_CLINICAL_IMPROVEMENT_METRIC,
)
from radoncgym.core.metric_schema import (
    absolute_clinical_objective,
    clinical_improvement,
)
from radoncgym.core.models import (
    HistoryEntry,
    ObjectiveWeights,
    Observation,
    SolverStatus,
)
from radoncgym.data.openkbp import PublicCaseFeatures
from radoncgym.clinical.protocol import CLINICAL_CRITERIA, ClinicalCriterion
from radoncgym.performance.tpp import validate_tpp_distribution_profile


PREDICTION_GOAL_DEVIATION = "prediction_goal_deviation_v1"
PROTOCOL_ARITHMETIC = "protocol_arithmetic_v1"
VERIFIED_RESPONSE_TREND = "verified_response_trend_v1"
OPTIMIZER_PRIORS = "optimizer_priors_v1"
AGENT_VISIBLE_FIDELITY_VALUE = "agent_visible_fidelity_value_v1"
M2M_TPP_DISTRIBUTION_GUIDANCE = "m2m_tpp_distribution_guidance_v1"
ACTIVE_TPP_RESPONSE_GUIDANCE = "active_tpp_response_guidance_v1"
INVERSE_PLANNING_DUAL_FEEDBACK = "inverse_planning_dual_feedback_v1"
STRUCTURED_DVH_PLAN_REVIEW = "structured_dvh_plan_review_v1"
GENERATED_SPATIAL_QUALITY_REVIEW = "generated_spatial_quality_review_v1"
CLINICAL_PRIORITY_HIERARCHY = "clinical_priority_hierarchy_v1"
STAGNATION_ADAPTIVE_ADJUSTMENT = "stagnation_adaptive_adjustment_v1"
COARSE_TO_REFINE_PLANNING = "coarse_to_refine_planning_v1"
EXPLICIT_HYPOTHESIS_EXPERIMENT_CYCLE = (
    "explicit_hypothesis_experiment_cycle_v1"
)
SOURCE_AWARE_PREDICT_ACT_COMPARE = "source_aware_predict_act_compare_v1"
SCORE_SORTED_EXPERIMENT_TABLE = "score_sorted_experiment_table_v1"
CRITERION_RESPONSE_UNCERTAINTY_PANEL = (
    "criterion_response_uncertainty_panel_v1"
)
RESPONSE_SCALED_ACTION_MAGNITUDE = "response_scaled_action_magnitude_v1"
OBJECTIVE_ACTIVITY_STATE = "objective_activity_state_v1"
EXPLORATION_REFINEMENT_LEDGER = "exploration_refinement_ledger_v1"
RECEDING_HORIZON_ACTION_PLANNING = "receding_horizon_action_planning_v1"
CONSTRAINT_REVISION_LEDGER = "constraint_revision_ledger_v1"
GOAL_CONSTRAINT_RECENCY_ANCHOR = "goal_constraint_recency_anchor_v1"

PERFORMANCE_MODULE_IDS = (
    PREDICTION_GOAL_DEVIATION,
    PROTOCOL_ARITHMETIC,
    VERIFIED_RESPONSE_TREND,
    OPTIMIZER_PRIORS,
    AGENT_VISIBLE_FIDELITY_VALUE,
    M2M_TPP_DISTRIBUTION_GUIDANCE,
    ACTIVE_TPP_RESPONSE_GUIDANCE,
    INVERSE_PLANNING_DUAL_FEEDBACK,
    STRUCTURED_DVH_PLAN_REVIEW,
    GENERATED_SPATIAL_QUALITY_REVIEW,
    CLINICAL_PRIORITY_HIERARCHY,
    STAGNATION_ADAPTIVE_ADJUSTMENT,
    COARSE_TO_REFINE_PLANNING,
    EXPLICIT_HYPOTHESIS_EXPERIMENT_CYCLE,
    SOURCE_AWARE_PREDICT_ACT_COMPARE,
    SCORE_SORTED_EXPERIMENT_TABLE,
    CRITERION_RESPONSE_UNCERTAINTY_PANEL,
    RESPONSE_SCALED_ACTION_MAGNITUDE,
    OBJECTIVE_ACTIVITY_STATE,
    EXPLORATION_REFINEMENT_LEDGER,
    RECEDING_HORIZON_ACTION_PLANNING,
    CONSTRAINT_REVISION_LEDGER,
    GOAL_CONSTRAINT_RECENCY_ANCHOR,
)


def build_performance_support(
    observation: Observation,
    public_case: PublicCaseFeatures,
    *,
    module_ids: Iterable[str],
    visible_step_indices: tuple[int, ...],
    fidelity_value_profile: Mapping[str, Any] | None = None,
    tpp_distribution_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    pass

    requested = tuple(dict.fromkeys(module_ids))
    unknown = sorted(set(requested) - set(PERFORMANCE_MODULE_IDS))
    if unknown:
        raise ContractError(f"unknown performance support modules: {unknown}")
    visible = tuple(observation.history[index] for index in visible_step_indices)
    support: dict[str, Any] = {}
    if PREDICTION_GOAL_DEVIATION in requested:
        support[PREDICTION_GOAL_DEVIATION] = _prediction_goal_deviations(
            public_case
        )
    if PROTOCOL_ARITHMETIC in requested:
        support[PROTOCOL_ARITHMETIC] = _protocol_arithmetic(visible)
    if VERIFIED_RESPONSE_TREND in requested:
        support[VERIFIED_RESPONSE_TREND] = _verified_response_trends(visible)
    if OPTIMIZER_PRIORS in requested:
        support[OPTIMIZER_PRIORS] = _optimizer_priors(observation)
    if AGENT_VISIBLE_FIDELITY_VALUE in requested:
        if fidelity_value_profile is None:
            raise ContractError(
                "agent-visible fidelity value requires a calibration profile"
            )
        support[AGENT_VISIBLE_FIDELITY_VALUE] = _agent_visible_fidelity_value(
            observation,
            visible,
            fidelity_value_profile,
        )
    if M2M_TPP_DISTRIBUTION_GUIDANCE in requested:
        if tpp_distribution_profile is None:
            raise ContractError(
                "M2M TPP guidance requires a training distribution profile"
            )
        support[M2M_TPP_DISTRIBUTION_GUIDANCE] = _m2m_tpp_guidance(
            tpp_distribution_profile
        )
    if ACTIVE_TPP_RESPONSE_GUIDANCE in requested:
        if tpp_distribution_profile is None:
            raise ContractError(
                "active TPP response guidance requires a training distribution profile"
            )
        support[ACTIVE_TPP_RESPONSE_GUIDANCE] = _active_tpp_response_guidance(
            visible,
            public_case,
            tpp_distribution_profile,
        )
    if INVERSE_PLANNING_DUAL_FEEDBACK in requested:
        support[INVERSE_PLANNING_DUAL_FEEDBACK] = (
            _inverse_planning_dual_feedback(visible)
        )
    if STRUCTURED_DVH_PLAN_REVIEW in requested:
        support[STRUCTURED_DVH_PLAN_REVIEW] = _structured_dvh_plan_review(
            visible
        )
    if GENERATED_SPATIAL_QUALITY_REVIEW in requested:
        support[GENERATED_SPATIAL_QUALITY_REVIEW] = (
            _generated_spatial_quality_review(visible)
        )
    if CLINICAL_PRIORITY_HIERARCHY in requested:
        support[CLINICAL_PRIORITY_HIERARCHY] = _clinical_priority_hierarchy(
            visible
        )
    if STAGNATION_ADAPTIVE_ADJUSTMENT in requested:
        support[STAGNATION_ADAPTIVE_ADJUSTMENT] = (
            _stagnation_adaptive_adjustment(visible)
        )
    if COARSE_TO_REFINE_PLANNING in requested:
        support[COARSE_TO_REFINE_PLANNING] = _coarse_to_refine_planning(
            observation,
            visible,
        )
    if EXPLICIT_HYPOTHESIS_EXPERIMENT_CYCLE in requested:
        support[EXPLICIT_HYPOTHESIS_EXPERIMENT_CYCLE] = {
            "source": "agent_submitted_hypotheses_and_public_environment_outcomes",
            "purpose": (
                "make the active planning claim, expected response, feedback "
                "choice, and revision trigger explicit before execution"
            ),
            "evidence_rule": (
                "a hypothesis is a prediction, not evidence; retain, revise, or "
                "reject it only after comparing with returned cheap or full feedback"
            ),
            "action_authority": "language_agent_only",
            "fidelity_decision_owner": "language_agent_only",
        }
    if SOURCE_AWARE_PREDICT_ACT_COMPARE in requested:
        support[SOURCE_AWARE_PREDICT_ACT_COMPARE] = {
            "source": "agent_forecast_and_exact_public_environment_comparison",
            "purpose": (
                "predict only a metric returned by the selected verifier and "
                "compare it with one exact earlier same-source step"
            ),
            "cheap_forecast_scope": [
                PREDICTED_CLINICAL_IMPROVEMENT_METRIC
            ],
            "full_forecast_scope": (
                f"{FULL_CLINICAL_IMPROVEMENT_METRIC}, absolute objective, "
                "multi-criterion tradeoff, or one returned protocol metric"
            ),
            "comparison_anchor": "agent_selected_exact_prior_step",
            "action_authority": "language_agent_only",
            "fidelity_decision_owner": "language_agent_only",
        }
    if SCORE_SORTED_EXPERIMENT_TABLE in requested:
        support[SCORE_SORTED_EXPERIMENT_TABLE] = (
            _score_sorted_experiment_table(visible)
        )
    if CRITERION_RESPONSE_UNCERTAINTY_PANEL in requested:
        support[CRITERION_RESPONSE_UNCERTAINTY_PANEL] = (
            _criterion_response_uncertainty_panel(visible)
        )
    if RESPONSE_SCALED_ACTION_MAGNITUDE in requested:
        support[RESPONSE_SCALED_ACTION_MAGNITUDE] = (
            _response_scaled_action_magnitude(visible)
        )
    if OBJECTIVE_ACTIVITY_STATE in requested:
        support[OBJECTIVE_ACTIVITY_STATE] = _objective_activity_state(visible)
    if EXPLORATION_REFINEMENT_LEDGER in requested:
        support[EXPLORATION_REFINEMENT_LEDGER] = (
            _exploration_refinement_ledger(observation, visible)
        )
    if RECEDING_HORIZON_ACTION_PLANNING in requested:
        support[RECEDING_HORIZON_ACTION_PLANNING] = (
            _receding_horizon_action_planning(observation, visible)
        )
    if CONSTRAINT_REVISION_LEDGER in requested:
        support[CONSTRAINT_REVISION_LEDGER] = _constraint_revision_ledger(
            visible
        )
    if GOAL_CONSTRAINT_RECENCY_ANCHOR in requested:
        support[GOAL_CONSTRAINT_RECENCY_ANCHOR] = {
            "purpose": (
                "repeat the episode goal and binding constraints immediately "
                "before action selection"
            ),
            "primary_goal": (
                "minimize generated-plan protocol violation; lower is better"
            ),
            "clinical_priority": [
                "preserve or repair target coverage first",
                "then reduce serial-OAR violations",
                "then improve parallel-OAR and other soft goals",
            ],
            "feedback_source_rule": (
                "cheap is an aligned learned estimate; full is generated-plan "
                "physics feedback; neither exposes hidden reference error"
            ),
            "remaining_steps": max(
                0,
                observation.budget_config.max_steps
                - observation.budget_state.steps_used,
            ),
            "remaining_full_calls": max(
                0,
                observation.budget_config.max_full_calls
                - observation.budget_state.full_calls_attempted,
            ),
            "action_authority": "language_agent_only",
            "fidelity_decision_owner": "language_agent_only",
        }
    return support


def _constraint_revision_ledger(
    history: tuple[HistoryEntry, ...],
) -> dict[str, Any]:
    pass

    solved = _full_solved(history)
    if not solved:
        return {
            "available": False,
            "reason": "no_visible_solved_full_response",
            "full_steps": [],
            "criteria": [],
        }

    snapshots = []
    for entry in solved:
        assert entry.outcome is not None and entry.outcome.full is not None
        snapshots.append(
            (entry.step_index, entry.outcome.full.constraints.to_dict())
        )
    criterion_names = sorted(
        {name for _step, values in snapshots for name in values}
    )
    tolerance = 1e-9
    rows = []
    for name in criterion_names:
        trajectory = [
            {
                "step": step,
                "violation": max(0.0, float(values.get(name, 0.0))),
            }
            for step, values in snapshots
        ]
        current = trajectory[-1]["violation"]
        if len(trajectory) == 1:
            status = (
                "active_at_first_anchor"
                if current > tolerance
                else "satisfied_at_first_anchor"
            )
            change = None
        else:
            previous = trajectory[-2]["violation"]
            change = current - previous
            active_before_previous = any(
                row["violation"] > tolerance for row in trajectory[:-2]
            )
            if previous <= tolerance and current > tolerance:
                status = (
                    "relapsed" if active_before_previous else "newly_active"
                )
            elif previous > tolerance and current <= tolerance:
                status = "resolved"
            elif previous > tolerance and current > tolerance:
                if current < previous - tolerance:
                    status = "improving_but_persistent"
                elif current > previous + tolerance:
                    status = "worsening_and_persistent"
                else:
                    status = "unchanged_and_persistent"
            else:
                status = "satisfied"
        rows.append(
            {
                "criterion": name,
                "priority_group": _constraint_priority_group(name),
                "status": status,
                "current_violation": current,
                "latest_violation_change": change,
                "change_interpretation": "negative_is_better",
                "trajectory": trajectory,
            }
        )

    active_statuses = {
        "active_at_first_anchor",
        "newly_active",
        "relapsed",
        "improving_but_persistent",
        "worsening_and_persistent",
        "unchanged_and_persistent",
    }
    rows.sort(
        key=lambda row: (
            row["status"] not in active_statuses,
            -row["current_violation"],
            row["criterion"],
        )
    )
    return {
        "available": True,
        "source": "visible_generated_plan_full_constraint_vectors",
        "full_steps": [step for step, _values in snapshots],
        "criteria_evaluated": len(rows),
        "currently_active": sum(
            row["current_violation"] > tolerance for row in rows
        ),
        "resolved_on_latest_transition": sum(
            row["status"] == "resolved" for row in rows
        ),
        "new_or_relapsed_on_latest_transition": sum(
            row["status"] in {"newly_active", "relapsed"} for row in rows
        ),
        "criteria": rows,
        "revision_principles": [
            "Track every active generated-plan constraint across full solves instead of diagnosing only the latest plan.",
            "Treat a resolved criterion as evidence to preserve, and a relapse as evidence that the latest tradeoff was not monotone.",
            "Use cheap feedback for screening, but do not infer constraint resolution until a full generated-plan result reports it.",
            "The ledger summarizes evidence and never chooses weights, fidelity, or stopping.",
        ],
        "action_authority": "language_agent_only",
        "fidelity_decision_owner": "language_agent_only",
    }


def _constraint_priority_group(name: str) -> str:
    if name.startswith("PTV"):
        return "target_coverage"
    if any(value in name for value in ("SpinalCord", "Brainstem", "Mandible")):
        return "serial_or_max_dose_oar"
    return "parallel_or_mean_dose_oar"


def _receding_horizon_action_planning(
    observation: Observation,
    history: tuple[HistoryEntry, ...],
) -> dict[str, Any]:
    pass

    solved_full = _full_solved(history)
    full_objectives = []
    for entry in solved_full:
        assert entry.outcome is not None and entry.outcome.full is not None
        objective = entry.outcome.full.metrics.get("objective")
        if objective is not None:
            full_objectives.append(float(objective))
    best_full = min(full_objectives, default=None)
    latest_full = full_objectives[-1] if full_objectives else None
    if not solved_full:
        evidence_regime = "surrogate_screening_without_physics_anchor"
    elif len(solved_full) == 1:
        evidence_regime = "one_physics_anchor"
    elif latest_full is not None and latest_full <= best_full + 1e-9:
        evidence_regime = "latest_physics_result_is_incumbent"
    else:
        evidence_regime = "latest_physics_result_did_not_improve_incumbent"

    remaining_steps = max(
        0,
        observation.budget_config.max_steps - observation.budget_state.steps_used,
    )
    remaining_full = max(
        0,
        observation.budget_config.max_full_calls
        - observation.budget_state.full_calls_attempted,
    )
    lookahead_depth = min(2, remaining_steps)
    return {
        "available": remaining_steps > 0,
        "source": "visible_public_episode_state_and_feedback",
        "evidence_regime": evidence_regime,
        "remaining_steps": remaining_steps,
        "remaining_full_calls": remaining_full,
        "planning_horizon": lookahead_depth,
        "receding_horizon_protocol": [
            "Before choosing the current action, consider the most likely useful next action if its returned evidence supports the active hypothesis.",
            "Also consider the next action if the returned evidence refutes the active hypothesis or reveals a hard-criterion regression.",
            "Execute only the current canonical action; after feedback arrives, discard stale branches and replan from the authoritative observation.",
            "Use a full call only when physics feedback can distinguish clinically consequential branches, establish an anchor, or confirm a final candidate.",
            "Do not invent unobserved outcomes or treat an internal forecast as evidence.",
        ],
        "adaptation_boundary": (
            "RAFA motivates plan-ahead/act-now replanning, while radiotherapy "
            "foresight planning motivates evaluating future DVO adjustments. "
            "This text-only two-branch protocol is a RadOncGym adaptation and "
            "does not provide a learned transition model or free counterfactual solves."
        ),
        "action_authority": "language_agent_only",
        "fidelity_decision_owner": "language_agent_only",
        "executes_only_first_action": True,
    }


def _exploration_refinement_ledger(
    observation: Observation,
    history: tuple[HistoryEntry, ...],
) -> dict[str, Any]:
    pass

    accepted = tuple(
        entry
        for entry in history
        if entry.accepted and entry.outcome is not None
    )
    control_names = tuple(ObjectiveWeights.all_ones().to_dict())
    vectors = [
        entry.outcome.executed_weights.to_dict()
        for entry in accepted
        if entry.outcome is not None
    ]
    unique_keys = {
        tuple(round(float(vector[name]), 8) for name in control_names)
        for vector in vectors
    }
    pairwise_distances = [
        _mean_log_weight_distance(left, right, control_names)
        for left_index, left in enumerate(vectors)
        for right in vectors[left_index + 1 :]
    ]

    coverage_rows = []
    never_changed = []
    one_sided = []
    for control in control_names:
        values = [float(vector[control]) for vector in vectors]
        increased = any(value > 1.0 + 1e-9 for value in values)
        decreased = any(value < 1.0 - 1e-9 for value in values)
        if not increased and not decreased:
            direction_coverage = "not_changed"
            never_changed.append(control)
        elif increased and decreased:
            direction_coverage = "both_directions"
        else:
            direction_coverage = "increase_only" if increased else "decrease_only"
            one_sided.append(control)
        coverage_rows.append(
            {
                "control": control,
                "direction_coverage": direction_coverage,
                "minimum_weight": min(values, default=1.0),
                "maximum_weight": max(values, default=1.0),
                "distinct_values": len({round(value, 8) for value in values}),
            }
        )

    full_rows = []
    for entry in accepted:
        assert entry.outcome is not None
        if entry.outcome.full is None:
            continue
        objective = entry.outcome.full.metrics.get("objective")
        full_rows.append(
            {
                "step": entry.step_index,
                "objective": objective,
                "constraint_sum": entry.outcome.full.constraints.sum(),
                "weights": entry.outcome.executed_weights.to_dict(),
            }
        )
    full_objectives = [
        float(row["objective"])
        for row in full_rows
        if row["objective"] is not None
    ]
    improving_full_transitions = sum(
        current < previous - 1e-9
        for previous, current in zip(full_objectives, full_objectives[1:])
    )
    full_remaining = max(
        0,
        observation.budget_config.max_full_calls
        - observation.budget_state.full_calls_attempted,
    )
    if not full_rows:
        evidence_regime = "no_physics_anchor_yet"
    elif improving_full_transitions:
        evidence_regime = "verified_improving_direction_exists"
    elif len(full_rows) >= 2:
        evidence_regime = "verified_candidates_not_yet_improving"
    else:
        evidence_regime = "one_physics_anchor_available"

    return {
        "available": bool(accepted),
        "source": "visible_public_episode_actions_and_feedback",
        "accepted_candidates": len(accepted),
        "unique_candidates": len(unique_keys),
        "exact_duplicate_candidates": len(vectors) - len(unique_keys),
        "mean_pairwise_log_weight_distance": (
            None
            if not pairwise_distances
            else sum(pairwise_distances) / len(pairwise_distances)
        ),
        "minimum_pairwise_log_weight_distance": (
            None if not pairwise_distances else min(pairwise_distances)
        ),
        "controls_never_changed": never_changed,
        "controls_with_one_sided_evidence": one_sided,
        "control_coverage": coverage_rows,
        "full_evidence": {
            "rows": full_rows,
            "improving_transitions": improving_full_transitions,
            "evidence_regime": evidence_regime,
            "full_calls_remaining": full_remaining,
        },
        "decision_principles": [
            "Exploit a verified response only when authoritative feedback improved the intended criterion without unacceptable hard-criterion regression.",
            "Explore a materially different clinically plausible hypothesis when evidence is concentrated, duplicated, or repeatedly non-improving.",
            "Use cheap feedback to screen and compare hypotheses; request full physics when the result can change a later action or final selection.",
            "Control coverage is descriptive, not a requirement to perturb clinically irrelevant objectives.",
        ],
        "adaptation_boundary": (
            "The cited radiotherapy studies motivate explicit exploration and "
            "exploitation. This coverage ledger and its statistics are a "
            "RadOncGym adaptation requiring matched evaluation."
        ),
        "action_authority": "language_agent_only",
        "fidelity_decision_owner": "language_agent_only",
    }


def _mean_log_weight_distance(
    left: Mapping[str, float],
    right: Mapping[str, float],
    controls: tuple[str, ...],
) -> float:
    return sum(
        abs(math.log(float(left[name])) - math.log(float(right[name])))
        for name in controls
    ) / len(controls)


def _objective_activity_state(
    history: tuple[HistoryEntry, ...],
) -> dict[str, Any]:
    pass

    solved = _full_solved(history)
    if not solved:
        return {
            "available": False,
            "reason": "no_visible_solved_full_response",
            "latest_full_step": None,
            "controls": [],
        }
    latest = solved[-1]
    assert latest.outcome is not None and latest.outcome.full is not None
    metrics = latest.outcome.full.metrics.to_dict()
    prefix = "objective_activity:positive_excess:"
    controls = []
    for name, positive_excess in metrics.items():
        if not name.startswith(prefix):
            continue
        control = name[len(prefix) :]
        controls.append(
            {
                "control": control,
                "positive_excess_in_objective_units": positive_excess,
                "weighted_contribution": metrics.get(
                    f"objective_activity:weighted_contribution:{control}"
                ),
                "active": positive_excess > 0.0,
            }
        )
    controls.sort(
        key=lambda row: (
            -float(row["positive_excess_in_objective_units"]),
            row["control"],
        )
    )
    if not controls:
        return {
            "available": False,
            "reason": "full_backend_did_not_export_objective_activity",
            "latest_full_step": latest.step_index,
            "controls": [],
        }
    return {
        "available": True,
        "source": "latest_visible_full_solve_saved_openkbp_opt_objective_rows",
        "latest_full_step": latest.step_index,
        "active_controls": int(
            round(metrics["objective_activity:active_controls"])
        ),
        "reconstructed_mean": metrics.get(
            "objective_activity:reconstructed_mean"
        ),
        "saved_gap_value": metrics.get("objective_activity:saved_gap_value"),
        "absolute_gap_error": metrics.get(
            "objective_activity:absolute_gap_error"
        ),
        "controls": controls,
        "interpretation": (
            "Positive excess measures how far an optimized OpenKBP-Opt "
            "dose-mimicking objective remains above its selected predicted-dose "
            "input objective. It is optimizer activity, not a protocol violation."
        ),
        "adaptation_boundary": (
            "The motivating head-and-neck DRL paper used commercial TPS structure "
            "objective losses. These are reconstructed absolute-mean dose-mimicking "
            "contributions and must not be described as the same signal."
        ),
        "action_authority": "language_agent_only",
        "fidelity_decision_owner": "language_agent_only",
    }


def _response_scaled_action_magnitude(
    history: tuple[HistoryEntry, ...],
) -> dict[str, Any]:
    pass

    arithmetic = _protocol_arithmetic(history)
    if not arithmetic.get("available"):
        return {
            "available": False,
            "reason": "no_visible_solved_full_response",
            "latest_full_step": None,
            "active_criteria": [],
            "action_authority": "language_agent_only",
            "fidelity_decision_owner": "language_agent_only",
        }
    trends = _verified_response_trends(history)
    transitions = trends.get("transitions", [])
    latest_transition = transitions[-1] if transitions else None
    latest_margin_changes = (
        {}
        if latest_transition is None
        else latest_transition.get("protocol_margin_change_gy", {})
    )
    active = []
    for row in arithmetic["criteria_by_worst_margin"]:
        threshold = abs(float(row["threshold_gy"]))
        margin = float(row["margin_gy"])
        relative_distance = abs(margin) / threshold if threshold else None
        if row["satisfied"] and (relative_distance is None or relative_distance > 0.03):
            continue
        regime = _response_magnitude_regime(
            satisfied=bool(row["satisfied"]),
            relative_distance=relative_distance,
        )
        observed_margin_change = latest_margin_changes.get(row["criterion"])
        active.append(
            {
                **row,
                "absolute_margin_fraction_of_threshold": relative_distance,
                "revision_scale": regime,
                "latest_verified_margin_change_gy": observed_margin_change,
                "latest_verified_response": (
                    "not_observed"
                    if observed_margin_change is None
                    else (
                        "improved"
                        if float(observed_margin_change) > 0.0
                        else "worsened_or_unchanged"
                    )
                ),
            }
        )
    return {
        "available": True,
        "source": "declared_protocol_and_visible_generated_plan_full_response",
        "latest_full_step": arithmetic["latest_full_step"],
        "active_criteria": active,
        "scale_interpretation": {
            "broad": "criterion is more than 10% of its threshold outside the allowed boundary",
            "moderate": "criterion is 3% to 10% of its threshold outside the allowed boundary",
            "fine": "criterion is within 3% of its threshold or already satisfied near the boundary",
        },
        "agent_instruction": (
            "Use the revision scale only to choose the relative size of a clinically "
            "directed weight change. Reduce adjustment amplitude near the protocol "
            "boundary; use observed full-response direction to avoid repeating an "
            "ineffective change. You retain the exact weight vector and full-request decision."
        ),
        "adaptation_boundary": (
            "The source study changed TPS dose goals in cGy. RadOncGym changes "
            "relative OpenKBP-Opt objective multipliers, so its broad/moderate/fine "
            "3% and 10% bands are a benchmark hypothesis requiring matched ablation, "
            "not published clinical settings."
        ),
        "action_authority": "language_agent_only",
        "fidelity_decision_owner": "language_agent_only",
    }


def _response_magnitude_regime(
    *,
    satisfied: bool,
    relative_distance: float | None,
) -> str:
    if satisfied or relative_distance is None or relative_distance < 0.03:
        return "fine"
    if relative_distance < 0.10:
        return "moderate"
    return "broad"


def _criterion_response_uncertainty_panel(
    history: tuple[HistoryEntry, ...],
) -> dict[str, Any]:
    pass

    latest = next(
        (
            entry
            for entry in reversed(history)
            if entry.accepted and entry.outcome is not None
        ),
        None,
    )
    if latest is None or latest.outcome is None:
        return {
            "available": False,
            "reason": "no_visible_accepted_cheap_response",
            "rows": [],
            "action_authority": "language_agent_only",
            "fidelity_decision_owner": "language_agent_only",
        }
    cheap = latest.outcome.cheap
    metrics = cheap.metrics.to_dict()
    constraints = cheap.constraints.to_dict()
    rows: list[dict[str, Any]] = []
    for criterion in CLINICAL_CRITERIA:
        delta_key = f"predicted_violation_delta:{criterion.key}"
        uncertainty_key = f"criterion_uncertainty_q90:{criterion.key}"
        ensemble_key = f"criterion_ensemble_std:{criterion.key}"
        delta = constraints.get(delta_key)
        uncertainty = metrics.get(uncertainty_key)
        if delta is None or uncertainty is None:
            continue
        lower = float(delta - uncertainty)
        upper = float(delta + uncertainty)
        if upper < 0.0:
            classification = "confident_predicted_violation_reduction"
        elif lower > 0.0:
            classification = "confident_predicted_violation_increase"
        else:
            classification = "interval_crosses_no_change"
        rows.append(
            {
                "criterion": criterion.key,
                "priority_group": _priority_group(criterion),
                "predicted_violation_delta_gy": float(delta),
                "residual_calibrated_q90_gy": float(uncertainty),
                "ensemble_std_gy": metrics.get(ensemble_key),
                "interval_gy": [lower, upper],
                "classification": classification,
            }
        )
    rows.sort(
        key=lambda row: (
            row["classification"] != "confident_predicted_violation_increase",
            -float(row["predicted_violation_delta_gy"]),
            row["criterion"],
        )
    )
    return {
        "available": bool(rows),
        "source": "latest_visible_criterion_response_hgb_prediction",
        "latest_step": latest.step_index,
        "rows": rows,
        "interpretation": (
            "Negative violation delta predicts improvement relative to all-ones. "
            "The interval combines ensemble spread with held-out training residual "
            "error; it is advisory and is not a physics result."
        ),
        "calibration_boundary": (
            "This is a residual-calibrated ensemble interval, not a claimed "
            "epistemic/aleatoric decomposition or a guarantee of coverage."
        ),
        "action_authority": "language_agent_only",
        "fidelity_decision_owner": "language_agent_only",
    }


def _score_sorted_experiment_table(
    history: tuple[HistoryEntry, ...],
    *,
    maximum_rows_per_table: int = 5,
) -> dict[str, Any]:
    pass

    cheap_rows: list[dict[str, Any]] = []
    full_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    for entry in history:
        if not entry.accepted or entry.outcome is None:
            failed_rows.append(
                {
                    "step": entry.step_index,
                    "validation_error": entry.validation_error,
                    "requested_full": entry.action.request_full_verifier,
                }
            )
            continue
        outcome = entry.outcome
        common = {
            "step": entry.step_index,
            "weights": outcome.executed_weights.to_dict(),
            "duplicate_of_step": entry.duplicate_of_step,
        }
        cheap_rows.append(
            {
                **common,
                "clinical_improvement": clinical_improvement(outcome.cheap),
                "constraint_sum": outcome.cheap.constraints.sum(),
                "status": outcome.cheap.status.value,
            }
        )
        if outcome.full is not None:
            full_rows.append(
                {
                    **common,
                    "clinical_improvement": clinical_improvement(outcome.full),
                    "absolute_clinical_objective": absolute_clinical_objective(
                        outcome.full
                    ),
                    "constraint_sum": outcome.full.constraints.sum(),
                    "status": outcome.full.status.value,
                }
            )

    def ranked(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            rows,
            key=lambda row: (
                float("inf")
                if row["clinical_improvement"] is None
                else -float(row["clinical_improvement"]),
                float(row["constraint_sum"]),
                int(row["step"]),
            ),
        )[:maximum_rows_per_table]

    return {
        "source": "visible_public_environment_history",
        "cheap_elite_table": ranked(cheap_rows),
        "full_elite_table": ranked(full_rows),
        "recent_accepted_steps": [
            row["step"] for row in cheap_rows[-maximum_rows_per_table:]
        ],
        "recent_failed_table": failed_rows[-maximum_rows_per_table:],
        "scale_warning": (
            "Cheap and full tables both report positive-is-better clinical "
            "improvement, but raw cross-source magnitudes should be compared "
            "only after the declared calibration."
        ),
        "raw_history_preserved": True,
        "action_authority": "language_agent_only",
        "fidelity_decision_owner": "language_agent_only",
    }


def _coarse_to_refine_planning(
    observation: Observation,
    history: tuple[HistoryEntry, ...],
) -> dict[str, Any]:
    pass

    solved = _full_solved(history)
    hierarchy = _clinical_priority_hierarchy(history)
    stagnation = _stagnation_adaptive_adjustment(history)
    steps_remaining = max(
        0,
        observation.budget_config.max_steps
        - observation.budget_state.steps_used,
    )
    full_remaining = max(
        0,
        observation.budget_config.max_full_calls
        - observation.budget_state.full_calls_attempted,
    )
    if stagnation["consecutive_nonimproving_full_transitions"] >= 2:
        phase = "diversity_reset_after_verified_stagnation"
        action_scope = "one broader clinically plausible tradeoff hypothesis"
        verifier_role = (
            "learn whether the broader hypothesis changes physics response"
        )
    elif not solved:
        phase = "coarse_hypothesis_search"
        action_scope = (
            "meaningful multi-control probes with distinct clinical hypotheses"
        )
        verifier_role = (
            "verify one informative candidate early enough to guide revision"
        )
    elif hierarchy["active_stage"] == "repair_hard_requirements":
        phase = "criterion_focused_repair"
        action_scope = (
            "targeted controls tied to the worst target or serial-OAR margins"
        )
        verifier_role = (
            "measure whether the repair worked before discretionary refinement"
        )
    elif steps_remaining <= 2 or full_remaining <= 1:
        phase = "late_refinement_or_confirmation"
        action_scope = (
            "small local revision or exact confirmation of the preferred incumbent"
        )
        verifier_role = (
            "confirm final generated-plan quality when decision value remains"
        )
    else:
        phase = "feasible_tradeoff_refinement"
        action_scope = (
            "criterion-focused soft-OAR or spatial refinement without hard regression"
        )
        verifier_role = (
            "check a clinically meaningful tradeoff before final selection"
        )
    return {
        "available": True,
        "source": "declared_budget_and_visible_generated_plan_response",
        "planning_phase": phase,
        "solved_full_responses": len(solved),
        "steps_remaining": steps_remaining,
        "full_calls_remaining": full_remaining,
        "active_clinical_stage": hierarchy["active_stage"],
        "recommended_action_scope": action_scope,
        "recommended_verifier_role": verifier_role,
        "phase_is_advisory": True,
        "agent_retains_weight_and_fidelity_control": True,
        "local_adaptation_note": (
            "The source papers motivate hierarchical and diversity/refinement "
            "planning. These phase names and transition rules are RadOncGym "
            "benchmark adaptations and must be evaluated by ablation."
        ),
    }


def _generated_spatial_quality_review(
    history: tuple[HistoryEntry, ...],
) -> dict[str, Any]:
    solved = _full_solved(history)
    if not solved:
        return {
            "available": False,
            "reason": "no_visible_solved_full_response",
            "latest_full_step": None,
            "target_rows": [],
            "falloff_shells": [],
        }
    latest = solved[-1]
    assert latest.outcome is not None and latest.outcome.full is not None
    metrics = latest.outcome.full.metrics.to_dict()
    if "plan_max_dose_gy" not in metrics:
        return {
            "available": False,
            "reason": "latest_full_response_predates_public_plan_metric_schema_0_2_0",
            "latest_full_step": latest.step_index,
            "target_rows": [],
            "falloff_shells": [],
        }
    target_rows = []
    for target in ("PTV56", "PTV63", "PTV70"):
        row = {
            "structure": target,
            "D_2_gy": metrics.get(f"D_2:{target}"),
            "D_98_gy": metrics.get(f"D_98:{target}"),
            "homogeneity_index": metrics.get(f"homogeneity_index:{target}"),
            "conformity_ratio": metrics.get(f"conformity_ratio:{target}"),
        }
        if any(value is not None for key, value in row.items() if key != "structure"):
            target_rows.append(row)
    falloff_shells = []
    for shell in ("0_5_mm", "5_10_mm", "10_20_mm"):
        mean = metrics.get(f"falloff_mean_gy:{shell}")
        p95 = metrics.get(f"falloff_p95_gy:{shell}")
        if mean is None and p95 is None:
            continue
        falloff_shells.append(
            {"outside_target_shell": shell, "mean_gy": mean, "p95_gy": p95}
        )
    return {
        "available": True,
        "source": "latest_visible_generated_dose_and_public_anatomy",
        "latest_full_step": latest.step_index,
        "plan_max_dose_gy": metrics["plan_max_dose_gy"],
        "plan_max_dose_percent_of_PTV70_prescription": metrics.get(
            "plan_max_dose_percent_of_PTV70_prescription"
        ),
        "high_dose_spill_cc_above_73_5_outside_PTV70": metrics.get(
            "high_dose_spill_cc_above_73_5_outside_PTV70"
        ),
        "target_rows": target_rows,
        "falloff_shells": falloff_shells,
        "interpretation": {
            "homogeneity_index": "lower_is_better; (D2-D98)/prescription",
            "conformity_ratio": (
                "prescription_isodose_volume/PTV_volume; values near 1 are preferred "
                "only when target coverage is simultaneously adequate"
            ),
            "falloff": "lower outside-target shell dose is preferred",
            "information_boundary": "no expert-reference dose used",
        },
    }


def _structured_dvh_plan_review(
    history: tuple[HistoryEntry, ...],
) -> dict[str, Any]:
    solved = _full_solved(history)
    if not solved:
        return {
            "available": False,
            "reason": "no_visible_solved_full_response",
            "latest_full_step": None,
            "target_rows": [],
            "oar_rows": [],
        }
    latest = solved[-1]
    assert latest.outcome is not None and latest.outcome.full is not None
    metrics = latest.outcome.full.metrics.to_dict()
    thresholds = {criterion.key: criterion for criterion in CLINICAL_CRITERIA}
    target_rows = []
    for target in ("PTV56", "PTV63", "PTV70"):
        values = {
            metric: metrics.get(f"{metric}:{target}")
            for metric in ("D_99", "D_95", "D_1")
        }
        if not any(value is not None for value in values.values()):
            continue
        criterion = thresholds.get(f"D_99:{target}")
        target_rows.append(
            {
                "structure": target,
                "dvh_gy": values,
                "D_99_threshold_gy": (
                    None if criterion is None else criterion.threshold_gy
                ),
                "D_99_margin_gy": (
                    None
                    if criterion is None or values["D_99"] is None
                    else _criterion_margin(criterion, values["D_99"])
                ),
            }
        )
    oar_rows = []
    oars = (
        "Brainstem",
        "SpinalCord",
        "RightParotid",
        "LeftParotid",
        "Esophagus",
        "Larynx",
        "Mandible",
    )
    for oar in oars:
        values = {
            metric: metrics.get(f"{metric}:{oar}")
            for metric in ("mean", "D_0.1_cc")
        }
        if not any(value is not None for value in values.values()):
            continue
        criteria = []
        for metric in ("mean", "D_0.1_cc"):
            criterion = thresholds.get(f"{metric}:{oar}")
            value = values[metric]
            if criterion is None or value is None:
                continue
            criteria.append(
                {
                    "metric": metric,
                    "threshold_gy": criterion.threshold_gy,
                    "margin_gy": _criterion_margin(criterion, value),
                }
            )
        oar_rows.append(
            {
                "structure": oar,
                "dvh_gy": values,
                "criteria": criteria,
            }
        )
    return {
        "available": True,
        "source": "latest_visible_generated_plan_full_response",
        "latest_full_step": latest.step_index,
        "target_rows": target_rows,
        "oar_rows": oar_rows,
        "interpretation": (
            "This is a tabular summary of generated-plan DVH endpoints from "
            "the full physics solve. It contains no expert-reference dose."
        ),
    }


def _clinical_priority_hierarchy(
    history: tuple[HistoryEntry, ...],
) -> dict[str, Any]:
    arithmetic = _protocol_arithmetic(history)
    rows = arithmetic.get("criteria_by_worst_margin", [])
    hard = [
        row
        for row in rows
        if row["priority_group"]
        in {"target_coverage", "serial_or_max_dose_oar"}
    ]
    soft = [
        row
        for row in rows
        if row["priority_group"] == "parallel_or_mean_dose_oar"
    ]
    hard_violations = [row for row in hard if not row["satisfied"]]
    active_stage = "establish_full_plan_state"
    if arithmetic.get("available"):
        active_stage = (
            "repair_hard_requirements"
            if hard_violations
            else "improve_soft_oar_tradeoffs_without_hard_regression"
        )
    return {
        "available": bool(arithmetic.get("available")),
        "source": "declared_protocol_and_latest_visible_full_response",
        "latest_full_step": arithmetic.get("latest_full_step"),
        "active_stage": active_stage,
        "hard_requirement_rows": hard,
        "soft_goal_rows": soft,
        "decision_order": [
            "restore or preserve target coverage",
            "satisfy serial and maximum-dose OAR requirements",
            "improve parallel-OAR mean-dose goals without regressing hard requirements",
            "only then refine secondary falloff or homogeneity tradeoffs",
        ],
        "action_authority": "language_agent_only",
    }


def _stagnation_adaptive_adjustment(
    history: tuple[HistoryEntry, ...],
    *,
    improvement_tolerance: float = 1e-6,
) -> dict[str, Any]:
    solved = _full_solved(history)
    objectives = []
    for entry in solved:
        assert entry.outcome is not None and entry.outcome.full is not None
        objective = entry.outcome.full.metrics.get("objective")
        if objective is not None:
            objectives.append((entry.step_index, objective))
    consecutive_nonimproving = 0
    for previous, current in reversed(tuple(zip(objectives, objectives[1:]))):
        if current[1] < previous[1] - improvement_tolerance:
            break
        consecutive_nonimproving += 1
    if len(objectives) < 2:
        regime = "insufficient_full_response_history"
    elif consecutive_nonimproving >= 2:
        regime = "broaden_one_clinically_plausible_hypothesis"
    else:
        regime = "targeted_local_revision"
    return {
        "available": len(objectives) >= 2,
        "source": "visible_generated_plan_full_objective_history",
        "full_objective_by_step": [
            {"step": step, "objective": objective}
            for step, objective in objectives
        ],
        "consecutive_nonimproving_full_transitions": consecutive_nonimproving,
        "adjustment_regime": regime,
        "guidance": (
            "Use small targeted changes while verified progress continues. "
            "After two authoritative non-improving transitions, test one "
            "broader but clinically plausible alternative instead of repeating "
            "the same local adjustment. This signal is advisory and never "
            "requests a verifier or stops the episode."
        ),
    }


def _inverse_planning_dual_feedback(
    history: tuple[HistoryEntry, ...],
) -> dict[str, Any]:
    solved = _full_solved(history)
    if not solved:
        return {
            "available": False,
            "reason": "no_visible_solved_full_response",
            "latest_full_step": None,
            "active_controls": [],
        }
    latest = solved[-1]
    assert latest.outcome is not None and latest.outcome.full is not None
    metrics = latest.outcome.full.metrics.to_dict()
    prefix = "inverse_dual:active_rows:"
    active_controls = []
    for name, value in metrics.items():
        if not name.startswith(prefix) or value <= 0:
            continue
        control = name[len(prefix) :]
        active_controls.append(
            {
                "control": control,
                "materially_active_objective_rows": int(round(value)),
                "maximum_fraction_of_outer_multiplier_bound": metrics.get(
                    f"inverse_dual:max_upper_fraction:{control}"
                ),
            }
        )
    active_controls.sort(
        key=lambda row: (
            -row["materially_active_objective_rows"],
            row["control"],
        )
    )
    if not any(name.startswith("inverse_dual:") for name in metrics):
        return {
            "available": False,
            "reason": "full_backend_did_not_export_inverse_duals",
            "latest_full_step": latest.step_index,
            "active_controls": [],
        }
    return {
        "available": True,
        "source": "latest_visible_full_solve_openkbp_opt_dual_vector",
        "latest_full_step": latest.step_index,
        "materially_active_controls": int(
            round(metrics["inverse_dual:materially_active_controls"])
        ),
        "materially_active_objective_rows": int(
            round(metrics["inverse_dual:materially_active_rows"])
        ),
        "active_controls": active_controls,
        "interpretation": (
            "These are objectives supporting the generated full-solver plan "
            "under OpenKBP-Opt duality. They are optimizer-response evidence, "
            "not clinical preferences, hidden reference scores, or recommended "
            "outer-loop multipliers."
        ),
    }


def _m2m_tpp_guidance(profile: Mapping[str, Any]) -> dict[str, Any]:
    validate_tpp_distribution_profile(profile)
    return {
        "source": "training_split_generated_plan_parameter_distributions",
        "profile_id": profile["profile_id"],
        "planning_formulation": profile["planning_formulation"],
        "coverage": profile["coverage"],
        "selection": profile["selection"],
        "parameter_distributions": profile["parameter_distributions"],
        "stable_generated_plan_associations": profile[
            "stable_generated_plan_associations"
        ],
        "association_interpretation": profile["association_interpretation"],
        "usage": profile["usage"],
        "agent_instruction": (
            "Condition candidate weights on these parameter-wise training "
            "distributions and stable associations. They are a soft prior, not "
            "hard bounds. Current-patient cheap and full response evidence takes "
            "precedence, and you still decide whether to request full physics."
        ),
    }


def _active_tpp_response_guidance(
    history: tuple[HistoryEntry, ...],
    public_case: PublicCaseFeatures,
    profile: Mapping[str, Any],
    *,
    maximum_active_criteria: int = 4,
    maximum_associations_per_criterion: int = 3,
) -> dict[str, Any]:
    pass

    validate_tpp_distribution_profile(profile)
    protocol = _protocol_arithmetic(history)
    if protocol.get("available"):
        candidates = list(protocol["criteria_by_worst_margin"])
        source = "latest_visible_generated_plan_full_response"
        criterion_key = "criterion"
        margin_key = "margin_gy"
    else:
        prediction = _prediction_goal_deviations(public_case)
        candidates = list(prediction["criteria_by_worst_margin"])
        source = "selected_prediction_initialization_estimate"
        criterion_key = "criterion"
        margin_key = "estimated_margin_gy"

    violated = [row for row in candidates if float(row[margin_key]) < 0.0]
    active = (violated or candidates)[:maximum_active_criteria]
    distributions = {
        str(row["objective"]): row
        for row in profile["parameter_distributions"]
    }
    associations_by_criterion: dict[str, list[Mapping[str, Any]]] = {}
    for association in profile["stable_generated_plan_associations"]:
        associations_by_criterion.setdefault(
            str(association["criterion"]), []
        ).append(association)

    rows = []
    for criterion in active:
        name = str(criterion[criterion_key])
        associations = sorted(
            associations_by_criterion.get(name, ()),
            key=lambda row: (
                -abs(float(row["median_spearman"])),
                str(row["objective"]),
            ),
        )[:maximum_associations_per_criterion]
        compact_associations = []
        for association in associations:
            objective = str(association["objective"])
            distribution = distributions[objective]
            compact_associations.append(
                {
                    "objective": objective,
                    "higher_weight_associated_with": association[
                        "higher_weight_associated_with"
                    ],
                    "median_spearman": association["median_spearman"],
                    "sign_consistency": association["sign_consistency"],
                    "context_count": association["context_count"],
                    "training_q10_q90": [
                        distribution["q10"],
                        distribution["q90"],
                    ],
                }
            )
        rows.append(
            {
                "criterion": name,
                "current_margin_gy": float(criterion[margin_key]),
                "priority_group": criterion["priority_group"],
                "stable_training_associations": compact_associations,
            }
        )
    return {
        "source": source,
        "training_profile_id": profile["profile_id"],
        "active_criteria": rows,
        "selection_rule": (
            "up to four currently violated criteria by worst margin; if none "
            "are violated, the four smallest positive margins"
        ),
        "interpretation": (
            "Associations summarize randomized training plans and are not "
            "isolated causal effects. Use them as compact directional evidence, "
            "then prefer current-patient full response when available."
        ),
        "action_authority": "language_agent_only",
    }


def _prediction_goal_deviations(public_case: PublicCaseFeatures) -> dict[str, Any]:
    roi_by_name = {roi.name: roi for roi in public_case.rois if roi.present}
    rows = []
    for criterion in CLINICAL_CRITERIA:
        roi = roi_by_name.get(criterion.roi_name)
        if roi is None:
            continue
        if criterion.metric_name == "mean":
            estimate = roi.predicted_dose.mean
            estimator = "prediction_structure_mean"
        elif criterion.metric_name == "D_99":
            estimate = roi.predicted_dose.p01
            estimator = "prediction_structure_p01_approximation"
        elif criterion.metric_name == "D_0.1_cc":
            estimate = roi.predicted_dose.p99
            estimator = "prediction_structure_p99_approximation"
        else:
            continue
        rows.append(
            {
                "criterion": criterion.key,
                "estimated_value_gy": estimate,
                "threshold_gy": criterion.threshold_gy,
                "estimated_margin_gy": _criterion_margin(criterion, estimate),
                "estimator": estimator,
                "priority_group": _priority_group(criterion),
            }
        )
    rows.sort(key=lambda row: (row["estimated_margin_gy"], row["criterion"]))
    return {
        "source": "selected_prediction_and_structure_masks_not_generated_plan",
        "interpretation": (
            "Negative margin suggests a predicted-dose weakness for initialization; "
            "D_99 and D_0.1_cc rows are percentile approximations, not verifier results."
        ),
        "criteria_by_worst_margin": rows,
    }


def _protocol_arithmetic(history: tuple[HistoryEntry, ...]) -> dict[str, Any]:
    solved = _full_solved(history)
    if not solved:
        return {
            "available": False,
            "reason": "no_visible_solved_full_response",
            "latest_full_step": None,
            "criteria_by_worst_margin": [],
        }
    latest = solved[-1]
    assert latest.outcome is not None and latest.outcome.full is not None
    full = latest.outcome.full
    metrics = full.metrics.to_dict()
    rows = []
    for criterion in CLINICAL_CRITERIA:
        value = metrics.get(criterion.key)
        if value is None:
            continue
        margin = _criterion_margin(criterion, value)
        rows.append(
            {
                "criterion": criterion.key,
                "value_gy": value,
                "threshold_gy": criterion.threshold_gy,
                "margin_gy": margin,
                "violation_gy": max(0.0, -margin),
                "satisfied": margin >= 0.0,
                "priority_group": _priority_group(criterion),
            }
        )
    rows.sort(key=lambda row: (row["margin_gy"], row["criterion"]))
    return {
        "available": True,
        "source": "latest_visible_generated_plan_full_response",
        "latest_full_step": latest.step_index,
        "criteria_satisfied": sum(bool(row["satisfied"]) for row in rows),
        "criteria_evaluated": len(rows),
        "criteria_by_worst_margin": rows,
    }


def _verified_response_trends(history: tuple[HistoryEntry, ...]) -> dict[str, Any]:
    solved = _full_solved(history)
    transitions = []
    for previous, current in zip(solved, solved[1:]):
        assert previous.outcome is not None and previous.outcome.full is not None
        assert current.outcome is not None and current.outcome.full is not None
        previous_metrics = previous.outcome.full.metrics.to_dict()
        current_metrics = current.outcome.full.metrics.to_dict()
        prior_weights = previous.outcome.executed_weights.to_dict()
        current_weights = current.outcome.executed_weights.to_dict()
        changed = {
            name: current_weights[name] - prior_weights[name]
            for name in current_weights
            if abs(current_weights[name] - prior_weights[name]) > 1e-12
        }
        margin_changes: dict[str, float] = {}
        for criterion in CLINICAL_CRITERIA:
            before = previous_metrics.get(criterion.key)
            after = current_metrics.get(criterion.key)
            if before is None or after is None:
                continue
            margin_changes[criterion.key] = (
                _criterion_margin(criterion, after)
                - _criterion_margin(criterion, before)
            )
        l1_change = sum(abs(value) for value in changed.values())
        max_margin_response = max(
            (abs(value) for value in margin_changes.values()), default=0.0
        )
        previous_objective = previous_metrics.get("objective")
        current_objective = current_metrics.get("objective")
        transitions.append(
            {
                "from_step": previous.step_index,
                "to_step": current.step_index,
                "weight_delta": changed,
                "weight_l1_change": l1_change,
                "protocol_margin_change_gy": margin_changes,
                "margin_change_interpretation": "positive_is_better",
                "full_objective_change": (
                    None
                    if previous_objective is None or current_objective is None
                    else current_objective - previous_objective
                ),
                "full_objective_change_interpretation": "negative_is_better",
                "low_observed_protocol_response": (
                    l1_change >= 0.25 and max_margin_response <= 0.25
                ),
            }
        )
    return {
        "source": "consecutive_visible_solved_full_responses",
        "solved_full_steps": [entry.step_index for entry in solved],
        "transitions": transitions,
        "causal_caution": (
            "Each transition changes a joint weight vector; treat responses as "
            "local evidence rather than isolated causal effects."
        ),
    }


def _optimizer_priors(observation: Observation) -> dict[str, Any]:
    steps_remaining = max(
        0, observation.budget_config.max_steps - observation.budget_state.steps_used
    )
    full_remaining = max(
        0,
        observation.budget_config.max_full_calls
        - observation.budget_state.full_calls_attempted,
    )
    progress = observation.budget_state.steps_used / observation.budget_config.max_steps
    phase = "early_probe" if progress < 1 / 3 else "midcourse_response"
    if progress >= 2 / 3:
        phase = "late_refinement"
    return {
        "source": "declared_optimizer_semantics_and_episode_budget",
        "planning_phase": phase,
        "steps_remaining": steps_remaining,
        "full_calls_remaining": full_remaining,
        "turns_available_after_full_request_now": max(0, steps_remaining - 1),
        "weight_semantics": (
            "Weights are relative objective penalties, not dose prescriptions. "
            "Increasing a control increases that objective's priority inside the "
            "joint inverse solve."
        ),
        "search_prior": (
            "Use broader, clinically plausible probes early to learn interactions; "
            "use smaller targeted revisions after verified responses or near the "
            "horizon. Reverse or relax a change when dose response plateaus while "
            "target or critical-OAR margins worsen."
        ),
        "verification_prior": (
            "A full call has decision value when later turns remain to respond to "
            "its physics feedback; final confirmation is a distinct use."
        ),
    }


def _agent_visible_fidelity_value(
    observation: Observation,
    history: tuple[HistoryEntry, ...],
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    pass

    if profile.get("schema_version") != "fidelity-value-profile-0.1.0":
        raise ContractError("unsupported fidelity-value calibration profile")
    calibration = profile.get("cheap_calibration")
    costs = profile.get("measured_cost_seconds")
    if not isinstance(calibration, Mapping) or not isinstance(costs, Mapping):
        raise ContractError("fidelity-value profile is missing calibration or cost")

    cheap_seconds = float(costs.get("cheap_warm_mean", -1.0))
    full_seconds = float(costs.get("full_median", -1.0))
    if cheap_seconds <= 0.0 or full_seconds <= 0.0:
        raise ContractError("fidelity-value profile costs must be positive")

    latest_cheap = next(
        (
            entry
            for entry in reversed(history)
            if entry.accepted
            and entry.outcome is not None
        ),
        None,
    )
    current: dict[str, Any]
    if latest_cheap is None:
        current = {
            "available": False,
            "reason": "no_visible_cheap_evaluation",
        }
    else:
        assert latest_cheap.outcome is not None
        cheap = latest_cheap.outcome.cheap
        current = {
            "available": True,
            "step": latest_cheap.step_index,
            "predicted_clinical_improvement_from_all_ones": cheap.metrics.get(
                "predicted_clinical_improvement_from_all_ones"
            ),
            "ensemble_std": cheap.metrics.get("ensemble_std"),
            "reported_uncertainty": cheap.uncertainty,
        }

    steps_remaining = max(
        0, observation.budget_config.max_steps - observation.budget_state.steps_used
    )
    full_remaining = max(
        0,
        observation.budget_config.max_full_calls
        - observation.budget_state.full_calls_attempted,
    )
    return {
        "source": "frozen_training_split_calibration_and_measured_runtime",
        "profile_id": str(profile.get("profile_id", "unknown")),
        "scope_warning": str(profile.get("scope_warning", "")),
        "cheap_calibration": {
            "patient_disjoint_contexts": int(
                calibration["patient_disjoint_contexts"]
            ),
            "mean_context_spearman": float(calibration["mean_context_spearman"]),
            "pairwise_preference_accuracy": float(
                calibration["pairwise_preference_accuracy"]
            ),
            "oracle_best_in_predicted_top3_rate": float(
                calibration["oracle_best_in_predicted_top3_rate"]
            ),
            "median_selection_regret": float(
                calibration["median_selection_regret"]
            ),
            "absolute_residual_q90": float(
                calibration["absolute_residual_q90"]
            ),
        },
        "measured_cost_seconds": {
            "cheap_warm_mean": cheap_seconds,
            "full_median": full_seconds,
            "full_to_cheap_ratio": full_seconds / cheap_seconds,
        },
        "current_visible_cheap_evidence": current,
        "remaining_opportunity": {
            "steps_remaining": steps_remaining,
            "full_calls_remaining": full_remaining,
            "turns_available_after_full_request_now": max(0, steps_remaining - 1),
        },
        "decision_factors_not_a_routing_rule": [
            "whether the candidate is promising under relative cheap ranking",
            "whether reported uncertainty or weak calibration makes cheap evidence insufficient",
            "whether the target/OAR tradeoff is clinically consequential",
            "whether later turns remain to revise the plan after physics feedback",
            "whether a distinct final confirmation use should be preserved",
        ],
        "routing_authority": "language_agent_only",
        "automatic_request_recommendation": None,
    }


def _full_solved(history: tuple[HistoryEntry, ...]) -> tuple[HistoryEntry, ...]:
    return tuple(
        entry
        for entry in history
        if entry.accepted
        and entry.outcome is not None
        and entry.outcome.full is not None
        and entry.outcome.full.status is SolverStatus.SOLVED
    )


def _criterion_margin(criterion: ClinicalCriterion, value: float) -> float:
    if criterion.relation == "at_least":
        return float(value - criterion.threshold_gy)
    return float(criterion.threshold_gy - value)


def _priority_group(criterion: ClinicalCriterion) -> str:
    if criterion.roi_name.startswith("PTV"):
        return "target_coverage"
    if criterion.roi_name in {"SpinalCord", "Brainstem", "Mandible"}:
        return "serial_or_max_dose_oar"
    return "parallel_or_mean_dose_oar"
