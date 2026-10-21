

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

from radoncgym.core.constants import (
    FULL_CLINICAL_IMPROVEMENT_METRIC,
    MAX_OBJECTIVE_WEIGHT,
    MIN_OBJECTIVE_WEIGHT,
    OBJECTIVE_NAMES,
    PREDICTED_CLINICAL_IMPROVEMENT_METRIC,
)
from radoncgym.core.errors import ContractError
from radoncgym.core.models import (
    CandidateAction,
    FeedbackSource,
    FidelityControl,
    ObjectiveWeights,
    Observation,
)
from radoncgym.clinical import clinical_preference_profile
from radoncgym.data.openkbp import PublicCaseFeatures
from radoncgym.adapters.dvh import DVHObservation
from radoncgym.evaluators.generated_plan import CLINICAL_CRITERIA
from radoncgym.memory import (
    CriterionResponseContrastMemory,
    FullHistoryMemory,
    ManagedMemoryRequest,
    MemoryPolicy,
    MemoryView,
)
from radoncgym.performance import build_performance_support
from radoncgym.performance.guidance import PREDICTION_GOAL_DEVIATION
from radoncgym.memory.policies import NoHistoryMemory, RecencyWindowMemory
from radoncgym.trajectories.serialization import observation_to_dict


LANGUAGE_SCHEMA_VERSION = "language-json-memory-view-1.1.0"

PLANNING_HYPOTHESIS_MODULE_ID = "explicit_hypothesis_experiment_cycle_v1"
SOURCE_AWARE_FORECAST_MODULE_ID = "source_aware_predict_act_compare_v1"
GOAL_CONSTRAINT_RECENCY_ANCHOR_MODULE_ID = (
    "goal_constraint_recency_anchor_v1"
)
PLANNING_HYPOTHESIS_METRICS = tuple(
    criterion.key for criterion in CLINICAL_CRITERIA
) + (
    PREDICTED_CLINICAL_IMPROVEMENT_METRIC,
    FULL_CLINICAL_IMPROVEMENT_METRIC,
    "objective",
    "multi_criterion_tradeoff",
    "initial_screening",
    "stop_decision",
)
PLANNING_HYPOTHESIS_DIRECTIONS = (
    "improve",
    "remain_within_margin",
    "diagnostic_only",
    "not_applicable",
)
PLANNING_HYPOTHESIS_FEEDBACK_SOURCES = (
    "cheap_only",
    "cheap_and_full",
    "none",
)
SOURCE_AWARE_CHEAP_METRICS = (
    PREDICTED_CLINICAL_IMPROVEMENT_METRIC,
)
SOURCE_AWARE_FULL_METRICS = tuple(
    criterion.key for criterion in CLINICAL_CRITERIA
) + (
    FULL_CLINICAL_IMPROVEMENT_METRIC,
    "objective",
    "multi_criterion_tradeoff",
)
SOURCE_AWARE_FORECAST_DIRECTIONS = (
    "improve",
    "not_improve",
    "diagnostic_only",
    "not_applicable",
)

CHEAP_FEEDBACK_SEMANTICS = {
    "aligned_generated_clinical_multifidelity": {
        "cheap_target": (
            "predicted reduction in generated-plan mean protocol violation "
            "relative to all-ones; positive values indicate improvement"
        ),
        "full_target": (
            "generated-plan clinical improvement relative to all-ones plus "
            "protocol metrics from a fresh physics optimization; improvement "
            "is positive-is-better and violations are lower-is-better"
        ),
        "alignment": "cheap and full target the same generated clinical objective",
    },
}


def _scalar_clinical_memory_policy(memory_policy: MemoryPolicy) -> MemoryPolicy:
    pass
    if isinstance(memory_policy, (FullHistoryMemory, NoHistoryMemory, RecencyWindowMemory)):
        return memory_policy
    if isinstance(memory_policy, CriterionResponseContrastMemory):
        if memory_policy.scalar_clinical_only:
            return memory_policy
        return CriterionResponseContrastMemory(
            recent_steps=memory_policy.recent_steps,
            scalar_clinical_only=True,
        )
    raise ContractError(
        "scalar clinical-only feedback requires history-only or scalar "
        "criterion-response contrast memory"
    )


def _validate_scalar_clinical_memory_view(memory: MemoryView) -> None:
    scalar_contrast = (
        memory.module_id.startswith("criterion_response_contrast_v1:")
        and memory.module_id.endswith(":mode=scalar_clinical:active=none")
    )
    history_only = memory.module_id in {
        FullHistoryMemory.module_id,
        NoHistoryMemory.module_id,
    } or memory.module_id.startswith("recency_window_v1:L=")
    if (
        not (history_only or scalar_contrast)
        or memory.retrieved_episodes
        or memory.reflections
        or memory.semantic_atlases
        or memory.page_operations
        or memory.managed_records
        or memory.managed_operations
        or memory.verified_responses
        or memory.pareto_candidates
        or memory.clinical_subgoals
    ):
        raise ContractError(
            "scalar clinical-only feedback cannot include detailed or external memory"
        )


def _apply_agent_feedback_profile(
    serialized_observation: dict[str, Any],
    *,
    full_feedback_profile: str,
) -> None:
    history = serialized_observation.get("history")
    if not isinstance(history, list):
        raise ContractError("serialized observation history must be a list")
    for entry in history:
        if not isinstance(entry, dict):
            continue
        outcome = entry.get("outcome")
        if not isinstance(outcome, dict):
            continue
        cheap = outcome.get("cheap")
        if isinstance(cheap, dict) and isinstance(cheap.get("metrics"), list):
            improvement = [
                item
                for item in cheap["metrics"]
                if isinstance(item, dict)
                and item.get("name")
                == PREDICTED_CLINICAL_IMPROVEMENT_METRIC
            ]
            if improvement or full_feedback_profile == "scalar_clinical_only":
                cheap["metrics"] = improvement
        if full_feedback_profile != "scalar_clinical_only":
            continue
        if isinstance(cheap, dict):
            cheap["constraints"] = []
        full = outcome.get("full")
        if not isinstance(full, dict):
            continue
        metrics = full.get("metrics")
        full["metrics"] = (
            [
                item
                for item in metrics
                if isinstance(item, dict)
                and item.get("name") == FULL_CLINICAL_IMPROVEMENT_METRIC
            ]
            if isinstance(metrics, list)
            else []
        )
        full["constraints"] = []


SYSTEM_PROMPT = """You are a radiotherapy objective-weight planning agent in RadOncGym.
Choose exactly one next action from the supplied JSON observation.
Return one JSON object only: no Markdown, prose outside JSON, assistant tags,
copied observation text, or hidden reasoning blocks.

Obey the supplied task and action rules. For an agent-choice task, propose
one complete objective-weight vector and decide whether that identical candidate
should also receive a budgeted full physics-verifier call. For a full-only task,
every proposed candidate must request full verification. There is no external
router, and the environment does not force or rewrite an action.

Evidence and planning priorities. Cheap feedback is an approximate learned
estimate. Full feedback comes from a physics optimization and contains
generated-plan protocol metrics, but never hidden expert-reference errors.
Higher reported improvement values and lower constraint violations are better.
Preserve target coverage before pursuing discretionary OAR gains. Use the
explicitly identified memory view to avoid unproductive duplicates. Any retrieved
example comes only from public training-split trajectories and includes its source
and retrieval score.

Meaning of the controls. Interpret weights as relative priorities inside the
inverse-planning objective, not as direct dose prescriptions. Increasing a
target-coverage weight raises the relative penalty for missing that target
objective. Increasing an OAR-sparing weight raises the relative penalty attached
to that OAR objective. The ring and posterior-neck controls emphasize dose
falloff and avoidance in their derived structures. Because all effects interact
through patient geometry, infer direction from the observed weight-response
history rather than assuming independent or monotonic improvement in every
metric.

Use of the expensive verifier. Use cheap evaluations to explore and compare
candidates. A full request is scarce and should be attached to the exact candidate
whose generated-plan DVH and protocol response would materially inform a later
decision or final choice. To escalate a previously cheap-evaluated candidate,
resubmit its exact weights with request_full_verifier=true; this deliberate
duplicate is accepted and returns cheap and full feedback for identical weights.
Do not request a full evaluation merely because budget remains, and do not avoid
one when cheap and observed physics evidence disagree or an important trade-off
requires confirmation. A full call on the last turn cannot guide a revision. When
several full calls are available, distribute their information value across the
remaining horizon so that at least an earlier call can inform later proposals. The
planner owns this decision.

If memory-management rules are present, memory operations are advisory,
evidence-checked requests with a separate budget. They become visible only on a
later turn, never replace the required planning action, and never alter the
environment's immutable public history.
"""


@dataclass(frozen=True, slots=True)
class LanguagePrompt:
    system: str
    user: str
    payload: Mapping[str, Any]
    schema_version: str = LANGUAGE_SCHEMA_VERSION

    def sha256(self) -> str:
        canonical = json.dumps(
            {
                "schema_version": self.schema_version,
                "system": self.system,
                "payload": self.payload,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_language_prompt(
    observation: Observation,
    public_case: PublicCaseFeatures,
    *,
    memory_policy: MemoryPolicy | None = None,
    memory_view: MemoryView | None = None,
    dvh_observation: DVHObservation | None = None,
    performance_modules: tuple[str, ...] = (PREDICTION_GOAL_DEVIATION,),
    fidelity_value_profile: Mapping[str, Any] | None = None,
    tpp_distribution_profile: Mapping[str, Any] | None = None,
    planner_critique: Mapping[str, Any] | None = None,
    prospective_strategy: Mapping[str, Any] | None = None,
    action_portfolio: Mapping[str, Any] | None = None,
    planning_hypothesis_ledger: tuple[Mapping[str, Any], ...] | None = None,
    source_aware_forecast_ledger: tuple[Mapping[str, Any], ...] | None = None,
    cheap_feedback_semantics: str = "aligned_generated_clinical_multifidelity",
    full_feedback_profile: str = "detailed_generated_plan",
    system_prompt: str = SYSTEM_PROMPT,
) -> LanguagePrompt:
    pass

    if observation.context.patient_id != public_case.patient_id:
        raise ContractError("observation and public case patient IDs differ")
    if observation.context.prediction_set_id != public_case.prediction_set_id:
        raise ContractError("observation and public case prediction IDs differ")
    try:
        feedback_semantics = CHEAP_FEEDBACK_SEMANTICS[cheap_feedback_semantics]
    except KeyError as error:
        raise ContractError(
            f"unsupported cheap feedback semantics: {cheap_feedback_semantics!r}"
        ) from error
    if full_feedback_profile not in {
        "detailed_generated_plan",
        "scalar_clinical_only",
    }:
        raise ContractError(
            f"unsupported full feedback profile: {full_feedback_profile!r}"
        )
    if dvh_observation is not None:
        if dvh_observation.patient_id != observation.context.patient_id:
            raise ContractError("DVH observation and episode patient IDs differ")
        if (
            dvh_observation.prediction_set_id
            != observation.context.prediction_set_id
        ):
            raise ContractError("DVH observation and episode prediction IDs differ")
        if dvh_observation.source is not FeedbackSource.CONTEXT:
            raise ContractError(
                "initial language DVH must be selected-prediction context"
            )

    if memory_policy is not None and memory_view is not None:
        raise ContractError("supply memory_policy or memory_view, not both")
    selected_memory = memory_policy or FullHistoryMemory()
    if full_feedback_profile == "scalar_clinical_only":
        if set(performance_modules) - {PREDICTION_GOAL_DEVIATION}:
            raise ContractError(
                "scalar clinical-only feedback permits only prediction goal deviations as performance support modules"
            )
        selected_memory = _scalar_clinical_memory_policy(selected_memory)
    memory = memory_view or selected_memory.select(observation)
    if full_feedback_profile == "scalar_clinical_only":
        _validate_scalar_clinical_memory_view(memory)
    clinical_preference = clinical_preference_profile(
        observation.context.clinical_profile_id
    )
    serialized_observation = observation_to_dict(observation)
    _apply_agent_feedback_profile(
        serialized_observation,
        full_feedback_profile=full_feedback_profile,
    )
    history = serialized_observation["history"]
    evaluated_candidates = sum(
        entry.accepted and entry.outcome is not None
        for entry in observation.history
    )
    required_candidates_before_stop = (
        0
        if observation.task_config.allow_stop_before_candidate
        else observation.task_config.minimum_evaluated_candidates_before_stop
    )
    stop_is_available = (
        observation.task_config.stop_enabled
        and evaluated_candidates >= required_candidates_before_stop
    )
    serialized_observation["history"] = [
        history[index] for index in memory.selected_step_indices
    ]
    if not memory.include_incumbent_pointers:
        serialized_observation["incumbents"] = {
            "best_cheap_step": None,
            "best_full_step": None,
        }
    payload: dict[str, Any] = {
        "schema_version": LANGUAGE_SCHEMA_VERSION,
        "task": {
            "task_id": observation.task_config.task_id,
            "kind": observation.task_config.kind.value,
            "fidelity_control": observation.task_config.fidelity_control.value,
            "full_incumbent_rule": (
                observation.task_config.full_incumbent_rule.value
            ),
            "stop_requires_evaluated_candidate": (
                required_candidates_before_stop > 0
            ),
            "minimum_evaluated_candidates_before_stop": (
                required_candidates_before_stop
            ),
            "remaining_evaluated_candidates_before_stop": max(
                0,
                required_candidates_before_stop - evaluated_candidates,
            ),
        },
        "information_policy": {
            "history": memory.module_id,
            "hidden_reference_feedback_available": False,
            "external_router": False,
            "automatic_full_call_forcing": False,
            "feedback_sources": {
                "cheap": feedback_semantics["cheap_target"],
                "full": "generated_plan_physics_and_protocol_metrics",
                "audit": "hidden_post_episode_only",
            },
            "cheap_full_semantics": {
                "profile": cheap_feedback_semantics,
                **feedback_semantics,
            },
        },
        "action_contract": {
            "objective_names_in_order": list(OBJECTIVE_NAMES),
            "weight_minimum": MIN_OBJECTIVE_WEIGHT,
            "weight_maximum": MAX_OBJECTIVE_WEIGHT,
            "complete_weight_vector_required": True,
            "agent_controls_full_request": (
                observation.task_config.fidelity_control
                is FidelityControl.AGENT_CHOICE
            ),
            "candidate_must_request_full": (
                observation.task_config.fidelity_control
                is FidelityControl.FULL_ONLY
            ),
            "allowed_actions": (
                ["propose_weights", "stop"]
                if stop_is_available
                else ["propose_weights"]
            ),
        },
        "clinical_preference": clinical_preference.to_dict(),
        "objective_semantics": {
            "PTV56_coverage": "relative penalty for the PTV56 coverage objective",
            "PTV63_coverage": "relative penalty for the PTV63 coverage objective when present",
            "PTV70_coverage": "relative penalty for the high-dose PTV70 coverage objective",
            "Brainstem_sparing": "relative penalty for the Brainstem sparing objective",
            "SpinalCord_sparing": "relative penalty for the SpinalCord sparing objective",
            "RightParotid_sparing": "relative penalty for the RightParotid sparing objective",
            "LeftParotid_sparing": "relative penalty for the LeftParotid sparing objective",
            "Esophagus_sparing": "relative penalty for the Esophagus sparing objective",
            "Larynx_sparing": "relative penalty for the Larynx sparing objective",
            "Mandible_sparing": "relative penalty for the Mandible sparing objective",
            "ring_falloff": "relative penalty on derived PTV falloff-ring objectives",
            "posterior_neck_sparing": "relative penalty on the derived posterior-neck avoidance objective",
            "causal_warning": "weights interact through one inverse solve; observed local responses are more reliable than assuming independent monotonic effects"
        },
        "verifier_strategy": {
            "cheap_role": feedback_semantics["cheap_target"],
            "full_role": "obtain positive-is-better clinical improvement, generated-plan physics, DVH metrics, and lower-is-better criterion violations for the identical candidate",
            "source_alignment": feedback_semantics["alignment"],
            "sign_convention": {
                PREDICTED_CLINICAL_IMPROVEMENT_METRIC: "higher_is_better",
                FULL_CLINICAL_IMPROVEMENT_METRIC: "higher_is_better",
                "criterion_violations": "lower_is_better",
                "absolute_plan_objective": "lower_is_better",
            },
            "decision_owner": "agent",
            "exact_escalation": "resubmit an earlier cheap candidate's exact weights with request_full_verifier=true; deliberate escalation duplicates are accepted",
            "recommended_information_value_tests": [
                "candidate is promising under cheap feedback",
                "candidate represents a clinically important target/OAR tradeoff",
                "cheap and prior full evidence disagree",
                "physics feedback can change a later proposal or final choice",
                "enough turns remain to act on this feedback unless this is deliberate final confirmation"
            ],
            "horizon_guidance": "do not defer every full call to episode end; use at least one early or middle full observation when its physics feedback can guide later proposals",
            "forced_schedule": None,
        },
        "clinical_protocol": [
            {
                "metric": criterion.metric_name,
                "roi": criterion.roi_name,
                "relation": criterion.relation,
                "threshold_gy": criterion.threshold_gy,
            }
            for criterion in CLINICAL_CRITERIA
        ],
        "public_case": public_case.to_dict(),
        "public_dvh": (
            None if dvh_observation is None else dvh_observation.to_dict()
        ),
        "memory_view": memory.to_dict(),
        "performance_support": build_performance_support(
            observation,
            public_case,
            module_ids=performance_modules,
            visible_step_indices=memory.selected_step_indices,
            fidelity_value_profile=fidelity_value_profile,
            tpp_distribution_profile=tpp_distribution_profile,
        ),
        "episode_observation": serialized_observation,
    }
    if memory.managed_operation_budget_remaining is not None:
        payload["memory_management_contract"] = {
            "enabled": True,
            "separate_from_planning_action": True,
            "visible_from_next_turn": True,
            "authoritative_history_writable": False,
            "operation_budget_remaining": (
                memory.managed_operation_budget_remaining
            ),
            "operations": {
                "add": "Create a new evidence-backed note.",
                "update": "Create a new version of an active record.",
                "link": "Create a new version with links to active records.",
                "discard": "Hide an active record without deleting its versions.",
                "retrieve": "Choose which active records appear next turn.",
                "summarize": "Create a summary citing at least two source steps.",
            },
            "evidence_fields": {
                "source_step_index": "zero-based completed public step",
                "kind": "typed public evidence kind from the observation",
                "key": "metric/field key when required by the evidence kind",
                "observed_value": "exact value copied from public history",
            },
            "request_fields": [
                "operation",
                "record_id",
                "expected_version",
                "text",
                "kind",
                "evidence",
                "target_record_ids",
                "reason",
            ],
        }
    if planner_critique is not None:
        payload["planner_critique"] = planner_critique
        payload["planner_critique_contract"] = {
            "role": "advisory_evaluator_output",
            "action_authority": False,
            "planner_may_disagree": True,
            "feedback_source": "same_public_observation",
        }
    if prospective_strategy is not None:
        payload["prospective_strategy"] = prospective_strategy
        payload["prospective_strategy_contract"] = {
            "role": "persistent_advisory_episode_plan",
            "action_authority": False,
            "mandatory_schedule": False,
            "planner_may_revise_or_ignore": True,
            "source": "initial_public_observation",
        }
    if action_portfolio is not None:
        payload["action_portfolio"] = action_portfolio
        payload["action_portfolio_contract"] = {
            "role": "advisory_candidate_set",
            "action_authority": False,
            "environment_ranked": False,
            "planner_may_select_modify_or_ignore": True,
            "fidelity_decision_owner": "planner",
            "source": "same_public_observation",
        }
    if PLANNING_HYPOTHESIS_MODULE_ID in performance_modules:
        payload["planning_hypothesis_ledger"] = list(
            planning_hypothesis_ledger or ()
        )
        payload["planning_hypothesis_contract"] = {
            "role": "agent_submitted_testable_planning_hypothesis",
            "action_authority": False,
            "required_on_every_turn": True,
            "raw_environment_history_remains_authoritative": True,
            "allowed_active_metrics": list(PLANNING_HYPOTHESIS_METRICS),
            "allowed_expected_directions": list(
                PLANNING_HYPOTHESIS_DIRECTIONS
            ),
            "feedback_source_must_match_action": True,
            "fields": {
                "active_metric": (
                    "one primary public criterion, objective, tradeoff, initial "
                    "screen, or stop decision"
                ),
                "control_hypothesis": (
                    "brief claim linking the proposed weight change to the "
                    "active planning problem"
                ),
                "expected_direction": (
                    "testable expected response of the active metric"
                ),
                "feedback_source": (
                    "cheap_only for a cheap action, cheap_and_full for a full "
                    "request, and none for stop"
                ),
                "revision_trigger": (
                    "public outcome that would cause the next proposal to change"
                ),
            },
            "instruction": (
                "Treat each proposed plan as an experiment. State one compact "
                "hypothesis before acting, then use the ledger's actual outcome "
                "on later turns to retain, revise, or reject it. Do not treat "
                "your prediction as evidence and do not change the action contract."
            ),
        }
    if SOURCE_AWARE_FORECAST_MODULE_ID in performance_modules:
        payload["source_aware_forecast_ledger"] = list(
            source_aware_forecast_ledger or ()
        )
        payload["source_aware_forecast_contract"] = {
            "role": "agent_submitted_source_matched_outcome_prediction",
            "action_authority": False,
            "required_on_every_turn": True,
            "raw_environment_history_remains_authoritative": True,
            "cheap_action_metrics": list(SOURCE_AWARE_CHEAP_METRICS),
            "full_action_metrics": list(SOURCE_AWARE_FULL_METRICS),
            "allowed_expected_directions": list(
                SOURCE_AWARE_FORECAST_DIRECTIONS
            ),
            "comparison_step_must_be_explicit": True,
            "feedback_source_must_match_action": True,
            "fields": {
                "active_metric": (
                    "predicted clinical improvement for cheap-only actions; "
                    "full clinical improvement, absolute objective, tradeoff, "
                    "or one protocol metric for a requested full action"
                ),
                "comparison_step": (
                    "exact earlier same-source step used as the forecast anchor; "
                    "null only for diagnostic or stop forecasts"
                ),
                "expected_direction": (
                    "whether the candidate will improve the named metric versus "
                    "the exact comparison step"
                ),
                "feedback_source": (
                    "cheap_only for a cheap action, cheap_and_full for a full "
                    "request, and none for stop"
                ),
                "forecast_basis": (
                    "brief public evidence supporting the prediction"
                ),
                "revision_trigger": (
                    "public result that would change the next proposal"
                ),
            },
            "instruction": (
                "Predict only an outcome that the selected verifier will return. "
                "Bind every directional prediction to one exact earlier step "
                "with the same feedback source. On the next turn, compare the "
                "forecast with the authoritative observed outcome before acting."
            ),
        }
    rendered = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False)
    final_decision_anchor = ""
    if GOAL_CONSTRAINT_RECENCY_ANCHOR_MODULE_ID in performance_modules:
        anchor = payload["performance_support"][
            GOAL_CONSTRAINT_RECENCY_ANCHOR_MODULE_ID
        ]
        final_decision_anchor = (
            "\n\nFinal decision anchor (intentionally repeated near the "
            "action):\n"
            + json.dumps(anchor, sort_keys=True, allow_nan=False)
        )
    user = (
        "RadOncGym observation follows. The memory_view states exactly which "
        "authoritative history steps were selected or omitted.\n\n"
        + rendered
        + final_decision_anchor
        + (
            "\n\nReturn exactly one JSON object containing the planning action"
            " and, only when enabled above, optional memory_operations."
        )
    )
    return LanguagePrompt(system=system_prompt, user=user, payload=payload)


@dataclass(frozen=True, slots=True)
class ParsedLanguageTurn:
    pass

    action: CandidateAction
    memory_requests: tuple[ManagedMemoryRequest, ...] = ()
    planning_hypothesis: PlanningHypothesis | None = None
    outcome_forecast: OutcomeForecast | None = None


@dataclass(frozen=True, slots=True)
class PlanningHypothesis:
    pass

    active_metric: str
    control_hypothesis: str
    expected_direction: str
    feedback_source: str
    revision_trigger: str

    def __post_init__(self) -> None:
        if self.active_metric not in PLANNING_HYPOTHESIS_METRICS:
            raise ContractError(
                f"unsupported planning-hypothesis metric: {self.active_metric!r}"
            )
        if self.expected_direction not in PLANNING_HYPOTHESIS_DIRECTIONS:
            raise ContractError(
                "unsupported planning-hypothesis expected direction: "
                f"{self.expected_direction!r}"
            )
        if self.feedback_source not in PLANNING_HYPOTHESIS_FEEDBACK_SOURCES:
            raise ContractError(
                "unsupported planning-hypothesis feedback source: "
                f"{self.feedback_source!r}"
            )
        for name, value in (
            ("control_hypothesis", self.control_hypothesis),
            ("revision_trigger", self.revision_trigger),
        ):
            if not value.strip():
                raise ContractError(f"planning-hypothesis {name} cannot be empty")
            if len(value) > 500:
                raise ContractError(
                    f"planning-hypothesis {name} exceeds 500 characters"
                )

    def to_dict(self) -> dict[str, str]:
        return {
            "active_metric": self.active_metric,
            "control_hypothesis": self.control_hypothesis,
            "expected_direction": self.expected_direction,
            "feedback_source": self.feedback_source,
            "revision_trigger": self.revision_trigger,
        }


@dataclass(frozen=True, slots=True)
class OutcomeForecast:
    pass

    active_metric: str
    comparison_step: int | None
    expected_direction: str
    feedback_source: str
    forecast_basis: str
    revision_trigger: str

    def __post_init__(self) -> None:
        if self.active_metric not in set(SOURCE_AWARE_CHEAP_METRICS).union(
            SOURCE_AWARE_FULL_METRICS, {"stop_decision"}
        ):
            raise ContractError(
                f"unsupported source-aware forecast metric: {self.active_metric!r}"
            )
        if self.expected_direction not in SOURCE_AWARE_FORECAST_DIRECTIONS:
            raise ContractError(
                "unsupported source-aware forecast direction: "
                f"{self.expected_direction!r}"
            )
        if self.feedback_source not in PLANNING_HYPOTHESIS_FEEDBACK_SOURCES:
            raise ContractError(
                "unsupported source-aware feedback source: "
                f"{self.feedback_source!r}"
            )
        if self.comparison_step is not None and self.comparison_step < 0:
            raise ContractError("forecast comparison_step cannot be negative")
        directional = self.expected_direction in {"improve", "not_improve"}
        if directional != (self.comparison_step is not None):
            raise ContractError(
                "directional forecasts require comparison_step; diagnostic and "
                "stop forecasts require comparison_step=null"
            )
        for name, value in (
            ("forecast_basis", self.forecast_basis),
            ("revision_trigger", self.revision_trigger),
        ):
            if not value.strip():
                raise ContractError(f"source-aware forecast {name} cannot be empty")
            if len(value) > 500:
                raise ContractError(
                    f"source-aware forecast {name} exceeds 500 characters"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_metric": self.active_metric,
            "comparison_step": self.comparison_step,
            "expected_direction": self.expected_direction,
            "feedback_source": self.feedback_source,
            "forecast_basis": self.forecast_basis,
            "revision_trigger": self.revision_trigger,
        }


def parse_language_turn(
    text: str,
    *,
    allow_memory_operations: bool = False,
    require_planning_hypothesis: bool = False,
    require_outcome_forecast: bool = False,
) -> ParsedLanguageTurn:
    pass

    if not isinstance(text, str) or not text.strip():
        raise ContractError("language action must be non-empty text")
    stripped = text.strip()
    decoder = json.JSONDecoder()
    try:
        value, end = decoder.raw_decode(stripped)
    except json.JSONDecodeError as error:
        raise ContractError(f"invalid action JSON: {error.msg}") from error
    if stripped[end:].strip():
        raise ContractError("action contains trailing text or multiple objects")
    if not isinstance(value, Mapping):
        raise ContractError("language action must be a JSON object")
    allowed_keys = {"action", "weights", "request_full_verifier"}
    if allow_memory_operations:
        allowed_keys.add("memory_operations")
    if require_planning_hypothesis:
        allowed_keys.add("planning_hypothesis")
    if require_outcome_forecast:
        allowed_keys.add("outcome_forecast")
    unknown_keys = sorted(set(value) - allowed_keys)
    if unknown_keys:
        raise ContractError(f"unknown language action fields: {unknown_keys}")

    raw_memory = value.get("memory_operations", [])
    if not isinstance(raw_memory, list) or any(
        not isinstance(item, Mapping) for item in raw_memory
    ):
        raise ContractError("memory_operations must be a JSON object list")
    memory_requests = tuple(
        _parse_managed_memory_request(item) for item in raw_memory
    )

    planning_hypothesis = (
        _parse_planning_hypothesis(value.get("planning_hypothesis"))
        if require_planning_hypothesis
        else None
    )
    outcome_forecast = (
        _parse_outcome_forecast(value.get("outcome_forecast"))
        if require_outcome_forecast
        else None
    )

    action_name = value.get("action")
    request_full = value.get("request_full_verifier")
    if not isinstance(request_full, bool):
        raise ContractError("request_full_verifier must be a JSON boolean")
    if action_name == "stop":
        if request_full:
            raise ContractError("stop cannot request full verification")
        if value.get("weights") is not None:
            raise ContractError("stop action must omit weights")
        if (
            planning_hypothesis is not None
            and planning_hypothesis.feedback_source != "none"
        ):
            raise ContractError(
                "stop planning hypothesis must use feedback_source='none'"
            )
        if outcome_forecast is not None and (
            outcome_forecast.feedback_source != "none"
            or outcome_forecast.active_metric != "stop_decision"
            or outcome_forecast.expected_direction != "not_applicable"
        ):
            raise ContractError(
                "stop outcome forecast must use stop_decision, not_applicable, "
                "feedback_source='none', and comparison_step=null"
            )
        return ParsedLanguageTurn(
            action=CandidateAction(stop=True),
            memory_requests=memory_requests,
            planning_hypothesis=planning_hypothesis,
            outcome_forecast=outcome_forecast,
        )
    if action_name != "propose_weights":
        raise ContractError(f"unsupported language action: {action_name!r}")

    raw_weights = value.get("weights")
    if not isinstance(raw_weights, Mapping):
        raise ContractError("propose_weights requires a complete weights object")
    weights = ObjectiveWeights.from_mapping(
        {str(name): float(weight) for name, weight in raw_weights.items()}
    )
    if not weights.in_bounds():
        raise ContractError(
            f"weights must lie in [{MIN_OBJECTIVE_WEIGHT}, {MAX_OBJECTIVE_WEIGHT}]"
        )
    expected_feedback_source = (
        "cheap_and_full" if request_full else "cheap_only"
    )
    if (
        planning_hypothesis is not None
        and planning_hypothesis.feedback_source != expected_feedback_source
    ):
        raise ContractError(
            "planning-hypothesis feedback source does not match the action"
        )
    if outcome_forecast is not None:
        if outcome_forecast.feedback_source != expected_feedback_source:
            raise ContractError(
                "outcome-forecast feedback source does not match the action"
            )
        allowed_metrics = (
            SOURCE_AWARE_FULL_METRICS
            if request_full
            else SOURCE_AWARE_CHEAP_METRICS
        )
        if outcome_forecast.active_metric not in allowed_metrics:
            raise ContractError(
                "outcome-forecast metric is not observable from the selected "
                "feedback source"
            )
    return ParsedLanguageTurn(
        action=CandidateAction(
            weights=weights,
            request_full_verifier=request_full,
            stop=False,
        ),
        memory_requests=memory_requests,
        planning_hypothesis=planning_hypothesis,
        outcome_forecast=outcome_forecast,
    )


def parse_language_action(text: str) -> CandidateAction:
    pass

    return parse_language_turn(text, allow_memory_operations=False).action


def _parse_planning_hypothesis(value: Any) -> PlanningHypothesis:
    if not isinstance(value, Mapping):
        raise ContractError("planning_hypothesis must be a JSON object")
    required = {
        "active_metric",
        "control_hypothesis",
        "expected_direction",
        "feedback_source",
        "revision_trigger",
    }
    unknown = sorted(set(value) - required)
    missing = sorted(required - set(value))
    if unknown or missing:
        raise ContractError(
            "planning_hypothesis fields differ from contract: "
            f"missing={missing}, unknown={unknown}"
        )
    return PlanningHypothesis(
        active_metric=str(value["active_metric"]),
        control_hypothesis=str(value["control_hypothesis"]),
        expected_direction=str(value["expected_direction"]),
        feedback_source=str(value["feedback_source"]),
        revision_trigger=str(value["revision_trigger"]),
    )


def _parse_outcome_forecast(value: Any) -> OutcomeForecast:
    if not isinstance(value, Mapping):
        raise ContractError("outcome_forecast must be a JSON object")
    required = {
        "active_metric",
        "comparison_step",
        "expected_direction",
        "feedback_source",
        "forecast_basis",
        "revision_trigger",
    }
    unknown = sorted(set(value) - required)
    missing = sorted(required - set(value))
    if unknown or missing:
        raise ContractError(
            "outcome_forecast fields differ from contract: "
            f"missing={missing}, unknown={unknown}"
        )
    comparison_step = value["comparison_step"]
    if comparison_step is not None and not isinstance(comparison_step, int):
        raise ContractError("outcome_forecast comparison_step must be integer or null")
    return OutcomeForecast(
        active_metric=str(value["active_metric"]),
        comparison_step=comparison_step,
        expected_direction=str(value["expected_direction"]),
        feedback_source=str(value["feedback_source"]),
        forecast_basis=str(value["forecast_basis"]),
        revision_trigger=str(value["revision_trigger"]),
    )


def _parse_managed_memory_request(
    value: Mapping[str, Any],
) -> ManagedMemoryRequest:
    allowed = {
        "operation",
        "record_id",
        "expected_version",
        "text",
        "kind",
        "evidence",
        "target_record_ids",
        "reason",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ContractError(f"unknown managed memory fields: {unknown}")
    evidence = value.get("evidence", [])
    if isinstance(evidence, list):
        evidence_allowed = {
            "source_step_index",
            "kind",
            "key",
            "observed_value",
        }
        for item in evidence:
            if isinstance(item, Mapping):
                evidence_unknown = sorted(set(item) - evidence_allowed)
                if evidence_unknown:
                    raise ContractError(
                        "unknown managed memory evidence fields: "
                        f"{evidence_unknown}"
                    )
    return ManagedMemoryRequest.from_dict(value)
