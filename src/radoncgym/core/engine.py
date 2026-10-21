

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any

from radoncgym.backends.base import CheapBackend, FullBackend
from radoncgym.core.errors import BackendContractError, ContractError, EpisodeStateError
from radoncgym.core.incumbents import full_incumbent_key
from radoncgym.core.constants import FULL_CLINICAL_IMPROVEMENT_METRIC
from radoncgym.core.metric_schema import (
    absolute_clinical_objective,
    clinical_improvement,
)
from radoncgym.core.models import (
    BudgetConfig,
    BudgetState,
    CandidateAction,
    ComputeCost,
    EndReason,
    EvaluationResult,
    FeedbackSource,
    FidelityControl,
    FullIncumbentRule,
    HistoryEntry,
    Incumbents,
    MetricVector,
    Observation,
    PlanningContext,
    PublicContextFeatures,
    SolverStatus,
    StepOutcome,
    StepResult,
    TaskConfig,
    TaskKind,
)
from radoncgym.core.validation import validate_action, validate_task_action
from radoncgym.rewards.base import RewardFunction, ZeroReward


_CHEAP_SOURCES = {
    FeedbackSource.CHEAP_LEARNED_SURROGATE,
    FeedbackSource.APPROXIMATE_CACHE_SURROGATE,
}
_FULL_SOURCES = {
    FeedbackSource.FULL_LIVE_OPTIMIZER,
    FeedbackSource.FULL_EXACT_CACHE,
}


class RadOncEngine:
    pass

    def __init__(
        self,
        *,
        context: PlanningContext,
        budget: BudgetConfig,
        cheap_backend: CheapBackend,
        full_backend: FullBackend,
        public_context: PublicContextFeatures | None = None,
        reward_function: RewardFunction | None = None,
        task_config: TaskConfig | None = None,
        all_ones_clinical_objective: float | None = None,
    ) -> None:
        self._context = context
        self._budget_config = budget
        self._cheap_backend = cheap_backend
        self._full_backend = full_backend
        self._public_context = public_context or PublicContextFeatures.empty()
        self._reward_function = reward_function or ZeroReward()
        self._task_config = task_config or TaskConfig(
            task_id="BudgetedMultiFidelity-v0",
            kind=TaskKind.BUDGETED_MULTI_FIDELITY,
            fidelity_control=FidelityControl.AGENT_CHOICE,
        )
        if all_ones_clinical_objective is not None and (
            not math.isfinite(all_ones_clinical_objective)
            or all_ones_clinical_objective < 0.0
        ):
            raise ContractError(
                "all_ones_clinical_objective must be finite and nonnegative"
            )
        self._all_ones_clinical_objective = all_ones_clinical_objective
        self._validate_task_budget()
        self._observation: Observation | None = None
        self._episode_ended = False

    @property
    def observation(self) -> Observation:
        if self._observation is None:
            raise EpisodeStateError("reset() must be called before reading observation")
        return self._observation

    @property
    def reward_version(self) -> str:
        return self._reward_function.version

    @property
    def budget_config(self) -> BudgetConfig:
        pass

        return self._budget_config

    @property
    def context(self) -> PlanningContext:
        pass

        return self._context

    @property
    def task_config(self) -> TaskConfig:
        pass

        return self._task_config

    @property
    def public_context(self) -> PublicContextFeatures:
        pass

        return self._public_context

    def reset(self, *, seed: int | None = None) -> tuple[Observation, dict[str, Any]]:
        pass

        self._episode_ended = False
        self._observation = Observation(
            context=self._context,
            public_context=self._public_context,
            task_config=self._task_config,
            budget_config=self._budget_config,
            budget_state=BudgetState(),
            history=(),
            incumbents=Incumbents(),
        )
        return self._observation, {
            "seed": seed,
            "reward_version": self.reward_version,
            "environment_version": self._context.environment_version,
            "task_id": self._task_config.task_id,
            "task_kind": self._task_config.kind.value,
            "clinical_profile_id": self._context.clinical_profile_id,
            "public_context_schema_version": self._public_context.schema_version,
            "public_context_feature_names": tuple(
                name for name, _ in self._public_context.items
            ),
        }

    def step(self, action: CandidateAction) -> StepResult:
        pass

        if self._observation is None:
            raise EpisodeStateError("reset() must be called before step()")
        if self._episode_ended:
            raise EpisodeStateError("reset() is required after an episode ends")

        before = self._observation
        if self._task_config.task_id == "FixedClinicalOnly-v0" and action.weights is not None:
            full_turn_count = min(
                before.budget_config.max_full_calls, before.budget_config.max_steps
            )
            action = CandidateAction(
                weights=action.weights,
                request_full_verifier=(
                    before.budget_state.steps_used
                    >= before.budget_config.max_steps - full_turn_count
                ),
                stop=False,
            )
        validation = validate_task_action(action, self._task_config, before.history)
        if validation.accepted:
            validation = validate_action(
                action,
                before.budget_config,
                before.budget_state,
                before.history,
            )
        step_index = before.budget_state.steps_used

        if not validation.accepted:
            return self._record_invalid_action(
                before=before,
                action=action,
                step_index=step_index,
                error=validation.error or "invalid_action",
            )

        if action.stop:
            return self._record_stop(before=before, action=action, step_index=step_index)

        if action.weights is None:
            raise AssertionError("validated candidate action has no weights")

        cheap_is_skipped = (
            action.request_full_verifier
            and not self._task_config.evaluate_cheap_on_full_request
        )
        if cheap_is_skipped:
            cheap = EvaluationResult(
                source=FeedbackSource.ENVIRONMENT_BOOKKEEPING,
                status=SolverStatus.NOT_APPLICABLE,
                cost=ComputeCost(),
                uncertainty=None,
                message="surrogate_skipped_for_direct_full_schedule",
            )
        else:
            previous_cheap = next(
                (
                    entry.outcome.cheap
                    for entry in reversed(before.history)
                    if action.request_full_verifier
                    and entry.accepted
                    and entry.outcome is not None
                    and entry.outcome.executed_weights == action.weights
                    and entry.outcome.cheap.status in {
                        SolverStatus.SOLVED, SolverStatus.NOT_APPLICABLE
                    }
                    and clinical_improvement(entry.outcome.cheap) is not None
                    and entry.outcome.cheap.source in _CHEAP_SOURCES
                ),
                None,
            )
            if previous_cheap is None:
                cheap = self._cheap_backend.evaluate(self._context, action.weights)
                self._validate_cheap_result(cheap)
            else:
                cheap = replace(previous_cheap, cost=ComputeCost())

        full: EvaluationResult | None = None
        if action.request_full_verifier:
            full = self._full_backend.evaluate(self._context, action.weights)
            self._validate_full_result(full)
            full = self._attach_full_improvement(full)

        outcome = StepOutcome(
            executed_weights=action.weights,
            cheap=cheap,
            full=full,
        )
        budget_state = self._advance_budget(
            before.budget_state,
            outcome,
            full_requested=action.request_full_verifier,
        )
        terminated, truncated, end_reason = self._budget_end_state(budget_state)

        provisional_entry = HistoryEntry(
            step_index=step_index,
            action=action,
            accepted=True,
            validation_error=None,
            duplicate_of_step=validation.duplicate_of_step,
            outcome=outcome,
            terminated=terminated,
            truncated=truncated,
            end_reason=end_reason,
        )
        history = before.history + (provisional_entry,)
        incumbents = self._select_incumbents(
            history, self._task_config.full_incumbent_rule
        )
        observation = Observation(
            context=before.context,
            public_context=before.public_context,
            task_config=before.task_config,
            budget_config=before.budget_config,
            budget_state=budget_state,
            history=history,
            incumbents=incumbents,
        )
        reward = float(self._reward_function(before, outcome, True))
        if not math.isfinite(reward):
            raise BackendContractError("Reward function returned a non-finite value")

        self._observation = observation
        self._episode_ended = terminated or truncated
        return StepResult(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={
                "accepted": True,
                "duplicate_of_step": validation.duplicate_of_step,
                "cheap_source": cheap.source.value,
                "full_source": full.source.value if full is not None else None,
                "end_reason": end_reason.value if end_reason is not None else None,
                "reward_version": self.reward_version,
                "task_id": self._task_config.task_id,
            },
        )

    def _record_invalid_action(
        self,
        *,
        before: Observation,
        action: CandidateAction,
        step_index: int,
        error: str,
    ) -> StepResult:
        budget_state = replace(
            before.budget_state,
            steps_used=before.budget_state.steps_used + 1,
            invalid_actions=before.budget_state.invalid_actions + 1,
        )
        terminated = (
            budget_state.invalid_actions >= before.budget_config.invalid_action_limit
        )
        truncated = False
        end_reason: EndReason | None = None
        if terminated:
            end_reason = EndReason.INVALID_ACTION_LIMIT
        elif budget_state.steps_used >= before.budget_config.max_steps:
            truncated = True
            end_reason = EndReason.MAX_STEPS

        entry = HistoryEntry(
            step_index=step_index,
            action=action,
            accepted=False,
            validation_error=error,
            duplicate_of_step=None,
            outcome=None,
            terminated=terminated,
            truncated=truncated,
            end_reason=end_reason,
        )
        observation = replace(
            before,
            budget_state=budget_state,
            history=before.history + (entry,),
        )
        reward = float(self._reward_function(before, None, False))
        if not math.isfinite(reward):
            raise BackendContractError("Reward function returned a non-finite value")

        self._observation = observation
        self._episode_ended = terminated or truncated
        return StepResult(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={
                "accepted": False,
                "validation_error": error,
                "end_reason": end_reason.value if end_reason is not None else None,
                "reward_version": self.reward_version,
                "task_id": self._task_config.task_id,
            },
        )

    def _record_stop(
        self,
        *,
        before: Observation,
        action: CandidateAction,
        step_index: int,
    ) -> StepResult:
        budget_state = replace(
            before.budget_state,
            steps_used=before.budget_state.steps_used + 1,
        )
        entry = HistoryEntry(
            step_index=step_index,
            action=action,
            accepted=True,
            validation_error=None,
            duplicate_of_step=None,
            outcome=None,
            terminated=True,
            truncated=False,
            end_reason=EndReason.AGENT_STOP,
        )
        observation = replace(
            before,
            budget_state=budget_state,
            history=before.history + (entry,),
        )
        reward = float(self._reward_function(before, None, True))
        if not math.isfinite(reward):
            raise BackendContractError("Reward function returned a non-finite value")

        self._observation = observation
        self._episode_ended = True
        return StepResult(
            observation=observation,
            reward=reward,
            terminated=True,
            truncated=False,
            info={
                "accepted": True,
                "end_reason": EndReason.AGENT_STOP.value,
                "reward_version": self.reward_version,
                "task_id": self._task_config.task_id,
            },
        )

    def _advance_budget(
        self,
        before: BudgetState,
        outcome: StepOutcome,
        *,
        full_requested: bool,
    ) -> BudgetState:
        full_solved = int(
            outcome.full is not None and outcome.full.status is SolverStatus.SOLVED
        )
        full_failed = int(outcome.full is not None and not full_solved)
        return BudgetState(
            steps_used=before.steps_used + 1,
            cheap_calls_used=before.cheap_calls_used + outcome.cheap.cost.cheap_calls,
            full_calls_attempted=before.full_calls_attempted + int(full_requested),
            full_calls_solved=before.full_calls_solved + full_solved,
            full_calls_failed=before.full_calls_failed + full_failed,
            invalid_actions=before.invalid_actions,
            wall_seconds=before.wall_seconds + outcome.total_cost.wall_seconds,
            solver_seconds=before.solver_seconds + outcome.total_cost.solver_seconds,
        )

    def _budget_end_state(
        self, budget_state: BudgetState
    ) -> tuple[bool, bool, EndReason | None]:
        config = self._budget_config
        if budget_state.steps_used >= config.max_steps:
            return False, True, EndReason.MAX_STEPS
        if (
            config.max_cheap_calls is not None
            and budget_state.cheap_calls_used >= config.max_cheap_calls
        ):
            return False, True, EndReason.CHEAP_BUDGET_EXHAUSTED
        if (
            config.max_wall_seconds is not None
            and budget_state.wall_seconds >= config.max_wall_seconds
        ):
            return False, True, EndReason.WALL_TIME_BUDGET_EXHAUSTED
        return False, False, None

    def _validate_task_budget(self) -> None:
        task = self._task_config
        budget = self._budget_config
        if task.fidelity_control is FidelityControl.DATASET_ONLY:
            raise ContractError(
                "offline trajectory tasks are replay datasets, not interactive engines"
            )
        if task.kind is TaskKind.SINGLE_SHOT and budget.max_steps != 1:
            raise ContractError("single-shot tasks require max_steps=1")
        if task.minimum_evaluated_candidates_before_stop > budget.max_steps:
            raise ContractError(
                "task stop minimum cannot exceed the episode step budget"
            )
        if (
            task.fidelity_control is FidelityControl.FULL_ONLY
            and budget.max_full_calls < budget.max_steps
        ):
            raise ContractError(
                "full-only tasks require at least one full call per possible step"
            )
        if (
            budget.max_cheap_calls is not None
            and budget.max_cheap_calls < budget.max_steps
        ):
            raise ContractError(
                "interactive task budgets require one cheap call per possible step"
            )

    @staticmethod
    def _validate_cheap_result(result: EvaluationResult) -> None:
        if result.source not in _CHEAP_SOURCES:
            raise BackendContractError(
                f"Cheap backend returned forbidden source: {result.source.value}"
            )
        if result.cost.cheap_calls != 1 or result.cost.full_calls != 0:
            raise BackendContractError(
                "Cheap backend must report exactly one cheap call and zero full calls"
            )

    @staticmethod
    def _validate_full_result(result: EvaluationResult) -> None:
        if result.source not in _FULL_SOURCES:
            raise BackendContractError(
                f"Full backend returned forbidden source: {result.source.value}"
            )
        if result.cost.full_calls != 1 or result.cost.cheap_calls != 0:
            raise BackendContractError(
                "Full backend must report exactly one full call and zero cheap calls"
            )

    @staticmethod
    def _select_incumbents(
        history: tuple[HistoryEntry, ...], rule: FullIncumbentRule
    ) -> Incumbents:
        cheap_candidates: list[tuple[float, int]] = []
        full_candidates: list[tuple[tuple[float, ...], int]] = []
        for entry in history:
            if not entry.accepted or entry.outcome is None:
                continue
            cheap_gain = clinical_improvement(entry.outcome.cheap)
            if cheap_gain is not None:
                cheap_candidates.append((-cheap_gain, entry.step_index))

            full = entry.outcome.full
            if full is None or full.status is not SolverStatus.SOLVED:
                continue
            full_objective = full.metrics.get("objective")
            if full_objective is None:
                continue
            full_candidates.append(
                (
                    full_incumbent_key(
                        rule=rule,
                        constraints=full.constraints,
                        objective=full_objective,
                        step_index=entry.step_index,
                    ),
                    entry.step_index,
                )
            )

        best_cheap = min(cheap_candidates)[1] if cheap_candidates else None
        best_full = min(full_candidates)[1] if full_candidates else None
        return Incumbents(best_cheap_step=best_cheap, best_full_step=best_full)

    def _attach_full_improvement(
        self,
        result: EvaluationResult,
    ) -> EvaluationResult:
        if result.status is not SolverStatus.SOLVED:
            return result
        if clinical_improvement(result) is not None:
            return result
        if self._all_ones_clinical_objective is None:
            return result
        objective = absolute_clinical_objective(result)
        if objective is None:
            raise BackendContractError(
                "solved full result lacks a clinical objective for anchoring"
            )
        metrics = result.metrics.to_dict()
        metrics[FULL_CLINICAL_IMPROVEMENT_METRIC] = (
            self._all_ones_clinical_objective - objective
        )
        return replace(result, metrics=MetricVector.from_mapping(metrics))
