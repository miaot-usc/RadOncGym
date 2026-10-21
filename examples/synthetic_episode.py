from __future__ import annotations

import json

from radoncgym.agents import FullCallStrategy, SobolSearchAgent
from radoncgym.envs import make_synthetic_env
from radoncgym.protocol import select_final_weights


def main() -> None:
    env = make_synthetic_env(max_steps=16, max_full_calls=3, max_cheap_calls=16)
    observation, _ = env.engine.reset(seed=23)
    agent = SobolSearchAgent(full_call_strategy=FullCallStrategy.CHEAP_THEN_VERIFY)
    agent.reset(observation, seed=23)
    while True:
        result = env.engine.step(agent.act(observation))
        observation = result.observation
        if not result.info["accepted"]:
            raise RuntimeError(result.info.get("validation_error"))
        if result.terminated or result.truncated:
            break
    selection = select_final_weights(observation.history)
    full_turns = [
        entry.step_index + 1
        for entry in observation.history
        if entry.outcome is not None and entry.outcome.full is not None
    ]
    assert observation.budget_state.steps_used == 16
    assert full_turns == [14, 15, 16]
    assert observation.budget_state.full_calls_solved == 3
    assert not selection.used_all_ones_fallback
    print(
        json.dumps(
            {
                "backend": "synthetic",
                "turns": observation.budget_state.steps_used,
                "full_request_turns": full_turns,
                "successful_full_requests": observation.budget_state.full_calls_solved,
                "selected_turn": selection.step_index + 1,
                "selected_weights": selection.weights.to_dict(),
            },
            indent=2,
        )
    )
    env.close()


if __name__ == "__main__":
    main()
