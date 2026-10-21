

from __future__ import annotations

from typing import Protocol

from radoncgym.core.models import CandidateAction, Observation
from radoncgym.trajectories.schema import PolicyProvenance


class Agent(Protocol):
    pass

    @property
    def provenance(self) -> PolicyProvenance:
        pass

    def reset(self, observation: Observation, *, seed: int) -> None:
        pass

    def act(self, observation: Observation) -> CandidateAction:
        pass
