

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from radoncgym.data.openkbp import (
    OPENKBP_GRID_SHAPE,
    OpenKBPOptDataLayout,
)
from radoncgym.data.voxel_io import load_dense_sparse_values


@dataclass(frozen=True, slots=True)
class HiddenReferenceDose:
    pass

    patient_id: str
    values: np.ndarray
    source_path: str


class HiddenReferenceDoseLoader:
    pass

    def __init__(
        self,
        repo_root: str | Path = "datasets/open-kbp-opt",
        *,
        grid_shape: tuple[int, int, int] = OPENKBP_GRID_SHAPE,
    ) -> None:
        self.layout = OpenKBPOptDataLayout.from_repo_root(repo_root)
        self.grid_shape = grid_shape

    def load(self, patient_id: str) -> HiddenReferenceDose:
        path = self.layout.patient_directory(patient_id) / "dose.csv"
        values = load_dense_sparse_values(path, self.grid_shape)
        values.setflags(write=False)
        return HiddenReferenceDose(
            patient_id=patient_id,
            values=values,
            source_path=str(path),
        )
