

from __future__ import annotations

from pathlib import Path

import numpy as np

from radoncgym.core.errors import ContractError


def load_mask_indices(path: str | Path) -> np.ndarray:
    resolved = Path(path)
    indices = np.loadtxt(
        resolved, delimiter=",", skiprows=1, usecols=(0,), dtype=np.int64
    )
    return np.atleast_1d(indices).astype(np.int64, copy=False)


def load_optional_mask_indices(path: str | Path) -> np.ndarray:
    resolved = Path(path)
    if not resolved.is_file():
        return np.empty((0,), dtype=np.int64)
    return load_mask_indices(resolved)


def load_dense_sparse_values(
    path: str | Path, grid_shape: tuple[int, int, int]
) -> np.ndarray:
    resolved = Path(path)
    rows = np.loadtxt(resolved, delimiter=",", skiprows=1, dtype=np.float64)
    rows = np.atleast_2d(rows)
    if rows.shape[1] != 2:
        raise ContractError(f"expected sparse index/value CSV: {resolved}")
    indices = rows[:, 0].astype(np.int64)
    values = rows[:, 1].astype(np.float64, copy=False)
    size = int(np.prod(grid_shape))
    if indices.size and (indices.min() < 0 or indices.max() >= size):
        raise ContractError(f"voxel index outside grid in {resolved}")
    dense = np.zeros((size,), dtype=np.float64)
    dense[indices] = values
    return dense.reshape(grid_shape)
