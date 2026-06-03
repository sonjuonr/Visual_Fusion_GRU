"""Dataset utilities for imitation-learning rollouts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def _stack_record_values(values: list[Any]) -> np.ndarray:
    first = values[0]
    if isinstance(first, np.ndarray):
        return np.stack([np.asarray(value) for value in values], axis=0)
    return np.asarray(values)


def write_rollout_dataset(path: str | Path, records: list[dict[str, Any]], metadata: dict[str, Any]) -> Path:
    if not records:
        raise ValueError("Cannot write an empty rollout dataset.")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    for key in records[0]:
        arrays[key] = _stack_record_values([record[key] for record in records])
    arrays["metadata_json"] = np.asarray(json.dumps(metadata), dtype=np.str_)
    np.savez_compressed(output_path, **arrays)
    return output_path


def resolve_dataset_paths(entries: list[str] | tuple[str, ...]) -> list[Path]:
    paths: list[Path] = []
    for entry in entries:
        path = Path(entry)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.npz")))
        else:
            paths.append(path)
    unique_paths: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique_paths.append(path)
    if not unique_paths:
        raise FileNotFoundError("No dataset shards were found.")
    return unique_paths


def load_rollout_arrays(entries: list[str] | tuple[str, ...]) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[Path]]:
    arrays_by_key: dict[str, list[np.ndarray]] = {}
    metadata: list[dict[str, Any]] = []
    resolved_paths = resolve_dataset_paths(entries)

    for path in resolved_paths:
        with np.load(path, allow_pickle=False) as shard:
            shard_files = set(shard.files)
            if "metadata_json" in shard_files:
                metadata.append(json.loads(str(shard["metadata_json"].item())))
                shard_files.remove("metadata_json")
            else:
                metadata.append({})
            for key in shard_files:
                arrays_by_key.setdefault(key, []).append(np.asarray(shard[key]))

    merged = {key: np.concatenate(parts, axis=0) for key, parts in arrays_by_key.items()}
    return merged, metadata, resolved_paths


def action_histogram(actions: np.ndarray) -> dict[int, int]:
    values, counts = np.unique(actions.astype(np.int64), return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}


class HeatmapActionDataset(Dataset):
    def __init__(self, observations: np.ndarray, actions: np.ndarray) -> None:
        self.observations = torch.from_numpy(np.asarray(observations, dtype=np.float32))
        self.actions = torch.from_numpy(np.asarray(actions, dtype=np.int64))

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.observations[index], self.actions[index]


class HeatmapActionSequenceDataset(Dataset):
    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        episode_keys: np.ndarray,
        step_indices: np.ndarray,
        sequence_length: int,
        stride: int = 1,
    ) -> None:
        self.observations = torch.from_numpy(np.asarray(observations, dtype=np.float32))
        self.actions = torch.from_numpy(np.asarray(actions, dtype=np.int64))
        self.sequence_length = int(sequence_length)
        self.windows: list[np.ndarray] = []

        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")

        stride_int = max(1, int(stride))
        keys = np.asarray(episode_keys)
        steps = np.asarray(step_indices, dtype=np.int64)
        for key in np.unique(keys):
            indices = np.flatnonzero(keys == key)
            indices = indices[np.argsort(steps[indices])]
            if len(indices) == 0:
                continue
            if len(indices) <= self.sequence_length:
                pad_count = self.sequence_length - len(indices)
                if pad_count > 0:
                    indices = np.concatenate(
                        [
                            np.full((pad_count,), indices[0], dtype=indices.dtype),
                            indices,
                        ]
                    )
                self.windows.append(indices)
                continue
            last_start = len(indices) - self.sequence_length
            for start in range(0, last_start + 1, stride_int):
                self.windows.append(indices[start : start + self.sequence_length])
            if self.windows[-1][-1] != indices[-1]:
                self.windows.append(indices[-self.sequence_length :])

        if not self.windows:
            raise ValueError("No sequence windows could be built from rollout data.")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.windows[index]
        return self.observations[window], self.actions[window]


def build_episode_keys(arrays: dict[str, np.ndarray]) -> np.ndarray:
    required = ("iteration_index", "seed", "episode_index")
    missing = [key for key in required if key not in arrays]
    if missing:
        raise KeyError(f"Cannot build sequence dataset; missing keys: {missing}")
    parts = [np.asarray(arrays[key]).astype(np.int64) for key in required]
    return np.asarray([f"{it}:{seed}:{ep}" for it, seed, ep in zip(*parts)], dtype=np.str_)
