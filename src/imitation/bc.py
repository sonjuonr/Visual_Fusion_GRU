"""Behavior cloning trainer for ViT-heatmap action classification."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from imitation.config import StudentBCConfig
from imitation.constants import NUM_ACTIONS
from imitation.dataset import (
    HeatmapActionDataset,
    HeatmapActionSequenceDataset,
    action_histogram,
    build_episode_keys,
    load_rollout_arrays,
)
from imitation.models import build_heatmap_action_model


def _build_class_weights(actions: np.ndarray, strategy: str, num_classes: int) -> torch.Tensor | None:
    normalized = str(strategy).strip().lower()
    if normalized in {"", "none"}:
        return None
    if normalized != "balanced":
        raise ValueError(f"Unsupported class-weighting strategy: {strategy}")

    actions_int = actions.astype(np.int64)
    counts = np.bincount(actions_int, minlength=int(num_classes)).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (float(num_classes) * counts)
    return torch.as_tensor(weights, dtype=torch.float32)


def _split_indices(num_samples: int, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(num_samples, dtype=np.int64)
    if num_samples <= 1 or val_ratio <= 0.0:
        return indices, np.empty((0,), dtype=np.int64)

    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    raw_val_count = int(round(float(num_samples) * float(val_ratio)))
    val_count = min(max(raw_val_count, 1), num_samples - 1)
    return indices[val_count:], indices[:val_count]


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = torch.argmax(logits, dim=-1)
    return float((predictions == labels).float().mean().item())


def _make_run_name(base_name: str, suffix: str | None) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if suffix:
        return f"{base_name}_{suffix}_{timestamp}"
    return f"{base_name}_{timestamp}"


def save_student_checkpoint(
    path: str | Path,
    model: nn.Module,
    config: StudentBCConfig,
    metrics: dict[str, float],
    class_hist: dict[int, int],
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_type": str(config.model_type),
        "model_config": asdict(config.model),
        "training_config": asdict(config),
        "metrics": metrics,
        "class_histogram": class_hist,
    }
    torch.save(checkpoint, output_path)
    return output_path


def train_behavior_cloning(
    config: StudentBCConfig,
    *,
    dataset_paths: list[str] | None = None,
    output_path: str | None = None,
    resume_from: str | None = None,
    run_name_suffix: str | None = None,
) -> dict[str, object]:
    dataset_entries = list(dataset_paths if dataset_paths is not None else config.dataset_paths)
    if not dataset_entries:
        raise ValueError("Behavior cloning requires at least one dataset path.")

    arrays, dataset_metadata, resolved_paths = load_rollout_arrays(dataset_entries)
    observations = np.asarray(arrays["student_obs"], dtype=np.float32)
    actions = np.asarray(arrays["teacher_action"], dtype=np.int64)
    if observations.shape[0] != actions.shape[0]:
        raise ValueError("Observation/action count mismatch in BC dataset.")

    class_hist = action_histogram(actions)
    class_weights = _build_class_weights(actions, config.class_weighting, NUM_ACTIONS)

    model_type = str(config.model_type).strip().lower()
    is_recurrent = model_type == "gru"

    train_idx, val_idx = _split_indices(len(actions), config.val_ratio, config.shuffle_seed)
    if is_recurrent:
        if "step_index" not in arrays:
            raise KeyError("GRU student training requires step_index in rollout datasets.")
        episode_keys = build_episode_keys(arrays)
        full_dataset = HeatmapActionSequenceDataset(
            observations,
            actions,
            episode_keys=episode_keys,
            step_indices=np.asarray(arrays["step_index"], dtype=np.int64),
            sequence_length=int(config.sequence_length),
            stride=int(config.sequence_stride),
        )
        sequence_indices = np.arange(len(full_dataset), dtype=np.int64)
        rng = np.random.default_rng(config.shuffle_seed)
        rng.shuffle(sequence_indices)
        raw_val_count = int(round(float(len(sequence_indices)) * float(config.val_ratio)))
        val_count = min(max(raw_val_count, 1), max(0, len(sequence_indices) - 1)) if len(sequence_indices) > 1 else 0
        train_idx = sequence_indices[val_count:]
        val_idx = sequence_indices[:val_count]
    else:
        full_dataset = HeatmapActionDataset(observations, actions)
    train_dataset = Subset(full_dataset, train_idx.tolist())
    val_dataset = Subset(full_dataset, val_idx.tolist())

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config.batch_size),
        shuffle=True,
        num_workers=int(config.num_workers),
    )
    val_loader = None
    if len(val_idx) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(config.batch_size),
            shuffle=False,
            num_workers=int(config.num_workers),
        )

    device = torch.device(config.device)
    model = build_heatmap_action_model(model_type, asdict(config.model)).to(device)
    if resume_from:
        checkpoint = torch.load(Path(resume_from), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss(weight=None if class_weights is None else class_weights.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )

    run_name = _make_run_name(config.tb_run_name, run_name_suffix)
    tb_dir = Path(config.tensorboard_log_dir) / run_name
    tb_dir.parent.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))

    best_metrics = {
        "best_epoch": 0.0,
        "best_val_loss": float("inf"),
        "best_val_accuracy": 0.0,
    }
    final_metrics = {
        "final_train_loss": 0.0,
        "final_train_accuracy": 0.0,
        "final_val_loss": 0.0,
        "final_val_accuracy": 0.0,
    }

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_output = Path(output_path if output_path is not None else config.output_path)
    final_output.parent.mkdir(parents=True, exist_ok=True)

    try:
        for epoch in range(1, int(config.epochs) + 1):
            model.train()
            train_loss_sum = 0.0
            train_acc_sum = 0.0
            train_count = 0
            for batch_obs, batch_actions in train_loader:
                batch_obs = batch_obs.to(device)
                batch_actions = batch_actions.to(device)
                optimizer.zero_grad(set_to_none=True)
                if is_recurrent:
                    logits, _ = model(batch_obs)
                    loss = criterion(logits.reshape(-1, NUM_ACTIONS), batch_actions.reshape(-1))
                    train_acc_batch = _accuracy(logits.reshape(-1, NUM_ACTIONS).detach(), batch_actions.reshape(-1))
                    batch_size = int(batch_actions.numel())
                else:
                    logits = model(batch_obs)
                    loss = criterion(logits, batch_actions)
                    train_acc_batch = _accuracy(logits.detach(), batch_actions)
                    batch_size = int(batch_actions.shape[0])
                loss.backward()
                optimizer.step()

                train_loss_sum += float(loss.item()) * batch_size
                train_acc_sum += train_acc_batch * batch_size
                train_count += batch_size

            train_loss = train_loss_sum / max(1, train_count)
            train_acc = train_acc_sum / max(1, train_count)

            val_loss = 0.0
            val_acc = 0.0
            if val_loader is not None:
                model.eval()
                val_loss_sum = 0.0
                val_acc_sum = 0.0
                val_count = 0
                with torch.inference_mode():
                    for batch_obs, batch_actions in val_loader:
                        batch_obs = batch_obs.to(device)
                        batch_actions = batch_actions.to(device)
                        if is_recurrent:
                            logits, _ = model(batch_obs)
                            loss = criterion(logits.reshape(-1, NUM_ACTIONS), batch_actions.reshape(-1))
                            val_acc_batch = _accuracy(logits.reshape(-1, NUM_ACTIONS), batch_actions.reshape(-1))
                            batch_size = int(batch_actions.numel())
                        else:
                            logits = model(batch_obs)
                            loss = criterion(logits, batch_actions)
                            val_acc_batch = _accuracy(logits, batch_actions)
                            batch_size = int(batch_actions.shape[0])

                        val_loss_sum += float(loss.item()) * batch_size
                        val_acc_sum += val_acc_batch * batch_size
                        val_count += batch_size
                val_loss = val_loss_sum / max(1, val_count)
                val_acc = val_acc_sum / max(1, val_count)

            writer.add_scalar("bc/train_loss", train_loss, epoch)
            writer.add_scalar("bc/train_accuracy", train_acc, epoch)
            writer.add_scalar("bc/val_loss", val_loss, epoch)
            writer.add_scalar("bc/val_accuracy", val_acc, epoch)

            final_metrics.update(
                {
                    "final_train_loss": train_loss,
                    "final_train_accuracy": train_acc,
                    "final_val_loss": val_loss,
                    "final_val_accuracy": val_acc,
                }
            )

            score_loss = val_loss if val_loader is not None else train_loss
            score_acc = val_acc if val_loader is not None else train_acc
            if score_loss < float(best_metrics["best_val_loss"]):
                best_metrics = {
                    "best_epoch": float(epoch),
                    "best_val_loss": float(score_loss),
                    "best_val_accuracy": float(score_acc),
                }
                save_student_checkpoint(final_output, model, config, {**best_metrics, **final_metrics}, class_hist)

            if int(config.checkpoint_every_epochs) > 0 and epoch % int(config.checkpoint_every_epochs) == 0:
                ckpt_path = checkpoint_dir / f"{run_name}_epoch_{epoch:03d}.pt"
                save_student_checkpoint(ckpt_path, model, config, {**best_metrics, **final_metrics}, class_hist)
    finally:
        writer.close()

    result = {
        "output_path": str(final_output),
        "tensorboard_dir": str(tb_dir),
        "num_samples": int(len(actions)),
        "num_train_samples": int(len(train_idx)),
        "num_val_samples": int(len(val_idx)),
        "dataset_paths": [str(path) for path in resolved_paths],
        "dataset_metadata": dataset_metadata,
        "class_histogram": class_hist,
        **best_metrics,
        **final_metrics,
    }
    return result
