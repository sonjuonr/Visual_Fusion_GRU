"""Fine-tune the GRU student from an already collected hard-seed dataset."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from imitation.bc import train_behavior_cloning
from imitation.config import DaggerConfig, load_json_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GRU student from an existing hard-seed dataset.")
    parser.add_argument("--base-config", default="configs/imitation/dagger_gru_balanced.json")
    parser.add_argument("--hard-dataset", required=True)
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_json_config(args.base_config, DaggerConfig)

    output_root = Path(args.output_root)
    checkpoints_dir = output_root / "checkpoints"
    summaries_dir = output_root / "summaries"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    previous_dagger_shards = sorted(Path(config.output_root).glob("datasets/dagger_iter_*.npz"))
    dataset_paths = (
        list(config.seed_dataset_paths)
        + [str(path) for path in previous_dagger_shards]
        + [str(Path(args.hard_dataset))]
    )
    output_checkpoint = checkpoints_dir / "student_hard_seed.pt"
    bc_config = replace(
        config.bc,
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        output_path=str(output_checkpoint),
        checkpoint_dir=str(checkpoints_dir / "epoch_checkpoints"),
    )
    train_result = train_behavior_cloning(
        bc_config,
        dataset_paths=dataset_paths,
        output_path=str(output_checkpoint),
        resume_from=str(args.student_checkpoint),
        run_name_suffix="hard_seed_dataset",
    )

    summary = {
        "base_config": str(args.base_config),
        "hard_dataset": str(args.hard_dataset),
        "student_checkpoint": str(args.student_checkpoint),
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "dataset_paths": dataset_paths,
        "training": train_result,
        "output_checkpoint": str(output_checkpoint),
    }
    summary_path = summaries_dir / "hard_seed_dataset_train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
