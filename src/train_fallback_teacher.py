"""CLI entrypoint for fallback-teacher RL training."""

from __future__ import annotations

import argparse
import json

from imitation.config import TeacherTrainConfig, load_json_config
from imitation.teacher_training import train_fallback_teacher


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a fallback-only teacher policy.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a JSON config matching imitation.config.TeacherTrainConfig.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    config = load_json_config(args.config, TeacherTrainConfig)
    result = train_fallback_teacher(config)
    print(json.dumps(result, indent=2))
