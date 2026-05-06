"""CLI entrypoint for collecting teacher-labeled heatmap datasets."""

from __future__ import annotations

import argparse
import json

from imitation.collector import collect_teacher_seed_dataset
from imitation.config import CollectionConfig, load_json_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect a teacher-labeled dataset for imitation learning.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a JSON config matching imitation.config.CollectionConfig.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    config = load_json_config(args.config, CollectionConfig)
    result = collect_teacher_seed_dataset(config)
    print(json.dumps(result, indent=2))
