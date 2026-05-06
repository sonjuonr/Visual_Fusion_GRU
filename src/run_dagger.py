"""CLI entrypoint for running DAgger iterations."""

from __future__ import annotations

import argparse
import json

from imitation.config import DaggerConfig, load_json_config
from imitation.dagger import run_dagger


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DAgger using a fallback teacher and ViT-heatmap student.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a JSON config matching imitation.config.DaggerConfig.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    config = load_json_config(args.config, DaggerConfig)
    result = run_dagger(config)
    print(json.dumps(result, indent=2))
