"""CLI entrypoint for evaluating teacher or student policies."""

from __future__ import annotations

import argparse
import json

from imitation.config import EvaluationConfig, load_json_config
from imitation.evaluation import evaluate_policy


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a teacher or student policy in the fish environment.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a JSON config matching imitation.config.EvaluationConfig.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    config = load_json_config(args.config, EvaluationConfig)
    result = evaluate_policy(config)
    print(json.dumps(result, indent=2))
