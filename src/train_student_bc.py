"""CLI entrypoint for behavior cloning on teacher-labeled datasets."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

from imitation.bc import train_behavior_cloning
from imitation.config import StudentBCConfig, load_json_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PyTorch BC student on heatmap/action data.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a JSON config matching imitation.config.StudentBCConfig.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="*",
        default=None,
        help="Optional dataset path override(s).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional checkpoint output path override.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Optional student checkpoint to warm-start from.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    config = load_json_config(args.config, StudentBCConfig)
    if args.dataset is not None:
        config = replace(config, dataset_paths=list(args.dataset))
    if args.output_path is not None:
        config = replace(config, output_path=args.output_path)
    if args.resume_from is not None:
        config = replace(config, resume_from=args.resume_from)
    result = train_behavior_cloning(
        config,
        dataset_paths=config.dataset_paths,
        output_path=config.output_path,
        resume_from=config.resume_from,
    )
    print(json.dumps(result, indent=2))
