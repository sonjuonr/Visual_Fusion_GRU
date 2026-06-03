"""Evaluate several student checkpoints with the same seed schedule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from imitation.config import EnvRuntimeConfig, EvaluationConfig
from imitation.evaluation import _evaluate_policy_in_env
from imitation.policies import TorchStudentPolicyAdapter
from imitation.runtime import SimulationAppSession, build_underwater_env


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multiple Torch student checkpoints.")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Checkpoint paths or glob patterns, e.g. 'imitation_runs/.../student_iter_*.pt'.",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--start-seed", type=int, default=200000)
    parser.add_argument("--output-dir", type=str, default="imitation_data/eval/dagger_sweep")
    parser.add_argument("--observation-view-name", type=str, default="clip")
    parser.add_argument("--renderer", type=str, default="HydraStorm")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _resolve_checkpoints(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        path = Path(pattern)
        if any(char in pattern for char in "*?[]"):
            paths.extend(sorted(Path().glob(pattern)))
        else:
            paths.append(path)
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    if not unique:
        raise FileNotFoundError("No checkpoints matched the provided patterns.")
    return unique


if __name__ == "__main__":
    args = _parse_args()
    checkpoints = _resolve_checkpoints(args.checkpoints)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env_config = EnvRuntimeConfig(renderer=args.renderer, headless=bool(args.headless))
    summaries: list[dict[str, object]] = []

    with SimulationAppSession(env_config):
        env = build_underwater_env(env_config)
        try:
            for checkpoint_path in checkpoints:
                policy = TorchStudentPolicyAdapter(str(checkpoint_path), deterministic=True)
                output_path = output_dir / f"{checkpoint_path.stem}_eval_{int(args.episodes)}eps.json"
                config = EvaluationConfig(
                    env=env_config,
                    policy_type="torch_student",
                    policy_path=str(checkpoint_path),
                    observation_view_name=args.observation_view_name,
                    episodes=int(args.episodes),
                    start_seed=int(args.start_seed),
                    deterministic=True,
                    output_path=str(output_path),
                )
                summary = _evaluate_policy_in_env(env, policy, config)
                summaries.append(summary)
        finally:
            env.close()

    ranking = sorted(
        (
            {
                "policy_path": item["policy_path"],
                "success_rate": item["success_rate"],
                "mean_reward": item["mean_reward"],
                "mean_steps": item["mean_steps"],
                "episodes": item["episodes"],
                "action_histogram": item["action_histogram"],
            }
            for item in summaries
        ),
        key=lambda item: (float(item["success_rate"]), float(item["mean_reward"])),
        reverse=True,
    )
    ranking_path = output_dir / f"student_checkpoint_ranking_{int(args.episodes)}eps.json"
    ranking_path.write_text(json.dumps(ranking, indent=2), encoding="utf-8")
    print(json.dumps(ranking, indent=2))
