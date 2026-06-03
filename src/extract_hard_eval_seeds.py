"""Extract failure, near-miss, and slow-success seeds from an eval summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract hard-case seeds from an eval summary JSON.")
    parser.add_argument("--eval-summary", required=True, help="Path to *_eval_*eps.json.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--near-miss-distance", type=float, default=0.6)
    parser.add_argument("--slow-success-steps", type=int, default=100)
    return parser.parse_args()


def _episode_record(item: dict[str, object], reason: str) -> dict[str, object]:
    return {
        "reason": reason,
        "episode_index": int(item["episode_index"]),
        "seed": int(item["seed"]),
        "steps": int(item["steps"]),
        "reward": float(item["reward"]),
        "success": bool(item.get("success", False)),
        "final_distance_xy_m": float(item.get("final_distance_xy_m", 0.0)),
        "terminated": bool(item.get("terminated", False)),
        "truncated": bool(item.get("truncated", False)),
    }


def main() -> None:
    args = _parse_args()
    payload = json.loads(Path(args.eval_summary).read_text(encoding="utf-8"))
    episodes = list(payload.get("episode_summaries", []))

    hard_cases: list[dict[str, object]] = []
    seen_seeds: set[int] = set()

    def add_case(item: dict[str, object], reason: str) -> None:
        seed = int(item["seed"])
        if seed in seen_seeds:
            return
        seen_seeds.add(seed)
        hard_cases.append(_episode_record(item, reason))

    for item in episodes:
        success = bool(item.get("success", False))
        steps = int(item.get("steps", 0))
        final_distance = float(item.get("final_distance_xy_m", 0.0))
        if not success:
            add_case(item, "failure")
        elif final_distance >= float(args.near_miss_distance):
            add_case(item, "near_miss")
        elif steps >= int(args.slow_success_steps):
            add_case(item, "slow_success")

    counts = {
        "failure": sum(1 for item in hard_cases if item["reason"] == "failure"),
        "near_miss": sum(1 for item in hard_cases if item["reason"] == "near_miss"),
        "slow_success": sum(1 for item in hard_cases if item["reason"] == "slow_success"),
    }
    output = {
        "eval_summary": str(args.eval_summary),
        "policy_path": payload.get("policy_path"),
        "episodes": int(payload.get("episodes", len(episodes))),
        "success_rate": float(payload.get("success_rate", 0.0)),
        "near_miss_distance": float(args.near_miss_distance),
        "slow_success_steps": int(args.slow_success_steps),
        "hard_case_count": len(hard_cases),
        "counts": counts,
        "hard_case_seeds": [int(item["seed"]) for item in hard_cases],
        "hard_cases": hard_cases,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
