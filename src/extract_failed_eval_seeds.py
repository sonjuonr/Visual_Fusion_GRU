"""Extract failed episode seeds from an imitation evaluation summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write failed seeds from an eval summary JSON.")
    parser.add_argument("--eval-summary", required=True, help="Path to *_eval_*eps.json.")
    parser.add_argument("--output", required=True, help="Output JSON path for failed seeds.")
    parser.add_argument(
        "--max-final-distance",
        type=float,
        default=None,
        help="Optional filter for near-miss failures only.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = json.loads(Path(args.eval_summary).read_text(encoding="utf-8"))
    episodes = payload.get("episode_summaries", [])
    failed: list[dict[str, object]] = []
    for item in episodes:
        if bool(item.get("success", False)):
            continue
        final_distance = float(item.get("final_distance_xy_m", 0.0))
        if args.max_final_distance is not None and final_distance > float(args.max_final_distance):
            continue
        failed.append(
            {
                "episode_index": int(item["episode_index"]),
                "seed": int(item["seed"]),
                "steps": int(item["steps"]),
                "reward": float(item["reward"]),
                "final_distance_xy_m": final_distance,
                "terminated": bool(item.get("terminated", False)),
                "truncated": bool(item.get("truncated", False)),
            }
        )

    output = {
        "eval_summary": str(args.eval_summary),
        "policy_path": payload.get("policy_path"),
        "episodes": int(payload.get("episodes", len(episodes))),
        "success_rate": float(payload.get("success_rate", 0.0)),
        "failed_count": len(failed),
        "failed_seeds": [int(item["seed"]) for item in failed],
        "failed_episodes": failed,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
