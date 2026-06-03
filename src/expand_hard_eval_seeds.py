"""Expand hard-case seeds with nearby neighbor seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand hard-case seeds with seed +/- radius.")
    parser.add_argument("--input", required=True, help="Input hard-case JSON.")
    parser.add_argument("--output", required=True, help="Output expanded hard-case JSON.")
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--min-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    raw_seeds = payload.get("hard_case_seeds") or payload.get("failed_seeds") or []
    base_seeds = sorted({int(seed) for seed in raw_seeds})
    if not base_seeds:
        raise ValueError(f"No hard_case_seeds/failed_seeds found in {args.input}.")

    expanded: set[int] = set()
    radius = int(args.radius)
    min_seed = int(args.min_seed)
    for seed in base_seeds:
        for offset in range(-radius, radius + 1):
            candidate = seed + offset
            if candidate >= min_seed:
                expanded.add(candidate)

    output = {
        "source": str(args.input),
        "radius": radius,
        "base_seed_count": len(base_seeds),
        "expanded_seed_count": len(expanded),
        "hard_case_seeds": sorted(expanded),
        "base_hard_case_seeds": base_seeds,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
