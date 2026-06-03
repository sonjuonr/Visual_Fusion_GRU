"""Capture a GitHub-ready camera screenshot from a student policy rollout."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from imitation.config import EnvRuntimeConfig
from imitation.policies import TorchStudentPolicyAdapter
from imitation.runtime import SimulationAppSession, build_underwater_env


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture a PNG screenshot from a student policy rollout.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="docs/assets/current_student_simulation.png")
    parser.add_argument("--seed", type=int, default=601000)
    parser.add_argument("--capture-step", type=int, default=20)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--renderer", type=str, default="HydraStorm")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _save_rgb(path: Path, rgb: np.ndarray, scale: int) -> None:
    image = np.asarray(rgb, dtype=np.uint8)
    if int(scale) > 1:
        height, width = image.shape[:2]
        image = cv2.resize(image, (width * int(scale), height * int(scale)), interpolation=cv2.INTER_CUBIC)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def main() -> None:
    args = _parse_args()
    output_path = Path(args.output)
    env_config = EnvRuntimeConfig(renderer=args.renderer, headless=bool(args.headless))
    policy = TorchStudentPolicyAdapter(str(args.checkpoint), deterministic=True)

    with SimulationAppSession(env_config):
        env = build_underwater_env(env_config)
        try:
            policy.reset()
            env.reset(seed=int(args.seed))
            capture_step = max(0, int(args.capture_step))
            frame = env.get_observation_views(
                force_refresh=True,
                include_fallback=False,
                include_clip=True,
                strict_clip=True,
            )["rgb"]

            success = False
            steps = 0
            for step_index in range(capture_step):
                views_payload = env.get_observation_views(
                    force_refresh=False,
                    include_fallback=False,
                    include_clip=True,
                    strict_clip=True,
                )
                frame = views_payload["rgb"]
                obs = np.asarray(views_payload["views"]["clip"]["obs"], dtype=np.float32)
                action = int(policy.predict(obs))
                _, _, terminated, truncated, info = env.step(action)
                steps = step_index + 1
                frame = env.get_observation_views(
                    force_refresh=True,
                    include_fallback=False,
                    include_clip=True,
                    strict_clip=True,
                )["rgb"]
                if terminated or truncated:
                    success = bool(info.get("is_success", False))
                    break

            _save_rgb(output_path, np.asarray(frame), int(args.scale))
            print(
                f"[Screenshot] saved={output_path} seed={int(args.seed)} "
                f"steps={steps} success={int(success)} scale={int(args.scale)}",
                flush=True,
            )
        finally:
            env.close()


if __name__ == "__main__":
    main()
