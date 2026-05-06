"""Evaluation helpers for imitation policies."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from imitation.config import EvaluationConfig
from imitation.dataset import action_histogram
from imitation.policies import SB3PolicyAdapter, TorchStudentPolicyAdapter
from imitation.runtime import SimulationAppSession, build_underwater_env


def _load_policy_adapter(config: EvaluationConfig):
    policy_type = str(config.policy_type).strip().lower()
    if policy_type == "teacher_sb3":
        return SB3PolicyAdapter(
            config.policy_path,
            algorithm=config.policy_algorithm,
            deterministic=config.deterministic,
        )
    if policy_type == "torch_student":
        return TorchStudentPolicyAdapter(
            config.policy_path,
            deterministic=config.deterministic,
        )
    raise ValueError(f"Unsupported evaluation policy type: {config.policy_type}")


def _evaluate_policy_in_env(env, policy, config: EvaluationConfig) -> dict[str, object]:
    episode_summaries: list[dict[str, object]] = []
    executed_actions: list[int] = []
    print(
        f"[Eval] start policy_type={config.policy_type} "
        f"policy_path={config.policy_path} episodes={int(config.episodes)} "
        f"obs_view={config.observation_view_name}",
        flush=True,
    )
    for episode_index in range(int(config.episodes)):
        seed = int(config.start_seed) + int(episode_index)
        policy.reset()
        _, info = env.reset(seed=seed)
        total_reward = 0.0
        steps = 0

        while True:
            views_payload = env.get_observation_views(
                force_refresh=False,
                include_fallback=(config.observation_view_name == "fallback"),
                include_clip=(config.observation_view_name == "clip"),
                include_hybrid=(config.observation_view_name == "hybrid"),
                strict_clip=(config.observation_view_name in {"clip", "hybrid"}),
            )
            obs = np.asarray(views_payload["views"][config.observation_view_name]["obs"], dtype=np.float32)
            action = int(policy.predict(obs))
            executed_actions.append(action)
            _, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            if terminated or truncated:
                episode_success = bool(info.get("is_success", False))
                episode_summaries.append(
                    {
                        "episode_index": int(episode_index),
                        "seed": int(seed),
                        "steps": int(steps),
                        "reward": float(total_reward),
                        "success": episode_success,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "final_distance_xy_m": float(info["distance_xy_m"]),
                    }
                )
                success_count = sum(1 for item in episode_summaries if item["success"])
                print(
                    f"[Eval] episode={episode_index + 1}/{int(config.episodes)} "
                    f"seed={seed} steps={steps} reward={total_reward:.3f} "
                    f"success={int(episode_success)} "
                    f"success_rate_so_far={success_count / float(len(episode_summaries)):.3f}",
                    flush=True,
                )
                break

    success_rate = (
        float(np.mean([1.0 if item["success"] else 0.0 for item in episode_summaries]))
        if episode_summaries
        else 0.0
    )
    mean_reward = float(np.mean([item["reward"] for item in episode_summaries])) if episode_summaries else 0.0
    mean_steps = float(np.mean([item["steps"] for item in episode_summaries])) if episode_summaries else 0.0
    summary = {
        "policy_type": config.policy_type,
        "policy_path": config.policy_path,
        "observation_view_name": config.observation_view_name,
        "episodes": int(config.episodes),
        "success_rate": success_rate,
        "mean_reward": mean_reward,
        "mean_steps": mean_steps,
        "action_histogram": action_histogram(np.asarray(executed_actions, dtype=np.int64)),
        "episode_summaries": episode_summaries,
    }

    if config.output_path:
        output_path = Path(config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[Eval] summary_saved={output_path}", flush=True)
    return summary


def evaluate_policy(config: EvaluationConfig) -> dict[str, object]:
    policy = _load_policy_adapter(config)
    summary: dict[str, object] | None = None

    with SimulationAppSession(config.env):
        env = build_underwater_env(config.env)
        try:
            summary = _evaluate_policy_in_env(env, policy, config)
        finally:
            env.close()

    if summary is None:
        raise RuntimeError("Evaluation finished without producing a summary.")
    return summary
