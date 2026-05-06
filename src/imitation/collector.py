"""Rollout collection helpers for teacher seeding and DAgger aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from imitation.config import CollectionConfig
from imitation.constants import ACTION_ID_TO_NAME
from imitation.dataset import action_histogram, write_rollout_dataset
from imitation.policies import PolicyAdapter, SB3PolicyAdapter
from imitation.runtime import SimulationAppSession, build_underwater_env


def _scenario_positions(env) -> dict[str, np.ndarray]:
    z = float(env.env_cfg.ball_default_position[2])
    return {
        # Relative to fish facing +X at reset.
        "left": np.array([1.8, 1.2, z], dtype=np.float32),
        "center": np.array([1.9, 0.0, z], dtype=np.float32),
        "right": np.array([1.8, -1.2, z], dtype=np.float32),
        # Behind the fish at reset.
        "hidden": np.array([-1.8, 0.0, z], dtype=np.float32),
        "hidden_left": np.array([-1.6, 1.4, z], dtype=np.float32),
        "hidden_right": np.array([-1.6, -1.4, z], dtype=np.float32),
    }


def _build_collection_plan(config: CollectionConfig, env) -> list[dict[str, object]]:
    plan_mode = str(config.scenario_plan).strip().lower()
    if plan_mode == "random":
        return [
            {
                "seed": int(config.start_seed) + int(episode_index) * int(config.seed_stride),
                "scenario_name": "random",
                "reset_options": None,
            }
            for episode_index in range(int(config.episodes))
        ]

    positions = _scenario_positions(env)
    in_view_cycle = ["left", "center", "right"]
    out_of_view_cycle = ["hidden", "hidden_left", "hidden_right"]

    plan: list[dict[str, object]] = []

    def add_entries(scenario_names: list[str], count: int) -> None:
        for run_index in range(max(0, int(count))):
            scenario_name = scenario_names[run_index % len(scenario_names)]
            plan.append(
                {
                    "seed": int(config.start_seed) + len(plan) * int(config.seed_stride),
                    "scenario_name": scenario_name,
                    "reset_options": {
                        "ball_position": positions[scenario_name].copy(),
                    },
                }
            )

    if plan_mode == "balanced":
        add_entries(in_view_cycle, int(config.in_view_runs))
        add_entries(out_of_view_cycle, int(config.out_of_view_runs))
    elif plan_mode == "in_view":
        add_entries(in_view_cycle, int(config.in_view_runs or config.episodes))
    elif plan_mode == "out_of_view":
        add_entries(out_of_view_cycle, int(config.out_of_view_runs or config.episodes))
    elif plan_mode == "custom":
        if not config.scenario_names:
            raise ValueError("CollectionConfig.scenario_names must be set when scenario_plan='custom'.")
        unknown_names = sorted(set(config.scenario_names) - set(positions))
        if unknown_names:
            raise ValueError(f"Unknown custom scenario names: {unknown_names}")
        add_entries(list(config.scenario_names), int(config.runs_per_scenario))
    else:
        raise ValueError(
            f"Unsupported collection scenario_plan: {config.scenario_plan}. "
            "Use 'random', 'balanced', 'in_view', 'out_of_view', or 'custom'."
        )

    if not plan:
        raise ValueError("Collection episode plan is empty. Increase run counts or use scenario_plan='random'.")
    return plan


def _build_rollout_record(
    *,
    episode_index: int,
    step_index: int,
    iteration_index: int,
    seed: int,
    scenario_name: str,
    reward: float,
    terminated: bool,
    truncated: bool,
    before_info: dict[str, object],
    after_info: dict[str, object],
    teacher_view: dict[str, object],
    student_view: dict[str, object],
    teacher_action: int,
    student_action: int,
    executed_action: int,
) -> dict[str, object]:
    return {
        "student_obs": np.asarray(student_view["obs"], dtype=np.float32),
        "teacher_obs": np.asarray(teacher_view["obs"], dtype=np.float32),
        "teacher_action": int(teacher_action),
        "student_action": int(student_action),
        "executed_action": int(executed_action),
        "episode_index": int(episode_index),
        "step_index": int(step_index),
        "iteration_index": int(iteration_index),
        "seed": int(seed),
        "scenario_name": str(scenario_name),
        "reward": float(reward),
        "distance_xy_m": float(before_info["distance_xy_m"]),
        "heading_error_rad": float(before_info["heading_error_rad"]),
        "teacher_attention_center_x": float(teacher_view["attention_center_x"]),
        "student_attention_center_x": float(student_view["attention_center_x"]),
        "teacher_peak": float(teacher_view["peak"]),
        "student_peak": float(student_view["peak"]),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "is_success": bool(after_info.get("is_success", False)),
    }


def collect_labeled_episodes(
    env,
    *,
    config: CollectionConfig | None = None,
    teacher_policy: PolicyAdapter,
    output_path: str,
    teacher_view_name: str,
    student_view_name: str,
    episodes: int,
    start_seed: int,
    deterministic_teacher: bool,
    student_policy: PolicyAdapter | None = None,
    beta: float = 1.0,
    iteration_index: int = 0,
) -> dict[str, object]:
    collection_config = config or CollectionConfig(
        output_path=output_path,
        episodes=int(episodes),
        start_seed=int(start_seed),
        deterministic_teacher=bool(deterministic_teacher),
        teacher_view_name=teacher_view_name,
        student_view_name=student_view_name,
    )
    records: list[dict[str, object]] = []
    episode_summaries: list[dict[str, object]] = []
    executed_actions: list[int] = []
    collection_plan = _build_collection_plan(collection_config, env)
    rng = np.random.default_rng(int(start_seed) + int(iteration_index) * 10_000)

    print(
        f"[Collect] start episodes={len(collection_plan)} "
        f"teacher_view={teacher_view_name} student_view={student_view_name} "
        f"iteration={int(iteration_index)} beta={float(beta):.3f} "
        f"scenario_plan={collection_config.scenario_plan}",
        flush=True,
    )

    for episode_index, episode_plan in enumerate(collection_plan):
        episode_seed = int(episode_plan["seed"])
        scenario_name = str(episode_plan["scenario_name"])
        teacher_policy.reset()
        if student_policy is not None:
            student_policy.reset()
        _, info = env.reset(seed=episode_seed, options=episode_plan["reset_options"])
        episode_reward = 0.0
        step_count = 0

        while True:
            views_payload = env.get_observation_views(
                force_refresh=False,
                include_fallback=(teacher_view_name == "fallback" or student_view_name == "fallback"),
                include_clip=(teacher_view_name == "clip" or student_view_name == "clip"),
                include_hybrid=(teacher_view_name == "hybrid" or student_view_name == "hybrid"),
                strict_clip=(teacher_view_name in {"clip", "hybrid"} or student_view_name in {"clip", "hybrid"}),
            )
            teacher_view = views_payload["views"][teacher_view_name]
            student_view = views_payload["views"][student_view_name]

            teacher_action = int(teacher_policy.predict(np.asarray(teacher_view["obs"], dtype=np.float32)))
            if student_policy is None:
                student_action = teacher_action
            else:
                student_action = int(student_policy.predict(np.asarray(student_view["obs"], dtype=np.float32)))

            if student_policy is None:
                executed_action = teacher_action
            else:
                executed_action = teacher_action if rng.random() < float(beta) else student_action

            before_info = info
            _, reward, terminated, truncated, info = env.step(executed_action)
            records.append(
                _build_rollout_record(
                    episode_index=episode_index,
                    step_index=step_count,
                    iteration_index=iteration_index,
                    seed=episode_seed,
                    scenario_name=scenario_name,
                    reward=float(reward),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    before_info=before_info,
                    after_info=info,
                    teacher_view=teacher_view,
                    student_view=student_view,
                    teacher_action=teacher_action,
                    student_action=student_action,
                    executed_action=executed_action,
                )
            )
            executed_actions.append(int(executed_action))
            episode_reward += float(reward)
            step_count += 1

            if terminated or truncated:
                episode_success = bool(info.get("is_success", False))
                episode_summaries.append(
                    {
                        "episode_index": int(episode_index),
                        "seed": int(episode_seed),
                        "scenario_name": scenario_name,
                        "steps": int(step_count),
                        "episode_reward": float(episode_reward),
                        "success": episode_success,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "final_distance_xy_m": float(info["distance_xy_m"]),
                    }
                )
                success_count = sum(1 for item in episode_summaries if item["success"])
                print(
                    f"[Collect] episode={episode_index + 1}/{len(collection_plan)} "
                    f"seed={episode_seed} scenario={scenario_name} steps={step_count} "
                    f"reward={episode_reward:.3f} success={int(episode_success)} "
                    f"records={len(records)} success_rate_so_far={success_count / float(len(episode_summaries)):.3f}",
                    flush=True,
                )
                break

    summary = {
        "episodes": int(len(collection_plan)),
        "records": int(len(records)),
        "teacher_view_name": teacher_view_name,
        "student_view_name": student_view_name,
        "deterministic_teacher": bool(deterministic_teacher),
        "beta": float(beta),
        "iteration_index": int(iteration_index),
        "scenario_plan": str(collection_config.scenario_plan),
        "seed_stride": int(collection_config.seed_stride),
        "action_histogram": action_histogram(np.asarray(executed_actions, dtype=np.int64)),
        "episode_summaries": episode_summaries,
    }
    metadata = {
        "summary": summary,
        "action_names": ACTION_ID_TO_NAME,
    }
    dataset_path = write_rollout_dataset(output_path, records, metadata)
    print(
        f"[Collect] done output={dataset_path} episodes={len(collection_plan)} records={len(records)}",
        flush=True,
    )
    return {
        "output_path": str(dataset_path),
        **summary,
    }


def collect_teacher_seed_dataset(config: CollectionConfig) -> dict[str, object]:
    teacher_policy = SB3PolicyAdapter(
        config.teacher_model_path,
        algorithm=config.teacher_algorithm,
        deterministic=config.deterministic_teacher,
    )
    result: dict[str, object] | None = None
    with SimulationAppSession(config.env):
        env = build_underwater_env(config.env)
        try:
            result = collect_labeled_episodes(
                env,
                config=config,
                teacher_policy=teacher_policy,
                output_path=config.output_path,
                teacher_view_name=config.teacher_view_name,
                student_view_name=config.student_view_name,
                episodes=config.episodes,
                start_seed=config.start_seed,
                deterministic_teacher=config.deterministic_teacher,
                student_policy=None,
                beta=1.0,
                iteration_index=0,
            )
        finally:
            env.close()
        summary_path = Path(config.output_path).with_suffix(".summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[Collect] summary_saved={summary_path}", flush=True)
    if result is None:
        raise RuntimeError("Collection finished without producing a result summary.")
    return result
