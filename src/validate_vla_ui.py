import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from parameter import ENV_CONFIG, SIM_CONFIG, TRAINING_CONFIG, VLA_CONFIG


def _resolve_model_path(path_str: str | None) -> Path:
    if path_str is None:
        path = Path(TRAINING_CONFIG.model_output_path)
    else:
        path = Path(path_str)
    if path.suffix != ".zip":
        path = path.with_suffix(".zip")
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return path


def _scenario_positions() -> dict[str, np.ndarray]:
    z = float(ENV_CONFIG.ball_default_position[2])
    return {
        # Relative to fish facing +X at center, +Y is usually left in camera view.
        "left": np.array([1.8, 1.2, z], dtype=np.float32),
        "center": np.array([1.9, 0.0, z], dtype=np.float32),
        "right": np.array([1.8, -1.2, z], dtype=np.float32),
        # Behind the fish at reset, usually not visible initially.
        "hidden": np.array([-1.8, 0.0, z], dtype=np.float32),
        "hidden_left": np.array([-1.6, 1.4, z], dtype=np.float32),
        "hidden_right": np.array([-1.6, -1.4, z], dtype=np.float32),
    }


def _exploration_action(step_idx: int) -> int:
    # Simple scan pattern: left sweep -> right sweep -> short forward probe.
    phase = (step_idx // 30) % 3
    if phase == 0:
        return 1
    if phase == 1:
        return 2
    return 0


def _is_in_view_scenario(scenario_name: str) -> bool:
    return scenario_name in {"left", "center", "right"}


def _build_episode_plan(args, positions: dict[str, np.ndarray]) -> list[dict[str, object]]:
    in_view_cycle = ["left", "center", "right"]
    out_of_view_cycle = ["hidden", "hidden_left", "hidden_right"]
    compatibility_all = ["left", "right", "hidden"]

    plan: list[dict[str, object]] = []

    def add_entries(scenario_names: list[str], count: int) -> None:
        for run_idx in range(max(0, int(count))):
            scenario_name = scenario_names[run_idx % len(scenario_names)]
            group = "in_view" if _is_in_view_scenario(scenario_name) else "out_of_view"
            plan.append(
                {
                    "scenario_name": scenario_name,
                    "group": group,
                    "seed": int(args.seed + args.seed_stride * len(plan)),
                }
            )

    if args.scenario in positions:
        add_entries([args.scenario], max(1, int(args.runs_per_scenario)))
    elif args.scenario == "all":
        for scenario_name in compatibility_all:
            add_entries([scenario_name], max(1, int(args.runs_per_scenario)))
    elif args.scenario == "in_view":
        add_entries(in_view_cycle, max(1, int(args.in_view_runs)))
    elif args.scenario == "out_of_view":
        add_entries(out_of_view_cycle, max(1, int(args.out_of_view_runs)))
    elif args.scenario == "balanced":
        add_entries(in_view_cycle, max(1, int(args.in_view_runs)))
        add_entries(out_of_view_cycle, max(1, int(args.out_of_view_runs)))
    else:
        raise ValueError(f"Unsupported scenario plan: {args.scenario}")

    if not plan:
        raise ValueError("Episode plan is empty. Increase run counts.")
    return plan


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained model zip (defaults to TRAINING_CONFIG.model_output_path + .zip).",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=[
            "left",
            "center",
            "right",
            "hidden",
            "hidden_left",
            "hidden_right",
            "all",
            "in_view",
            "out_of_view",
            "balanced",
        ],
        default="balanced",
        help=(
            "Validation plan. "
            "Use 'balanced' to run both in-view and out-of-view multi-episode checks."
        ),
    )
    parser.add_argument(
        "--steps-per-scenario",
        type=int,
        default=500,
        help="Max env steps to run for each scenario.",
    )
    parser.add_argument(
        "--runs-per-scenario",
        type=int,
        default=1,
        help="Repeat count when using a single scenario or --scenario all.",
    )
    parser.add_argument(
        "--in-view-runs",
        type=int,
        default=4,
        help="Episode count for --scenario in_view or --scenario balanced.",
    )
    parser.add_argument(
        "--out-of-view-runs",
        type=int,
        default=4,
        help="Episode count for --scenario out_of_view or --scenario balanced.",
    )
    parser.add_argument(
        "--seed-stride",
        type=int,
        default=101,
        help="Seed increment between episodes to diversify random lighting and resets.",
    )
    parser.add_argument(
        "--decision-interval-steps",
        type=int,
        default=1,
        help=(
            "Hold the chosen action for N env steps before making a new decision. "
            "Use 60 for ~1 decision per second at 60Hz simulation."
        ),
    )
    parser.add_argument(
        "--forward-speed-scale",
        type=float,
        default=1.0,
        help="Scale forward speed during validation only (e.g. 0.4 for slower motion).",
    )
    parser.add_argument(
        "--turn-rate-scale",
        type=float,
        default=1.0,
        help="Scale turn rate during validation only (e.g. 0.4 for slower yaw).",
    )
    parser.add_argument(
        "--visibility-threshold",
        type=float,
        default=0.02,
        help="Heatmap max threshold to decide whether the red ball is 'seen'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used in reset.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without UI window. Default is UI mode.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Override instruction text (default uses VLA_CONFIG.instruction_text).",
    )
    return parser.parse_args()


def _load_policy(model_path: Path):
    algo = TRAINING_CONFIG.algorithm.strip().lower()
    if algo == "recurrent_ppo":
        from sb3_contrib import RecurrentPPO

        model = RecurrentPPO.load(str(model_path), device=TRAINING_CONFIG.device)
        return model, True
    if algo == "ppo":
        from stable_baselines3 import PPO

        model = PPO.load(str(model_path), device=TRAINING_CONFIG.device)
        return model, False
    raise ValueError(f"Unsupported algorithm for validation: {TRAINING_CONFIG.algorithm}")


def run_validation() -> None:
    args = _parse_args()

    from isaacsim import SimulationApp

    sim_app = SimulationApp({"headless": args.headless, "renderer": SIM_CONFIG.renderer})

    env = None
    try:
        from underwater_env import UnderwaterReachEnv

        model_path = _resolve_model_path(args.model_path)
        model, is_recurrent = _load_policy(model_path)
        instruction = args.instruction or VLA_CONFIG.instruction_text

        forward_scale = max(1e-6, float(args.forward_speed_scale))
        turn_scale = max(1e-6, float(args.turn_rate_scale))
        env_cfg = ENV_CONFIG
        if forward_scale != 1.0 or turn_scale != 1.0:
            env_cfg = replace(
                ENV_CONFIG,
                forward_speed_mps=float(ENV_CONFIG.forward_speed_mps) * forward_scale,
                turn_rate_radps=float(ENV_CONFIG.turn_rate_radps) * turn_scale,
            )

        env = UnderwaterReachEnv(env_cfg=env_cfg, instruction_text=instruction)
        positions = _scenario_positions()
        episode_plan = _build_episode_plan(args, positions)
        in_view_count = sum(1 for item in episode_plan if item["group"] == "in_view")
        out_view_count = sum(1 for item in episode_plan if item["group"] == "out_of_view")

        print(f"[Validate] model={model_path}")
        print(f"[Validate] algorithm={TRAINING_CONFIG.algorithm} recurrent={is_recurrent}")
        print(f"[Validate] instruction='{instruction}'")
        print(
            f"[Validate] scenario_plan={args.scenario} "
            f"episodes_total={len(episode_plan)} in_view={in_view_count} out_of_view={out_view_count}"
        )
        print(
            f"[Validate] fish_start={ENV_CONFIG.fish_initial_position} "
            f"semantic_update_interval={ENV_CONFIG.semantic_update_interval_steps}"
        )
        print(
            f"[Validate] speed_scale: forward={forward_scale:.3f} turn={turn_scale:.3f} | "
            f"effective_forward={env.env_cfg.forward_speed_mps:.3f}m/s "
            f"effective_turn={env.env_cfg.turn_rate_radps:.3f}rad/s"
        )

        episode_summaries: list[dict[str, object]] = []

        for episode_idx, item in enumerate(episode_plan, start=1):
            scenario_name = str(item["scenario_name"])
            group = str(item["group"])
            run_seed = int(item["seed"])
            ball_pos = positions[scenario_name]
            print(
                f"\n[Scenario] ({episode_idx}/{len(episode_plan)}) "
                f"{scenario_name} group={group} seed={run_seed} ball_position={ball_pos.tolist()}"
            )

            obs, info = env.reset(seed=run_seed, options={"ball_position": ball_pos})
            lstm_state = None
            episode_start = np.ones((1,), dtype=bool)
            seen_once = False
            first_seen_step = None
            decision_interval_steps = max(1, int(args.decision_interval_steps))
            held_action_int = 0
            held_source = "init"
            policy_steps = 0
            explore_steps = 0
            visible_steps = 0
            action_counts = np.zeros(3, dtype=np.int32)
            ended_by = "max_steps"
            final_step = int(args.steps_per_scenario)

            for step_idx in range(args.steps_per_scenario):
                heatmap = info["semantic_heatmap_14x14"]
                heatmap_peak = float(np.max(heatmap))
                visible = heatmap_peak >= args.visibility_threshold
                if visible:
                    visible_steps += 1

                should_decide = (step_idx % decision_interval_steps == 0) or step_idx == 0
                if should_decide:
                    if visible:
                        if is_recurrent:
                            action, lstm_state = model.predict(
                                obs,
                                state=lstm_state,
                                episode_start=episode_start,
                                deterministic=True,
                            )
                        else:
                            action, _ = model.predict(obs, deterministic=True)
                        source = "policy"
                    else:
                        action = _exploration_action(step_idx)
                        source = "explore"
                    held_action_int = int(np.asarray(action).item())
                    held_source = source

                action_int = held_action_int
                source = held_source
                if source == "policy":
                    policy_steps += 1
                elif source == "explore":
                    explore_steps += 1
                if 0 <= action_int < 3:
                    action_counts[action_int] += 1
                obs, reward, terminated, truncated, info = env.step(action_int)

                if visible and not seen_once:
                    seen_once = True
                    first_seen_step = step_idx + 1
                    print(
                        f"[Scenario][{scenario_name}] first_seen_step={step_idx + 1} "
                        f"peak={heatmap_peak:.4f} attn_x={info['attention_center_x']:.3f}"
                    )

                if (step_idx + 1) % 20 == 0 or terminated or truncated:
                    print(
                        f"[Scenario][{scenario_name}] step={step_idx + 1} "
                        f"source={source} action={action_int} "
                        f"visible={int(visible)} peak={heatmap_peak:.4f} "
                        f"distance={info['distance_xy_m']:.3f} reward={reward:.3f}"
                    )

                episode_start[:] = bool(terminated or truncated)
                if terminated or truncated:
                    ended_by = "terminated" if terminated else "truncated"
                    final_step = step_idx + 1
                    print(f"[Scenario][{scenario_name}] ended_by={ended_by} step={final_step}")
                    break
            else:
                print(f"[Scenario][{scenario_name}] max steps reached without episode end.")

            visible_ratio = visible_steps / float(max(1, final_step))
            print(
                f"[Scenario][{scenario_name}] summary "
                f"steps={final_step} seen_once={int(seen_once)} first_seen={first_seen_step} "
                f"policy_steps={policy_steps} explore_steps={explore_steps} "
                f"visible_ratio={visible_ratio:.3f} actions=[F:{int(action_counts[0])},L:{int(action_counts[1])},R:{int(action_counts[2])}]"
            )
            episode_summaries.append(
                {
                    "group": group,
                    "scenario_name": scenario_name,
                    "seen_once": bool(seen_once),
                    "first_seen_step": first_seen_step,
                    "steps": final_step,
                    "policy_steps": policy_steps,
                    "explore_steps": explore_steps,
                    "visible_ratio": visible_ratio,
                    "ended_by": ended_by,
                }
            )

        if episode_summaries:
            print("\n[Summary] Per-group aggregate:")
            for group in ("in_view", "out_of_view"):
                group_items = [item for item in episode_summaries if item["group"] == group]
                if not group_items:
                    continue
                seen_rate = float(sum(1 for item in group_items if item["seen_once"])) / float(len(group_items))
                avg_visible_ratio = float(np.mean([item["visible_ratio"] for item in group_items]))
                avg_explore_ratio = float(
                    np.mean(
                        [item["explore_steps"] / float(max(1, int(item["steps"]))) for item in group_items]
                    )
                )
                first_seen_values = [
                    int(item["first_seen_step"])
                    for item in group_items
                    if item["first_seen_step"] is not None
                ]
                avg_first_seen = float(np.mean(first_seen_values)) if first_seen_values else float("nan")
                print(
                    f"[Summary][{group}] episodes={len(group_items)} "
                    f"seen_rate={seen_rate:.3f} avg_visible_ratio={avg_visible_ratio:.3f} "
                    f"avg_explore_ratio={avg_explore_ratio:.3f} avg_first_seen={avg_first_seen:.1f}"
                )

    finally:
        if env is not None:
            env.close()
        sim_app.close()


if __name__ == "__main__":
    run_validation()
