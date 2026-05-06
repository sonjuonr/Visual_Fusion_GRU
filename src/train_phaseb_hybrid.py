import argparse
import json
import os
import traceback
from dataclasses import asdict, replace
from pathlib import Path

from parameter import ENV_CONFIG, SIM_CONFIG, TRAINING_CONFIG, VLA_CONFIG


def _zip_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.suffix.lower() != ".zip":
        path = path.with_suffix(".zip")
    return path


def _find_default_resume_model() -> Path | None:
    preferred = Path("models/my_underwater_robot_policy_continue_170k_mixsearch.zip")
    if preferred.exists():
        return preferred
    candidates = sorted(Path("models").glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _parse_alpha_targets(alpha_targets: str | None) -> list[float] | None:
    if alpha_targets is None:
        return None
    raw = str(alpha_targets).strip()
    if not raw:
        return None
    values: list[float] = []
    for token in raw.split(","):
        t = token.strip()
        if not t:
            continue
        values.append(float(t))
    return values if values else None


def _ensure_training_dirs(model_output_path: str, monitor_log_path: str) -> None:
    Path(TRAINING_CONFIG.tensorboard_log_dir).mkdir(parents=True, exist_ok=True)
    Path(TRAINING_CONFIG.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(monitor_log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)


def _dump_phaseb_run_config(
    instruction_text: str,
    total_timesteps: int,
    model_output_path: str,
    monitor_log_path: str,
    env_cfg_phaseb,
    vla_cfg_phaseb,
    curriculum_config: dict[str, object],
) -> None:
    config_path = Path(model_output_path).parent / "run_config_phaseb_hybrid.json"
    payload = {
        "phase": "B_hybrid_fusion_curriculum",
        "instruction_text": instruction_text,
        "total_timesteps": total_timesteps,
        "sim_config": asdict(SIM_CONFIG),
        "env_config": asdict(env_cfg_phaseb),
        "vla_config_phaseb": asdict(vla_cfg_phaseb),
        "training_config": asdict(TRAINING_CONFIG),
        "monitor_log_path": monitor_log_path,
        "curriculum_config": curriculum_config,
    }
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_simulation_app_config(renderer: str) -> dict[str, object]:
    config: dict[str, object] = {"headless": SIM_CONFIG.headless, "renderer": renderer}
    # Work around RTX startup crashes observed on Windows + Vulkan by forcing D3D12.
    if os.name == "nt":
        config["extra_args"] = ["--/app/vulkan=false"]
    return config


def run_phaseb_training(
    smoke_steps: int = 0,
    timesteps: int | None = None,
    learning_rate: float | None = None,
    instruction_text: str | None = None,
    resume_from: str | None = None,
    model_output_path: str | None = None,
    monitor_log_path: str = "monitor_logs/monitor_phaseb.monitor.csv",
    tb_run_name: str = "fish_vla_phaseb",
    alpha_start: float = 0.05,
    alpha_max: float = 1.0,
    alpha_step: float = 0.1,
    alpha_targets: str | None = None,
    success_threshold: float = 0.7,
    success_window: int = 20,
    min_episodes_before_update: int = 20,
    cooldown_episodes: int = 5,
    reset_window_on_alpha_change: bool = True,
    allow_alpha_decrease: bool = True,
    success_lower_threshold: float = 0.2,
    alpha_decrease_step: float | None = None,
    min_episodes_before_decrease: int | None = None,
    clip_required: bool = True,
    clip_local_files_only: bool = True,
    semantic_update_interval_steps: int | None = None,
    action_repeat_steps: int | None = None,
    progress_bar: bool | None = None,
    renderer: str | None = None,
) -> None:
    from isaacsim import SimulationApp

    sim_renderer = SIM_CONFIG.renderer if renderer is None else str(renderer)
    simulation_app = SimulationApp(_build_simulation_app_config(sim_renderer))

    env = None
    raw_env = None
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
        from stable_baselines3.common.monitor import Monitor

        from training_callbacks import (
            ConsoleProgressCallback,
            FusionAlphaCurriculumCallback,
            HyperparameterTensorboardCallback,
            InfoTensorboardCallback,
        )
        from underwater_env import UnderwaterReachEnv

        output_path = (
            "models/my_underwater_robot_policy_phaseb_hybrid"
            if model_output_path is None
            else model_output_path
        )
        alpha_target_values = _parse_alpha_targets(alpha_targets)
        _ensure_training_dirs(model_output_path=output_path, monitor_log_path=monitor_log_path)
        instruction = instruction_text or VLA_CONFIG.instruction_text

        vla_cfg_phaseb = replace(
            VLA_CONFIG,
            semantic_backend_mode="hybrid",
            fusion_alpha=float(alpha_start),
            clip_required=bool(clip_required),
            clip_local_files_only=bool(clip_local_files_only),
        )
        env_cfg_updates = {}
        if semantic_update_interval_steps is not None:
            env_cfg_updates["semantic_update_interval_steps"] = max(1, int(semantic_update_interval_steps))
        if action_repeat_steps is not None:
            env_cfg_updates["action_repeat_steps"] = max(1, int(action_repeat_steps))
        env_cfg_phaseb = replace(ENV_CONFIG, **env_cfg_updates) if env_cfg_updates else ENV_CONFIG

        raw_env = UnderwaterReachEnv(env_cfg=env_cfg_phaseb, vla_cfg=vla_cfg_phaseb, instruction_text=instruction)
        env = Monitor(
            raw_env,
            filename=monitor_log_path,
            info_keywords=("is_success", "fusion_alpha"),
        )

        if smoke_steps > 0:
            obs, info = env.reset(seed=TRAINING_CONFIG.seed)
            print(
                f"[Smoke][PhaseB] obs_dim={obs.shape[0]}, obs_sum={float(obs.sum()):.4f}, "
                f"distance={info['distance_xy_m']:.3f}m, backend={info['semantic_backend']}, "
                f"alpha={info.get('fusion_alpha', -1.0):.3f}"
            )
            for i in range(smoke_steps):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                print(
                    f"[Smoke][PhaseB] step={i + 1} action={action} reward={reward:.4f} "
                    f"distance={info['distance_xy_m']:.3f}m heading={info['heading_error_rad']:.3f} "
                    f"attn_x={info['attention_center_x']:.3f} alpha={info.get('fusion_alpha', -1.0):.3f}"
                )
                if terminated or truncated:
                    obs, info = env.reset(seed=TRAINING_CONFIG.seed)
            print("[Smoke][PhaseB] Environment stepping finished successfully.")
            return

        total_timesteps = TRAINING_CONFIG.total_timesteps if timesteps is None else timesteps
        effective_learning_rate = TRAINING_CONFIG.learning_rate if learning_rate is None else float(learning_rate)
        effective_progress_bar = TRAINING_CONFIG.progress_bar if progress_bar is None else bool(progress_bar)

        algorithm_name = TRAINING_CONFIG.algorithm.strip().lower()
        if algorithm_name == "recurrent_ppo":
            from sb3_contrib import RecurrentPPO

            model_cls = RecurrentPPO
            policy_name = TRAINING_CONFIG.recurrent_policy
        elif algorithm_name == "ppo":
            model_cls = PPO
            policy_name = TRAINING_CONFIG.policy
        else:
            raise ValueError(f"Unsupported algorithm: {TRAINING_CONFIG.algorithm}")

        resume_zip_path = None
        if resume_from is not None:
            resume_zip_path = _zip_path(resume_from)
            if not resume_zip_path.exists():
                raise FileNotFoundError(f"Resume model not found: {resume_zip_path}")
        else:
            auto_resume = _find_default_resume_model()
            if auto_resume is not None:
                resume_zip_path = auto_resume
                print(f"[PhaseB] Auto resume from latest model: {resume_zip_path}")

        if resume_zip_path is not None:
            model = model_cls.load(str(resume_zip_path), env=env, device=TRAINING_CONFIG.device)
            model.set_env(env)
            model.learning_rate = effective_learning_rate
            model.lr_schedule = lambda _: effective_learning_rate
            if hasattr(model, "policy") and hasattr(model.policy, "optimizer"):
                for param_group in model.policy.optimizer.param_groups:
                    param_group["lr"] = effective_learning_rate
            print(f"[Train][PhaseB] Resuming from: {resume_zip_path}")
        else:
            model = model_cls(
                policy_name,
                env,
                verbose=TRAINING_CONFIG.verbose,
                learning_rate=effective_learning_rate,
                n_steps=TRAINING_CONFIG.n_steps,
                batch_size=TRAINING_CONFIG.batch_size,
                n_epochs=TRAINING_CONFIG.n_epochs,
                gamma=TRAINING_CONFIG.gamma,
                gae_lambda=TRAINING_CONFIG.gae_lambda,
                clip_range=TRAINING_CONFIG.clip_range,
                ent_coef=TRAINING_CONFIG.ent_coef,
                vf_coef=TRAINING_CONFIG.vf_coef,
                max_grad_norm=TRAINING_CONFIG.max_grad_norm,
                tensorboard_log=TRAINING_CONFIG.tensorboard_log_dir,
                device=TRAINING_CONFIG.device,
                seed=TRAINING_CONFIG.seed,
            )

        checkpoint_callback = CheckpointCallback(
            save_freq=TRAINING_CONFIG.checkpoint_freq_steps,
            save_path=TRAINING_CONFIG.checkpoint_dir,
            name_prefix="fish_vla_phaseb",
        )
        custom_info_callback = InfoTensorboardCallback(
            info_keys=[
                "distance_xy_m",
                "heading_error_rad",
                "attention_center_x",
                "wall_collision_hit",
                "wall_clearance_m",
                "reward_progress",
                "reward_heading",
                "reward_turn",
                "reward_forward_center",
                "reward_search",
                "reward_wall_collision",
                "reward_success",
                "reward_timeout",
                "reward_total",
                "target_visible",
                "heatmap_peak",
                "clip_peak",
                "fallback_peak",
                "fused_peak",
                "is_success",
                "fusion_alpha",
            ]
        )
        hparam_callback = HyperparameterTensorboardCallback(
            hparams={
                "phase": "B_hybrid_fusion_curriculum",
                "policy": TRAINING_CONFIG.policy,
                "effective_policy": policy_name,
                "algorithm": TRAINING_CONFIG.algorithm,
                "learning_rate": effective_learning_rate,
                "n_steps": TRAINING_CONFIG.n_steps,
                "batch_size": TRAINING_CONFIG.batch_size,
                "n_epochs": TRAINING_CONFIG.n_epochs,
                "gamma": TRAINING_CONFIG.gamma,
                "gae_lambda": TRAINING_CONFIG.gae_lambda,
                "clip_range": TRAINING_CONFIG.clip_range,
                "ent_coef": TRAINING_CONFIG.ent_coef,
                "vf_coef": TRAINING_CONFIG.vf_coef,
                "max_grad_norm": TRAINING_CONFIG.max_grad_norm,
                "instruction_text": instruction,
                "semantic_backend_mode": vla_cfg_phaseb.semantic_backend_mode,
                "alpha_start": alpha_start,
                "alpha_max": alpha_max,
                "alpha_step": alpha_step,
                "alpha_targets": alpha_target_values,
                "success_threshold": success_threshold,
                "success_window": success_window,
                "min_episodes_before_update": min_episodes_before_update,
                "cooldown_episodes": cooldown_episodes,
                "reset_window_on_alpha_change": reset_window_on_alpha_change,
                "allow_alpha_decrease": allow_alpha_decrease,
                "success_lower_threshold": success_lower_threshold,
                "alpha_decrease_step": alpha_decrease_step,
                "min_episodes_before_decrease": min_episodes_before_decrease,
                "clip_required": clip_required,
                "clip_local_files_only": clip_local_files_only,
                "semantic_update_interval_steps": env_cfg_phaseb.semantic_update_interval_steps,
                "action_repeat_steps": env_cfg_phaseb.action_repeat_steps,
                "sim_renderer": sim_renderer,
            }
        )
        curriculum_callback = FusionAlphaCurriculumCallback(
            alpha_start=alpha_start,
            alpha_max=alpha_max,
            alpha_step=alpha_step,
            alpha_targets=alpha_target_values,
            success_rate_threshold=success_threshold,
            success_window_episodes=success_window,
            min_episodes_before_update=min_episodes_before_update,
            cooldown_episodes=cooldown_episodes,
            reset_window_on_alpha_change=reset_window_on_alpha_change,
            allow_alpha_decrease=allow_alpha_decrease,
            success_rate_lower_threshold=success_lower_threshold,
            alpha_decrease_step=alpha_decrease_step,
            min_episodes_before_decrease=min_episodes_before_decrease,
        )
        console_callback = ConsoleProgressCallback(print_freq_steps=TRAINING_CONFIG.console_log_freq_steps)
        callback = CallbackList(
            [hparam_callback, custom_info_callback, curriculum_callback, checkpoint_callback, console_callback]
        )

        model.learn(
            total_timesteps=total_timesteps,
            progress_bar=effective_progress_bar,
            callback=callback,
            tb_log_name=tb_run_name,
            reset_num_timesteps=(resume_zip_path is None),
        )
        model.save(output_path)
        curriculum_config = {
            "alpha_start": alpha_start,
            "alpha_max": alpha_max,
            "alpha_step": alpha_step,
            "alpha_targets": alpha_target_values,
            "success_threshold": success_threshold,
            "success_window": success_window,
            "min_episodes_before_update": min_episodes_before_update,
            "cooldown_episodes": cooldown_episodes,
            "reset_window_on_alpha_change": reset_window_on_alpha_change,
            "allow_alpha_decrease": allow_alpha_decrease,
            "success_lower_threshold": success_lower_threshold,
            "alpha_decrease_step": alpha_decrease_step,
            "min_episodes_before_decrease": min_episodes_before_decrease,
        }
        _dump_phaseb_run_config(
            instruction_text=instruction,
            total_timesteps=total_timesteps,
            model_output_path=output_path,
            monitor_log_path=monitor_log_path,
            env_cfg_phaseb=env_cfg_phaseb,
            vla_cfg_phaseb=vla_cfg_phaseb,
            curriculum_config=curriculum_config,
        )
        print(f"[Train][PhaseB] Done. Model saved to: {output_path}")
        print(f"[Train][PhaseB] TensorBoard log dir: {TRAINING_CONFIG.tensorboard_log_dir}")
        print(f"[Train][PhaseB] Checkpoints dir: {TRAINING_CONFIG.checkpoint_dir}")
    except Exception as exc:
        print(f"[Train][PhaseB][Fatal] {type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        if raw_env is not None:
            raw_env.close()
        simulation_app.close()


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=0,
        help="Run environment-only smoke steps and exit (no PPO training).",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=300000,
        help="Override total PPO timesteps for Phase-B curriculum.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override PPO learning rate (also applied when resuming from checkpoint).",
    )
    parser.add_argument(
        "--progress-bar",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override SB3 progress bar. Disable to avoid rich/tqdm monitor threads.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Override VLA text instruction, e.g. 'Find the red ball'.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume training from an existing model zip path. Default: auto-detect latest model.",
    )
    parser.add_argument(
        "--model-output-path",
        type=str,
        default="models/my_underwater_robot_policy_phaseb_hybrid",
        help="Output path prefix for saving the Phase-B model.",
    )
    parser.add_argument(
        "--monitor-log-path",
        type=str,
        default="monitor_logs/monitor_phaseb.monitor.csv",
        help="Monitor csv path for this Phase-B run.",
    )
    parser.add_argument(
        "--tb-run-name",
        type=str,
        default="fish_vla_phaseb",
        help="TensorBoard run name for Phase-B training.",
    )
    parser.add_argument(
        "--alpha-start",
        type=float,
        default=0.05,
        help="Initial fusion alpha: obs = alpha*CLIP + (1-alpha)*fallback.",
    )
    parser.add_argument(
        "--alpha-max",
        type=float,
        default=1.0,
        help="Maximum alpha in curriculum.",
    )
    parser.add_argument(
        "--alpha-step",
        type=float,
        default=0.1,
        help="Alpha increment each time success-rate threshold is satisfied.",
    )
    parser.add_argument(
        "--alpha-targets",
        type=str,
        default=None,
        help=(
            "Comma-separated alpha milestones, e.g. '0.96,0.97,1.0'. "
            "When set, curriculum raises alpha to next target only."
        ),
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=0.7,
        help="Raise alpha when window success rate reaches this threshold.",
    )
    parser.add_argument(
        "--success-window",
        type=int,
        default=20,
        help="Episode window size for success-rate estimation.",
    )
    parser.add_argument(
        "--min-episodes-before-update",
        type=int,
        default=20,
        help="Minimum observed episodes before first alpha update.",
    )
    parser.add_argument(
        "--cooldown-episodes",
        type=int,
        default=5,
        help="Minimum episode gap between two alpha updates.",
    )
    parser.add_argument(
        "--reset-window-on-alpha-change",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, clear success window after each alpha update.",
    )
    parser.add_argument(
        "--allow-alpha-decrease",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, decrease alpha when success rate collapses.",
    )
    parser.add_argument(
        "--success-lower-threshold",
        type=float,
        default=0.2,
        help="Lower-threshold for triggering alpha decrease.",
    )
    parser.add_argument(
        "--alpha-decrease-step",
        type=float,
        default=None,
        help="Alpha decrement step. Default uses alpha-step.",
    )
    parser.add_argument(
        "--min-episodes-before-decrease",
        type=int,
        default=None,
        help="Minimum episodes at current alpha before allowing decrease.",
    )
    parser.add_argument(
        "--clip-required",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, fail fast when CLIP branch cannot initialize.",
    )
    parser.add_argument(
        "--clip-local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, only load CLIP from local cache.",
    )
    parser.add_argument(
        "--semantic-update-interval-steps",
        type=int,
        default=None,
        help="Override semantic refresh interval in env decision steps (higher is faster, less reactive).",
    )
    parser.add_argument(
        "--action-repeat-steps",
        type=int,
        default=None,
        help="Override action repeat steps (higher is faster, but coarser control).",
    )
    parser.add_argument(
        "--renderer",
        type=str,
        default=None,
        help="Override SimulationApp renderer (e.g. HydraStorm, RayTracedLighting).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_phaseb_training(
        smoke_steps=args.smoke_steps,
        timesteps=args.timesteps,
        learning_rate=args.learning_rate,
        instruction_text=args.instruction,
        resume_from=args.resume_from,
        model_output_path=args.model_output_path,
        monitor_log_path=args.monitor_log_path,
        tb_run_name=args.tb_run_name,
        alpha_start=args.alpha_start,
        alpha_max=args.alpha_max,
        alpha_step=args.alpha_step,
        alpha_targets=args.alpha_targets,
        success_threshold=args.success_threshold,
        success_window=args.success_window,
        min_episodes_before_update=args.min_episodes_before_update,
        cooldown_episodes=args.cooldown_episodes,
        reset_window_on_alpha_change=args.reset_window_on_alpha_change,
        allow_alpha_decrease=args.allow_alpha_decrease,
        success_lower_threshold=args.success_lower_threshold,
        alpha_decrease_step=args.alpha_decrease_step,
        min_episodes_before_decrease=args.min_episodes_before_decrease,
        clip_required=args.clip_required,
        clip_local_files_only=args.clip_local_files_only,
        semantic_update_interval_steps=args.semantic_update_interval_steps,
        action_repeat_steps=args.action_repeat_steps,
        progress_bar=args.progress_bar,
        renderer=args.renderer,
    )
