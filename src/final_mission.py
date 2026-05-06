import argparse
import json
import os
import traceback
from dataclasses import asdict
from pathlib import Path

from parameter import ENV_CONFIG, SIM_CONFIG, TRAINING_CONFIG, VLA_CONFIG


def _zip_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.suffix.lower() != ".zip":
        path = path.with_suffix(".zip")
    return path


def _ensure_training_dirs(model_output_path: str) -> None:
    Path(TRAINING_CONFIG.tensorboard_log_dir).mkdir(parents=True, exist_ok=True)
    Path(TRAINING_CONFIG.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(TRAINING_CONFIG.monitor_log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)


def _dump_run_config(instruction_text: str, total_timesteps: int, model_output_path: str) -> None:
    config_path = Path(model_output_path).parent / "run_config.json"
    payload = {
        "instruction_text": instruction_text,
        "total_timesteps": total_timesteps,
        "sim_config": asdict(SIM_CONFIG),
        "env_config": asdict(ENV_CONFIG),
        "vla_config": asdict(VLA_CONFIG),
        "training_config": asdict(TRAINING_CONFIG),
    }
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_simulation_app_config(renderer: str) -> dict[str, object]:
    config: dict[str, object] = {"headless": SIM_CONFIG.headless, "renderer": renderer}
    # Work around RTX startup crashes observed on Windows + Vulkan by forcing D3D12.
    if os.name == "nt":
        config["extra_args"] = ["--/app/vulkan=false"]
    return config


def run_training(
    smoke_steps: int = 0,
    timesteps: int | None = None,
    instruction_text: str | None = None,
    resume_from: str | None = None,
    model_output_path: str | None = None,
) -> None:
    from isaacsim import SimulationApp

    # No-UI mode required by the task.
    simulation_app = SimulationApp(_build_simulation_app_config(SIM_CONFIG.renderer))

    env = None
    raw_env = None
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
        from stable_baselines3.common.monitor import Monitor

        from training_callbacks import (
            ConsoleProgressCallback,
            HyperparameterTensorboardCallback,
            InfoTensorboardCallback,
        )
        from underwater_env import UnderwaterReachEnv

        output_path = TRAINING_CONFIG.model_output_path if model_output_path is None else model_output_path
        _ensure_training_dirs(model_output_path=output_path)
        instruction = instruction_text or VLA_CONFIG.instruction_text

        raw_env = UnderwaterReachEnv(instruction_text=instruction)
        env = Monitor(raw_env, filename=TRAINING_CONFIG.monitor_log_path)

        if smoke_steps > 0:
            obs, info = env.reset(seed=TRAINING_CONFIG.seed)
            print(
                f"[Smoke] obs_dim={obs.shape[0]}, obs_sum={float(obs.sum()):.4f}, "
                f"distance={info['distance_xy_m']:.3f}m, backend={info['semantic_backend']}"
            )
            for i in range(smoke_steps):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                print(
                    f"[Smoke] step={i + 1} action={action} reward={reward:.4f} "
                    f"distance={info['distance_xy_m']:.3f}m heading={info['heading_error_rad']:.3f} "
                    f"attn_x={info['attention_center_x']:.3f}"
                )
                if terminated or truncated:
                    obs, info = env.reset(seed=TRAINING_CONFIG.seed)
            print("[Smoke] Environment stepping finished successfully.")
            return

        total_timesteps = TRAINING_CONFIG.total_timesteps if timesteps is None else timesteps

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
            model = model_cls.load(str(resume_zip_path), env=env, device=TRAINING_CONFIG.device)
            model.set_env(env)
            print(f"[Train] Resuming from: {resume_zip_path}")
        else:
            model = model_cls(
                policy_name,
                env,
                verbose=TRAINING_CONFIG.verbose,
                learning_rate=TRAINING_CONFIG.learning_rate,
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
            name_prefix="fish_vla",
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
            ]
        )
        hparam_callback = HyperparameterTensorboardCallback(
            hparams={
                "policy": TRAINING_CONFIG.policy,
                "effective_policy": policy_name,
                "algorithm": TRAINING_CONFIG.algorithm,
                "learning_rate": TRAINING_CONFIG.learning_rate,
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
                "camera_width": ENV_CONFIG.camera_resolution[0],
                "camera_height": ENV_CONFIG.camera_resolution[1],
                "action_repeat_steps": ENV_CONFIG.action_repeat_steps,
                "forward_speed_mps": ENV_CONFIG.forward_speed_mps,
                "turn_rate_radps": ENV_CONFIG.turn_rate_radps,
                "curriculum_front_spawn_only": ENV_CONFIG.curriculum_front_spawn_only,
                "front_spawn_probability": ENV_CONFIG.front_spawn_probability,
                "search_visibility_peak_threshold": ENV_CONFIG.search_visibility_peak_threshold,
                "search_turn_reward": ENV_CONFIG.search_turn_reward,
                "search_forward_penalty": ENV_CONFIG.search_forward_penalty,
                "heatmap_h": VLA_CONFIG.heatmap_height,
                "heatmap_w": VLA_CONFIG.heatmap_width,
                "semantic_model": VLA_CONFIG.clip_model_name,
                "wall_x_min": ENV_CONFIG.wall_x_min,
                "wall_x_max": ENV_CONFIG.wall_x_max,
                "wall_y_min": ENV_CONFIG.wall_y_min,
                "wall_y_max": ENV_CONFIG.wall_y_max,
                "wall_collision_margin_m": ENV_CONFIG.wall_collision_margin_m,
                "wall_collision_penalty": ENV_CONFIG.wall_collision_penalty,
                "terminate_on_wall_collision": ENV_CONFIG.terminate_on_wall_collision,
            }
        )
        console_callback = ConsoleProgressCallback(print_freq_steps=TRAINING_CONFIG.console_log_freq_steps)
        callback = CallbackList([hparam_callback, custom_info_callback, checkpoint_callback, console_callback])

        model.learn(
            total_timesteps=total_timesteps,
            progress_bar=TRAINING_CONFIG.progress_bar,
            callback=callback,
            tb_log_name=TRAINING_CONFIG.tb_run_name,
            reset_num_timesteps=(resume_zip_path is None),
        )
        model.save(output_path)
        _dump_run_config(instruction_text=instruction, total_timesteps=total_timesteps, model_output_path=output_path)
        print(f"[Train] Done. Model saved to: {output_path}")
        print(f"[Train] TensorBoard log dir: {TRAINING_CONFIG.tensorboard_log_dir}")
        print(f"[Train] Checkpoints dir: {TRAINING_CONFIG.checkpoint_dir}")
    except Exception as exc:
        print(f"[Train][Fatal] {type(exc).__name__}: {exc}")
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
        default=None,
        help="Override total PPO timesteps.",
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
        help="Resume training from an existing model zip path.",
    )
    parser.add_argument(
        "--model-output-path",
        type=str,
        default=None,
        help="Output path prefix for saving the trained model (without .zip or with .zip).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_training(
        smoke_steps=args.smoke_steps,
        timesteps=args.timesteps,
        instruction_text=args.instruction,
        resume_from=args.resume_from,
        model_output_path=args.model_output_path,
    )
