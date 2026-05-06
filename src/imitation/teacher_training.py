"""Fallback-teacher training using the existing RL environment."""

from __future__ import annotations

from pathlib import Path

from imitation.config import TeacherTrainConfig, save_json_config
from imitation.runtime import SimulationAppSession, build_underwater_env, ensure_parent_dir, resolve_zip_path


def _build_model(config: TeacherTrainConfig, env):
    algorithm_name = config.ppo.algorithm.strip().lower()
    if algorithm_name == "recurrent_ppo":
        from sb3_contrib import RecurrentPPO

        model_cls = RecurrentPPO
        policy_name = "MlpLstmPolicy"
    elif algorithm_name == "ppo":
        from stable_baselines3 import PPO

        model_cls = PPO
        policy_name = "MlpPolicy"
    else:
        raise ValueError(f"Unsupported teacher algorithm: {config.ppo.algorithm}")

    resume_path = None
    if config.resume_from:
        resume_path = resolve_zip_path(config.resume_from)
        model = model_cls.load(str(resume_path), env=env, device=config.ppo.device)
        model.set_env(env)
        model.learning_rate = float(config.ppo.learning_rate)
        model.lr_schedule = lambda _: float(config.ppo.learning_rate)
        if hasattr(model, "policy") and hasattr(model.policy, "optimizer"):
            for group in model.policy.optimizer.param_groups:
                group["lr"] = float(config.ppo.learning_rate)
    else:
        model = model_cls(
            policy_name,
            env,
            verbose=int(config.ppo.verbose),
            learning_rate=float(config.ppo.learning_rate),
            n_steps=int(config.ppo.n_steps),
            batch_size=int(config.ppo.batch_size),
            n_epochs=int(config.ppo.n_epochs),
            gamma=float(config.ppo.gamma),
            gae_lambda=float(config.ppo.gae_lambda),
            clip_range=float(config.ppo.clip_range),
            ent_coef=float(config.ppo.ent_coef),
            vf_coef=float(config.ppo.vf_coef),
            max_grad_norm=float(config.ppo.max_grad_norm),
            tensorboard_log=str(config.tensorboard_log_dir),
            device=str(config.ppo.device),
            seed=int(config.ppo.seed),
        )
    return model, policy_name, resume_path


def train_fallback_teacher(config: TeacherTrainConfig) -> dict[str, object]:
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor

    from training_callbacks import ConsoleProgressCallback, HyperparameterTensorboardCallback, InfoTensorboardCallback

    output_path = ensure_parent_dir(config.model_output_path)
    ensure_parent_dir(config.monitor_log_path)
    Path(config.tensorboard_log_dir).mkdir(parents=True, exist_ok=True)
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    with SimulationAppSession(config.env):
        env = None
        raw_env = None
        try:
            raw_env = build_underwater_env(config.env)
            env = Monitor(raw_env, filename=str(config.monitor_log_path), info_keywords=("is_success",))
            model, policy_name, resume_path = _build_model(config, env)

            callback = CallbackList(
                [
                    HyperparameterTensorboardCallback(
                        {
                            "teacher_algorithm": config.ppo.algorithm,
                            "learning_rate": config.ppo.learning_rate,
                            "semantic_backend_mode": config.env.semantic_backend_mode,
                            "instruction_text": config.env.instruction_text,
                        }
                    ),
                    InfoTensorboardCallback(
                        info_keys=[
                            "distance_xy_m",
                            "heading_error_rad",
                            "attention_center_x",
                            "reward_total",
                            "target_visible",
                            "heatmap_peak",
                            "is_success",
                        ]
                    ),
                    CheckpointCallback(
                        save_freq=int(config.checkpoint_freq_steps),
                        save_path=str(config.checkpoint_dir),
                        name_prefix="fish_imitation_fallback_teacher",
                    ),
                    ConsoleProgressCallback(print_freq_steps=int(config.console_log_freq_steps)),
                ]
            )

            model.learn(
                total_timesteps=int(config.total_timesteps),
                progress_bar=bool(config.progress_bar),
                callback=callback,
                tb_log_name=str(config.tb_run_name),
                reset_num_timesteps=(resume_path is None),
            )
            model.save(str(output_path))
        finally:
            if env is not None:
                env.close()
            if raw_env is not None:
                raw_env.close()

    config_path = output_path.parent / "fallback_teacher_run_config.json"
    save_json_config(config_path, config)
    return {
        "model_output_path": str(resolve_zip_path(output_path)),
        "policy_name": policy_name,
        "monitor_log_path": str(config.monitor_log_path),
        "tensorboard_log_dir": str(config.tensorboard_log_dir),
        "config_path": str(config_path),
    }
