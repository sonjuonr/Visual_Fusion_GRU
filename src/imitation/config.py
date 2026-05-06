"""Dataclass-based config loading for imitation-learning scripts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

from parameter import ENV_CONFIG, SIM_CONFIG, TRAINING_CONFIG, VLA_CONFIG


T = TypeVar("T")


@dataclass(frozen=True)
class EnvRuntimeConfig:
    instruction_text: str = VLA_CONFIG.instruction_text
    semantic_backend_mode: str = "clip"
    fusion_alpha: float = 1.0
    clip_required: bool = True
    clip_local_files_only: bool = True
    semantic_update_interval_steps: int | None = ENV_CONFIG.semantic_update_interval_steps
    action_repeat_steps: int | None = ENV_CONFIG.action_repeat_steps
    renderer: str = SIM_CONFIG.renderer
    headless: bool = SIM_CONFIG.headless


@dataclass(frozen=True)
class PPOConfig:
    algorithm: str = TRAINING_CONFIG.algorithm
    learning_rate: float = TRAINING_CONFIG.learning_rate
    n_steps: int = TRAINING_CONFIG.n_steps
    batch_size: int = TRAINING_CONFIG.batch_size
    n_epochs: int = TRAINING_CONFIG.n_epochs
    gamma: float = TRAINING_CONFIG.gamma
    gae_lambda: float = TRAINING_CONFIG.gae_lambda
    clip_range: float = TRAINING_CONFIG.clip_range
    ent_coef: float = TRAINING_CONFIG.ent_coef
    vf_coef: float = TRAINING_CONFIG.vf_coef
    max_grad_norm: float = TRAINING_CONFIG.max_grad_norm
    device: str = TRAINING_CONFIG.device
    verbose: int = TRAINING_CONFIG.verbose
    seed: int = TRAINING_CONFIG.seed


@dataclass(frozen=True)
class TeacherTrainConfig:
    env: EnvRuntimeConfig = field(
        default_factory=lambda: EnvRuntimeConfig(
            semantic_backend_mode="fallback",
            fusion_alpha=0.0,
            clip_required=False,
        )
    )
    ppo: PPOConfig = field(default_factory=PPOConfig)
    total_timesteps: int = TRAINING_CONFIG.total_timesteps
    model_output_path: str = "models/fallback_teacher"
    monitor_log_path: str = "monitor_logs/fallback_teacher.monitor.csv"
    tensorboard_log_dir: str = "tensorboard"
    tb_run_name: str = "fish_imitation_fallback_teacher"
    checkpoint_dir: str = "checkpoints"
    checkpoint_freq_steps: int = TRAINING_CONFIG.checkpoint_freq_steps
    console_log_freq_steps: int = TRAINING_CONFIG.console_log_freq_steps
    progress_bar: bool = TRAINING_CONFIG.progress_bar
    resume_from: str | None = None


@dataclass(frozen=True)
class CollectionConfig:
    env: EnvRuntimeConfig = field(
        default_factory=lambda: EnvRuntimeConfig(
            semantic_backend_mode="fallback",
            fusion_alpha=0.0,
            clip_required=True,
        )
    )
    teacher_model_path: str = "models/fallback_teacher.zip"
    teacher_algorithm: str = TRAINING_CONFIG.algorithm
    output_path: str = "imitation_data/teacher_seed/teacher_seed_dataset.npz"
    episodes: int = 100
    start_seed: int = TRAINING_CONFIG.seed
    seed_stride: int = 1
    deterministic_teacher: bool = True
    teacher_view_name: str = "fallback"
    student_view_name: str = "clip"
    scenario_plan: str = "random"
    scenario_names: list[str] = field(default_factory=list)
    runs_per_scenario: int = 1
    in_view_runs: int = 0
    out_of_view_runs: int = 0


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int = VLA_CONFIG.heatmap_height * VLA_CONFIG.heatmap_width
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
    dropout: float = 0.0
    activation: str = "relu"


@dataclass(frozen=True)
class StudentBCConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset_paths: list[str] = field(default_factory=list)
    output_path: str = "models/student_bc.pt"
    resume_from: str | None = None
    batch_size: int = 512
    epochs: int = 25
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_ratio: float = 0.1
    num_workers: int = 0
    shuffle_seed: int = TRAINING_CONFIG.seed
    class_weighting: str = "balanced"
    device: str = "cpu"
    tensorboard_log_dir: str = "tensorboard"
    tb_run_name: str = "fish_imitation_student_bc"
    checkpoint_dir: str = "checkpoints/imitation"
    checkpoint_every_epochs: int = 5


@dataclass(frozen=True)
class DaggerConfig:
    env: EnvRuntimeConfig = field(
        default_factory=lambda: EnvRuntimeConfig(
            semantic_backend_mode="clip",
            fusion_alpha=1.0,
            clip_required=True,
        )
    )
    bc: StudentBCConfig = field(default_factory=StudentBCConfig)
    teacher_model_path: str = "models/fallback_teacher.zip"
    teacher_algorithm: str = TRAINING_CONFIG.algorithm
    seed_dataset_paths: list[str] = field(default_factory=list)
    initial_student_checkpoint: str | None = None
    output_root: str = "imitation_runs/dagger"
    total_iterations: int = 5
    rollout_episodes_per_iteration: int = 25
    start_seed: int = TRAINING_CONFIG.seed
    deterministic_teacher: bool = True
    deterministic_student: bool = True
    beta_start: float = 1.0
    beta_end: float = 0.0
    resume_student_from_previous: bool = True
    evaluate_episodes: int = 10


@dataclass(frozen=True)
class EvaluationConfig:
    env: EnvRuntimeConfig = field(default_factory=EnvRuntimeConfig)
    policy_type: str = "torch_student"
    policy_path: str = "models/student_bc.pt"
    policy_algorithm: str = TRAINING_CONFIG.algorithm
    observation_view_name: str = "clip"
    episodes: int = 20
    start_seed: int = TRAINING_CONFIG.seed
    deterministic: bool = True
    output_path: str | None = "imitation_data/eval/eval_summary.json"


def _coerce_value(type_hint: Any, value: Any) -> Any:
    origin = get_origin(type_hint)
    args = get_args(type_hint)

    if type_hint is Any or value is None:
        return value

    if origin is Union:
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            return _coerce_value(non_none_args[0], value)

    if is_dataclass(type_hint):
        return dataclass_from_dict(type_hint, value)

    if origin is list:
        inner = args[0] if args else Any
        return [_coerce_value(inner, item) for item in value]

    if origin is tuple:
        inner = args[0] if args else Any
        return tuple(_coerce_value(inner, item) for item in value)

    return value


def dataclass_from_dict(cls: type[T], payload: dict[str, Any]) -> T:
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass type")

    type_hints = get_type_hints(cls)
    known_fields = {item.name for item in fields(cls)}
    unknown_keys = sorted(set(payload) - known_fields)
    if unknown_keys:
        raise KeyError(f"Unknown keys for {cls.__name__}: {unknown_keys}")

    kwargs: dict[str, Any] = {}
    for item in fields(cls):
        if item.name not in payload:
            continue
        kwargs[item.name] = _coerce_value(type_hints.get(item.name, Any), payload[item.name])
    return cls(**kwargs)


def load_json_config(path: str | Path, cls: type[T]) -> T:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return dataclass_from_dict(cls, payload)


def save_json_config(path: str | Path, config: Any) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(config) if is_dataclass(config) else config
    path_obj.write_text(json.dumps(payload, indent=2), encoding="utf-8")
