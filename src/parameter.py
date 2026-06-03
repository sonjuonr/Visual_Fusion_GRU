"""Centralized parameters for Isaac Sim robotic fish VLA training."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_STAGE_CANDIDATES = (
    _REPO_ROOT / "final_project" / "watertank.usd",
    _REPO_ROOT.parent / "final_project" / "watertank.usd",
)


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _default_stage_path() -> str:
    override = os.getenv("FISH_TANK_USD_PATH")
    if override:
        return override
    for candidate in _DEFAULT_STAGE_CANDIDATES:
        if candidate.is_file():
            return str(candidate)
    return str(_DEFAULT_STAGE_CANDIDATES[0])


def _default_clip_model_name() -> str:
    return os.getenv("FISH_CLIP_MODEL_PATH", "openai/clip-vit-base-patch16")


@dataclass(frozen=True)
class SimConfig:
    # SimulationApp (must be headless for no-UI runs)
    headless: bool = True
    renderer: str = "RayTracedLighting"

    # SimulationContext timing
    stage_units_in_meters: float = 1.0
    physics_dt: float = 1.0 / 60.0
    rendering_dt: float = 1.0 / 60.0
    warmup_steps: int = 5
    sensor_warmup_steps: int = 3


@dataclass(frozen=True)
class SceneConfig:
    # USD scene and prim paths
    # Default to the repo-local stage copy and allow overriding at launch time.
    usd_path: str = _default_stage_path()
    fish_prim_path: str = "/World/robotic_fish_fixed/base_link"
    ball_prim_path: str = "/World/ObstaclePool/Sphere_mini"
    camera_prim_path: str = "/World/robotic_fish_fixed/base_link/Camera"
    light_prim_paths: Tuple[str, ...] = (
        "/World/DomeLight",
        "/World/DistantLight",
        "/World/RectLight",
        "/World/DiskLight",
    )


@dataclass(frozen=True)
class VLAConfig:
    # Language instruction for the semantic branch.
    instruction_text: str = "Find the red ball"

    # CLIP ViT-B/16 settings.
    # semantic_backend_mode: "clip", "fallback", or "hybrid"
    semantic_backend_mode: str = "fallback"
    # For semantic_backend_mode="hybrid":
    # fused_heatmap = fusion_alpha * clip_heatmap + (1 - fusion_alpha) * fallback_heatmap
    fusion_alpha: float = 0.0
    clip_model_name: str = _default_clip_model_name()
    clip_device: str = "cuda"
    clip_local_files_only: bool = _env_flag("FISH_CLIP_LOCAL_FILES_ONLY", True)
    clip_required: bool = _env_flag("FISH_CLIP_REQUIRED", True)

    # Semantic output: 14x14 -> 196.
    heatmap_height: int = 14
    heatmap_width: int = 14


@dataclass(frozen=True)
class EnvConfig:
    # Camera: keep 224x224 to match CLIP ViT-B/16 expectation.
    camera_resolution: Tuple[int, int] = (224, 224)
    # Refresh semantic image processing once every K decision steps.
    # With action_repeat_steps=6 at 60Hz physics, K=1 means ~10Hz semantic updates.
    semantic_update_interval_steps: int = 1
    # Apply each chosen action for N physics steps.
    # With physics_dt=1/60 and N=6, policy decision frequency is ~10Hz.
    action_repeat_steps: int = 6

    # Action magnitudes
    forward_speed_mps: float = 0.8
    turn_rate_radps: float = 1.2

    # Episode setup
    fish_initial_position: Tuple[float, float, float] = (0.0, 0.0, 2.5)
    fish_initial_orientation_wxyz: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    ball_default_position: Tuple[float, float, float] = (2.0, 0.0, 2.5)
    auto_create_ball_if_missing: bool = True
    ball_visual_radius_m: float = 0.08
    ball_visual_color_rgb: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    randomize_ball_xy: bool = True
    ball_xy_min: Tuple[float, float] = (-2.3, -2.3)
    ball_xy_max: Tuple[float, float] = (2.3, 2.3)
    # Stage-1 curriculum: mixed spawn (front-visible majority + full-scene minority).
    curriculum_front_spawn_only: bool = True
    front_spawn_probability: float = 0.7
    front_spawn_x_min: float = 0.8
    front_spawn_x_max: float = 2.3
    front_spawn_y_abs_max: float = 1.2
    max_episode_steps: int = 600

    # Pool XY bounds used for wall-collision logic.
    wall_x_min: float = -2.5
    wall_x_max: float = 2.5
    wall_y_min: float = -2.5
    wall_y_max: float = 2.5
    wall_collision_margin_m: float = 0.08
    wall_collision_penalty: float = -10.0
    terminate_on_wall_collision: bool = True

    # Domain randomization
    randomize_lighting: bool = True
    light_intensity_min: float = 1000.0
    light_intensity_max: float = 6000.0
    light_color_jitter: float = 0.25

    # Reward shaping and termination
    success_distance_m: float = 0.45
    distance_reward_scale: float = 4.5
    heading_reward_scale: float = 0.15
    turn_alignment_reward: float = 0.06
    turn_misalignment_penalty: float = -0.03
    forward_centered_reward: float = 0.02
    # Search reward when target is not visible.
    search_visibility_peak_threshold: float = 0.008
    search_turn_reward: float = 0.03
    search_forward_penalty: float = -0.02
    attention_turn_threshold: float = 0.03
    success_reward: float = 50.0
    timeout_penalty: float = -5.0


@dataclass(frozen=True)
class TrainingConfig:
    # Algorithm options: "ppo" or "recurrent_ppo" (sb3-contrib, LSTM-based).
    algorithm: str = "recurrent_ppo"
    policy: str = "MlpPolicy"
    recurrent_policy: str = "MlpLstmPolicy"
    learning_rate: float = 3e-4
    total_timesteps: int = 1_000_000
    # For SB3 MLP/RecurrentPPO in simulation-heavy workloads, CPU is typically faster.
    device: str = "cpu"
    verbose: int = 1
    seed: int = 42
    progress_bar: bool = True
    console_log_freq_steps: int = 1000

    # PPO internals
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Logging and checkpoints
    tensorboard_log_dir: str = "tensorboard"
    tb_run_name: str = "fish_vla"
    monitor_log_path: str = "monitor_logs/monitor.csv"
    checkpoint_dir: str = "checkpoints"
    checkpoint_freq_steps: int = 50_000

    # Final model output
    model_output_path: str = "models/my_underwater_robot_policy"


SIM_CONFIG = SimConfig()
SCENE_CONFIG = SceneConfig()
VLA_CONFIG = VLAConfig()
ENV_CONFIG = EnvConfig()
TRAINING_CONFIG = TrainingConfig()
