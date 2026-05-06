"""Runtime helpers for imitation-learning scripts."""

from __future__ import annotations

import os
from contextlib import AbstractContextManager
from dataclasses import replace
from pathlib import Path

from imitation.config import EnvRuntimeConfig
from parameter import ENV_CONFIG, SIM_CONFIG, VLA_CONFIG


def _build_simulation_app_config(runtime_cfg: EnvRuntimeConfig) -> dict[str, object]:
    config: dict[str, object] = {
        "headless": bool(runtime_cfg.headless),
        "renderer": str(runtime_cfg.renderer),
    }
    if os.name == "nt":
        config["extra_args"] = ["--/app/vulkan=false"]
    return config


class SimulationAppSession(AbstractContextManager):
    def __init__(self, runtime_cfg: EnvRuntimeConfig) -> None:
        self.runtime_cfg = runtime_cfg
        self._simulation_app = None

    def __enter__(self):
        from isaacsim import SimulationApp

        self._simulation_app = SimulationApp(_build_simulation_app_config(self.runtime_cfg))
        return self._simulation_app

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._simulation_app is not None:
            self._simulation_app.close()
        self._simulation_app = None
        return None


def build_underwater_env(runtime_cfg: EnvRuntimeConfig):
    from underwater_env import UnderwaterReachEnv

    env_cfg_updates = {}
    if runtime_cfg.semantic_update_interval_steps is not None:
        env_cfg_updates["semantic_update_interval_steps"] = max(1, int(runtime_cfg.semantic_update_interval_steps))
    if runtime_cfg.action_repeat_steps is not None:
        env_cfg_updates["action_repeat_steps"] = max(1, int(runtime_cfg.action_repeat_steps))
    env_cfg = replace(ENV_CONFIG, **env_cfg_updates) if env_cfg_updates else ENV_CONFIG
    vla_cfg = replace(
        VLA_CONFIG,
        semantic_backend_mode=str(runtime_cfg.semantic_backend_mode),
        fusion_alpha=float(runtime_cfg.fusion_alpha),
        clip_required=bool(runtime_cfg.clip_required),
        clip_local_files_only=bool(runtime_cfg.clip_local_files_only),
    )
    return UnderwaterReachEnv(env_cfg=env_cfg, vla_cfg=vla_cfg, instruction_text=runtime_cfg.instruction_text)


def resolve_zip_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.suffix.lower() != ".zip":
        path = path.with_suffix(".zip")
    return path


def ensure_parent_dir(path_str: str | Path) -> Path:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
