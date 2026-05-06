"""Custom callbacks for TensorBoard logging."""

from __future__ import annotations

import time
from collections import deque
from typing import Dict, Iterable

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class HyperparameterTensorboardCallback(BaseCallback):
    """Write static hyperparameters to TensorBoard once at training start."""

    def __init__(self, hparams: Dict[str, object], verbose: int = 0):
        super().__init__(verbose)
        self.hparams = hparams

    def _on_training_start(self) -> None:
        for key, value in self.hparams.items():
            if isinstance(value, (int, float, bool, str)):
                self.logger.record(f"config/{key}", value)

    def _on_step(self) -> bool:
        return True


class InfoTensorboardCallback(BaseCallback):
    """Log selected info dict values returned by env.step()."""

    def __init__(self, info_keys: Iterable[str], verbose: int = 0):
        super().__init__(verbose)
        self.info_keys = list(info_keys)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for done, info in zip(dones, infos):
            for key in self.info_keys:
                if key not in info:
                    continue
                value = info[key]
                if isinstance(value, (int, float, bool, np.number)):
                    # Use mean aggregation across rollout steps instead of keeping
                    # only the latest value before each logger dump.
                    self.logger.record_mean(f"env/{key}", float(value))

            # Episode-level success signal (only at terminal steps).
            if bool(done):
                self.logger.record_mean("episode/is_success", float(bool(info.get("is_success", False))))
        return True


class ConsoleProgressCallback(BaseCallback):
    """Print periodic wall-clock progress in terminal."""

    def __init__(self, print_freq_steps: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.print_freq_steps = int(print_freq_steps)
        self._start_time = 0.0
        self._last_print_step = 0
        self._start_step = 0
        self._target_step = 0

    def _on_training_start(self) -> None:
        self._start_time = time.time()
        self._start_step = int(getattr(self.model, "num_timesteps", 0))
        self._target_step = int(getattr(self.model, "_total_timesteps", self._start_step))
        self._last_print_step = self._start_step
        print(
            f"[Train] Start: start_timesteps={self._start_step} "
            f"target_timesteps={self._target_step} "
            f"delta={max(0, self._target_step - self._start_step)}"
        )

    def _on_step(self) -> bool:
        if self.print_freq_steps <= 0:
            return True

        step_now = int(self.num_timesteps)
        if step_now - self._last_print_step < self.print_freq_steps:
            return True

        elapsed = max(1e-6, time.time() - self._start_time)
        done_delta = max(0, step_now - self._start_step)
        total_delta = max(1, self._target_step - self._start_step)
        speed = done_delta / elapsed
        ratio = min(1.0, done_delta / total_delta)
        eta = max(0.0, (total_delta - done_delta) / max(1e-6, speed))
        print(
            f"[Train] progress={ratio * 100:.2f}% "
            f"timesteps={done_delta}/{total_delta} "
            f"elapsed={elapsed / 60.0:.1f}min "
            f"eta={eta / 60.0:.1f}min "
            f"fps={speed:.1f}"
        )
        self._last_print_step = step_now
        return True


class FusionAlphaCurriculumCallback(BaseCallback):
    """
    Curriculum for hybrid semantic fusion:
    fused_heatmap = alpha * CLIP + (1 - alpha) * fallback.

    Alpha increases when recent episode success rate stays above threshold.
    """

    def __init__(
        self,
        alpha_start: float = 0.05,
        alpha_max: float = 1.0,
        alpha_step: float = 0.1,
        alpha_targets: list[float] | None = None,
        success_rate_threshold: float = 0.7,
        success_window_episodes: int = 20,
        min_episodes_before_update: int = 20,
        cooldown_episodes: int = 5,
        reset_window_on_alpha_change: bool = True,
        allow_alpha_decrease: bool = True,
        success_rate_lower_threshold: float = 0.2,
        alpha_decrease_step: float | None = None,
        min_episodes_before_decrease: int | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.alpha_start = float(alpha_start)
        self.alpha_max = float(alpha_max)
        self.alpha_step = float(alpha_step)
        self.success_rate_threshold = float(success_rate_threshold)
        self.success_window_episodes = max(1, int(success_window_episodes))
        self.min_episodes_before_update = max(1, int(min_episodes_before_update))
        self.cooldown_episodes = max(0, int(cooldown_episodes))
        self.reset_window_on_alpha_change = bool(reset_window_on_alpha_change)
        self.allow_alpha_decrease = bool(allow_alpha_decrease)
        self.success_rate_lower_threshold = float(success_rate_lower_threshold)
        self.alpha_decrease_step = (
            float(alpha_decrease_step)
            if alpha_decrease_step is not None
            else float(alpha_step)
        )
        self.min_episodes_before_decrease = (
            max(1, int(min_episodes_before_decrease))
            if min_episodes_before_decrease is not None
            else max(1, int(min_episodes_before_update))
        )
        self.alpha_targets = None
        self._alpha_target_idx = 0
        if alpha_targets is not None:
            cleaned_targets: list[float] = []
            for value in alpha_targets:
                v = float(np.clip(float(value), 0.0, 1.0))
                if not cleaned_targets or abs(v - cleaned_targets[-1]) > 1e-9:
                    cleaned_targets.append(v)
            if cleaned_targets:
                self.alpha_targets = cleaned_targets
                self.alpha_start = cleaned_targets[0]
                self.alpha_max = cleaned_targets[-1]

        self._alpha = 0.0
        self._episode_count = 0
        self._last_update_episode = 0
        self._episodes_since_alpha_change = 0
        self._recent_success = deque(maxlen=self.success_window_episodes)

    def _set_env_alpha(self, alpha: float) -> None:
        alpha_clamped = float(np.clip(alpha, 0.0, 1.0))
        self.training_env.env_method("set_semantic_fusion_alpha", alpha_clamped)
        self._alpha = alpha_clamped

    def _window_success_rate(self) -> float:
        if len(self._recent_success) == 0:
            return 0.0
        return float(np.mean(self._recent_success))

    def _on_training_start(self) -> None:
        self._episode_count = 0
        self._last_update_episode = 0
        self._episodes_since_alpha_change = 0
        self._recent_success.clear()
        self._alpha_target_idx = 0
        self._set_env_alpha(self.alpha_start)
        target_text = (
            ",".join(f"{v:.3f}" for v in self.alpha_targets)
            if self.alpha_targets is not None
            else "none"
        )
        print(
            f"[Curriculum] start alpha={self._alpha:.3f} max={self.alpha_max:.3f} "
            f"step={self.alpha_step:.3f} threshold_up={self.success_rate_threshold:.3f} "
            f"threshold_down={self.success_rate_lower_threshold:.3f} "
            f"window={self.success_window_episodes} cooldown={self.cooldown_episodes} "
            f"alpha_targets={target_text}"
        )

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for done, info in zip(dones, infos):
            if not bool(done):
                continue
            self._episode_count += 1
            self._episodes_since_alpha_change += 1
            success = bool(info.get("is_success", False))
            self._recent_success.append(1.0 if success else 0.0)
            success_rate = self._window_success_rate()

            enough_raise_history = (
                len(self._recent_success) >= self.min_episodes_before_update
                and self._episodes_since_alpha_change >= self.min_episodes_before_update
            )
            enough_lower_history = (
                len(self._recent_success) >= self.min_episodes_before_decrease
                and self._episodes_since_alpha_change >= self.min_episodes_before_decrease
            )
            cooldown_ok = (self._episode_count - self._last_update_episode) >= self.cooldown_episodes
            next_target_exists = (
                (self._alpha_target_idx < len(self.alpha_targets) - 1)
                if self.alpha_targets is not None
                else (self._alpha + 1e-9 < self.alpha_max)
            )
            should_raise = (
                enough_raise_history
                and cooldown_ok
                and success_rate >= self.success_rate_threshold
                and next_target_exists
            )
            should_lower = (
                self.allow_alpha_decrease
                and enough_lower_history
                and cooldown_ok
                and success_rate <= self.success_rate_lower_threshold
                and self._alpha - 1e-9 > self.alpha_start
            )

            if should_raise:
                old_alpha = self._alpha
                if self.alpha_targets is not None:
                    self._alpha_target_idx = min(self._alpha_target_idx + 1, len(self.alpha_targets) - 1)
                    new_alpha = float(self.alpha_targets[self._alpha_target_idx])
                else:
                    new_alpha = min(self.alpha_max, self._alpha + self.alpha_step)
                self._set_env_alpha(new_alpha)
                self._last_update_episode = self._episode_count
                self._episodes_since_alpha_change = 0
                if self.reset_window_on_alpha_change:
                    self._recent_success.clear()
                print(
                    f"[Curriculum] episode={self._episode_count} "
                    f"success_rate_window={success_rate:.3f} alpha_up {old_alpha:.3f}->{self._alpha:.3f}"
                )
            elif should_lower:
                old_alpha = self._alpha
                lowered_alpha = max(self.alpha_start, self._alpha - self.alpha_decrease_step)
                self._set_env_alpha(lowered_alpha)
                if self.alpha_targets is not None:
                    idx = 0
                    for i, target_alpha in enumerate(self.alpha_targets):
                        if target_alpha <= self._alpha + 1e-9:
                            idx = i
                    self._alpha_target_idx = idx
                self._last_update_episode = self._episode_count
                self._episodes_since_alpha_change = 0
                if self.reset_window_on_alpha_change:
                    self._recent_success.clear()
                print(
                    f"[Curriculum] episode={self._episode_count} "
                    f"success_rate_window={success_rate:.3f} alpha_down {old_alpha:.3f}->{self._alpha:.3f}"
                )

        self.logger.record("curriculum/fusion_alpha", float(self._alpha))
        self.logger.record("curriculum/success_rate_window", self._window_success_rate())
        self.logger.record("curriculum/episode_count", float(self._episode_count))
        self.logger.record("curriculum/episodes_since_alpha_change", float(self._episodes_since_alpha_change))
        self.logger.record("curriculum/alpha_target_idx", float(self._alpha_target_idx))
        return True
