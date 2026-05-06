"""Policy wrappers shared by collection, evaluation, and DAgger."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from imitation.models import HeatmapActionMLP


class PolicyAdapter:
    def reset(self) -> None:
        return None

    def predict(self, observation: np.ndarray) -> int:
        raise NotImplementedError


class SB3PolicyAdapter(PolicyAdapter):
    def __init__(self, model_path: str, algorithm: str, deterministic: bool = True, device: str = "cpu") -> None:
        algo = str(algorithm).strip().lower()
        if algo == "recurrent_ppo":
            from sb3_contrib import RecurrentPPO

            self._model = RecurrentPPO.load(str(model_path), device=device)
            self._is_recurrent = True
        elif algo == "ppo":
            from stable_baselines3 import PPO

            self._model = PPO.load(str(model_path), device=device)
            self._is_recurrent = False
        else:
            raise ValueError(f"Unsupported SB3 algorithm: {algorithm}")

        self.deterministic = bool(deterministic)
        self._lstm_state = None
        self._episode_start = np.ones((1,), dtype=bool)

    def reset(self) -> None:
        self._lstm_state = None
        self._episode_start = np.ones((1,), dtype=bool)

    def predict(self, observation: np.ndarray) -> int:
        obs = np.asarray(observation, dtype=np.float32)
        if self._is_recurrent:
            action, self._lstm_state = self._model.predict(
                obs,
                state=self._lstm_state,
                episode_start=self._episode_start,
                deterministic=self.deterministic,
            )
            self._episode_start[:] = False
        else:
            action, _ = self._model.predict(obs, deterministic=self.deterministic)
        return int(np.asarray(action).item())


class TorchStudentPolicyAdapter(PolicyAdapter):
    def __init__(self, checkpoint_path: str, deterministic: bool = True, device: str = "cpu") -> None:
        checkpoint = torch.load(Path(checkpoint_path), map_location=device)
        model_config = checkpoint["model_config"]
        self._model = HeatmapActionMLP(**model_config)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.to(device)
        self._model.eval()
        self._device = torch.device(device)
        self.deterministic = bool(deterministic)

    def predict(self, observation: np.ndarray) -> int:
        obs = torch.as_tensor(np.asarray(observation, dtype=np.float32), device=self._device).reshape(1, -1)
        with torch.inference_mode():
            logits = self._model(obs)
            action = int(torch.argmax(logits, dim=-1).item())
        return action
