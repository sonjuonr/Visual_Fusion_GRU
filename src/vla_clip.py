"""CLIP ViT-B/16 semantic heatmap encoder for VLA observations."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _huggingface_cache_roots() -> list[Path]:
    roots: list[Path] = []

    explicit_hub_cache = os.getenv("HUGGINGFACE_HUB_CACHE")
    if explicit_hub_cache:
        roots.append(Path(explicit_hub_cache).expanduser())

    hf_home = os.getenv("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home).expanduser() / "hub")

    roots.append(Path.home() / ".cache" / "huggingface" / "hub")

    seen: set[str] = set()
    unique_roots: list[Path] = []
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        unique_roots.append(root)
    return unique_roots


def _find_cached_snapshot(model_name: str) -> str | None:
    model_cache_dir = f"models--{model_name.replace('/', '--')}"
    required_files = ("config.json", "preprocessor_config.json")
    weight_files = ("model.safetensors", "pytorch_model.bin")

    for cache_root in _huggingface_cache_roots():
        snapshots_dir = cache_root / model_cache_dir / "snapshots"
        if not snapshots_dir.is_dir():
            continue

        snapshots = sorted(
            (path for path in snapshots_dir.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for snapshot in snapshots:
            if not all((snapshot / name).is_file() for name in required_files):
                continue
            if not any((snapshot / name).is_file() for name in weight_files):
                continue
            return str(snapshot)
    return None


def _resolve_model_source(model_name: str) -> tuple[str, bool]:
    override = os.getenv("FISH_CLIP_MODEL_PATH")
    if override:
        override_path = Path(override).expanduser()
        if override_path.is_dir():
            return str(override_path), True

    explicit_path = Path(model_name).expanduser()
    if explicit_path.is_dir():
        return str(explicit_path), True

    cached_snapshot = _find_cached_snapshot(model_name)
    if cached_snapshot is not None:
        return cached_snapshot, True

    return model_name, False


class CLIPHeatmapEncoder:
    """Encode (RGB image, text instruction) into a 14x14 semantic heatmap."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        grid_hw: Tuple[int, int] = (14, 14),
        local_files_only: bool = False,
        use_fp16_on_cuda: bool = True,
    ) -> None:
        if local_files_only:
            # Avoid background Hub requests in restricted-network environments.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        try:
            from transformers import CLIPModel, CLIPProcessor
            from transformers.utils import logging as hf_logging
        except Exception as exc:  # pragma: no cover - dependency check
            raise ImportError(
                "transformers is required for CLIP ViT-B/16 semantic encoding. "
                "Install it with: pip install transformers"
            ) from exc
        hf_logging.set_verbosity_error()

        if device.startswith("cuda") and not torch.cuda.is_available():
            self._device = torch.device("cpu")
        else:
            self._device = torch.device(device)

        resolved_model_name, resolved_is_local = _resolve_model_source(model_name)
        resolved_local_only = bool(local_files_only or resolved_is_local)

        self.grid_hw = grid_hw
        self._use_fp16_on_cuda = bool(use_fp16_on_cuda and self._device.type == "cuda")
        self._text_embed_cache: dict[str, torch.Tensor] = {}
        self._processor = CLIPProcessor.from_pretrained(
            resolved_model_name,
            local_files_only=resolved_local_only,
            use_fast=False,
        )
        self._model = CLIPModel.from_pretrained(
            resolved_model_name,
            local_files_only=resolved_local_only,
        ).to(self._device)
        self._model.eval()
        for param in self._model.parameters():
            param.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, rgb: np.ndarray, instruction_text: str) -> np.ndarray:
        """Return normalized 2D semantic heatmap (H, W) as float32."""
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 RGB image, got shape={rgb.shape}")

        rgb_uint8 = np.asarray(rgb, dtype=np.uint8)
        image_inputs = self._processor(images=rgb_uint8, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].to(self._device, non_blocking=True)

        text_embed = self._text_embed_cache.get(instruction_text)
        if text_embed is None:
            text_inputs = self._processor.tokenizer(
                [instruction_text],
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            text_inputs = {k: v.to(self._device, non_blocking=True) for k, v in text_inputs.items()}
            try:
                text_outputs = self._model.text_model(
                    input_ids=text_inputs["input_ids"],
                    attention_mask=text_inputs["attention_mask"],
                    return_dict=True,
                )
                text_pooler = text_outputs.pooler_output
            except TypeError:
                # Older transformers versions may not support return_dict kwarg.
                text_outputs = self._model.text_model(
                    input_ids=text_inputs["input_ids"],
                    attention_mask=text_inputs["attention_mask"],
                )
                text_pooler = text_outputs[1]
            text_embed = self._model.text_projection(text_pooler)
            text_embed = text_embed / (text_embed.norm(dim=-1, keepdim=True) + 1e-8)
            self._text_embed_cache[instruction_text] = text_embed

        autocast_enabled = self._use_fp16_on_cuda
        with torch.autocast(device_type=self._device.type, dtype=torch.float16, enabled=autocast_enabled):
            try:
                vision_outputs = self._model.vision_model(pixel_values=pixel_values, return_dict=True)
                vision_last_hidden = vision_outputs.last_hidden_state
            except TypeError:
                # Older transformers versions may not support return_dict kwarg.
                vision_outputs = self._model.vision_model(pixel_values=pixel_values)
                vision_last_hidden = vision_outputs[0]
        patch_tokens = vision_last_hidden[:, 1:, :]  # [B, 196, C] for ViT-B/16 @ 224x224
        patch_embeds = self._model.visual_projection(patch_tokens)

        patch_embeds = patch_embeds / (patch_embeds.norm(dim=-1, keepdim=True) + 1e-8)

        similarity = torch.matmul(patch_embeds, text_embed.unsqueeze(-1)).squeeze(-1)  # [B, N]
        heat = torch.softmax(similarity, dim=-1)

        patch_count = heat.shape[-1]
        side = int(round(float(np.sqrt(patch_count))))
        if side * side == patch_count:
            heat_2d = heat.reshape(-1, 1, side, side)
        else:
            heat_2d = heat.reshape(-1, 1, patch_count, 1)
        heat_2d = F.adaptive_avg_pool2d(heat_2d, self.grid_hw)

        result = heat_2d[0, 0].float().cpu().numpy().astype(np.float32)
        result_sum = float(result.sum())
        if result_sum > 0.0:
            result /= result_sum
        return result
