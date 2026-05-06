import gymnasium as gym
import numpy as np
import omni.kit.app
import omni.replicator.core as rep
import omni.usd
import torch
import torch.nn.functional as F
from omni.isaac.core import SimulationContext
from omni.isaac.core.prims import RigidPrim, XFormPrim
from omni.isaac.core.utils.numpy.rotations import quats_to_rot_matrices
from omni.isaac.core.utils.prims import is_prim_path_valid

from parameter import ENV_CONFIG, SCENE_CONFIG, SIM_CONFIG, VLA_CONFIG


class UnderwaterReachEnv(gym.Env):
    """RL env for a robotic fish searching a red ball in a pool."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        sim_cfg=SIM_CONFIG,
        scene_cfg=SCENE_CONFIG,
        env_cfg=ENV_CONFIG,
        vla_cfg=VLA_CONFIG,
        instruction_text: str | None = None,
    ):
        super().__init__()
        self.sim_cfg = sim_cfg
        self.scene_cfg = scene_cfg
        self.env_cfg = env_cfg
        self.vla_cfg = vla_cfg
        self.instruction_text = instruction_text or self.vla_cfg.instruction_text

        # Discrete actions requested by the task:
        # 0=forward, 1=turn left, 2=turn right
        self.action_space = gym.spaces.Discrete(3)

        # Observation is the flattened CLIP semantic heatmap: 14x14 = 196.
        self._heatmap_hw = (self.vla_cfg.heatmap_height, self.vla_cfg.heatmap_width)
        self._obs_dim = self._heatmap_hw[0] * self._heatmap_hw[1]
        self.observation_space = gym.spaces.Box(
            low=np.zeros(self._obs_dim, dtype=np.float32),
            high=np.ones(self._obs_dim, dtype=np.float32),
            dtype=np.float32,
        )

        self.world = None
        self.fish_prim = None
        self.ball_prim = None
        self._rgb_annotator = None
        self._render_product = None

        self._clip_encoder = None
        self._semantic_backend = "color_fallback"
        self._clip_init_error = None
        self._semantic_backend_mode = str(getattr(self.vla_cfg, "semantic_backend_mode", "clip")).strip().lower()
        self._fusion_alpha = float(np.clip(getattr(self.vla_cfg, "fusion_alpha", 0.0), 0.0, 1.0))

        self._episode_step = 0
        self._last_distance = None
        self._semantic_update_interval_steps = max(1, int(self.env_cfg.semantic_update_interval_steps))
        self._action_repeat_steps = max(1, int(getattr(self.env_cfg, "action_repeat_steps", 1)))
        self._cached_obs = np.full(self._obs_dim, 1.0 / float(self._obs_dim), dtype=np.float32)
        self._cached_heatmap_2d = np.full(self._heatmap_hw, 1.0 / float(self._obs_dim), dtype=np.float32)
        self._cached_attention_center_x = 0.5
        self._cached_semantic_metrics = {
            "clip_peak": 0.0,
            "fallback_peak": float(1.0 / float(self._obs_dim)),
            "fused_peak": float(1.0 / float(self._obs_dim)),
        }
        width, height = self.env_cfg.camera_resolution
        self._cached_rgb = np.zeros((height, width, 3), dtype=np.uint8)

        self._init_isaac_sim()
        self._init_semantic_encoder()

    def _init_semantic_encoder(self) -> None:
        backend_mode = self._semantic_backend_mode
        if backend_mode == "fallback":
            self._clip_encoder = None
            self._clip_init_error = "forced_fallback_by_config"
            self._semantic_backend = "color_fallback_forced"
            return
        if backend_mode not in {"clip", "hybrid"}:
            raise ValueError(
                f"Unsupported semantic backend mode: {self.vla_cfg.semantic_backend_mode}. "
                "Use 'clip', 'fallback', or 'hybrid'."
            )

        try:
            from vla_clip import CLIPHeatmapEncoder

            self._clip_encoder = CLIPHeatmapEncoder(
                model_name=self.vla_cfg.clip_model_name,
                device=self.vla_cfg.clip_device,
                grid_hw=self._heatmap_hw,
                local_files_only=self.vla_cfg.clip_local_files_only,
            )
            if backend_mode == "hybrid":
                self._semantic_backend = "hybrid_clip_fallback"
            else:
                self._semantic_backend = "clip_vit_b16"
        except Exception as exc:
            self._clip_init_error = str(exc)
            self._clip_encoder = None
            if self.vla_cfg.clip_required:
                raise RuntimeError(
                    "CLIP is required by config but failed to initialize. "
                    f"Cause: {self._clip_init_error}"
                ) from exc
            if backend_mode == "hybrid":
                self._semantic_backend = "hybrid_fallback_only"
            else:
                self._semantic_backend = "color_fallback"

    def _ensure_clip_encoder(self, strict: bool = False) -> bool:
        """Lazily create the CLIP encoder when auxiliary views request it."""
        if self._clip_encoder is not None:
            return True

        try:
            from vla_clip import CLIPHeatmapEncoder

            self._clip_encoder = CLIPHeatmapEncoder(
                model_name=self.vla_cfg.clip_model_name,
                device=self.vla_cfg.clip_device,
                grid_hw=self._heatmap_hw,
                local_files_only=self.vla_cfg.clip_local_files_only,
            )
            self._clip_init_error = None
            return True
        except Exception as exc:
            self._clip_init_error = str(exc)
            self._clip_encoder = None
            if strict or self.vla_cfg.clip_required:
                raise RuntimeError(
                    "CLIP heatmap view was requested but the encoder failed to initialize. "
                    f"Cause: {self._clip_init_error}"
                ) from exc
            return False

    def _create_fallback_ball_prim(self) -> None:
        """Create a simple red sphere under the configured ball Xform path."""
        from pxr import Gf, Sdf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        parent_path = "/".join(self.scene_cfg.ball_prim_path.rstrip("/").split("/")[:-1])
        try:
            if parent_path and not is_prim_path_valid(parent_path):
                UsdGeom.Xform.Define(stage, parent_path)
            UsdGeom.Xform.Define(stage, self.scene_cfg.ball_prim_path)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot auto-create ball prim at {self.scene_cfg.ball_prim_path}. "
                "Set SCENE_CONFIG.ball_prim_path to an existing Xform in your USD."
            ) from exc
        sphere = UsdGeom.Sphere.Define(stage, f"{self.scene_cfg.ball_prim_path}/Visual")
        sphere.CreateRadiusAttr(float(self.env_cfg.ball_visual_radius_m))
        sphere.GetPrim().CreateAttribute("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray).Set(
            [Gf.Vec3f(*self.env_cfg.ball_visual_color_rgb)]
        )

    def _init_isaac_sim(self):
        """
        Initialize Isaac Sim state:
        1) open USD stage through omni.usd API,
        2) create SimulationContext (omni.isaac.core),
        3) wrap fish and ball prims with RigidPrim/XFormPrim.
        """
        usd_context = omni.usd.get_context()
        if not usd_context.open_stage(self.scene_cfg.usd_path):
            raise FileNotFoundError(f"Failed to open USD stage: {self.scene_cfg.usd_path}")

        # Stage loading is async; update app until loading finishes.
        app = omni.kit.app.get_app()
        while usd_context.get_stage_loading_status()[2] > 0:
            app.update()

        self.world = SimulationContext(
            physics_dt=self.sim_cfg.physics_dt,
            rendering_dt=self.sim_cfg.rendering_dt,
            stage_units_in_meters=self.sim_cfg.stage_units_in_meters,
            set_defaults=False,
        )
        self.world.initialize_physics()
        self.world.play()

        # Warm-up steps so rigid bodies/sensors have valid buffers.
        for _ in range(self.sim_cfg.warmup_steps):
            self.world.step(render=True)

        if not is_prim_path_valid(self.scene_cfg.fish_prim_path):
            raise ValueError(f"Fish prim not found: {self.scene_cfg.fish_prim_path}")
        if not is_prim_path_valid(self.scene_cfg.ball_prim_path):
            if self.env_cfg.auto_create_ball_if_missing:
                self._create_fallback_ball_prim()
                self.world.step(render=True)
            else:
                raise ValueError(f"Ball prim not found: {self.scene_cfg.ball_prim_path}")
        if not is_prim_path_valid(self.scene_cfg.ball_prim_path):
            raise ValueError(
                f"Ball prim still missing after fallback creation: {self.scene_cfg.ball_prim_path}"
            )
        if not is_prim_path_valid(self.scene_cfg.camera_prim_path):
            raise ValueError(f"Camera prim not found: {self.scene_cfg.camera_prim_path}")

        # Required wrappers from omni.isaac.core.prims
        self.fish_prim = RigidPrim(self.scene_cfg.fish_prim_path)
        self.ball_prim = XFormPrim(self.scene_cfg.ball_prim_path)
        self.fish_prim.initialize()
        self.ball_prim.initialize()

        # Use replicator RGB annotator attached to fish camera render product.
        self._render_product = rep.create.render_product(
            self.scene_cfg.camera_prim_path, self.env_cfg.camera_resolution
        )
        self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self._rgb_annotator.attach([self._render_product])

        for _ in range(self.sim_cfg.sensor_warmup_steps):
            self.world.step(render=True)

    def _get_camera_rgb(self) -> np.ndarray:
        """
        Fetch current RGB frame from the fish camera using omni.replicator.
        Returns HxWx3 uint8 numpy array.
        """
        width, height = self.env_cfg.camera_resolution
        fallback = np.zeros((height, width, 3), dtype=np.uint8)

        raw = self._rgb_annotator.get_data() if self._rgb_annotator is not None else None
        if raw is None:
            return fallback

        # Replicator returns ndarray for "rgb"; keep a dict fallback for robustness.
        if isinstance(raw, dict):
            raw = raw.get("data", raw.get("rgb", None))
            if raw is None:
                return fallback

        rgb = np.asarray(raw)
        if rgb.ndim == 1 and rgb.size == width * height * 4:
            rgb = rgb.reshape((height, width, 4))
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[..., None], 3, axis=2)
        if rgb.ndim != 3:
            return fallback
        if rgb.shape[2] >= 4:
            rgb = rgb[:, :, :3]
        elif rgb.shape[2] == 1:
            rgb = np.repeat(rgb, 3, axis=2)

        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(rgb)

    def _calculate_distance(self) -> float:
        """Compute fish-ball Euclidean distance in XY plane (ignore Z)."""
        fish_pos, _ = self.fish_prim.get_world_pose()
        ball_pos, _ = self.ball_prim.get_world_pose()
        delta_xy = np.asarray(ball_pos[:2], dtype=np.float64) - np.asarray(fish_pos[:2], dtype=np.float64)
        return float(np.linalg.norm(delta_xy))

    def _quat_wxyz_to_yaw(self, quat_wxyz: np.ndarray) -> float:
        w, x, y, z = [float(v) for v in quat_wxyz]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return float(np.arctan2(siny_cosp, cosy_cosp))

    def _get_heading_error(self) -> float:
        fish_pos, fish_quat = self.fish_prim.get_world_pose()
        ball_pos, _ = self.ball_prim.get_world_pose()
        delta_xy = np.asarray(ball_pos[:2], dtype=np.float64) - np.asarray(fish_pos[:2], dtype=np.float64)
        target_yaw = float(np.arctan2(delta_xy[1], delta_xy[0]))
        fish_yaw = self._quat_wxyz_to_yaw(np.asarray(fish_quat))
        return float((target_yaw - fish_yaw + np.pi) % (2.0 * np.pi) - np.pi)

    def _execute_action(self, action: int) -> None:
        """
        Apply discrete action in fish local frame:
        0: forward at env_cfg.forward_speed_mps
        1: turn left at env_cfg.turn_rate_radps
        2: turn right at -env_cfg.turn_rate_radps

        set_linear_velocity/set_angular_velocity use world frame values, so local
        commands are rotated to world frame with current fish orientation first.
        """
        if action == 0:
            local_linear = np.array([self.env_cfg.forward_speed_mps, 0.0, 0.0], dtype=np.float32)
            local_angular = np.zeros(3, dtype=np.float32)
        elif action == 1:
            local_linear = np.zeros(3, dtype=np.float32)
            local_angular = np.array([0.0, 0.0, self.env_cfg.turn_rate_radps], dtype=np.float32)
        elif action == 2:
            local_linear = np.zeros(3, dtype=np.float32)
            local_angular = np.array([0.0, 0.0, -self.env_cfg.turn_rate_radps], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported action: {action}")

        _, fish_quat_wxyz = self.fish_prim.get_world_pose()
        rot_world_from_local = quats_to_rot_matrices(np.asarray(fish_quat_wxyz, dtype=np.float64))

        world_linear = (rot_world_from_local @ local_linear).astype(np.float32)
        world_angular = (rot_world_from_local @ local_angular).astype(np.float32)

        for _ in range(self._action_repeat_steps):
            self.fish_prim.set_linear_velocity(world_linear)
            self.fish_prim.set_angular_velocity(world_angular)
            self.world.step(render=True)

    def _sample_ball_position(self) -> np.ndarray:
        if not self.env_cfg.randomize_ball_xy:
            return np.array(self.env_cfg.ball_default_position, dtype=np.float32)

        x_min = float(self.env_cfg.ball_xy_min[0])
        x_max = float(self.env_cfg.ball_xy_max[0])
        y_min = float(self.env_cfg.ball_xy_min[1])
        y_max = float(self.env_cfg.ball_xy_max[1])

        if bool(getattr(self.env_cfg, "curriculum_front_spawn_only", False)):
            front_x_min = max(x_min, float(self.env_cfg.front_spawn_x_min))
            front_x_max = min(x_max, float(self.env_cfg.front_spawn_x_max))
            front_y_abs = abs(float(self.env_cfg.front_spawn_y_abs_max))
            front_y_min = max(y_min, -front_y_abs)
            front_y_max = min(y_max, front_y_abs)
            front_prob = float(np.clip(getattr(self.env_cfg, "front_spawn_probability", 1.0), 0.0, 1.0))
            use_front_spawn = bool(self.np_random.random() < front_prob)
            if use_front_spawn and front_x_max > front_x_min and front_y_max > front_y_min:
                x = self.np_random.uniform(front_x_min, front_x_max)
                y = self.np_random.uniform(front_y_min, front_y_max)
            else:
                # Mixed curriculum: sample full scene for exploration.
                x = self.np_random.uniform(x_min, x_max)
                y = self.np_random.uniform(y_min, y_max)
        else:
            x = self.np_random.uniform(x_min, x_max)
            y = self.np_random.uniform(y_min, y_max)

        z = self.env_cfg.ball_default_position[2]
        return np.array([x, y, z], dtype=np.float32)

    def _randomize_lighting(self) -> None:
        if not self.env_cfg.randomize_lighting:
            return
        from pxr import Gf, UsdLux

        stage = omni.usd.get_context().get_stage()
        for light_path in self.scene_cfg.light_prim_paths:
            if not is_prim_path_valid(light_path):
                continue
            prim = stage.GetPrimAtPath(light_path)
            light_api = UsdLux.LightAPI(prim)
            if not light_api:
                continue

            intensity = float(self.np_random.uniform(self.env_cfg.light_intensity_min, self.env_cfg.light_intensity_max))
            intensity_attr = light_api.GetIntensityAttr()
            if not intensity_attr:
                intensity_attr = light_api.CreateIntensityAttr()
            intensity_attr.Set(intensity)

            jitter = self.env_cfg.light_color_jitter
            color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            color += self.np_random.uniform(-jitter, jitter, size=3).astype(np.float32)
            color = np.clip(color, 0.2, 1.5)
            color_attr = light_api.GetColorAttr()
            if not color_attr:
                color_attr = light_api.CreateColorAttr()
            color_attr.Set(Gf.Vec3f(float(color[0]), float(color[1]), float(color[2])))

    def _fallback_color_heatmap(self, rgb: np.ndarray) -> np.ndarray:
        # Heuristic fallback: red saliency map if CLIP backend is unavailable.
        red = rgb[:, :, 0].astype(np.float32)
        green = rgb[:, :, 1].astype(np.float32)
        blue = rgb[:, :, 2].astype(np.float32)
        saliency = np.clip(red - 0.5 * (green + blue), a_min=0.0, a_max=None)

        pooled = F.adaptive_avg_pool2d(
            torch.from_numpy(saliency).unsqueeze(0).unsqueeze(0),
            self._heatmap_hw,
        )[0, 0].cpu().numpy()
        pooled = np.clip(pooled, a_min=0.0, a_max=None)
        total = float(pooled.sum())
        if total <= 0.0:
            pooled = np.full(self._heatmap_hw, 1.0 / float(self._obs_dim), dtype=np.float32)
        else:
            pooled = (pooled / total).astype(np.float32)
        return pooled

    def set_semantic_fusion_alpha(self, alpha: float) -> None:
        self._fusion_alpha = float(np.clip(float(alpha), 0.0, 1.0))

    def get_semantic_fusion_alpha(self) -> float:
        return float(self._fusion_alpha)

    def _normalize_heatmap(self, heatmap: np.ndarray) -> np.ndarray:
        heatmap_arr = np.asarray(heatmap, dtype=np.float32)
        if heatmap_arr.shape != self._heatmap_hw:
            heatmap_arr = F.adaptive_avg_pool2d(
                torch.from_numpy(heatmap_arr).unsqueeze(0).unsqueeze(0),
                self._heatmap_hw,
            )[0, 0].cpu().numpy()
        heatmap_arr = np.clip(heatmap_arr, a_min=0.0, a_max=None)
        total = float(heatmap_arr.sum())
        if total <= 0.0:
            return np.full(self._heatmap_hw, 1.0 / float(self._obs_dim), dtype=np.float32)
        return (heatmap_arr / total).astype(np.float32)

    def _clip_semantic_heatmap(
        self,
        rgb: np.ndarray,
        *,
        allow_lazy_init: bool = False,
        strict: bool = False,
        update_backend_status: bool = True,
    ) -> np.ndarray | None:
        if self._clip_encoder is None and allow_lazy_init:
            self._ensure_clip_encoder(strict=strict)
        if self._clip_encoder is None:
            return None
        try:
            raw_heatmap = self._clip_encoder.encode(rgb=rgb, instruction_text=self.instruction_text)
        except Exception as exc:
            if strict or self.vla_cfg.clip_required:
                raise
            self._clip_init_error = str(exc)
            if update_backend_status:
                if self._semantic_backend_mode == "hybrid":
                    self._semantic_backend = "hybrid_fallback_runtime"
                else:
                    self._semantic_backend = "color_fallback_runtime"
            return None
        return self._normalize_heatmap(raw_heatmap)

    def _peak_or_zero(self, heatmap: np.ndarray | None) -> float:
        if heatmap is None:
            return 0.0
        return float(np.max(np.asarray(heatmap, dtype=np.float32)))

    def _semantic_metrics(
        self,
        *,
        clip_heatmap: np.ndarray | None,
        fallback_heatmap: np.ndarray | None,
        fused_heatmap: np.ndarray,
    ) -> dict[str, float]:
        return {
            "clip_peak": self._peak_or_zero(clip_heatmap),
            "fallback_peak": self._peak_or_zero(fallback_heatmap),
            "fused_peak": self._peak_or_zero(fused_heatmap),
        }

    def _get_semantic_heatmap(self, rgb: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        if self._semantic_backend_mode == "fallback":
            fallback_heatmap = self._fallback_color_heatmap(rgb)
            return fallback_heatmap, self._semantic_metrics(
                clip_heatmap=None,
                fallback_heatmap=fallback_heatmap,
                fused_heatmap=fallback_heatmap,
            )

        if self._semantic_backend_mode == "clip":
            clip_heatmap = self._clip_semantic_heatmap(rgb)
            if clip_heatmap is None:
                fallback_heatmap = self._fallback_color_heatmap(rgb)
                return fallback_heatmap, self._semantic_metrics(
                    clip_heatmap=None,
                    fallback_heatmap=fallback_heatmap,
                    fused_heatmap=fallback_heatmap,
                )
            self._semantic_backend = "clip_vit_b16"
            return clip_heatmap, self._semantic_metrics(
                clip_heatmap=clip_heatmap,
                fallback_heatmap=None,
                fused_heatmap=clip_heatmap,
            )

        if self._semantic_backend_mode == "hybrid":
            fallback_heatmap = self._fallback_color_heatmap(rgb)
            clip_heatmap = self._clip_semantic_heatmap(rgb)
            if clip_heatmap is None:
                return fallback_heatmap, self._semantic_metrics(
                    clip_heatmap=None,
                    fallback_heatmap=fallback_heatmap,
                    fused_heatmap=fallback_heatmap,
                )
            alpha = float(np.clip(self._fusion_alpha, 0.0, 1.0))
            fused = alpha * clip_heatmap + (1.0 - alpha) * fallback_heatmap
            self._semantic_backend = "hybrid_clip_fallback"
            fused_heatmap = self._normalize_heatmap(fused)
            return fused_heatmap, self._semantic_metrics(
                clip_heatmap=clip_heatmap,
                fallback_heatmap=fallback_heatmap,
                fused_heatmap=fused_heatmap,
            )

        raise ValueError(
            f"Unsupported semantic backend mode at runtime: {self._semantic_backend_mode}"
        )

    def _attention_center_x(self, heatmap_2d: np.ndarray) -> float:
        column_weight = heatmap_2d.sum(axis=0)
        denom = float(column_weight.sum())
        if denom <= 0.0:
            return 0.5
        x_coords = (np.arange(heatmap_2d.shape[1], dtype=np.float32) + 0.5) / float(heatmap_2d.shape[1])
        return float(np.sum(column_weight * x_coords) / denom)

    def _check_wall_collision(self) -> tuple[bool, float]:
        fish_pos, _ = self.fish_prim.get_world_pose()
        x = float(fish_pos[0])
        y = float(fish_pos[1])

        x_left_clearance = x - float(self.env_cfg.wall_x_min)
        x_right_clearance = float(self.env_cfg.wall_x_max) - x
        y_bottom_clearance = y - float(self.env_cfg.wall_y_min)
        y_top_clearance = float(self.env_cfg.wall_y_max) - y

        min_clearance = min(x_left_clearance, x_right_clearance, y_bottom_clearance, y_top_clearance)
        wall_hit = min_clearance <= float(self.env_cfg.wall_collision_margin_m)
        return bool(wall_hit), float(min_clearance)

    def _build_obs_from_rgb(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, dict[str, float]]:
        heatmap_2d, semantic_metrics = self._get_semantic_heatmap(rgb)
        obs = heatmap_2d.reshape(-1).astype(np.float32)
        attention_center_x = self._attention_center_x(heatmap_2d)
        return obs, heatmap_2d, attention_center_x, semantic_metrics

    def _refresh_semantic_cache(self) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, dict[str, float]]:
        rgb = self._get_camera_rgb()
        obs, heatmap_2d, attention_center_x, semantic_metrics = self._build_obs_from_rgb(rgb)
        self._cached_obs = obs
        self._cached_heatmap_2d = heatmap_2d
        self._cached_attention_center_x = attention_center_x
        self._cached_semantic_metrics = semantic_metrics
        self._cached_rgb = rgb
        return obs, heatmap_2d, attention_center_x, rgb, semantic_metrics

    def _get_semantic_observation(
        self, force_refresh: bool = False
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, dict[str, float], bool]:
        should_refresh = force_refresh or (self._episode_step % self._semantic_update_interval_steps == 0)
        if should_refresh:
            obs, heatmap_2d, attention_center_x, rgb, semantic_metrics = self._refresh_semantic_cache()
            return obs, heatmap_2d, attention_center_x, rgb, semantic_metrics, True
        return (
            self._cached_obs,
            self._cached_heatmap_2d,
            self._cached_attention_center_x,
            self._cached_rgb,
            self._cached_semantic_metrics,
            False,
        )

    def _pack_heatmap_view(self, heatmap_2d: np.ndarray) -> dict[str, object]:
        return {
            "obs": heatmap_2d.reshape(-1).astype(np.float32),
            "heatmap_2d": heatmap_2d.astype(np.float32),
            "attention_center_x": float(self._attention_center_x(heatmap_2d)),
            "peak": float(np.max(heatmap_2d)),
        }

    def get_observation_views(
        self,
        *,
        force_refresh: bool = False,
        include_fallback: bool = True,
        include_clip: bool = True,
        include_hybrid: bool = False,
        hybrid_alpha: float | None = None,
        strict_clip: bool = False,
    ) -> dict[str, object]:
        """
        Return multiple heatmap views from the same simulator frame.

        This is used by imitation learning to query fallback and ViT heatmaps
        from one underlying RGB observation without changing environment logic.
        """
        _, _, _, rgb, _, semantic_refreshed = self._get_semantic_observation(force_refresh=force_refresh)

        fallback_heatmap = None
        if include_fallback or include_hybrid:
            fallback_heatmap = self._fallback_color_heatmap(rgb)

        clip_heatmap = None
        if include_clip or include_hybrid:
            clip_heatmap = self._clip_semantic_heatmap(
                rgb,
                allow_lazy_init=True,
                strict=strict_clip,
                update_backend_status=False,
            )

        views: dict[str, dict[str, object]] = {}
        if include_fallback and fallback_heatmap is not None:
            views["fallback"] = self._pack_heatmap_view(fallback_heatmap)
        if include_clip and clip_heatmap is not None:
            views["clip"] = self._pack_heatmap_view(clip_heatmap)
        if include_hybrid:
            if fallback_heatmap is None:
                raise RuntimeError("Hybrid view requires the fallback heatmap.")
            if clip_heatmap is None:
                fused_heatmap = fallback_heatmap
            else:
                alpha = self._fusion_alpha if hybrid_alpha is None else float(hybrid_alpha)
                alpha_clamped = float(np.clip(alpha, 0.0, 1.0))
                fused = alpha_clamped * clip_heatmap + (1.0 - alpha_clamped) * fallback_heatmap
                fused_heatmap = self._normalize_heatmap(fused)
            views["hybrid"] = self._pack_heatmap_view(fused_heatmap)

        return {
            "rgb": rgb,
            "views": views,
            "semantic_refreshed": bool(semantic_refreshed),
            "clip_available": bool(clip_heatmap is not None),
            "clip_init_error": self._clip_init_error,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._episode_step = 0

        self.fish_prim.set_world_pose(
            position=np.array(self.env_cfg.fish_initial_position, dtype=np.float32),
            orientation=np.array(self.env_cfg.fish_initial_orientation_wxyz, dtype=np.float32),
        )
        self.fish_prim.set_linear_velocity(np.zeros(3, dtype=np.float32))
        self.fish_prim.set_angular_velocity(np.zeros(3, dtype=np.float32))

        ball_position = self._sample_ball_position()
        if options is not None and "ball_position" in options:
            ball_position = np.asarray(options["ball_position"], dtype=np.float32)
        self.ball_prim.set_world_pose(position=ball_position)

        self._randomize_lighting()
        self.world.step(render=True)
        self._last_distance = self._calculate_distance()
        heading_error = self._get_heading_error()

        obs, heatmap_2d, attention_center_x, rgb, semantic_metrics, semantic_refreshed = self._get_semantic_observation(
            force_refresh=True
        )
        wall_hit, wall_clearance_m = self._check_wall_collision()
        info = {
            "distance_xy_m": self._last_distance,
            "heading_error_rad": heading_error,
            "attention_center_x": attention_center_x,
            "wall_collision_hit": wall_hit,
            "wall_clearance_m": wall_clearance_m,
            "semantic_heatmap_14x14": heatmap_2d,
            "semantic_refreshed": semantic_refreshed,
            "semantic_update_interval_steps": self._semantic_update_interval_steps,
            "action_repeat_steps": self._action_repeat_steps,
            "semantic_backend_mode": self._semantic_backend_mode,
            "semantic_backend": self._semantic_backend,
            "fusion_alpha": float(self._fusion_alpha),
            "clip_peak": float(semantic_metrics["clip_peak"]),
            "fallback_peak": float(semantic_metrics["fallback_peak"]),
            "fused_peak": float(semantic_metrics["fused_peak"]),
            "instruction_text": self.instruction_text,
            "clip_init_error": self._clip_init_error,
            "rgb": rgb,
        }
        return obs, info

    def step(self, action):
        action = int(action)
        self._execute_action(action)
        self._episode_step += 1

        current_distance = self._calculate_distance()
        heading_error = self._get_heading_error()
        progress = 0.0 if self._last_distance is None else (self._last_distance - current_distance)

        obs, heatmap_2d, attention_center_x, rgb, semantic_metrics, semantic_refreshed = self._get_semantic_observation()
        attention_offset = attention_center_x - 0.5
        turn_threshold = float(self.env_cfg.attention_turn_threshold)

        reward_progress = progress * self.env_cfg.distance_reward_scale
        reward_heading = float(np.cos(heading_error)) * self.env_cfg.heading_reward_scale

        reward_turn = 0.0
        desired_turn_action = -1
        if abs(attention_offset) > turn_threshold:
            desired_turn_action = 1 if attention_offset < 0.0 else 2
            if action == desired_turn_action:
                reward_turn += self.env_cfg.turn_alignment_reward
            elif action in (1, 2):
                reward_turn += self.env_cfg.turn_misalignment_penalty

        reward_forward_center = 0.0
        if abs(attention_offset) <= turn_threshold and action == 0:
            reward_forward_center += self.env_cfg.forward_centered_reward

        # Encourage active scanning when the target is likely out of view.
        heatmap_peak = float(semantic_metrics["fused_peak"])
        visible_peak_threshold = float(self.env_cfg.search_visibility_peak_threshold)
        target_visible = heatmap_peak >= visible_peak_threshold
        reward_search = 0.0
        if not target_visible:
            if action in (1, 2):
                reward_search += float(self.env_cfg.search_turn_reward)
            elif action == 0:
                reward_search += float(self.env_cfg.search_forward_penalty)

        wall_hit, wall_clearance_m = self._check_wall_collision()
        reward_wall_collision = self.env_cfg.wall_collision_penalty if wall_hit else 0.0

        reward_success = 0.0
        reward_timeout = 0.0
        is_success = False
        reward = (
            reward_progress
            + reward_heading
            + reward_turn
            + reward_forward_center
            + reward_search
            + reward_wall_collision
        )

        terminated = False
        truncated = False
        if wall_hit and self.env_cfg.terminate_on_wall_collision:
            terminated = True
        elif current_distance <= self.env_cfg.success_distance_m:
            reward_success = self.env_cfg.success_reward
            reward += reward_success
            is_success = True
            terminated = True
        elif self._episode_step >= self.env_cfg.max_episode_steps:
            reward_timeout = self.env_cfg.timeout_penalty
            reward += reward_timeout
            truncated = True

        self._last_distance = current_distance
        info = {
            "distance_xy_m": current_distance,
            "heading_error_rad": heading_error,
            "attention_center_x": attention_center_x,
            "desired_turn_action": desired_turn_action,
            "wall_collision_hit": wall_hit,
            "wall_clearance_m": wall_clearance_m,
            "semantic_heatmap_14x14": heatmap_2d,
            "semantic_refreshed": semantic_refreshed,
            "semantic_update_interval_steps": self._semantic_update_interval_steps,
            "action_repeat_steps": self._action_repeat_steps,
            "semantic_backend_mode": self._semantic_backend_mode,
            "semantic_backend": self._semantic_backend,
            "fusion_alpha": float(self._fusion_alpha),
            "clip_peak": float(semantic_metrics["clip_peak"]),
            "fallback_peak": float(semantic_metrics["fallback_peak"]),
            "fused_peak": float(semantic_metrics["fused_peak"]),
            "instruction_text": self.instruction_text,
            "clip_init_error": self._clip_init_error,
            "reward_progress": reward_progress,
            "reward_heading": reward_heading,
            "reward_turn": reward_turn,
            "reward_forward_center": reward_forward_center,
            "reward_search": reward_search,
            "reward_wall_collision": reward_wall_collision,
            "reward_success": reward_success,
            "reward_timeout": reward_timeout,
            "reward_total": reward,
            "is_success": bool(is_success),
            "target_visible": bool(target_visible),
            "heatmap_peak": heatmap_peak,
            "rgb": rgb,
        }
        return obs, float(reward), terminated, truncated, info

    def close(self):
        try:
            if self._rgb_annotator is not None and self._render_product is not None:
                self._rgb_annotator.detach([self._render_product])
        except Exception:
            pass

        try:
            if self._render_product is not None and hasattr(self._render_product, "destroy"):
                self._render_product.destroy()
        except Exception:
            pass

        try:
            rep.orchestrator.stop()
        except Exception:
            pass

        if self.world is not None:
            self.world.stop()
