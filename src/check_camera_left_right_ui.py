import argparse
import time

import numpy as np

from parameter import ENV_CONFIG, SIM_CONFIG, VLA_CONFIG


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-seconds", type=float, default=10.0, help="Seconds to hold the ball on left side.")
    parser.add_argument("--right-seconds", type=float, default=10.0, help="Seconds to hold the ball on right side.")
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=3.0,
        help="Seconds to keep UI open after phases are finished.",
    )
    parser.add_argument("--ball-x", type=float, default=1.8, help="Ball X position for left/right phases.")
    parser.add_argument("--ball-y", type=float, default=1.2, help="Ball |Y| offset for left/right phases.")
    parser.add_argument(
        "--ball-z",
        type=float,
        default=float(ENV_CONFIG.ball_default_position[2]),
        help="Ball Z position for left/right phases.",
    )
    parser.add_argument(
        "--log-hz",
        type=float,
        default=2.0,
        help="How often to print semantic stats (times per second).",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Override text instruction. Default uses VLA_CONFIG.instruction_text.",
    )
    return parser.parse_args()


def _set_viewport_to_fish_camera(camera_prim_path: str) -> bool:
    """Best-effort switch UI viewport to fish camera."""
    try:
        from omni.kit.viewport.utility import get_active_viewport

        viewport = get_active_viewport()
        if viewport is None:
            return False
        viewport.set_active_camera(camera_prim_path)
        return True
    except Exception:
        return False


def _run_phase(env, phase_name: str, ball_pos: np.ndarray, seconds: float, log_hz: float) -> None:
    _, info = env.reset(seed=42, options={"ball_position": ball_pos})
    print(
        f"[Phase] {phase_name} start | ball={ball_pos.tolist()} "
        f"backend={info.get('semantic_backend')}"
    )

    fixed_pos = np.array(env.env_cfg.fish_initial_position, dtype=np.float32)
    fixed_quat = np.array(env.env_cfg.fish_initial_orientation_wxyz, dtype=np.float32)

    t0 = time.time()
    log_period = 1.0 / max(log_hz, 1e-6)
    next_log = t0
    while True:
        elapsed = time.time() - t0
        if elapsed >= seconds:
            break

        # Keep fish fixed at center and facing same direction for clear camera check.
        env.fish_prim.set_world_pose(position=fixed_pos, orientation=fixed_quat)
        env.fish_prim.set_linear_velocity(np.zeros(3, dtype=np.float32))
        env.fish_prim.set_angular_velocity(np.zeros(3, dtype=np.float32))
        env.world.step(render=True)

        now = time.time()
        if now >= next_log:
            _, heatmap_2d, attention_x, _, _ = env._get_semantic_observation(force_refresh=True)
            peak = float(np.max(heatmap_2d))
            base_uniform = 1.0 / float(heatmap_2d.size)
            print(
                f"[Phase] {phase_name} t={elapsed:5.2f}s "
                f"attn_x={attention_x:.4f} peak={peak:.5f} uniform_base={base_uniform:.5f}"
            )
            next_log = now + log_period

    print(f"[Phase] {phase_name} done.")


def run_left_right_ui_check() -> None:
    args = _parse_args()

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": False, "renderer": SIM_CONFIG.renderer})

    env = None
    try:
        from underwater_env import UnderwaterReachEnv

        instruction = args.instruction or VLA_CONFIG.instruction_text
        env = UnderwaterReachEnv(instruction_text=instruction)

        switched = _set_viewport_to_fish_camera(env.scene_cfg.camera_prim_path)
        if switched:
            print(f"[UI] Viewport switched to fish camera: {env.scene_cfg.camera_prim_path}")
        else:
            print(
                "[UI] Could not switch viewport automatically. "
                f"Please manually select camera: {env.scene_cfg.camera_prim_path}"
            )

        z = float(args.ball_z)
        x = float(args.ball_x)
        y = abs(float(args.ball_y))
        left_pos = np.array([x, y, z], dtype=np.float32)
        right_pos = np.array([x, -y, z], dtype=np.float32)

        print(f"[Check] instruction='{instruction}'")
        print(f"[Check] fish_start={ENV_CONFIG.fish_initial_position}")
        print(f"[Check] left={left_pos.tolist()} right={right_pos.tolist()}")

        _run_phase(env, "left", left_pos, seconds=float(args.left_seconds), log_hz=float(args.log_hz))
        _run_phase(env, "right", right_pos, seconds=float(args.right_seconds), log_hz=float(args.log_hz))

        end_time = time.time() + max(0.0, float(args.hold_seconds))
        print(f"[Check] finished. Holding UI for {args.hold_seconds:.1f}s...")
        while time.time() < end_time:
            env.world.step(render=True)
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    run_left_right_ui_check()
