from isaacsim import SimulationApp
# 必须最先运行，headless=False 可以让你看到训练窗口
simulation_app = SimulationApp({"headless": False}) 

import gymnasium as gym
import numpy as np
import torch
from omni.isaac.core import World
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.types import ArticulationAction
from stable_baselines3 import PPO  # 推荐使用 PPO 算法