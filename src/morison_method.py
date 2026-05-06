import numpy as np
from omni.isaac.core.utils.rotations import quat_to_rot_matrix
from omni.isaac.core.articulations import ArticulationView
import parameter as p

class MorisonHydrodynamics:
    def __init__(self, rho=p.rho, cd_normal= current_cd, cm = p.c_m):
        self.rho = rho          # 水的密度 (kg/m^3)
        self.cd_n = cd_normal   # 法向阻力系数 (类似平板受力)
        self.cm = cm            # 附加质量系数
    
    def get_dynamic_cd(v_total, n_world):
    # 1. 计算速度方向与法线的夹角余弦值
    # v_total 是切片总速度，n_world 是鳍面法线
        cos_theta = np.dot(v_total, n_world) / (np.linalg.norm(v_total) + 1e-6)
        sin_theta = np.sqrt(1 - cos_theta**2)
    
    # 2. 插值计算
        cd_min = 0.02
        cd_max = 1.2
    # 使用 sin 平方确保在 0-90 度之间平滑过渡
        current_cd = cd_min + (cd_max - cd_min) * (sin_theta**2)
        return current_cd

    def apply_link_forces(self, fish_view, link_index, link_length, num_strips):
        """
        link_index: 0 为 base_link, 1 为 tail_link
        link_length: 24cm 鱼的对应部分长度
        num_strips: 建议尾巴切 10-15 份
        """
        # 1. 获取该 Link 在世界坐标系下的线速度、角速度和姿态
        # velocities shape: (num_envs, num_links, 6) -> [v_x, v_y, v_z, w_x, w_y, w_z]
        all_vels = fish_view.get_velocities()
        link_vel = all_vels[0, link_index]  # 假设只有一只鱼
        
        # 获取旋转矩阵 (用于将力从局部转到世界空间)
        poses = fish_view.get_world_poses()
        quat = poses[1][0] # 姿态四元数
        rot_matrix = quat_to_rot_matrix(quat)

        v_world = link_vel[:3]   # 世界坐标系线速度
        w_world = link_vel[3:]   # 世界坐标系角速度

        total_force_world = np.zeros(3)
        total_torque_world = np.zeros(3)

        # --- 开始切片循环 (Slicing Loop) ---
        strip_thickness = p.l_body / p.n1
        
        for i in range(num_strips):
            # A. 定位切片 (Locating the strip)
            # 假设 Link 沿局部 X 轴延伸。r_local 是切片相对于 Link 原点的位移
            r_local = np.array([(i + 0.5) * strip_thickness, 0, 0])
            # 转到世界坐标系的位移向量
            r_world = rot_matrix @ r_local

            # B. 计算该切片的实时速度 (Linear velocity of the strip point)
            # 公式: v_strip = v_link + w x r
            v_strip_world = v_world + np.cross(w_world, r_world)

            # C. 计算莫里森受力 (Morison Equation)
            # 我们主要计算垂直于摆动平面的力。假设主要摆动在 XY 平面。
            # 这里简化演示：只取速度的模长进行二次型阻力计算
            speed_sq = np.linalg.norm(v_strip_world) * v_strip_world
            
            # 这里的面积 A 需要根据你鱼的几何形状动态确定
            # 假设每段切片的高度是 0.1m
            area_strip = 0.1 * strip_thickness 
            
            # F_drag = 0.5 * rho * Cd * A * v * |v|
            f_drag = -0.5 * self.rho * self.cd_n * area_strip * speed_sq
            
            # D. 累加力与力矩 (Summing up)
            total_force_world += f_drag
            total_torque_world += np.cross(r_world, f_drag)

        # 3. 将最终结果施加给 Isaac Sim
        # indices 对应当前的鱼，forces 和 torques 需要是世界坐标系
        fish_view.apply_forces(total_force_world, indices=[link_index])
        fish_view.apply_forces_and_torques_at_poses(
            forces=total_force_world.reshape(1, 3),
            torques=total_torque_world.reshape(1, 3),
            indices=[link_index]
        )