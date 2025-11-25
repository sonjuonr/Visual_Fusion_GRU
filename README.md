# 🐟 MMR-PPO: Multi-Modal Recurrent RL for Underwater Navigation

**MMR-PPO (Multi-Modal Recurrent PPO)** is a deep reinforcement learning framework designed for the autonomous navigation and obstacle avoidance of a biomimetic robotic fish. 

This project leverages **Unity ML-Agents** to simulate a high-fidelity underwater environment. It implements an end-to-end learning architecture that fuses visual data (via **CNN**) and sensor data (via **Sonar/IMU**) into a Gated Recurrent Unit (**GRU**) memory network, trained using Proximal Policy Optimization (**PPO**).

![Unity Environment](path/to/your/screenshot_or_gif.gif) 
*(Note: Replace this line with a GIF or Screenshot of your fish swimming)*

---

## 🚀 Key Features

* **Multi-Modal Sensor Fusion**: Combines high-dimensional visual inputs (Camera) with low-dimensional vector observations (Ray Perception Sonar + IMU) for robust perception.
* **Temporal Memory (GRU)**: Utilizes a Gated Recurrent Unit to handle partially observable environments and capture the temporal dynamics of the fish's motion (inertia/drift).
* **End-to-End Learning**: Maps raw sensor data directly to continuous control signals (Thrust & Turn Torque).
* **Domain Randomization**: Implements randomized obstacle textures, positions, and target locations during training to improve Sim-to-Real generalization capability.
* **Physics-Based Control**: Simulation based on rigid-body dynamics with linear/angular drag modeling to mimic underwater hydrodynamics.

## 🧠 Network Architecture

The agent's "Brain" is constructed using PyTorch via ML-Agents:

1.  **Visual Stream**: 64x64 Grayscale Input $\rightarrow$ **SimpleCNN** (NatureCNN) $\rightarrow$ 128-dim Feature Vector.
2.  **Vector Stream**: 12-dim State (Velocity, Orientation, Target Dir) + Sonar Rays $\rightarrow$ **MLP** $\rightarrow$ 128-dim Feature Vector.
3.  **Fusion Layer**: Concatenation of Visual and Vector streams.
4.  **Memory Layer**: **GRU** (Seq Len: 64, Hidden Size: 128) for temporal context.
5.  **Policy Head**: **PPO** outputs continuous actions (Forward Force, Yaw Torque).

## 🛠️ Environment & Setup

* **Engine**: Unity 2022.3 LTS (URP Pipeline)
* **ML Framework**: Unity ML-Agents Release 20 / PyTorch
* **Language**: C# (Environment), Python 3.9 (Training)

### Observation Space (Vector: 12 + Sonar)
* Relative direction to target (Normalized Vector3)
* Agent's local orientation (Transform.up)
* Current Linear Velocity & Angular Velocity
* Ray Perception Sensor 3D (Simulated Sonar)

### Action Space (Continuous: 2)
* `Action[0]`: Swim Force (Forward Thrust)
* `Action[1]`: Turn Torque (Yaw Rotation)

## 📈 Training Strategy

* **Algorithm**: PPO (Proximal Policy Optimization)
* **Reward Shaping**:
    * `+10.0`: Reached Target.
    * `-10.0`: Collision with Obstacle/Wall.
    * `-1/MaxStep`: Time penalty to encourage efficiency.
    * `Distance Reward`: Dense reward based on proximity to target to guide early learning.
* **Curriculum**: Training starts with a simplified environment (fixed spawn) and progresses to fully randomized scenarios.

## 📦 Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/YourUsername/Visual-Fusion-GRU.git](https://github.com/YourUsername/Visual-Fusion-GRU.git)
    ```
2.  Open the project folder in **Unity Hub**.
3.  Install Python dependencies:
    ```bash
    conda create -n fish-ai python=3.9
    pip install mlagents torch
    # Fix protobuf/setuptools compatibility if needed
    pip install protobuf==3.20.3 setuptools==65.5.0
    ```
4.  Start Training:
    ```bash
    mlagents-learn config/fish_config.yaml --run-id=MMR_Run1 --force
    ```

## 📝 License

This project is open-sourced under the MIT License.
