# 🐟 Visual-Fusion-GRU: Attention-Based Underwater Navigation

![Work In Progress](https://media.giphy.com/media/13HgwGsXF0aiGY/giphy.gif)

> **Status:** Active Development (Phase 2: Network Architecture Implementation)

**Visual-Fusion-GRU** is a deep reinforcement learning framework designed for autonomous robotic fish navigation in turbid underwater environments. 

It addresses the challenge of **sensor reliability**—where visual data degrades due to turbidity (fog) and sonar data lacks context—by implementing an **Adaptive Gating Mechanism**. This mechanism dynamically weights the importance of Visual (CNN) and Vector (Sonar/IMU) inputs before feeding them into a Temporal Memory (GRU) module.

---

## 🎥 Simulation Demo

Current environment setup in isaac-sim, showcasing the **Robot Fish's POV (Camera Sensor)** , **perspective POV with lidar** and the **physically simulated underwater environment**.

[**👉 Click here to watch the Initial Simulation Video**]([https://github.com/sonjuonr/Visual_Fusion_GRU/blob/main/initial%20simulation%202025-11-30_013224_200.mp4](https://github.com/sonjuonr/Visual_Fusion_GRU/blob/main/initial%20simulation%202025-11-30_013224_200.mp4))

*(The video demonstrates the agent passing an obstacle。)*

---
## Small test in simulation environment
mission decription: An underwater capsule robot learns to navigate toward a target object 5 meters away by processing 360-degree LiDAR scan data and relative goal coordinates to output continuous 2D movement actions using the PPO algorithm.  

[**👉 here is the RL Simulation Gif**]([[**👉 Click here to watch the Initial Simulation Video**](https://github.com/sonjuonr/Visual_Fusion_GRU/blob/main/1.gif)

## 🧠 Proposed System Architecture

The core innovation of this project is the **Multi-Modal Gating Network**. Instead of simple concatenation, the model learns *when* to trust its eyes (Camera) and *when* to trust its sensors (Sonar).

### Data Flow Pipeline

```mermaid
graph TD
    subgraph Environment [Unity URP Environment]
        IMG["Camera Input (64x64 Grayscale)"]
        VEC["Vector Input (Sonar + State)"]
    end

    subgraph Encoders [Feature Extraction]
        CNN[SimpleCNN Encoder]
        MLP[Vector Encoder]
    end

    subgraph Gating [Adaptive Attention Mechanism]
        ATTN{Gating Network}
        WEIGHT["Alpha (α)"]
    end

    subgraph Memory_Policy [Temporal Decision]
        FUSE[Weighted Fusion]
        GRU[GRU Memory Cell]
        ACT["Action Head (PPO)"]
    end

    IMG --> CNN --> FUSE
    VEC --> MLP --> FUSE
    
    %% The Gating Logic
    MLP --> ATTN
    CNN --> ATTN
    ATTN -->|Calculates Uncertainty| WEIGHT
    WEIGHT -.->|Modulates| FUSE
    
    FUSE --> GRU --> ACT -->|Thrust & Torque| Environment
