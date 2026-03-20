# 🐟 Visual-Fusion-GRU: SLA-Aware Underwater VLA Navigation

![Work In Progress](https://media.giphy.com/media/13HgwGsXF0aiGY/giphy.gif)

> **Status:** Phase A (Late Stage) — Fallback validation complete; focusing on search stability during target loss.

**Visual-Fusion-GRU** is a Deep Reinforcement Learning framework designed for autonomous robotic fish navigation in challenging underwater environments (e.g., high turbidity, dynamic lighting). The core of this project is a **VLA (Vision-Language-Action)** architecture that bridges high-level semantic understanding with low-level physical control.

---

## 🎥 Simulation Demos

### 1. Latest Progress: Fallback (Color Saliency) Mode Performance
In this video, the robotic fish uses a 14x14 color saliency heatmap as its primary observation.
*   **Performance:** Precise turning and tracking when the target is within the Field of View (FoV).
*   **Pain Point:** When the target leaves the FoV, the policy tends to oscillate (shaking the "head" left/right), falling into a local dead loop.
[**👉 Click here to watch: Fallback Tracking Demo (MP4)**](https://github.com/sonjuonr/Visual_Fusion_GRU/blob/main/Desktop%202026.03.20%20-%2012.49.03.05_compressed.mp4)

### 2. Initial Environment Validation
[**👉 Click here to watch: Early Environment Setup (MP4)**](https://github.com/sonjuonr/Visual_Fusion_GRU/blob/main/fixed_video_final.mp4)

---

## ⏳ Training Evolution Timeline

| Phase | Strategy | Observation Input (Obs) | Status & Feedback |
| :--- | :--- | :--- | :--- |
| **Initial** | Pure CLIP Semantic Training | Pure CLIP (ViT-B16) 196d | **Failed**. Semantic signals lacked discriminative power; policy jittered/failed to converge. |
| **Phase A (Current)** | Fallback Guided Training | Color Saliency Heatmap | **Functional but unstable**. Basic tracking achieved; search logic outside FoV remains weak. |
| **Phase B (Next)** | Hybrid Fusion Training | $\alpha \cdot CLIP + (1-\alpha) \cdot Fallback$ | **Planned**. Implementing smooth feature migration via alpha-weight annealing. |
| **Phase C (Final)** | Full Semantic VLA Navigation | Pure CLIP Heatmap | **Ultimate Goal**. Achieving robust, fully semantic-aligned underwater navigation. |

---

## 🧠 System Architecture

The architecture has evolved from traditional CNN encoders to an **Attention-Based Heatmap Fusion** design, utilizing temporal memory (GRU) to handle frequent visual target loss.

### Data Flow Pipeline

```mermaid
graph TD
    subgraph Environment [NVIDIA Isaac Sim]
        IMG["Camera RGB (224x224)"]
        TXT["Instruction: 'Find Red Ball'"]
        RADAR["Emergency Radar (16d LiDAR)"]
    end

    subgraph Perception [VLM Orchestrator]
        CLIP[CLIP ViT-B16 Encoder]
        HEAT["Semantic Heatmap (14x14)"]
    end

    subgraph Policy [Recurrent PPO]
        FUSE["Alpha Fusion (CLIP + Fallback)"]
        GRU[GRU Memory Cell]
        ACT{"Discrete Action Head"}
    end

    IMG & TXT --> CLIP --> HEAT
    HEAT --> FUSE
    FUSE --> GRU --> ACT
    
    ACT -->|0: Forward | Environment
    ACT -->|1: Turn Left| Environment
    ACT -->|2: Turn Right| Environment
    
    RADAR -.->|SLA Trigger: Low Vision| ACT
