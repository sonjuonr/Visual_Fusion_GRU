# Project Structure

```text
Visual_Fusion_GRU_github/
  README.md
  requirements.txt
  src/                         Core environment, VLA, RL, and IL code
  src/imitation/               BC, DAgger, dataset, policy, evaluation modules
  configs/imitation/           Teacher, BC, DAgger, and evaluation configs
  scripts/                     Historical and current launch scripts
  models/                      Compact trained model outputs and run configs
  monitor_logs/                RL and teacher monitor CSV records
  imitation_data/              Seed teacher datasets and evaluation summaries
  imitation_runs/              DAgger datasets, checkpoints, summaries
  artifacts_manifest/          Included file list and excluded checkpoint list
  docs/                        Upload and structure notes
```

The full Isaac Sim installation is not part of this folder. Keep this package
as the public project repo, and keep the Isaac Sim workspace as the local
runtime environment.

