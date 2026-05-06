# GitHub Upload Notes

This folder is the GitHub-ready package for the Visual-Fusion-GRU project.
It is intentionally curated from the full Isaac Sim workspace.

Included:

- Project README and current research story.
- Source code under `src/`.
- Training, evaluation, and launch scripts under `scripts/`.
- Imitation configs under `configs/imitation/`.
- Compact trained artifacts under `models/`.
- Imitation datasets and DAgger run outputs under `imitation_data/` and
  `imitation_runs/`.
- Monitor logs and run summaries needed to support the RL choke-point finding.
- File manifests under `artifacts_manifest/`.

Not included:

- The full rolling `my_projects/checkpoints/` directory. It is about `891M`
  and mostly contains intermediate RL snapshots. The manifest is preserved in
  `artifacts_manifest/excluded_checkpoints.tsv`.
- TensorBoard event directories. They are useful locally but noisy for GitHub.
- MP4 demos. The README links to the existing GitHub-hosted demo files.

Recommended upload strategy:

1. Create a fresh GitHub repository from this folder.
2. Commit this curated package first.
3. Upload large checkpoint snapshots as GitHub Release assets or manage them
   with Git LFS if you really want them versioned.
4. Choose a license before making the repo public.

Useful commands:

```bash
cd Visual_Fusion_GRU_github
git init
git add .
git commit -m "Prepare Visual-Fusion-GRU project release"
```

Release helper scripts:

```bash
./scripts/run_dagger_release.sh
./scripts/eval_student_release.sh
```

Runtime note:

The code expects NVIDIA Isaac Sim / Omniverse Python APIs to be available.
The `requirements.txt` file lists only the regular Python packages; it does
not install Isaac Sim itself.
