"""Collect failed-seed DAgger labels and fine-tune the GRU student."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from imitation.bc import train_behavior_cloning
from imitation.collector import collect_labeled_episodes
from imitation.config import CollectionConfig, DaggerConfig, load_json_config
from imitation.policies import SB3PolicyAdapter, TorchStudentPolicyAdapter
from imitation.runtime import SimulationAppSession, build_underwater_env


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a one-shot hard-seed DAgger update for GRU.")
    parser.add_argument("--base-config", default="configs/imitation/dagger_gru_balanced.json")
    parser.add_argument("--failed-seeds-json", required=True)
    parser.add_argument(
        "--student-checkpoint",
        default="imitation_runs/dagger_gru_balanced/checkpoints/student_iter_08.pt",
    )
    parser.add_argument("--output-root", default="imitation_runs/dagger_gru_hard_seeds")
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=None)
    return parser.parse_args()


def _load_failed_seeds(path: str | Path) -> list[int]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_seeds = payload.get("failed_seeds") or payload.get("hard_case_seeds") or []
    seeds = [int(seed) for seed in raw_seeds]
    if not seeds:
        raise ValueError(f"No failed/hard-case seeds found in {path}.")
    return seeds


def main() -> None:
    args = _parse_args()
    config = load_json_config(args.base_config, DaggerConfig)
    failed_seeds = _load_failed_seeds(args.failed_seeds_json)

    output_root = Path(args.output_root)
    datasets_dir = output_root / "datasets"
    checkpoints_dir = output_root / "checkpoints"
    summaries_dir = output_root / "summaries"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    shard_path = datasets_dir / "failed_seed_teacher_labels.npz"
    student_checkpoint = str(args.student_checkpoint)

    teacher_policy = SB3PolicyAdapter(
        config.teacher_model_path,
        algorithm=config.teacher_algorithm,
        deterministic=config.deterministic_teacher,
    )
    student_policy = TorchStudentPolicyAdapter(
        student_checkpoint,
        deterministic=config.deterministic_student,
    )

    with SimulationAppSession(config.env):
        env = build_underwater_env(config.env)
        try:
            collection_config = CollectionConfig(
                env=config.env,
                teacher_model_path=config.teacher_model_path,
                teacher_algorithm=config.teacher_algorithm,
                output_path=str(shard_path),
                episodes=len(failed_seeds),
                start_seed=failed_seeds[0],
                deterministic_teacher=config.deterministic_teacher,
                teacher_view_name="fallback",
                student_view_name="clip",
                scenario_plan="seed_list",
                explicit_seeds=failed_seeds,
            )
            collect_result = collect_labeled_episodes(
                env,
                config=collection_config,
                teacher_policy=teacher_policy,
                student_policy=student_policy,
                output_path=str(shard_path),
                teacher_view_name="fallback",
                student_view_name="clip",
                episodes=len(failed_seeds),
                start_seed=failed_seeds[0],
                deterministic_teacher=config.deterministic_teacher,
                beta=float(args.beta),
                iteration_index=1,
            )
        finally:
            env.close()

        previous_dagger_shards = sorted(Path(config.output_root).glob("datasets/dagger_iter_*.npz"))
        dataset_paths = (
            list(config.seed_dataset_paths)
            + [str(path) for path in previous_dagger_shards]
            + [str(shard_path)]
        )
        output_checkpoint = checkpoints_dir / "student_hard_seed.pt"
        bc_config = replace(
            config.bc,
            epochs=int(args.epochs),
            learning_rate=(
                float(args.learning_rate)
                if args.learning_rate is not None
                else float(config.bc.learning_rate)
            ),
            output_path=str(output_checkpoint),
            checkpoint_dir=str(checkpoints_dir / "epoch_checkpoints"),
        )
        train_result = train_behavior_cloning(
            bc_config,
            dataset_paths=dataset_paths,
            output_path=str(output_checkpoint),
            resume_from=student_checkpoint,
            run_name_suffix="hard_seeds",
        )

        summary = {
            "base_config": str(args.base_config),
            "failed_seeds_json": str(args.failed_seeds_json),
            "student_checkpoint": student_checkpoint,
            "beta": float(args.beta),
            "failed_seeds": failed_seeds,
            "collection": collect_result,
            "training": train_result,
            "output_checkpoint": str(output_checkpoint),
        }
        summary_path = summaries_dir / "hard_seed_dagger_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
