"""DAgger orchestration for the robotic fish imitation pipeline."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from imitation.bc import train_behavior_cloning
from imitation.collector import collect_labeled_episodes
from imitation.config import CollectionConfig, DaggerConfig, EvaluationConfig, save_json_config
from imitation.evaluation import _evaluate_policy_in_env
from imitation.policies import SB3PolicyAdapter, TorchStudentPolicyAdapter
from imitation.runtime import SimulationAppSession, build_underwater_env


def _beta_for_iteration(config: DaggerConfig, iteration_index: int) -> float:
    if int(config.total_iterations) <= 1:
        return float(config.beta_end)
    progress = float(iteration_index - 1) / float(max(1, int(config.total_iterations) - 1))
    beta = float(config.beta_start) + (float(config.beta_end) - float(config.beta_start)) * progress
    return float(max(0.0, min(1.0, beta)))


def run_dagger(config: DaggerConfig) -> dict[str, object]:
    output_root = Path(config.output_root)
    datasets_dir = output_root / "datasets"
    checkpoints_dir = output_root / "checkpoints"
    summaries_dir = output_root / "summaries"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)
    save_json_config(output_root / "dagger_config.json", config)

    dataset_paths = list(config.seed_dataset_paths)
    iteration_summaries: list[dict[str, object]] = []

    if config.initial_student_checkpoint is None:
        if not dataset_paths:
            raise ValueError(
                "DAgger requires either seed_dataset_paths or an initial_student_checkpoint."
            )
        seed_student_path = checkpoints_dir / "student_iter00_seed.pt"
        seed_train_result = train_behavior_cloning(
            replace(config.bc, output_path=str(seed_student_path)),
            dataset_paths=dataset_paths,
            output_path=str(seed_student_path),
            resume_from=None,
            run_name_suffix="iter00_seed",
        )
        current_student_checkpoint = str(seed_train_result["output_path"])
        iteration_summaries.append({"seed_bc": seed_train_result})
    else:
        current_student_checkpoint = str(config.initial_student_checkpoint)

    teacher_policy = SB3PolicyAdapter(
        config.teacher_model_path,
        algorithm=config.teacher_algorithm,
        deterministic=config.deterministic_teacher,
    )

    with SimulationAppSession(config.env):
        for iteration_index in range(1, int(config.total_iterations) + 1):
            beta = _beta_for_iteration(config, iteration_index)
            shard_path = datasets_dir / f"dagger_iter_{iteration_index:02d}.npz"
            student_policy = TorchStudentPolicyAdapter(
                current_student_checkpoint,
                deterministic=config.deterministic_student,
            )

            env = build_underwater_env(config.env)
            try:
                collection_config = CollectionConfig(
                    env=config.env,
                    teacher_model_path=config.teacher_model_path,
                    teacher_algorithm=config.teacher_algorithm,
                    output_path=str(shard_path),
                    episodes=int(config.rollout_episodes_per_iteration),
                    start_seed=int(config.start_seed)
                    + (iteration_index - 1) * int(config.rollout_episodes_per_iteration),
                    seed_stride=int(config.seed_stride),
                    deterministic_teacher=config.deterministic_teacher,
                    teacher_view_name="fallback",
                    student_view_name="clip",
                    scenario_plan=str(config.scenario_plan),
                    scenario_names=list(config.scenario_names),
                    runs_per_scenario=int(config.runs_per_scenario),
                    in_view_runs=int(config.in_view_runs_per_iteration),
                    out_of_view_runs=int(config.out_of_view_runs_per_iteration),
                )
                collect_result = collect_labeled_episodes(
                    env,
                    config=collection_config,
                    teacher_policy=teacher_policy,
                    student_policy=student_policy,
                    output_path=str(shard_path),
                    teacher_view_name="fallback",
                    student_view_name="clip",
                    episodes=int(config.rollout_episodes_per_iteration),
                    start_seed=int(config.start_seed) + (iteration_index - 1) * int(config.rollout_episodes_per_iteration),
                    deterministic_teacher=config.deterministic_teacher,
                    beta=beta,
                    iteration_index=iteration_index,
                )
            finally:
                env.close()

            dataset_paths.append(str(shard_path))
            next_student_path = checkpoints_dir / f"student_iter_{iteration_index:02d}.pt"
            bc_result = train_behavior_cloning(
                replace(config.bc, output_path=str(next_student_path)),
                dataset_paths=dataset_paths,
                output_path=str(next_student_path),
                resume_from=current_student_checkpoint if config.resume_student_from_previous else None,
                run_name_suffix=f"iter{iteration_index:02d}",
            )
            current_student_checkpoint = str(bc_result["output_path"])

            eval_result = None
            if int(config.evaluate_episodes) > 0:
                eval_policy = TorchStudentPolicyAdapter(
                    current_student_checkpoint,
                    deterministic=config.deterministic_student,
                )
                eval_env = build_underwater_env(config.env)
                try:
                    eval_result = _evaluate_policy_in_env(
                        eval_env,
                        eval_policy,
                        EvaluationConfig(
                            env=config.env,
                            policy_type="torch_student",
                            policy_path=current_student_checkpoint,
                            observation_view_name="clip",
                            episodes=int(config.evaluate_episodes),
                            start_seed=int(config.start_seed) + 100_000 + iteration_index * 1_000,
                            deterministic=config.deterministic_student,
                            output_path=str(summaries_dir / f"eval_iter_{iteration_index:02d}.json"),
                        ),
                    )
                finally:
                    eval_env.close()

            iteration_summaries.append(
                {
                    "iteration_index": int(iteration_index),
                    "beta": float(beta),
                    "collection": collect_result,
                    "bc": bc_result,
                    "evaluation": eval_result,
                }
            )

    summary = {
        "output_root": str(output_root),
        "final_student_checkpoint": current_student_checkpoint,
        "dataset_paths": dataset_paths,
        "iteration_summaries": iteration_summaries,
    }
    (summaries_dir / "dagger_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
