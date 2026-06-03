# Current Best Student

Current public checkpoint:

```text
models/student_gru_hard_cases_radius1_seed601000.pt
```

Original run checkpoint:

```text
imitation_runs/dagger_gru_hard_cases_radius1_seed601000/checkpoints/student_hard_seed.pt
```

The two files have the same SHA-256:

```text
1d69f4c51092f10fa3b2ca7f37242a2646a3f1c290c376b64040282cad1964a2
```

Reason:

- Best current long evaluation.
- `success_rate = 0.957` over `1000` pure-CLIP episodes.
- `mean_steps = 31.341`.
- `mean_reward = 56.392`.
- Action histogram:

```json
{
  "0": 17409,
  "1": 2404,
  "2": 11528
}
```

Evidence:

- Eval summary:
  `imitation_data/eval/dagger_gru_hard_cases_radius1_seed601000_1000eps/student_hard_seed_eval_1000eps.json`
- Hard-case report:
  `imitation_data/eval/dagger_gru_hard_cases_radius1_seed601000_1000eps/student_hard_seed_hard_cases.json`
- Training summary:
  `imitation_runs/dagger_gru_hard_cases_radius1_seed601000/summaries/hard_seed_dataset_train_summary.json`

Previous baselines:

```text
imitation_runs/dagger_gru_balanced/checkpoints/student_iter_08.pt
```

- `success_rate = 0.883` over `1000` episodes.
- `success_rate = 0.92` over `200` episodes.
- Balanced GRU DAgger baseline before failed-seed/radius-1 hard-case tuning.

```text
imitation_runs/dagger_balanced/checkpoints/student_iter_04.pt
```

- `success_rate = 0.66` over `100` episodes.
- Best MLP DAgger baseline.

Caution:

The current best is strong but not solved. The latest hard-case report contains
`43` failures and `5` slow successes over `1000` episodes. Future work should
mine those residual cases while checking that hard-case fine-tuning does not
regress the broader balanced benchmark.
