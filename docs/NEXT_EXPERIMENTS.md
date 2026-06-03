# Next Experiments

## Current Read

The project is now past the first successful GRU imitation milestone. The
current best model is:

```text
models/student_gru_hard_cases_radius1_seed601000.pt
```

Result ladder:

| Checkpoint | Eval episodes | Success rate | Mean steps | Read |
| :--- | ---: | ---: | ---: | :--- |
| `models/student_bc_balanced.pt` | 20 | 0.150 | 496.2 | BC-only baseline; heavy left/right oscillation. |
| `imitation_runs/dagger_balanced/checkpoints/student_iter_04.pt` | 100 | 0.660 | 46.1 | Best MLP DAgger baseline. |
| `imitation_runs/dagger_gru_balanced/checkpoints/student_iter_08.pt` | 200 | 0.920 | 29.4 | Good short sweep, but not enough seeds. |
| `imitation_runs/dagger_gru_balanced/checkpoints/student_iter_08.pt` | 1000 | 0.883 | 31.3 | Broad GRU baseline; exposed 119 hard-case seeds. |
| `imitation_runs/dagger_gru_hard_cases_1000/checkpoints/student_hard_seed.pt` | 1000 | 0.942 | 79.1 | First hard-case fine-tune. |
| `imitation_runs/dagger_gru_hard_cases_1000/checkpoints/student_hard_seed.pt` | 1000 | 0.946 | 69.2 | Recheck on seed family `601000`. |
| `models/student_gru_hard_cases_radius1_seed601000.pt` | 1000 | 0.957 | 31.3 | Current best after radius-1 hard-case tuning. |

The most important result is not only the `95.7%` number. The important story
is how we got there: RL found the bottleneck, DAgger corrected the student in
states it actually visits, GRU memory reduced target-loss oscillation, and
hard-case mining focused training on the remaining decision failures.

## Training Data

Current best training data:

- Balanced teacher seed dataset: `360` episodes, `11720` labeled records.
- Balanced GRU DAgger: `8` shards, `320` rollout episodes, `26444` labeled
  records.
- Radius-1 hard-case shard: `299` targeted episodes, `35573` labeled records.
- Total records in the final fine-tune summary: `73737`.

The radius-1 hard-case shard came from expanding failed/slow seeds around the
`601000` seed-family evaluation. This is intentionally narrow; it should fix
known misses without replacing the broad balanced dataset.

## Immediate Commands

Re-evaluate the public current-best checkpoint:

```bash
./scripts/eval_student_release.sh
```

Re-run the broad GRU checkpoint sweep:

```bash
./scripts/eval_dagger_gru_students_200eps.sh
```

Rebuild the original hard-case seed list from the balanced GRU checkpoint:

```bash
./scripts/eval_iter08_1000eps_and_extract_hard_cases.sh
```

Run the one-shot hard-seed DAgger update:

```bash
./scripts/run_gru_hard_seed_dagger.sh
```

## Next Step

Use the latest hard-case report:

```text
imitation_data/eval/dagger_gru_hard_cases_radius1_seed601000_1000eps/student_hard_seed_hard_cases.json
```

It contains `48` hard cases: `43` failures and `5` slow successes. The next
experiment should collect a small teacher-labeled shard for these residual
seeds, train conservatively for `1` to `2` epochs, and evaluate on both:

- the same `601000` family, to check whether the hard cases improve;
- a wider unseen seed range, to check that the fix does not overfit.

Avoid returning to the RL alpha ladder as the primary route. The RL result is
valuable because it showed that even `0.007` fallback weight at `alpha=0.993`
can carry decisive control signal, but the current path should keep fallback
as a teacher and let the deployed student run from pure CLIP observations.
