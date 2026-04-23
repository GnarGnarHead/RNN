# Checkpoint Manifest

Checkpoints are local training artifacts and are ignored by git. This file records what the known local checkpoint names mean and how to verify the ones that matter.

## Current Best

### `checkpoints/kinder_ABCDEFG_alltasks_ctxn_v2.pt`

Status: current best retained kindergarten milestone.

Model shape:

- `d_model=32`
- `n_layers=2`
- `k_settle=2`
- vocab from `input.txt`

Demonstrated with state reset before each quiz:

- `copy`: `A -> A ... G -> G`
- `copy2`: `AB -> AB ... GA -> GA`
- `next`: `N:ABCDEFG:A:n -> B ... N:ABCDEFG:G:n -> A`
- `next2`: `N:ABCDEFG:AB:n -> C ... N:ABCDEFG:GA:n -> B`

Verify:

```bash
python3 scripts/exam_checkpoint.py \
  --text-path input.txt \
  --checkpoint checkpoints/kinder_ABCDEFG_alltasks_ctxn_v2.pt \
  --targets ABCDEFG \
  --tasks copy,copy2,next,next2
```

Expected result: `28/28 passed`.

Note: this checkpoint was trained and verified with legacy generation re-step behavior, which is the default. Keep `--restep-generate` enabled when comparing against this milestone.

## Useful Historical Milestones

### `checkpoints/kinder_ABCDE.pt`

Status: retained A-E letter copy.

Use this as a clean-ish base for re-running the move from simple copy to mixed copy/next tasks.

### `checkpoints/kinder_ABCDE_copy_next_v2.pt`

Status: retained A-E copy plus legacy next-letter prediction.

Notes:

- Verified copy: `A -> A ... E -> E`
- Verified legacy next prompts: `T:N:A S: -> B ... T:N:E S: -> A`
- Predates contextual next prompts like `T:N:ABCDE:A:n S:`

### `checkpoints/kinder_ABCDEFGH_alltasks_ctxn_trial.pt`

Status: failed H expansion trial, kept as a process-bug artifact.

What it showed:

- A-G retention stayed intact.
- H did not bind correctly.
- Root cause was scheduler starvation in the old REPL behavior, not a clean model-limit result.

Current code schedules sequential target order plus cyclic task order as a target x task cross-product, so this exact starvation bug should not recur.

## Retention Rule

Retention checks should reset recurrent state before each quiz. That measures weights, not working memory:

```json
{"cmd":"reset"}
{"cmd":"ingest","text":"T:A S:"}
{"cmd":"generate","max_new_tokens":1,"temperature":0.0}
```

## When Adding New Checkpoints

For each new milestone, record:

- checkpoint path
- base checkpoint, if any
- targets and tasks
- important REPL settings
- exact exam command
- pass/fail summary
- what regressed, if anything
