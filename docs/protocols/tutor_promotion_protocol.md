# Tutor Promotion Protocol

The tutor may explore partial improvements, but it may only promote clean,
non-regressing checkpoints.

## Promotion Rule

A candidate is promoted only when:

- it passes the full requested reset-based exam
- it introduces no regressions against previously passing exam items
- its training command and exam result are recorded

Near misses are valuable diagnostic artifacts, but they are not milestones.

## Guarded Runner Pattern

Use `scripts/tutor_guarded_runner.py` for noninteractive repair searches:

```bash
.venv/bin/python scripts/tutor_guarded_runner.py \
  --text-path input.txt \
  --checkpoint <base.pt> \
  --out-dir runs/<run_name> \
  --targets <TARGETS> \
  --tasks copy,copy2,next,next2 \
  --save-best-rejected
```

Use `--save-best-rejected` so strict promotion does not discard informative
near misses.

## Restep Alignment

Current exams use legacy restep generation by default. When repairing under that
exam mode, align supervised examples with the effective prompt by adding:

```bash
--align-restep-training
```

This trains examples as `T:<prompt> S::<answer>`, matching the extra final-token
step used during generation.

## Failure Replay

Use exact failure replay for sharp last-mile repairs:

```bash
--failure-repeats 512
```

This should be paired with the full exam gate. Exact replay without a full
regression exam can overfit the failed item and damage nearby items.

## Required Notes

Each promoted or diagnostic run should record:

- question
- hypothesis
- base checkpoint
- command
- accepted candidate, if any
- full exam result
- failure transitions
- interpretation
- alternative explanations
- next action
