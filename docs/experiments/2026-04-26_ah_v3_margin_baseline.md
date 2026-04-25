# Experiment: A-H v3 Margin Baseline

Date: 2026-04-26

## Question

What is the logit-margin profile of the promoted A-H `32/32` checkpoint before
attempting A-I?

## Setup

- Checkpoint: `checkpoints/kinder_ABCDEFGH_alltasks_ctxn_v3.pt`
- Targets: `ABCDEFGH`
- Tasks: `copy,copy2,next,next2`
- Device: CPU
- Generation mode: legacy restep generation enabled
- Trace: top-5 logits per generated exam token

## Command

```bash
.venv/bin/python scripts/exam_checkpoint.py --text-path input.txt \
  --checkpoint checkpoints/kinder_ABCDEFGH_alltasks_ctxn_v3.pt \
  --targets ABCDEFGH --tasks copy,copy2,next,next2 \
  --jsonl --trace-top-k 5 > runs/ah_v3_margin_baseline/exam.jsonl
```

## Result

- Exam items: `32`
- Failures: `0`
- Minimum margin: `0.176428`
- Average margin: `4.044073`
- Artifact: `runs/ah_v3_margin_baseline/exam.jsonl`

## Weakest Items

```text
next:N:ABCDEFGH:H:n -> A, margin=0.176428
next2:N:ABCDEFGH:GH:n -> A, margin=0.568429
next2:N:ABCDEFGH:FG:n -> H, margin=0.599022
copy2:FG -> FG, margin=1.026517
copy2:HA -> HA, margin=1.194086
```

## Interpretation

The checkpoint is behaviorally clean, but the weakest margins cluster around the
successor wraparound and the late-sequence `next2` items. That makes A-I risky:
adding `I` is likely to disturb the same successor boundary unless training and
promotion explicitly watch margins, not just pass/fail.
