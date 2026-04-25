# Exam Protocol

This protocol defines what counts as a retained tutor milestone.

## Purpose

The exam measures behavior stored in weights, not transient recurrent state. Every
exam item must reset state before ingesting its prompt.

## Default Command

```bash
.venv/bin/python scripts/exam_checkpoint.py \
  --text-path input.txt \
  --checkpoint <checkpoint.pt> \
  --targets <TARGETS> \
  --tasks copy,copy2,next,next2
```

For current checkpoints, leave legacy restep generation enabled. That is the
default. Use `--no-restep-generate` only when explicitly comparing decode modes.

For diagnostic runs, use JSONL and keep the default top-5 logit trace:

```bash
.venv/bin/python scripts/exam_checkpoint.py \
  --text-path input.txt \
  --checkpoint <checkpoint.pt> \
  --targets <TARGETS> \
  --tasks copy,copy2,next,next2 \
  --jsonl --trace-top-k 5
```

## Passing Criteria

- All requested exam items must pass exactly.
- State must reset before every item.
- Temperature must be `0.0`.
- The checkpoint path, targets, tasks, and result must be recorded.

Partial scores are diagnostic. They do not promote a checkpoint.

## Standard Gates

- A-G with four tasks: `28/28`.
- A-H with four tasks: `32/32`.
- A-I with four tasks: `36/36`.

## Record These Details

- checkpoint path
- source/base checkpoint
- exact exam command
- target set and task set
- generation mode (`restep_generate`)
- failed items, if any
- minimum margin and weakest items
- pass count and total
- any qualitative notes about confidence, margins, or shifted failures
