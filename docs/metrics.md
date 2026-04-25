# Metrics

The project currently records enough to promote checkpoints, but not enough to
fully analyze learning dynamics. These are the metrics we should keep or add.

## Current Metrics

- pass/fail per exam item
- partial score per exam item
- average loss for tutor learning calls
- train steps and train tokens
- `delta_per_k`
- `logits_entropy`
- `state_norm`
- regressions against previously passing items
- per-item logit margin diagnostics from `generate_with_trace`
- top-k alternatives for each generated exam token
- failure transitions in guarded runner candidate events

## Add Next

### Candidate Phase Region

For sweeps, summarize each candidate as one of:

- unchanged
- improved but not clean
- shifted failure
- clean repair
- collapse/regression

Then compare regions over `lr`, `steps`, `focus`, and `failure_repeats`.

### Representation Snapshots

For selected checkpoints, save hidden states/logits for every exam prompt.
Useful later analyses:

- PCA of prompt states
- CKA or cosine similarity across seeds
- nearest-neighbor structure among copy/next/next2 prompts
- before/after repair deltas in activation space

## Promotion Metrics

Promotion still requires exact full-exam pass. Margins and representation
metrics are diagnostic, not substitutes for the gate.
