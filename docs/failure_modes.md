# Failure Modes

This file separates observed failures from inferred explanations.

## Attractor Collapse

Observation: after overtraining a new item, the model may answer the same
character for many prompts.

Likely cause: too much focused learning without enough maintenance rehearsal.

Mitigation:

- reduce learning rate
- increase broad maintenance
- reset and run the full exam after every candidate

## Scheduler Starvation

Observation: an old A-H attempt did not practice the actually failing H pairs.

Cause: sequential target order plus cyclic task order previously failed to cover
the target x task cross-product.

Mitigation:

- use the fixed flattened scheduler
- verify coverage in tests
- inspect exact practiced prompts when results look unchanged

## Shifted Failure

Observation: repairing `next2:EF -> G` shifted the failure to
`next2:FG -> H`.

Inference: neighboring successor tasks interfere and should be treated as a
coupled system.

Mitigation:

- save near misses
- repair shifted failures under the full exam
- log failure transitions as a graph

## Restep Misalignment

Observation: exact replay as `S:answer` did not repair the legacy exam item.

Cause: legacy generation re-steps the final `:` before sampling, so the exam is
closer to `S::answer`.

Mitigation:

- use `--align-restep-training` while `--restep-generate` is enabled
- document decode mode in every experiment

## Partial Promotion

Observation: `31/32` checkpoints can look convincing while hiding live
interference.

Cause: pass count alone misses shifted regressions and margin instability.

Mitigation:

- promote only clean full-exam checkpoints
- preserve best rejected checkpoints as diagnostics
- record failed items and shifted failures
