# Checkpoint Lineage

This file records the conceptual lineage of important local checkpoints. The
checkpoint files themselves are local artifacts and ignored by git.

## Current Lineage

```text
checkpoints/kinder_ABCDE.pt
  -> checkpoints/kinder_ABCDE_copy_next_v2.pt
  -> checkpoints/kinder_ABCDEFG_alltasks_ctxn_v2.pt
  -> runs/abcdefgh_targeted_try_local_v5.pt
  -> runs/guarded_h_full_from_v5_strict_run2/best_rejected.pt
  -> runs/guarded_h_diagnostic_ef_fixed_fg_regressed/best.pt
  -> runs/guarded_h_from_fg_regression_strict_run1/best.pt
  -> checkpoints/kinder_ABCDEFGH_alltasks_ctxn_v3.pt
```

## Current Best

`checkpoints/kinder_ABCDEFGH_alltasks_ctxn_v3.pt`

Verification:

```bash
.venv/bin/python scripts/exam_checkpoint.py --text-path input.txt \
  --checkpoint checkpoints/kinder_ABCDEFGH_alltasks_ctxn_v3.pt \
  --targets ABCDEFGH --tasks copy,copy2,next,next2
```

Expected result:

```text
Exam summary: 32/32 passed
```

## Important Diagnostic Branches

`runs/guarded_h_full_from_v5_strict_run2/best_rejected.pt`

- Result: `31/32`
- Failure: `next2:EF -> F`, expected `G`
- Use: near-miss showing original last-mile failure

`runs/guarded_h_diagnostic_ef_fixed_fg_regressed/best.pt`

- Result: `31/32`
- Failure: `next2:FG -> A`, expected `H`
- Use: diagnostic state showing shifted failure after repairing `EF -> G`
