# Progress log (research notes)

This file records **what has been trained, what worked, and what broke** so future agents (and future us) can pick up without re-discovering the same failure modes.

## What the task is

We are treating the model like a kindergartener:

- Start from **random weights**
- Teach **mimicry** (letter → same letter)
- Use that mimicry to bootstrap toward **digrams → words → sentences**
- Keep learning **semi-permanent** (weights persist), and **verify retention** frequently

The goal is not scale/performance; it’s understanding the dynamics of **feedback + structured lessons** without attention.

## How we teach (current method)

Lesson format (next-token char LM):

- Training example: `T:A S:A`
- Quiz prompt: `T:A S:` then generate 1 char

Important implementation details:

- Training uses **answer-only loss masking**: we only backprop on tokens after `S:`. This avoids an impossible objective when prompts vary.
- Lessons **do not end with `\n`** (so we don’t accidentally teach newline during the “letters” phase).
- Retention checks reset recurrent state each quiz (`{"cmd":"reset"}`), so we measure **weights**, not working memory.
- Legacy-compatible exams use `restep_generate=true`, which steps the final `:` once more before decoding. Last-mile repairs should use restep-aligned training examples (`S::answer`), now exposed by `scripts/tutor_guarded_runner.py --align-restep-training`.

## Known dynamics so far

- **Catastrophic forgetting / single-attractor**: if we train only the newest letter (e.g. only `D`), the model often starts answering `D` for everything.
- **Fix direction**: rehearsal is necessary, but plain interleaving is too crude. New targets need **maintenance bundles**: direct lessons, old mimicry, nearby copy/digram contrasts, prediction contrasts, and known confusions.
- **Learning rate**: aggressive `lr` learns fast but overwrites. As the target set grows, a lower `lr` with more steps is safer.
  - Early (A–C): `lr ≈ 3e-3` worked
  - Adding D/E: `lr ≈ 1e-3` was more stable for retention
- **BPTT vs detach**: for prompt-conditioned copy tasks, we typically want `detach_state=false` during `learn` (BPTT through the short example). Detaching can make it fail to bind the prompt to the answer.
- **Tutor role correction (2026-04-26)**: mimicry is not just the first phase. It is the permanent stabilizer. The tutor should detect forgetting, repair it dynamically, and only promote checkpoints after non-regressing retention exams.
- **Promotion rule**: partial improvement is diagnostic only. A final A-H checkpoint for `copy,copy2,next,next2` must pass `32/32` after reset; anything short of that remains a failed/diagnostic artifact.
- **A-H last-mile dynamic**: direct repair of `next2:EF -> G` can shift the single failure to `next2:FG -> H`. The successful run alternated through that interference instead of treating `31/32` as acceptable.
- **Margin baseline**: the clean A-H v3 checkpoint passes `32/32`, but the thinnest margins are successor wraparound items (`H -> A`, `GH -> A`, `FG -> H`). Future expansion should watch margins before pass/fail breaks.

## Milestones (retention)

### A–E letter mimicry achieved

Achieved retention on A–E using:

- `d_model=32`, `n_layers=2`, `k_settle=2`
- `loss_mode="answer_only"`, `detach_state=false` during learning
- rehearsal-weighted examples and `lr=1e-3` when adding D/E

Checkpoint (local): `checkpoints/kinder_ABCDE.pt`

Verification command (JSONL session):

```bash
.venv/bin/python session.py --text-path input.txt --checkpoint checkpoints/kinder_ABCDE.pt --d-model 32 --n-layers 2
```

Then send (one per line):

```json
{"cmd":"reset"}
{"cmd":"ingest","text":"T:A S:"}
{"cmd":"generate","max_new_tokens":1,"temperature":0.0}
```

Repeat for `B`, `C`, `D`, `E`.

### A–E copy + next-letter prediction achieved

We mixed “copy” and “predict next” *slowly* without conflicting supervision.

Rule: **never** train both `T:A S:A` and `T:A S:B` (same prompt, different answer). Instead tag prediction prompts:

- copy: `T:A S:A`
- next (legacy): `T:N:A S:B`
- next2 (legacy): `T:N:AB S:C`
- next (current): `T:N:<alphabet>:A:n S:B`
- next2 (current): `T:N:<alphabet>:AB:n S:C`

Including `<alphabet>` in the prompt avoids contradictory supervision as the tutor expands the known set (A–D vs A–E vs A–G): the prompt string changes when the rule changes.

Checkpoint (local): `checkpoints/kinder_ABCDE_copy_next_v2.pt`

Verified retention (reset before each quiz):

- copy: `A→A, B→B, C→C, D→D, E→E`
- next (legacy prompts): `N:A→B, N:B→C, N:C→D, N:D→E, N:E→A`

Verification command:

```bash
.venv/bin/python session.py --text-path input.txt --checkpoint checkpoints/kinder_ABCDE_copy_next_v2.pt --d-model 32 --n-layers 2
```

Then send (one per line):

```json
{"cmd":"reset"}
{"cmd":"ingest","text":"T:N:A S:"}
{"cmd":"generate","max_new_tokens":1,"temperature":0.0}
```

Notes:

- `checkpoints/kinder_ABCDE_copy_next.pt` was an earlier mixed checkpoint where `B`-copy drifted to `C`; it’s kept only as a “what can break” artifact.
- The stepper later switched prediction prompts to include `<alphabet>` (e.g. `T:N:ABCDE:A S:`) to avoid contradictory supervision as targets expand; `kinder_ABCDE_copy_next_v2.pt` predates that change.

The stepper supports mixing via:

```bash
python3 scripts/tutor_stepper.py --text-path input.txt --targets ABCDE --checkpoint checkpoints/kinder_ABCDE.pt --tasks copy,next --task-order cycle
```

### A–G letters + digrams + prediction

Checkpoint (local): `checkpoints/kinder_ABCDEFG_alltasks_ctxn_v2.pt`

Demonstrated (retention, reset before each quiz):

- copy: `A→A ... G→G`
- digram copy (copy2): `AB→AB, BC→BC, ... GA→GA`
- next (contextual + local marker): `N:ABCDEFG:A:n→B, ... N:ABCDEFG:G:n→A`
- next2 (contextual + local marker): `N:ABCDEFG:AB:n→C, ... N:ABCDEFG:GA:n→B`

Quick verify:

```bash
.venv/bin/python scripts/tutor_stepper.py --text-path input.txt --checkpoint checkpoints/kinder_ABCDEFG_alltasks_ctxn_v2.pt --targets ABCDEFG --tasks copy,copy2,next,next2
```

Then run `exam`.

### A–H letters + digrams + prediction (current best)

Checkpoint (local): `checkpoints/kinder_ABCDEFGH_alltasks_ctxn_v3.pt`

Demonstrated (retention, reset before each quiz):

- copy: `A->A ... H->H`
- digram copy (copy2): `AB->AB ... HA->HA`
- next: `N:ABCDEFGH:A:n->B ... N:ABCDEFGH:H:n->A`
- next2: `N:ABCDEFGH:AB:n->C ... N:ABCDEFGH:HA:n->B`

Final verification:

```bash
.venv/bin/python scripts/exam_checkpoint.py --text-path input.txt \
  --checkpoint checkpoints/kinder_ABCDEFGH_alltasks_ctxn_v3.pt \
  --targets ABCDEFGH --tasks copy,copy2,next,next2
```

Expected result: `32/32 passed`.

Margin baseline:

```bash
.venv/bin/python scripts/exam_checkpoint.py --text-path input.txt \
  --checkpoint checkpoints/kinder_ABCDEFGH_alltasks_ctxn_v3.pt \
  --targets ABCDEFGH --tasks copy,copy2,next,next2 \
  --jsonl --trace-top-k 5 > runs/ah_v3_margin_baseline/exam.jsonl
```

Result: `32/32`, minimum margin `0.176428`, average margin `4.044073`.
Weakest item: `next:N:ABCDEFGH:H:n -> A`.

Important route:

- Base clean A-G: `checkpoints/kinder_ABCDEFG_alltasks_ctxn_v2.pt`.
- Best diagnostic A-H near miss: `runs/guarded_h_full_from_v5_strict_run2/best_rejected.pt`, `31/32`, failing `next2:EF -> G` by copying `F`.
- Restep-aligned exact replay (`--align-restep-training --failure-repeats 512`) fixed `EF -> G` but moved the failure to `next2:FG -> H`.
- A second strict repair from that shifted failure found a clean candidate:
  - run: `runs/guarded_h_from_fg_regression_strict_run1`
  - accepted candidate: `steps=20`, `lr=2e-05`, `focus=1`
  - result: `32/32`, no regressions

### ABCDEFGH trial attempt (note: did not teach `H` yet)

Checkpoint artifact (local): `checkpoints/kinder_ABCDEFGH_alltasks_ctxn_trial.pt`

What happened:

- A–G retention stayed perfect.
- `H` *did not* bind correctly:
  - `copy:'H'` answered `'F'`
  - `copy2:'GH'` answered `'GD'`
  - `next:'...:F:n'` answered `'A'` (expected `'G'`)
  - `next2:'...:FG:n'` answered `'A'` (expected `'H'`)

Root cause (process bug, not necessarily a model limitation):

- The run used `--order sequential` + `--task-order cycle` with `focus GH`.
- With 2 focused targets and 2 tasks (`copy,copy2`), the REPL repeatedly paired them as:
  - `G` always got `copy`
  - `H` always got `copy2`
- That meant the operator never actually practiced the failing pairs (`copy:H`, `copy2:GH`), so the “30 rounds” didn’t apply meaningful updates for those prompts (baseline exam == post-round exam for the failing items).

Fix for next attempt:

- Current code now schedules `order seq` + `taskorder cycle` as a flattened target × task cross-product, so the exact starvation bug should not recur.
- `taskorder rand` remains useful for exploratory practice, especially after a regression.
- Or teach in short phases:
  1) `focus H` + `tasks copy` until `H→H`
  2) `focus G` + `tasks copy2,next` until `GH→GH` and `N:...:G:n→H`
  3) Re-enable full tasks and `exam`

## Next research step (proposed)

Use `checkpoints/kinder_ABCDEFGH_alltasks_ctxn_v3.pt` as the clean base before broadening. The next expansion should keep the same maintenance-first rule:

- add `I` only behind strict phase gates, then final `36/36` for `A-I` across `copy,copy2,next,next2`
- or start word-like mini-patterns while retaining the full A-H exam as a regression gate
- keep `--align-restep-training` enabled while exams use the legacy restep decoder
