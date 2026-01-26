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

## Known dynamics so far

- **Catastrophic forgetting / single-attractor**: if we train only the newest letter (e.g. only `D`), the model often starts answering `D` for everything.
- **Fix**: always interleave rehearsal (e.g. train on `A,B,C,D`, not just `D`), and *weight rehearsal* by repeating old targets in the `examples` list.
- **Learning rate**: aggressive `lr` learns fast but overwrites. As the target set grows, a lower `lr` with more steps is safer.
  - Early (A–C): `lr ≈ 3e-3` worked
  - Adding D/E: `lr ≈ 1e-3` was more stable for retention
- **BPTT vs detach**: for prompt-conditioned copy tasks, we typically want `detach_state=false` during `learn` (BPTT through the short example). Detaching can make it fail to bind the prompt to the answer.

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

### A–G letters + digrams + prediction (current best)

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

- Use `taskorder rand` (recommended) or `order rand` to cover the cross-product of targets × tasks, **especially** when using `focus`.
- Or teach in short phases:
  1) `focus H` + `tasks copy` until `H→H`
  2) `focus G` + `tasks copy2,next` until `GH→GH` and `N:...:G:n→H`
  3) Re-enable full tasks and `exam`

## Next research step (proposed)

Move from single letters to **digrams** while keeping retention:

- Example: `T:AB S:AB` and quiz `T:AB S:` → generate 2 chars.
- Start tiny (e.g. `AB`, `BA`, `AC`, `CA`) and aggressively interleave rehearsal of single-letter tasks.
