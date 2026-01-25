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
- next: `T:N:A S:B`
- next2: `T:N:AB S:C`

Checkpoint (local): `checkpoints/kinder_ABCDE_copy_next_v2.pt`

Verified retention (reset before each quiz):

- copy: `A→A, B→B, C→C, D→D, E→E`
- next: `N:A→B, N:B→C, N:C→D, N:D→E, N:E→A`

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

The stepper supports mixing via:

```bash
python3 scripts/tutor_stepper.py --text-path input.txt --targets ABCDE --checkpoint checkpoints/kinder_ABCDE.pt --tasks copy,next --task-order cycle
```

## Next research step (proposed)

Move from single letters to **digrams** while keeping retention:

- Example: `T:AB S:AB` and quiz `T:AB S:` → generate 2 chars.
- Start tiny (e.g. `AB`, `BA`, `AC`, `CA`) and aggressively interleave rehearsal of single-letter tasks.
