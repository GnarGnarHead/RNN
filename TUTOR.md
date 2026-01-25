# Tutor notes (kindergarten → words → sentences)

## What this is

We’re testing a simple hypothesis:

> Highly structured lessons + feedback + a persistent recurrent state can build capability without attention.

This is intentionally a **research loop**, not a “train once and declare victory” pipeline.

## Roles (researchers, different hats)

- **Codex CLI assistant (tutor/controller)**:
  - directly interacts with the model (quizzes, grades, picks a small correction)
  - runs the actual `learn` step(s) and reports what happened
  - stays conservative: small updates, frequent retention checks, checkpoint milestones
- **You (human, guiding researcher)**:
  - sets goals, constraints, and what counts as “good enough”
  - approves or rejects training steps (or asks for smaller/bigger steps)
  - decides when to broaden scope (A→E → digrams → words) or backtrack when the model forgets

The point is to learn about the model learning. Healthy paranoia is fine: ask “what exactly did you run?” and demand reproducible commands.

## Architecture boundary

This project separates responsibilities:

- The **model** (`rnn/model.py`) only settles + predicts next characters.
- The **session process** (`session.py`) holds persistent recurrent state and exposes a JSONL control surface.
- The **tutor** (this Codex CLI assistant + you, as co-developers) decides *what to teach*, *how to grade*, and *when to repeat or advance*.

The tutor is intentionally external so it can be strict, kind, or experimental without changing model internals.

## Tutor contract (kind + stable)

Treat the model like a kindergartener, but don’t reward collapse:

- **Be kind early:** partial credit for “close enough” answers so it stays engaged.
- **Tighten over time:** as moving-average score rises, require more exactness.
- **Stability is part of the grade:** prefer answers produced from a stable settle loop (lower/contracting `delta_per_k`) and avoid “always output the same thing” behavior.

In practice, the tutor uses the model’s output + cheap stats (`delta_per_k`, `logits_entropy`, repetition) to decide when to:

- repeat the same lesson
- slow down / reduce scope
- increase `k_settle`
- apply more (or fewer) supervised correction steps

## Two kinds of “win”

Track both from the beginning:

1. **Immediate recall (short-term)**: teach → quiz without restarting the process. (Weights + session dynamics.)
2. **Retention (long-term)**: teach → `reset` → quiz. (Weights only; recurrent state is cleared.)

For “semi-permanent” learning, retention is the important scoreboard.

## Prompt format for mimicry

For a next-token character LM, the simplest copy task is:

```
T:<prompt> S:<answer>
```

Example (copy a letter):

```
T:A S:A
```

When quizzing, you omit the answer and ask the model to generate it:

```
T:A S:
```

## Mixing mimicry + prediction (recommended)

If you teach both “copy” and “predict”, **don’t reuse the same prompt with different answers** (e.g. don’t do `T:A S:A` and `T:A S:B`). That is contradictory supervision and will usually collapse.

Instead, tag the prompt so tasks are distinguishable:

- **Copy**: `T:A S:A` (quiz: `T:A S:`)
- **Next-letter**: `T:N:A S:B` (quiz: `T:N:A S:`)
- **Two-letter next**: `T:N:AB S:C` (quiz: `T:N:AB S:`)

The stepper supports this directly via `--tasks`:

```bash
python3 scripts/tutor_stepper.py --text-path input.txt --targets ABCDE --tasks copy,next --task-order cycle
```

As a “kind” dynamic rubric, the tutor gives partial credit for reasonable mistakes, e.g. copying `A` when the expected next letter was `B`.

## Sentiment grading (kind but tightening)

Early on, grade like a child:

- Exact match: “Perfect!”
- Case-insensitive match (`A` vs `a`): “Good job!” (counts as a win early)
- Wrong letter (`A` vs `h`): “Nice try!” (small credit)
- Any printable output: “You answered!” (tiny credit)

As performance improves (moving average rises), tighten the rubric: case must match, then punctuation, then full-string exactness.

### Grading knobs (recommended)

When you start chaining letters → words → sentences, you’ll usually want at least these “teacher controls”:

- **Partial credit:** exact match > case-insensitive > “any alpha” > “any printable”.
- **Repetition penalty:** if the student repeats the same character too often, reduce score (prevents attractor collapse).
- **Confidence shaping:** very low entropy + wrong answer is a red flag (overconfident collapse); treat it as worse than a hesitant wrong answer early.
- **Settle health:** if `delta_per_k` doesn’t contract (or grows), treat it as “needs more settling / smaller lesson”, even if the character was lucky.

## Stepper mode (human in the loop)

For research-style tutoring, use the stepper REPL. It quizzes → grades → suggests a correction, then **waits for your approval** before applying any `learn` steps:

```bash
python3 scripts/tutor_stepper.py --text-path input.txt --targets A
```

Useful commands inside the REPL:

- `targets AB` (change practice set)
- `add C` (add letters)
- `k 4` / `kl 4` (change settle steps for quiz / learn)
- `detach on|off` (toggle detach during learn; default is off)
- `order seq` (practice in alphabetical order)
- `tasks copy,next` / `taskorder rand|cycle` (mix tasks slowly)
- `drill 200` (run an interleaved practice block across all targets)
- `save checkpoints/kinder.pt` / `load checkpoints/kinder.pt`

Note: the stepper trains with **answer-only loss masking** (it only backprops on the part after `S:`). This avoids an impossible objective (predicting the prompt content) when you practice multiple different prompts.

Also: for letter-copy / prompt-conditioned tasks, you usually want **BPTT through the example**, so the stepper defaults to `--no-detach-state` during learning. If you want faster but “shallow” learning that doesn’t propagate credit assignment through time, restart with `--detach-state`.

## How to run the kindergarten loop (recommended)

For each new concept (letters first):

1. **Retention quiz** (state reset each time): check old targets still work.
2. **Teach in small bursts**: one `learn` call, 25–150 steps.
3. **Re-quiz retention** immediately. If anything regresses, reduce `lr` and increase rehearsal weight.
4. **Checkpoint** after each milestone (e.g. `checkpoints/kinder_ABCDE.pt`).

Key tricks that prevent collapse/forgetting:

- **Interleave rehearsal**: when adding `D`, train on `A,B,C,D` (not just `D`).
- **Weight rehearsal**: duplicates in the `examples` list act like a simple sampler weight.
- **Lower `lr` as the set grows**: early letters tolerated `3e-3`; adding new letters often works better at `1e-3` with more steps.

## Underlying JSONL session (advanced)

Start the session:

```bash
python3 session.py --text-path input.txt
```

Commands are one JSON object per line on stdin, replies are one JSON object per line on stdout.

Required first command:

```json
{"cmd":"reset"}
```

### Learn (supervised, semi-permanent)

Train on one or more short examples:

```json
{"cmd":"learn","examples":[{"prompt":"A"},{"prompt":"B"}],"steps":200,"lr":0.001}
```

Each `{prompt}` is converted to `T:<prompt> S:<prompt>` (copy task).

### Quiz

```json
{"cmd":"reset"}
{"cmd":"ingest","text":"T:A S:"}
{"cmd":"generate","max_new_tokens":1,"temperature":0.0}
```

### Save / load

```json
{"cmd":"save","path":"checkpoints/kinder.pt"}
{"cmd":"load","path":"checkpoints/kinder.pt"}
```
