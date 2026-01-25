# Tutor notes (kindergarten → words → sentences)

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
- `save checkpoints/kinder.pt` / `load checkpoints/kinder.pt`

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
