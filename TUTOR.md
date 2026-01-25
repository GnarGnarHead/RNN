# Tutor notes (kindergarten → words → sentences)

This project separates responsibilities:

- The **model** (`rnn/model.py`) only settles + predicts next characters.
- The **session process** (`session.py`) holds persistent recurrent state and exposes a JSONL control surface.
- The **tutor** (you / Codex CLI / a driver script) decides *what to teach*, *how to grade*, and *when to repeat or advance*.

The tutor is intentionally external so it can be strict, kind, or experimental without changing model internals.

## Two kinds of “win”

Track both from the beginning:

1. **Immediate recall (short-term)**: teach → quiz without restarting the process. (Weights + session dynamics.)
2. **Retention (long-term)**: teach → `reset` → quiz. (Weights only; recurrent state is cleared.)

For “semi-permanent” learning, retention is the important scoreboard.

## Prompt format for mimicry

For a next-token character LM, the simplest copy task is:

```
T:<prompt>\n
S:<answer>\n
```

Example (copy a letter):

```
T:A
S:A
```

When quizzing, you omit the answer and ask the model to generate it:

```
T:A
S:
```

## Sentiment grading (kind but tightening)

Early on, grade like a child:

- Exact match: “Perfect!”
- Case-insensitive match (`A` vs `a`): “Good job!” (counts as a win early)
- Wrong letter (`A` vs `h`): “Nice try!” (small credit)
- Any printable output: “You answered!” (tiny credit)

As performance improves (moving average rises), tighten the rubric: case must match, then punctuation, then full-string exactness.

## JSONL commands (session protocol)

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

Each `{prompt}` is converted to `T:<prompt>\nS:<prompt>\n` (copy task).

### Quiz

```json
{"cmd":"reset"}
{"cmd":"ingest","text":"T:A\nS:"}
{"cmd":"generate","max_new_tokens":2,"temperature":0.0}
```

### Save / load

```json
{"cmd":"save","path":"checkpoints/kinder.pt"}
{"cmd":"load","path":"checkpoints/kinder.pt"}
```

## Quick demo run

Runs a tiny “teach A/B then quiz” loop against the JSONL session:

```bash
python3 scripts/brief_tutor_run.py --text-path input.txt
```

