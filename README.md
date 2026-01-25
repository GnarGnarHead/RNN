# RNN: settle-before-decode (no attention)

This repo is a small CPU-friendly prototype to test a **recurrent “settling loop”**: apply a shallow/wide MLP “core” **K times per token** before producing logits, and track whether the internal iterations converge (`delta[k]` decreases) or collapse/oscillate.

See `brief.md` for the full experiment writeup.

## Quickstart

1. Create a venv and install deps:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Add a training text file:

- Put your own `input.txt` next to `settle_rnn_charlm.py`, or
- Download Tiny Shakespeare:

```bash
python3 scripts/download_tiny_shakespeare.py --out input.txt
```

3. Train:

```bash
python3 settle_rnn_charlm.py --k-settle 2
```

By default the script detaches the recurrent state each timestep (no BPTT through time) and RMS-norms the state before injecting it. To enable full BPTT or disable normalization:

```bash
python3 settle_rnn_charlm.py --no-detach-state --no-state-norm
```

4. Sweep K:

```bash
python3 settle_rnn_charlm.py --steps 8000 --sweep-k 1 2 4 8
```

Optional: write logs + samples under `runs/`:

```bash
python3 settle_rnn_charlm.py --steps 8000 --sweep-k 1 2 4 8 --out-dir runs --run-name sweep
```

## What to look for

- `loss` vs steps
- `delta[k]` decreasing with `k` (per-token settle convergence)
- generation: less chaotic with larger K **without** repetition/mode collapse

## Tutor stepper (human in the loop)

Interactive REPL that quizzes → grades → suggests a correction, then waits for approval before applying any training steps:

```bash
python3 scripts/tutor_stepper.py --text-path input.txt --targets A
```

Tutoring notes (including the underlying JSONL session protocol used by the stepper) live in `TUTOR.md`.

## Research workflow (what we’re actually doing)

This repo is intentionally “lab notebook style”: small, inspectable experiments around **feedback + structured lessons** as an alternative to attention.

Roles:

- **Codex CLI assistant**: the *tutor/controller* that interacts with the model (quizzes, grades, applies tiny supervised updates).
- **You (human)**: the *guiding researcher* who sets goals, approves training, and decides when to broaden scope or backtrack.

Practical approach:

- Start from **random weights** and teach **mimicry** (A→A, B→B, …) with explicit retention checks (reset state, re-quiz).
- When adding new targets, always **interleave rehearsal** of old targets (otherwise catastrophic forgetting is common).
- As the target set grows, prefer **lower `lr` + more steps** over high `lr` (safer, less overwriting).
- Save **checkpoints** under `checkpoints/` (ignored by git) whenever you hit a milestone (e.g. A–E).

Current status and what has been demonstrated so far lives in `PROGRESS.md`.

## Notes

- If `pip install -r requirements.txt` installs a GPU build of PyTorch on your machine, install the CPU wheels instead (see the official PyTorch install instructions for your platform).
