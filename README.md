# RNN — settle-before-decode (no attention) + tutor loop

This repository is a small experimental lab exploring a specific combination of ideas:

1) An **attentionless** character-level language model with a recurrent **internal settling loop** (iterative inference per token)
2) A **human-in-the-loop tutor / stepper workflow** that treats training as ongoing *maintenance* (partial credit, regression, rehearsal)

It is not a production model, not a transformer replacement, and not a benchmark chase. It’s meant to be small, CPU-friendly, and inspectable.

If you want the longer design brief(s), start with `brief.md`.

## What this is (at a glance)

- Character-level LM (tiny vocab; abstractions must emerge in the hidden state)
- **K internal iterations per token** (shared weights), then decode
- Instrumentation for **convergence vs oscillation** inside the settle loop (`delta[k]`)
- Recurrent **state carried across tokens** (gated injection; optional detach/no-detach)
- Interactive tutor loop for structured lessons and retention checks

## Core idea: iterative inference per token

Instead of emitting logits after one pass, the model does a small “think loop” per token:

1. Embed current character
2. Optionally inject a recurrent state from the previous token (gated; optionally normalized)
3. Apply an **attentionless residual MLP core** (norm → MLP → residual) **K times**
4. Track per-iteration change magnitude (`delta[k]`)
5. Emit logits *after* the state has (hopefully) settled

This treats recurrence as **inference-time computation**, not just “memory across time.”

## What we actually study

This repo is intentionally scoped to small, sharp questions:

- Does increasing internal iteration K improve loss per unit compute?
- Do internal states converge, oscillate, or collapse to bland attractors?
- What changes when we detach vs backprop through time?
- Does “thinking longer” help on structured tasks vs free-form text?

No claims are made beyond what can be measured at this scale.

## Tutor loop: education-by-maintenance (why this isn’t just a toy RNN)

Alongside standard training, the repo includes a manual tutoring workflow intended to probe *continual learning* dynamics:

- Interactive prompt → model response
- Human grading (often **intent/partial-credit** rather than brittle exact-match)
- Small supervised updates (only with approval)
- Rehearsal / retention checks to surface forgetting
- Symptom-driven regression (if spelling breaks, regress to spelling drills; if copying breaks, regress to mimicry)

The hypothesis in the briefs is that a useful system isn’t one that “converges forever,” but one that can be **maintained**: detect drift, repair it, and keep going.

## What this is not

- Not SOTA language modeling
- Not a transformer replacement
- Not an AGI proposal
- Not benchmark-competitive

## Repo map

- `settle_rnn_charlm.py`: main CPU training script (K-sweep, delta logging, sampling)
- `scripts/tutor_stepper.py`: interactive tutoring stepper (mimicry-first curriculum)
- `TUTOR.md`: tutoring method + session protocol notes
- `PROGRESS.md`: what’s been demonstrated so far
- `rnn/`: reusable model/session/vocab helpers
- `checkpoints/`: local checkpoints (kept small; treated as local artifacts)
- `runs/`: optional logs + samples

## Quickstart

1) Create a venv and install deps:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Provide training text:

- Put your own `input.txt` next to `settle_rnn_charlm.py`, or
- Download Tiny Shakespeare:

```bash
python3 scripts/download_tiny_shakespeare.py --out input.txt
```

3) Train a baseline run:

```bash
python3 settle_rnn_charlm.py --k-settle 2
```

By default the script detaches recurrent state each timestep (no BPTT through time) and RMS-norms the state before injecting it. To enable full BPTT or disable state normalization:

```bash
python3 settle_rnn_charlm.py --no-detach-state --no-state-norm
```

4) Sweep K:

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
- `logits_entropy` and `state_norm` staying sane (avoid trivial collapse)
- samples: less chaotic with larger K without repetition/mode collapse

## Tutor stepper (human in the loop)

Interactive REPL that quizzes → grades → proposes a correction, then waits for approval before applying any training steps:

```bash
python3 scripts/tutor_stepper.py --text-path input.txt --targets A
```

Tutoring notes (including the session protocol used by the stepper) live in `TUTOR.md`.

## Experiments to try

- Compute-matched K sweep: `--sweep-k 1 2 4 8`, compare loss + `delta[k]` + samples
- Remove recurrence baseline: add `--no-state` and see what K buys you without cross-token state
- Detach vs full BPTT: compare `--detach-state` vs `--no-detach-state`
- State injection normalization: `--state-norm` vs `--no-state-norm`
- Tutor curriculum: start with mimicry (A→A), expand target set, and watch retention under interleaved rehearsal

## Planned directions (not implemented yet)

These are discussed in the briefs, but not fully built here:

- Replace full-width feedback with a low-bandwidth “context bus” (compressed recurrent signal)
- Add an explicit PAUSE/WAIT capability (extra internal steps without emitting output)

## Notes

- If `pip install -r requirements.txt` installs a GPU build of PyTorch on your machine, install CPU wheels instead (see the official PyTorch install instructions for your platform).
