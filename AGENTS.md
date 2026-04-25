# RNN (settle-before-decode) repo guidance

## What this repo is

CPU-only experiments around a **shallow/wide recurrent “settling loop”** for char-level language modeling (**no attention**).

Primary artifact: `settle_rnn_charlm.py`.

## Current research task (for future agents)

We’re building a **human-in-the-loop tutor loop** that teaches the model like a kindergartener:

- Start from random weights
- Keep mimicry (A→A, B→B, …) as a **permanent maintenance substrate**, not just phase 1
- Gradually expand to digrams → words → sentences
- For every expansion, train local maintenance bundles: direct examples, nearby copy/digram contrasts, prediction contrasts, and known confusions
- Detect forgetting after each update and dynamically repair it before promotion
- Promote final milestones only on clean reset-based exams (e.g. A-H `copy,copy2,next,next2` requires `32/32`; partial improvements are diagnostic artifacts)

The tutor/controller is external (Codex CLI assistant + human), and training should stay **interactive and inspectable**, not “autopilot”.

See: `TUTOR.md` (method) and `PROGRESS.md` (what has been achieved so far).

Current clean local milestone: `checkpoints/kinder_ABCDEFGH_alltasks_ctxn_v3.pt`
passes `32/32` on A-H `copy,copy2,next,next2` with reset and legacy restep
generation.

## Project constraints

- Keep everything **CPU-friendly** by default (`device=cpu`, small batch sizes, no GPU assumptions).
- Do **not** add attention mechanisms unless explicitly requested.
- Prefer **small, inspectable code** over frameworks (single-file scripts are fine).
- Do **not** commit large datasets. Treat `input.txt`/`data/` as local-only.

## Repro & logging

- Default to deterministic seeds when possible.
- Log: `loss`, time per log interval, and **delta per settle-iteration** (`delta[k]`).
- For tutor runs, log exact pass/fail exams, regressions, repair attempts, and promoted checkpoints.
- If adding run outputs, write to `runs/` and keep them ignored by git.

## Data

- Default training input is `input.txt` (user-provided).
- If you add a downloader, keep it optional (a script under `scripts/`) and avoid running it automatically.

## Code hygiene

- Prefer type hints and dataclasses for configs.
- If you add deps, update `requirements.txt` and `README.md`.
- Keep the “K sweep” easy (CLI flags or a small helper script).
