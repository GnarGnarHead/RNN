# RNN (settle-before-decode) repo guidance

## What this repo is

CPU-only experiments around a **shallow/wide recurrent “settling loop”** for char-level language modeling (**no attention**).

Primary artifact: `settle_rnn_charlm.py`.

## Project constraints

- Keep everything **CPU-friendly** by default (`device=cpu`, small batch sizes, no GPU assumptions).
- Do **not** add attention mechanisms unless explicitly requested.
- Prefer **small, inspectable code** over frameworks (single-file scripts are fine).
- Do **not** commit large datasets. Treat `input.txt`/`data/` as local-only.

## Repro & logging

- Default to deterministic seeds when possible.
- Log: `loss`, time per log interval, and **delta per settle-iteration** (`delta[k]`).
- If adding run outputs, write to `runs/` and keep them ignored by git.

## Data

- Default training input is `input.txt` (user-provided).
- If you add a downloader, keep it optional (a script under `scripts/`) and avoid running it automatically.

## Code hygiene

- Prefer type hints and dataclasses for configs.
- If you add deps, update `requirements.txt` and `README.md`.
- Keep the “K sweep” easy (CLI flags or a small helper script).

