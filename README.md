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

## Notes

- If `pip install -r requirements.txt` installs a GPU build of PyTorch on your machine, install the CPU wheels instead (see the official PyTorch install instructions for your platform).
