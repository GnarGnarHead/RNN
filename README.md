# RNN

This project tests whether a tiny, attention-free recurrent character model can learn through a dynamic external tutor rather than bulk training alone.

The model is deliberately small and CPU-friendly. The interesting part is the loop around it: a human/Codex tutor quizzes the model, grades the response with partial credit, applies small approved updates, and keeps checking whether old skills survive after state resets.

## Core Idea

The model is a character-level language model with no attention. For each input character it:

1. Embeds the current character.
2. Injects the previous recurrent state through a learned gate.
3. Runs a shared residual MLP core for `K` internal settle iterations.
4. Logs per-iteration movement as `delta[k]`.
5. Decodes the next character from the settled state.

The research question is not raw language-modeling performance. It is whether structured tutoring plus recurrent settling can produce inspectable learning dynamics: retention, forgetting, repair, and curriculum growth.

## Dynamic Tutor Loop

The tutor is meant to stay dynamic. It is not a fixed benchmark, reward script, or static curriculum.

The current workflow starts like a kindergarten lesson:

- teach copy tasks: `T:A S:A`
- quiz with the answer omitted: `T:A S:`
- reset state before retention checks, so success lives in weights rather than working memory
- expand slowly: letters -> next-letter prediction -> digrams -> words/sentences
- interleave rehearsal so new lessons do not overwrite old ones

The tutor can respond to symptoms:

- copy breaks -> regress to copy drills
- next-letter prediction collapses -> separate prompts and rehearse old targets
- output repeats across different prompts -> treat it as attractor collapse
- settle deltas expand -> reduce scope or increase settling/check dynamics

See [TUTOR.md](TUTOR.md) for the live teaching method and [PROGRESS.md](PROGRESS.md) for the checkpoints and observed failure modes.

## Current Best Milestone

The current best local milestone is:

```text
checkpoints/kinder_ABCDEFG_alltasks_ctxn_v2.pt
```

It demonstrates retention, with state reset before each quiz, for:

- `copy`: `A -> A ... G -> G`
- `copy2`: `AB -> AB ... GA -> GA`
- `next`: `N:ABCDEFG:A:n -> B ... N:ABCDEFG:G:n -> A`
- `next2`: `N:ABCDEFG:AB:n -> C ... N:ABCDEFG:GA:n -> B`

Run the noninteractive exam:

```bash
python3 scripts/exam_checkpoint.py \
  --text-path input.txt \
  --checkpoint checkpoints/kinder_ABCDEFG_alltasks_ctxn_v2.pt \
  --targets ABCDEFG \
  --tasks copy,copy2,next,next2
```

The checkpoints themselves are local artifacts and ignored by git. See [docs/checkpoints.md](docs/checkpoints.md) for the checkpoint manifest.

## Quickstart

Create a virtual environment and install the single runtime dependency:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Provide a local training corpus:

```bash
python3 scripts/download_tiny_shakespeare.py --out input.txt
```

Train a baseline character model:

```bash
python3 settle_rnn_charlm.py --text-path input.txt --k-settle 2
```

Run a K sweep and write logs/samples under `runs/`:

```bash
python3 settle_rnn_charlm.py \
  --text-path input.txt \
  --steps 8000 \
  --sweep-k 1 2 4 8 \
  --out-dir runs \
  --run-name sweep
```

Start the dynamic tutor stepper:

```bash
python3 scripts/tutor_stepper.py --text-path input.txt --targets A
```

Run the stepper against a checkpoint:

```bash
python3 scripts/tutor_stepper.py \
  --text-path input.txt \
  --checkpoint checkpoints/kinder_ABCDEFG_alltasks_ctxn_v2.pt \
  --targets ABCDEFG \
  --tasks copy,copy2,next,next2
```

Inside the REPL, use `exam` for a retention check, `drill <n>` for interleaved rehearsal, and `save checkpoints/name.pt` after a milestone.

By default, generation uses the original checkpoint-compatible behavior: it re-steps the last ingested token before sampling. Use `--no-restep-generate` when you want the cleaner mode that samples the first generated token from the cached logits produced during ingest.

## Repo Map

- [settle_rnn_charlm.py](settle_rnn_charlm.py): CPU training script for the base char LM, K sweeps, logging, and samples.
- [session.py](session.py): JSONL process that keeps recurrent state alive and exposes `reset`, `ingest`, `generate`, `learn`, `save`, and `load`.
- [scripts/tutor_stepper.py](scripts/tutor_stepper.py): interactive dynamic tutor REPL.
- [scripts/exam_checkpoint.py](scripts/exam_checkpoint.py): noninteractive retention exam for checkpoints.
- [rnn/model.py](rnn/model.py): attention-free recurrent settle-loop model.
- [rnn/session.py](rnn/session.py): session wrapper and supervised learning helper.
- [rnn/tutor.py](rnn/tutor.py): reusable lesson construction, scheduling, and grading helpers.
- [tests/](tests): focused tests for tutor task construction and scheduler coverage.
- [TUTOR.md](TUTOR.md): teaching protocol and operator notes.
- [PROGRESS.md](PROGRESS.md): research log and checkpoint history.
- [docs/checkpoints.md](docs/checkpoints.md): local checkpoint manifest and verification commands.
- [docs/briefs/](docs/briefs): older research briefs and design notes.

## What To Watch

During training or tutoring, track:

- `loss`
- `delta[k]` across settle iterations
- `logits_entropy`
- `state_norm`
- retention after `reset`
- repeated outputs across different prompts

Useful experiments:

- Compare `--sweep-k 1 2 4 8`.
- Compare `--detach-state` and `--no-detach-state`.
- Compare `--state-norm` and `--no-state-norm`.
- Disable recurrence with `--no-state`.
- Expand tutor targets gradually and keep rehearsal on.
- Compare legacy generation with `--restep-generate` vs `--no-restep-generate` before training new milestones.

## Notes

- Default device is CPU.
- Keep `input.txt`, `data/`, `runs/`, and checkpoints local-only.
- If you add dependencies, update [requirements.txt](requirements.txt).
- If a new training phase works, record the command and result in [PROGRESS.md](PROGRESS.md).
