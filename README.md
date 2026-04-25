# Settle RNN Tutor Lab

This is a tiny character-level recurrent model with an inner settling loop and
no attention. An external tutor teaches it small skills, checks whether they
survive after state resets, and repairs forgetting when it appears. The point is
not language-model performance; it is to watch retention, forgetting, and repair
in a system small enough to inspect.

## What Is Being Tested

The model is a character-level next-token predictor. The research loop around it
is the important part:

- teach a small skill, such as `T:A S:A`
- quiz with the answer omitted, such as `T:A S:`
- reset recurrent state before retention exams
- apply small supervised updates only when needed
- re-test old skills after every update
- promote checkpoints only when the full reset-based exam is clean

This makes retention, forgetting, repair, and curriculum growth visible instead
of hiding them inside one bulk training run.

## Model Shape

For each input character, the model:

1. embeds the current token
2. mixes in persistent recurrent state through a learned gate
3. runs a shared residual MLP core for `K` settle iterations
4. records movement per settle step as `delta[k]`
5. decodes the next character from the settled state

There is no attention mechanism. Keeping the architecture small is deliberate:
the goal is to see whether tutoring dynamics can be made clear enough to study.

## Tutor Loop

The tutor is external to the model. In this repo that can mean the interactive
stepper, the guarded runner, Codex CLI, and the human operator.

The current curriculum starts like a kindergarten lesson:

- mimicry: `A -> A`, `B -> B`, ...
- digram copy: `AB -> AB`, `BC -> BC`, ...
- next-letter prediction: `A -> B`, `B -> C`, ...
- two-letter successor prompts: `AB -> C`, `BC -> D`, ...

Mimicry is not treated as a phase to leave behind. It is a stabilizer that must
remain under rehearsal while new skills are added.

The guarded runner records whether candidate updates fixed failures, created new
failures, or shifted a failure to a neighboring prompt. That is the core
scientific loop: train, examine, detect forgetting, repair, and only then
promote.

## Current Status

The best local milestone is:

```text
checkpoints/kinder_ABCDEFGH_alltasks_ctxn_v3.pt
```

It passes a reset-based A-H exam across four task families:

- `copy`: `A -> A ... H -> H`
- `copy2`: `AB -> AB ... HA -> HA`
- `next`: `N:ABCDEFGH:A:n -> B ... N:ABCDEFGH:H:n -> A`
- `next2`: `N:ABCDEFGH:AB:n -> C ... N:ABCDEFGH:HA:n -> B`

Result:

```text
32/32 passed
```

The checkpoint itself is a local artifact and is ignored by git. The repo keeps
the code, protocols, and experiment records; generated data, runs, and `.pt`
files stay local. See [docs/checkpoints.md](docs/checkpoints.md) and
[PROGRESS.md](PROGRESS.md) for the local checkpoint manifest and history.

The current A-H margin baseline is recorded in
[docs/experiments/2026-04-26_ah_v3_margin_baseline.md](docs/experiments/2026-04-26_ah_v3_margin_baseline.md).
The weakest items are successor wraparound cases, especially `H -> A`, which is
useful context before attempting A-I.

## Quickstart

Create a virtual environment and install the minimal runtime dependency:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Provide a local character corpus:

```bash
python3 scripts/download_tiny_shakespeare.py --out input.txt
```

Run the basic settle-loop character model:

```bash
python3 settle_rnn_charlm.py --text-path input.txt --k-settle 2
```

Run a small `K` sweep:

```bash
python3 settle_rnn_charlm.py \
  --text-path input.txt \
  --steps 8000 \
  --sweep-k 1 2 4 8 \
  --out-dir runs \
  --run-name sweep
```

Start the interactive tutor:

```bash
python3 scripts/tutor_stepper.py --text-path input.txt --targets A
```

Inside the tutor REPL, useful commands include:

- `exam`: reset-based retention check
- `drill 200`: interleaved practice block
- `tasks copy,copy2,next,next2`: choose task families
- `focus EFG`: focus quizzes while retaining maintenance rehearsal
- `save checkpoints/name.pt`: save a local milestone

For the full operator protocol, see [TUTOR.md](TUTOR.md).

## Exam And Promotion

If you have a local checkpoint, run a noninteractive retention exam:

```bash
python3 scripts/exam_checkpoint.py \
  --text-path input.txt \
  --checkpoint checkpoints/kinder_ABCDEFGH_alltasks_ctxn_v3.pt \
  --targets ABCDEFGH \
  --tasks copy,copy2,next,next2
```

For diagnostic JSONL with logit margins and top-k alternatives:

```bash
python3 scripts/exam_checkpoint.py \
  --text-path input.txt \
  --checkpoint checkpoints/kinder_ABCDEFGH_alltasks_ctxn_v3.pt \
  --targets ABCDEFGH \
  --tasks copy,copy2,next,next2 \
  --jsonl --trace-top-k 5
```

Use the guarded runner when trying to promote a new milestone:

```bash
python3 scripts/tutor_guarded_runner.py \
  --text-path input.txt \
  --checkpoint checkpoints/kinder_ABCDEFGH_alltasks_ctxn_v3.pt \
  --out-dir runs/guarded_i_attempt \
  --targets ABCDEFGHI \
  --tasks copy,copy2,next,next2 \
  --align-restep-training
```

Promotion is strict: partial improvement is diagnostic only. A checkpoint should
not become the new milestone unless it passes the requested full exam after
state resets, without regressing previously passed items.

## Metrics To Watch

During training and tutoring, track:

- training loss
- `delta[k]` across settle iterations
- `logits_entropy`
- `state_norm`
- reset-based retention pass/fail
- per-item logit margin
- repeated outputs across different prompts
- failure transitions: resolved, persistent, and newly introduced failures

See [docs/metrics.md](docs/metrics.md) for the current metric plan.

## Repo Map

- [settle_rnn_charlm.py](settle_rnn_charlm.py): CPU training script for the base char LM, K sweeps, logging, and samples.
- [session.py](session.py): JSONL process exposing `reset`, `ingest`, `generate`, `learn`, `save`, and `load`.
- [scripts/tutor_stepper.py](scripts/tutor_stepper.py): interactive human-in-the-loop tutor REPL.
- [scripts/tutor_guarded_runner.py](scripts/tutor_guarded_runner.py): promotion-gated runner with forgetting detection and focused repairs.
- [scripts/exam_checkpoint.py](scripts/exam_checkpoint.py): noninteractive retention exams with optional logit traces.
- [rnn/model.py](rnn/model.py): attention-free recurrent settle-loop model.
- [rnn/session.py](rnn/session.py): session wrapper and supervised learning helper.
- [rnn/tutor.py](rnn/tutor.py): lesson construction, scheduling, grading, and maintenance examples.
- [tests/](tests): focused unit tests for model generation, sessions, and tutor logic.
- [TUTOR.md](TUTOR.md): teaching protocol and operator notes.
- [PROGRESS.md](PROGRESS.md): research log and checkpoint history.
- [docs/protocols/](docs/protocols): exam and promotion protocols.
- [docs/experiments/](docs/experiments): structured experiment records.
- [docs/failure_modes.md](docs/failure_modes.md): observed failures and mitigations.
- [docs/checkpoint_lineage.md](docs/checkpoint_lineage.md): local checkpoint ancestry.

## Development

Run tests:

```bash
python3 -m unittest discover -s tests
```

Local-only files:

- `input.txt`
- `data/`
- `runs/`
- `checkpoints/*.pt`

If a new training phase works, record the command, result, failures, and
interpretation in [PROGRESS.md](PROGRESS.md) or a new file under
[docs/experiments/](docs/experiments).
