# Development Brief

**Persistent Recurrent LM Session + Codex CLI Tutor Integration**

## Objective

Turn the current “offline training script” into a **persistent, interactive recurrent model** that:

* Maintains **internal state across turns**
* Supports **settle-before-decode**
* Can be **driven externally** by Codex CLI acting as a tutor/controller
* Remains simple, inspectable, and CPU-friendly

This is **not** about performance or scale yet. It’s about *control and dynamics*.

---

## Phase 1 — Architectural refactor (minimal, essential)

### 1. Split the model into two conceptual layers

**(A) Core model**

* `SettleCharLM`
* Pure computation: embeddings → recurrence → logits
* No training loop, no I/O, no CLI logic

**(B) Session wrapper**

* Holds **persistent recurrent state**
* Owns token stepping, generation, reset
* Is the only thing Codex CLI ever talks to

---

### 2. Add a persistent state API to the model

Add these methods (no behavior change yet):

```python
def reset_state(self, batch_size=1):
    self.state = torch.zeros(batch_size, d_model)

def step(self, token_id):
    """
    token_id: (B,) LongTensor
    returns logits, stats
    updates internal state
    """

def generate(self, max_tokens, temperature=1.0, k_settle=None):
    """
    Uses step() repeatedly
    """
```

Rules:

* **State lives on the model object**, not inside `forward`
* `step()` runs the **settle loop**
* `reset_state()` is explicit (never implicit)

This turns the model into a **true recurrent agent**, not a batch processor.

---

## Phase 2 — Session process (the key integration point)

### 3. Create a long-running session executable

New file:

```
session.py
```

This script:

* Loads model + vocab once
* Reads **JSON lines from stdin**
* Writes **JSON lines to stdout**
* Never exits unless told to

---

### 4. Define a dead-simple protocol (JSONL)

#### Input commands

```json
{"cmd":"reset"}
{"cmd":"ingest","text":"A B A B","k_settle":4}
{"cmd":"generate","max_new_tokens":50,"temperature":0.8}
{"cmd":"stats"}
{"cmd":"exit"}
```

#### Output format

```json
{
  "ok": true,
  "text": "generated output (if any)",
  "stats": {
    "delta_per_k": [0.18, 0.11, 0.07, 0.05],
    "state_norm": 3.42
  }
}
```

Design goals:

* Human-readable
* Scriptable
* No hidden state
* Easy to debug with `echo | python session.py`

---

## Phase 3 — Codex CLI as tutor/controller

### 5. Codex CLI’s role (very clear boundary)

Codex CLI:

* Decides *what to send*
* Evaluates responses
* Provides structured feedback
* Controls pacing (repeat / advance / reset)

The recurrent model:

* **Never decides curriculum**
* **Never self-prompts**
* Only settles, responds, updates state

This clean separation avoids “self-delusion loops.”

---

### 6. Example Codex-driven loop

1. Codex CLI sends:

```json
{"cmd":"reset"}
```

2. Codex CLI sends lesson:

```json
{"cmd":"ingest","text":"A A A A"}
```

3. Codex CLI checks stability:

```json
{"cmd":"stats"}
```

4. If stable → advance
5. If unstable → rephrase / repeat / simplify
6. Eventually:

```json
{"cmd":"generate","max_new_tokens":20}
```

This is your **K–12 + tutor** in practice.

---

## Phase 4 — Instrumentation (non-optional)

Add cheap introspection:

* `delta_per_k`
* `state_norm`
* optional entropy of logits
* optional repetition detector

Codex uses these to decide:

* “Let it settle more”
* “Repeat lesson”
* “Reset and backtrack”

This replaces hand-waving with control theory.

---

## Phase 5 — Guardrails (to avoid known failure modes)

Must-haves:

* Optional `state.detach()` mode (CPU sanity)
* RMSNorm or clamp on state before injection
* Hard max on settle iterations
* Explicit reset command (no silent carryover)

Failure modes you are explicitly defending against:

* attractor collapse
* oscillation
* semantic mush
* runaway confidence

---

## What this gives you

* A **living recurrent system**, not a script
* True **temporal consolidation**
* Externalized curriculum and supervision
* A clean on-ramp to:

  * richer curricula
  * photonic / analog thinking
  * scaled hardware later

And crucially:

* You can stop at *any phase* and still have something real.

