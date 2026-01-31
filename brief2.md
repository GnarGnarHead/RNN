Got it. Here’s a **single distilled research brief** you can hand straight to a co-researcher agent. I’ve kept it explicit, structural, and faithful to what you actually argued—not cleaned up into something generic.

---

## Research Brief

### Dynamic Tutored Recurrent Transformers & Education-by-Maintenance

### Core Premise

This work explores **how to build and maintain a “mind” structurally**, not how to maximize benchmark performance. Training is **tutor-driven and interactive**, not passive corpus ingestion. The system is intended to be **permanently tutored by an agent** (the tutor remains part of the system at runtime, not just during “pretraining”). The hypothesis is that **collapse in neural systems is primarily a product of rigid objectives and static training**, not an inevitable property of recurrence or continual learning. A dynamically tutored system—graded on *sentiment and intent rather than brittle semantics*—can actively **detect, regress, and repair conceptual drift**, preventing catastrophic forgetting through maintenance rather than architectural hard constraints.

---

## Model Architecture (High-Level)

* **Base model**: A small transformer-*style* residual stack (**no attention heads**) trained at **character level** rather than token/word level.

  * ~100 characters as the full vocabulary.
  * Avoids pre-baked abstractions in the token layer; abstractions must emerge in the hidden state.
* **Depth**: Target **3–8 transformer layers**, with a practical sweet spot estimated at **4–6 layers**, inspired loosely by neocortical depth constraints.

  * Rationale: shallow enough to be stable and interpretable under continual learning, deep enough for hierarchical operators.
* **Width**: Scaled to available compute (CPU now, GPU later). Width provides redundancy; depth provides structure.
* **Recurrence**:

  * The **final hidden state is fed back into the next step** to provide temporal grounding.
  * Current prototype (in this repo): **full-state feedback** (the full hidden vector), injected weakly with gating/damping.
  * Future direction: replace full feedback with a **low-bandwidth “context bus”**:

    * Final hidden state → dimensional collapse → a small number of latent signals.
    * Candidate implementations: low-rank linear projection; signed bilinear/quadratic summaries; possibly quantized/binary signals.
  * Goal: provide *contextual placement* (“where am I in thought?”) without creating a dominant attractor loop.
* **Gating & damping**:

  * Recurrent injection is weak, gated, and optionally noisy.
  * Prevents the model from ignoring new inputs or collapsing into self-reinforcing internal states.

---

## Tutor-Centric Learning Philosophy

### The Tutor *Is* the World (Early Stages)

At early and mid stages, **world-modeling and tutor-pleasing are intentionally identical**. The system optimizes for interaction with a dynamic tutor, not for abstract correctness. Later divergence to real-world data (internet, environments) is possible but explicitly out of scope for early prototyping.

### Sentiment over Semantics

* The tutor evaluates **intent, directionality, and structural correctness**, not strict logical or symbolic validity.
* Example:

  * Spelling “CAT” as “CAF” is *not a zero*.
  * It demonstrates correct semantic targeting but degraded mechanics.
* Rationale:

  * Semantic grading is brittle and riddled with edge cases.
  * Missing edge cases in rigid grading *actively weakens conceptual representations*.
  * Sentiment-based assessment allows **partial credit and repair**, not punishment.

### Symptom-Driven Regression

Errors are treated as **diagnostics**, not failures.

* Misspelling → regress to orthographic drills.
* Misuse of a word → regress to contextual usage.
* Drift in logic → reintroduce simpler exemplars.

Crucially:

* Regression **does not erase higher-level context**.
* Repairs occur *with meaning still present*, preserving conceptual graphs.

---

## Education as Dynamic Maintenance (Not Convergence)

> “A mind doesn’t converge. It persists.”

* Transformers do **not decay over time**; forgetting occurs via **gradient interference**, not entropy.
* Catastrophic forgetting is acceptable *because it is detectable and repairable*.
* The system is designed to require **ongoing maintenance**, mirroring biological cognition.
* Snapshots can be taken for deployment, but the research goal is understanding **what maintenance systems are required**, not producing a static pretrained artifact.

---

## Curriculum Strategy

### Tiered Training Worlds

**Tier A — Closed, Automatable**

* Mimicry (copy input).
* Basic prediction.
* Arithmetic and symbolic operators.
* Deterministic grading; tutor mostly audits.

**Tier B — Hybrid**

* Spelling with partial credit.
* Format compliance.
* Simple transformations.
* Automated grading + tutor intervention on edge cases.

**Tier C — Open, Tutor-Led**

* Language use.
* Conceptual explanations.
* Conversation.
* Ambiguous or pragmatic tasks.

This tiering is **tunable and reversible**. Tasks can move between tiers as confidence increases.

---

## Starting Domain: Mathematics (Before Language)

Rationale:

* Arithmetic has **small, enumerable state spaces**.
* Infinite synthetic data is trivial to generate.
* Edge cases are well understood and controllable.
* Allows early validation of:

  * operator learning,
  * regression without collapse,
  * retention under interference.

Focus is **operators, not facts**:

* Copy, reorder, increment, substitute.
* Only later: arithmetic semantics.

Language is introduced after operator stability is demonstrated.

---

## Pause / Null Response Capability

* The model should be allowed to **not respond**.
* A null or pause action provides:

  * internal recurrence without external commitment,
  * a crude analogue of “thinking time.”
* This is sentiment-graded, not rigidly penalized.
* Excessive or pathological pausing is handled dynamically by the tutor (“this is unacceptable—fail”).
* Formalization (explicit PAUSE action) is acceptable *only if tunable* and non-brittle (and is **not implemented yet**).

---

## Scaling Hypothesis

* A **7–10B parameter model** may be sufficient for a “university-level mind” under this regime.
* Justification:

  * No passive ingestion of the internet.
  * High compression via interaction.
  * Continuous repair prevents representational bloat.
* Training is slow in wall-clock time but **extremely sample-efficient**.
* Early results show:

  * mimicry and prediction emerging in **hundreds of interactive cycles**,
  * forgetting occurring but being **successfully repaired**.

---

## Core Research Questions

1. Can **dynamic tutorship** act as a structural stabilizer under continual learning?
2. Can regression repair mechanics **without destroying higher-order meaning**?
3. Does low-bandwidth recurrence provide temporal coherence **without collapse**?
4. How much redundancy is required for long-term conceptual stability?

---

## Guiding Principle

Rigid formalization too early causes collapse.

Dynamic, sentiment-driven interaction:

* tolerates edge cases,
* preserves meaning,
* and allows structure to *grow and heal*.

This work treats education not as data ingestion, but as **ongoing structural alignment**.

---

If you want, next step could be:

* compressing this into a **one-page technical abstract**, or
* turning it into a **pseudo-spec / experimental protocol** for replication.
