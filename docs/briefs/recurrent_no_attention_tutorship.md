Brief: Recurrent (No-Attention) Prototype with Dynamic Tutorship

Purpose and Scope
The conversation revolves around Scott exploring a proof‑of‑concept model that combines ideas from recurrent neural networks and human‑feedback tutoring to build a dynamically maintained “mind.” The goal is not to train a large language model, but to test whether interactive tutorship can build stable internal structures—starting from simple mimicry and prediction tasks and gradually scaling toward complex language use. The system is intended to be **permanently tutored by an agent**. Training is tutor-driven (interactive), not corpus-driven. The prototype is CPU-first and intended to scale to GPU hardware once the concept is validated.

Model Architecture (No Attention)
- Core: an attentionless, transformer-style residual stack (norm → MLP → residual), run as an internal “settle loop”.
- Recurrence (current code): the full final hidden state is fed forward into the next step’s first layer / input (full-width feedback), injected weakly via gating/damping.
- Recurrence (future direction): replace full-width feedback with a low-bandwidth “context bus” (e.g., low-rank linear projection, bilinear/quadratic summaries, possibly quantized/binary signals).
- Depth: target ~3–8 layers (practically ~4–6), loosely inspired by neocortical depth constraints.
- Width / Parameters: start tiny (CPU-scale) and scale later (~7–10B parameters) if the tutor loop and stability mechanics pan out.
- Vocabulary: character-level tokens (~100 characters) so abstractions must emerge in the hidden state.
- PAUSE / WAIT: planned but not implemented yet; intended to allow additional internal iterations without emitting output, with tutor-controlled cost/caps.
Tutor and Education Methodology


Dynamic, agentic tutor:  An external tutor evaluates each output using sentiment‑driven grading rather than rigid semantic correctness.  Feedback includes partial credit, and the tutor can adjust the curriculum when the model exhibits symptoms like typos or misused words.  This approach echoes reinforcement learning from human feedback (RLHF), where a reward model is trained on human preferences and then used to guide a policy.


Partial credit and regression:  Misspellings (e.g., spelling “cat” as “caf”) or near‑correct answers are not treated as total failures.  The tutor regresses the curriculum to earlier exercises (such as character mimicry or basic spelling) to reinforce the weak sub‑skill while preserving higher‑level knowledge.  This prevents catastrophic forgetting—a phenomenon in sequential learning where training on a new task overwrites weights important for previous tasks.


Curriculum stages:


Stage 1 – Mimicry and prediction:  The model learns to copy individual characters and predict simple sequences (e.g., “A”→“B”).  The tasks are closed‑form and can be graded automatically.  Arithmetic drills (e.g., “1+1=2”) are proposed as an alternative starting point; they allow deterministic grading and minimal edge cases.


Stage 2 – Pattern recognition and spelling:  Two‑ and three‑letter words and simple arithmetic expressions.  Misspellings trigger targeted reinforcement rather than full resets.


Stage 3 – Language and semantics:  Once pattern recognition is robust, curriculum moves to words, sentences, and basic language comprehension.  Tutor feedback focuses on appropriateness and intent, giving partial credit for near‑correct usage.




Sentiment over semantics:  The tutor grades responses based on alignment with intended meaning and sentiment, not strict semantic form.  This makes the system robust to edge cases and avoids brittle logical rules that can cause collapse.  Human feedback is easier to provide as “good” or “bad” judgments, similar to RLHF.


Dynamic maintenance:  Skills are periodically revisited (“spaced learning”).  If the model exhibits a typo or misused word, the tutor triggers refresher exercises to reinforce spelling or context.  This counters catastrophic forgetting by restoring weights associated with earlier tasks, analogous to synaptic consolidation strategies used in elastic weight consolidation.


Avoiding Collapse and Weight Dynamics


No inherent weight decay:  Neural network weights remain fixed unless training applies updates or regularization.  In ℓ2‑regularized models, weight decay shrinks weights toward zero during training.  There is no natural “decay” over time, so forgetting arises from conflicting updates rather than spontaneous drift.  The user’s concern about collapse therefore focuses on interference from new training rather than time‑based decay.


Catastrophic forgetting:  Sequentially training on task B can abruptly erase performance on task A because weights important for A are overwritten.  Dynamic tutoring counters this by selectively reinforcing older knowledge when symptoms appear.


Sentiment‑driven tutor vs. semantic grading:  Using human feedback to grade outputs is similar to RLHF, where a reward model is trained on human judgments and then used to optimize the policy.  This allows flexible grading (partial credit) and avoids the brittleness of rule‑based evaluation.


Recurrence, Halting and Bus Design


Full-width now, low-bandwidth later: The current prototype uses full-width feedback (the full hidden vector/state) between steps. A planned refinement is to compress feedback through a narrow, possibly nonlinear “bus” (e.g., a few low-rank or quadratic channels) so recurrence provides temporal grounding without overwhelming new inputs.


Gating the feedback:  A learned gate mixes the bus signal with new token embeddings so that the model can ignore the recurrence when it is irrelevant.  This prevents attractor collapse (the model producing the same output regardless of input).


Dynamic halting: Inspired by adaptive computation time, the design includes a PAUSE/WAIT mechanism so the model can run extra internal iterations before emitting an answer. This is planned but not implemented yet; the tutor can impose a small time cost and cap consecutive pauses to avoid infinite loops.


Scaling and Hardware Considerations


The prototype runs on a single‑threaded CPU with a tiny network.  Once the dynamic tutoring loop and recurrence design are validated, the architecture will be scaled to GPUs (e.g., NVIDIA A6000) with parameter counts up to ~7–10 billion.  The user believes this parameter range is sufficient to encode K‑12 and university‑level knowledge due to the efficient compression afforded by dynamic tutoring and spaced learning.


Implications and Next Steps


Demonstrate stability across tasks:  Train the model on mimicry and arithmetic, then introduce a new task (e.g., spelling) and observe if earlier skills degrade.  Use dynamic reinforcement when symptoms appear.  Validating that older skills can be recovered without full re‑training will show that the tutor loop mitigates catastrophic forgetting.


Implement explicit pause action:  Add a distinct WAIT token; allow the model to run additional recurrent steps while charging a small cost.  This will test whether dynamic halting improves performance on tasks requiring multiple reasoning steps.


Prototype bilinear bus:  Test low‑rank linear and bilinear projections from the hidden state into a small number of channels, combined with a gate.  Compare their impact on retention and stability.


Expand curriculum gradually:  Move from character mimicry and arithmetic to spelling, grammar, and simple conversations.  Keep the tutor flexible and rely on sentiment grading rather than strict semantics.  Continue to monitor for interference and use regression when needed.


Document and log:  Maintain detailed logs of training, regression events and performance.  This will inform future refinements and provide data for your co‑researcher to iteratively improve the system.


Summary
The conversation outlines a framework for building an attentionless recurrent “settle loop” model that uses a dynamic tutor to create and maintain cognitive structures. The current code uses full-width recurrent feedback, with a planned move toward low-bandwidth feedback (“context bus”). The tutor relies on sentiment-based grading, partial credit, regression, and spaced reinforcement to combat catastrophic forgetting. Curriculum starts from simple mimicry/arithmetic and gradually builds toward language use, with a planned PAUSE/WAIT capability for internal thinking time. Scaling is deferred until the prototype demonstrates stable retention and dynamic maintenance.

This document distills the key concepts, design choices and educational methodology we explored. Let me know if you'd like this summary adjusted or further expanded in any section.

Sources
