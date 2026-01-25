from __future__ import annotations

import argparse
import json
import math
import random
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _send(proc: subprocess.Popen[str], obj: Dict[str, Any]) -> Dict[str, Any]:
    proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("session.py exited unexpectedly")
    return json.loads(line)


def _succ(ch: str, alphabet: List[str]) -> str:
    if ch not in alphabet:
        raise ValueError(f"Character {ch!r} not in targets/alphabet")
    i = alphabet.index(ch)
    return alphabet[(i + 1) % len(alphabet)]


_CANON_ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _canon(ch: str) -> str | None:
    if not ch:
        return None
    c = ch[0].upper()
    if c in _CANON_ALPHA:
        return c
    return None


def _global_succ(ch: str) -> str | None:
    c = _canon(ch)
    if c is None:
        return None
    i = _CANON_ALPHA.index(c)
    return _CANON_ALPHA[(i + 1) % len(_CANON_ALPHA)]


def _grade_letter(expected: str, got: str) -> Tuple[float, str]:
    if not got:
        return 0.0, "No answer yet — let's try again."
    g0 = got[0]
    if g0 == expected:
        return 1.0, "Perfect!"
    if g0.lower() == expected.lower():
        return 0.9, "Good job (right letter, wrong case)."
    if g0.isalpha():
        return 0.2, "Nice try (wrong letter, but you answered)."
    return 0.1, "You answered!"


def _grade_task(
    expected: str,
    got: str,
    *,
    task: str,
    seq: str,
    alphabet: List[str],
    copy_continue_score: float,
    next_copy_score: float,
) -> Tuple[float, str]:
    """
    task:
      - copy: expect to echo seq (1-char)
      - copy2: expect to echo seq (2-char)
      - next / next2: expect successor of last char in seq (prompt is tagged with "N:")
    """
    if not got:
        return 0.0, "No answer yet — let's try again."

    # Do not reward newline during the kindergarten phase.
    if "\n" in got or "\r" in got:
        return 0.05, "Oops, a newline — let's stick to letters for now."

    # Derived “reasonable” answers from the underlying sequence (bound to prompt content).
    last = seq[-1] if seq else ""
    next1 = ""
    next2 = ""
    if last:
        try:
            next1 = _succ(last, alphabet)
            next2 = next1 + _succ(next1, alphabet)
        except Exception:
            next1 = ""
            next2 = ""

    # “Ahead of curriculum” continuations, based on the canonical A–Z alphabet.
    # This lets the student “discover” a new letter (e.g. 'H') before we explicitly add it to targets.
    g_next1 = _global_succ(last) or ""
    g_next2 = ""
    if g_next1:
        g_next2 = g_next1 + (_global_succ(g_next1) or "")

    g = got
    e = expected
    g_lo = g.lower()
    e_lo = e.lower()

    if g == e:
        return 1.0, "Perfect!"
    if g_lo == e_lo:
        return 0.9, "Good job (right letter, wrong case)."

    # If we generated longer than the “official” answer length, accept a correct prefix.
    # (E.g. expected 'F', got 'FG' — that's actually a good sign.)
    if e and g_lo.startswith(e_lo):
        if len(e) == 1 and len(g) >= 2:
            # Reward a correct continuation: expected 'F', got 'FG'.
            e0 = e[0]
            e_canon = next((a for a in alphabet if a.lower() == e0.lower()), None)
            if e_canon is not None:
                try:
                    e_succ = _succ(e_canon, alphabet)
                except Exception:
                    e_succ = ""
                if e_succ and g[1].lower() == e_succ.lower():
                    return 1.0, "Perfect (and you kept going)!"
        return 0.85, "Good (right start)."

    # Task-specific partial credit (kindergarten: “reasonable mix-up” still earns points).
    if task in {"copy", "copy2"}:
        # Copy asked, but they continued the alphabet.
        if next1 and g and g[0].lower() == next1.lower():
            if g_lo == next1.lower():
                return copy_continue_score, "Good (you continued to the next letter)."
            if next2 and g_lo.startswith(next2.lower()):
                return min(copy_continue_score + 0.1, 0.95), "Great (you continued two letters)!"
            return max(0.0, copy_continue_score - 0.05), "Good (you started continuing to the next letter)."
        # Continued the canonical alphabet beyond current targets (e.g. 'G' -> 'H').
        if g_next1 and g and g[0].lower() == g_next1.lower():
            if g_lo == g_next1.lower():
                return copy_continue_score, "Good (you continued to a new letter)."
            if g_next2 and g_lo.startswith(g_next2.lower()):
                return min(copy_continue_score + 0.1, 0.95), "Great (you continued two new letters)!"
            return max(0.0, copy_continue_score - 0.05), "Good (you started continuing to a new letter)."
        # For multi-char copy, getting the first character right is still good progress.
        if e and g and g[0].lower() == e[0].lower():
            return 0.55, "Nice start (first letter right)."

    if task.startswith("next"):
        # Next asked, but they copied the prompt (or its last letter).
        if seq and (g_lo == seq.lower() or g_lo == seq[-1].lower()):
            return next_copy_score, "Nice try (you copied instead of predicting the next)."
        # Ahead-of-curriculum: if we wrapped (expected is the first target), reward the
        # canonical next letter too (e.g. with targets 'ABCD', N:D expects 'A' but 'E' is a good guess).
        if g_next1 and g and g[0].lower() == g_next1.lower():
            if expected and expected[0].lower() != g_next1.lower():
                if g_next2 and g_lo.startswith(g_next2.lower()):
                    return 0.9, "Great (you predicted a new letter and kept going)!"
                return 0.8, "Good (you predicted a new letter beyond the current wrap-around)."
        # Off-by-one around the alphabet cycle (first char).
        try:
            prev = alphabet[(alphabet.index(expected[0]) - 1) % len(alphabet)]
            nxt = alphabet[(alphabet.index(expected[0]) + 1) % len(alphabet)]
        except Exception:
            prev = ""
            nxt = ""
        g0 = g[0]
        if prev and (g0 == prev or g0.lower() == prev.lower()):
            return 0.35, "Close (off by one: previous letter)."
        if nxt and (g0 == nxt or g0.lower() == nxt.lower()):
            return 0.35, "Close (off by one: next letter)."

    if g and g[0].isalpha():
        return 0.2, "Nice try (wrong letter, but you answered)."
    return 0.1, "You answered!"


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _steps_from_score(score: float, *, min_steps: int, max_steps: int) -> int:
    s = _clamp01(score)
    if max_steps < min_steps:
        raise ValueError("max_steps must be >= min_steps")
    # Lower score → more supervised correction.
    steps_f = min_steps + (max_steps - min_steps) * (1.0 - s)
    return int(round(steps_f))


def _grade_with_stats(
    expected: str,
    got: str,
    *,
    task: str,
    seq: str,
    alphabet: List[str],
    copy_continue_score: float,
    next_copy_score: float,
    stats: Dict[str, Any],
    recent_prompts: List[str],
    recent_outputs: List[str],
) -> Tuple[float, str]:
    base_score, base_msg = _grade_task(
        expected,
        got,
        task=task,
        seq=seq,
        alphabet=alphabet,
        copy_continue_score=copy_continue_score,
        next_copy_score=next_copy_score,
    )
    score = float(base_score)
    reasons = [base_msg]

    # --- Confidence shaping (entropy) ---
    entropy = _safe_float(stats.get("logits_entropy"))
    if not math.isnan(entropy) and base_score < 0.6:
        if entropy < 1.0:
            score *= 0.5
            reasons.append("Overconfident wrong answer (low entropy).")
        elif entropy > 3.5:
            score = min(1.0, score + 0.05)
            reasons.append("Good effort (uncertain, not collapsed).")

    # --- Settle health (delta_per_k contraction) ---
    deltas_any = stats.get("delta_per_k")
    if isinstance(deltas_any, list) and deltas_any:
        deltas = [_safe_float(d) for d in deltas_any]
        d0 = deltas[0]
        d_last = deltas[-1]
        if not math.isnan(d_last) and d_last > 10.0:
            score *= 0.7
            reasons.append("Very large settle step (unstable dynamics).")
        if len(deltas) >= 2 and not (math.isnan(d0) or math.isnan(d_last)):
            ratio = d_last / max(d0, 1e-8)
            if ratio > 1.05:
                score *= 0.85
                reasons.append("Settle loop expanding (delta not contracting).")
            elif ratio > 0.98:
                score *= 0.95
                reasons.append("Settle loop barely contracting (needs more settling).")
            elif ratio < 0.75:
                score *= 1.05
                reasons.append("Nice stable settling (contracting deltas).")

    # --- Repetition / mode collapse penalty across varying prompts ---
    if base_score < 0.9 and len(set(recent_prompts)) >= 2 and len(recent_outputs) >= 4:
        distinct_out = len(set(recent_outputs))
        if distinct_out == 1:
            score *= 0.5
            reasons.append("Looks like mode collapse (same output for different prompts).")
        else:
            counts: Dict[str, int] = {}
            for ch in recent_outputs:
                counts[ch] = counts.get(ch, 0) + 1
            dominant = max(counts.values())
            frac = dominant / max(len(recent_outputs), 1)
            if frac >= 0.9:
                score *= 0.8
                reasons.append("Low output diversity (repetition).")

    score = _clamp01(score)
    return score, " ".join(reasons)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Human-in-the-loop tutor stepper for session.py (mimicry first).")
    p.add_argument("--text-path", default="input.txt", help="Text used to build the vocab.")
    p.add_argument("--checkpoint", default="", help="Optional checkpoint to load (session format or raw state_dict).")

    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--k-settle", type=int, default=2, help="Settle steps during quiz/generate.")
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument("--targets", default="A", help="Characters to practice, e.g. 'A' or 'ABCD'.")
    p.add_argument(
        "--focus",
        default="",
        help="Optional subset of --targets to focus on during quizzes (rehearsal still uses full targets).",
    )
    p.add_argument(
        "--order",
        choices=["random", "sequential"],
        default="random",
        help="Target selection order (random or sequential cycle).",
    )
    p.add_argument(
        "--tasks",
        default="copy",
        help="Comma-separated tasks: copy,copy2,next,next2 (next2: 'N:AB' -> 'C'; copy2: 'AB' -> 'AB').",
    )
    p.add_argument(
        "--task-order",
        choices=["cycle", "random"],
        default="cycle",
        help="How to pick tasks when multiple are enabled (cycle is slower/more predictable).",
    )
    p.add_argument("--temperature", type=float, default=0.0, help="0.0 = greedy decode (recommended for grading).")
    p.add_argument(
        "--quiz-len",
        type=int,
        default=1,
        help="Max new tokens to generate during quiz (auto-increases for multi-char expected answers).",
    )

    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="AdamW weight decay used during learn() (default 0.0 for tutoring).",
    )
    p.add_argument("--grad-clip", type=float, default=1.0, help="Global grad-norm clip during learn().")
    p.add_argument("--k-settle-learn", type=int, default=2, help="Settle steps during learn().")
    p.add_argument(
        "--detach-state",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Detach recurrent state each timestep during learn() (faster; disables BPTT through time).",
    )

    p.add_argument("--min-steps", type=int, default=1, help="Min suggested learn steps when score is high.")
    p.add_argument("--max-steps", type=int, default=6, help="Max suggested learn steps when score is low.")
    p.add_argument(
        "--copy-continue-score",
        type=float,
        default=0.8,
        help="Partial credit for copy tasks when the student continues to the next letter (0..1).",
    )
    p.add_argument(
        "--next-copy-score",
        type=float,
        default=0.65,
        help="Partial credit for next tasks when the student copies instead of predicting (0..1).",
    )
    p.add_argument("--budget-steps", type=int, default=0, help="0 = unlimited; otherwise total learn-step budget.")
    p.add_argument(
        "--replay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When learning, mix in other targets as rehearsal (recommended for multi-letter).",
    )
    p.add_argument(
        "--rehearsal-mult",
        type=int,
        default=1,
        help="Rehearsal weighting during learn(): repeat rehearsal examples this many times (>=1).",
    )
    p.add_argument("--w-copy", type=int, default=1, help="Rehearsal/drill weight for copy examples (>=0).")
    p.add_argument("--w-copy2", type=int, default=1, help="Rehearsal/drill weight for copy2 examples (>=0).")
    p.add_argument("--w-next", type=int, default=1, help="Rehearsal/drill weight for next examples (>=0).")
    p.add_argument("--w-next2", type=int, default=1, help="Rehearsal/drill weight for next2 examples (>=0).")
    p.add_argument(
        "--reset-each-round",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reset recurrent state before each quiz (measures retention, not working memory).",
    )

    return p.parse_args()


def _print_help() -> None:
    print(
        "\nCommands:\n"
        "  <enter>           apply suggested learn steps\n"
        "  0 / skip          skip learning this round\n"
        "  <int>             apply that many learn steps\n"
        "  targets ABC       set practice set\n"
        "  add ABC           add characters to practice set\n"
        "  focus ABC         focus quiz on subset (or 'focus off')\n"
        "  order rand|seq    set target selection order\n"
        "  tasks copy,next   set enabled tasks (copy/copy2/next/next2)\n"
        "  taskorder rand|cycle  set task selection order\n"
        "  k <int>           set quiz k_settle\n"
        "  gen <int>         set quiz output length (max new tokens)\n"
        "  kl <int>          set learn k_settle\n"
        "  lr <float>        set learning rate\n"
        "  rehearsal <int>   set rehearsal weighting (>=1)\n"
        "  copycont <float>  set copy-continue partial credit (0..1)\n"
        "  nextcopy <float>  set next-copy partial credit (0..1)\n"
        "  wcopy <int>       set copy example weight (>=0)\n"
        "  wcopy2 <int>      set copy2 example weight (>=0)\n"
        "  wnext <int>       set next example weight (>=0)\n"
        "  wnext2 <int>      set next2 example weight (>=0)\n"
        "  wd <float>        set weight decay\n"
        "  gc <float>        set grad clip\n"
        "  detach on|off     toggle detach_state during learn\n"
        "  replay on|off     toggle rehearsal mix-in\n"
        "  reset on|off      toggle reset_each_round\n"
        "  drill <int>       run learn() across all targets\n"
        "  exam              quiz all targets (retention)\n"
        "  save <path>       save checkpoint\n"
        "  load <path>       load checkpoint\n"
        "  stats             print session stats\n"
        "  help              show this message\n"
        "  quit              exit\n"
    )


def main() -> None:
    args = _parse_args()
    if not Path(args.text_path).exists():
        raise SystemExit(f"Missing {args.text_path!r}.")

    targets = [ch for ch in str(args.targets) if ch.strip()]
    if not targets:
        raise SystemExit("--targets must include at least 1 character")

    tasks_raw = [t.strip().lower() for t in str(args.tasks).split(",") if t.strip()]
    aliases = {
        "c": "copy",
        "copy": "copy",
        "copy2": "copy2",
        "pair": "copy2",
        "digram": "copy2",
        "bigram": "copy2",
        "n": "next",
        "next": "next",
        "next2": "next2",
        "predict": "next",
    }
    tasks: List[str] = []
    for t in tasks_raw:
        tt = aliases.get(t)
        if tt is None:
            raise SystemExit(f"Unknown task {t!r}. Use copy,copy2,next,next2.")
        if tt not in tasks:
            tasks.append(tt)
    if not tasks:
        raise SystemExit("--tasks must include at least one task")
    if any(t in {"next", "next2", "copy2"} for t in tasks) and len(targets) < 2:
        raise SystemExit("Need at least 2 --targets for copy2/next/next2 tasks.")

    k_settle = int(args.k_settle)
    k_settle_learn = int(args.k_settle_learn)
    quiz_len = int(args.quiz_len)
    if quiz_len < 1:
        raise SystemExit("--quiz-len must be >= 1")
    lr = float(args.lr)
    weight_decay = float(args.weight_decay)
    grad_clip = float(args.grad_clip)
    temperature = float(args.temperature)
    reset_each_round = bool(args.reset_each_round)
    detach_state = bool(args.detach_state)
    replay = bool(args.replay)
    rehearsal_mult = int(args.rehearsal_mult)
    if rehearsal_mult < 1:
        raise SystemExit("--rehearsal-mult must be >= 1")

    w_copy = int(args.w_copy)
    w_copy2 = int(args.w_copy2)
    w_next = int(args.w_next)
    w_next2 = int(args.w_next2)
    if min(w_copy, w_copy2, w_next, w_next2) < 0:
        raise SystemExit("Weights must be >= 0 (use 0 to disable a category).")

    copy_continue_score = _clamp01(float(args.copy_continue_score))
    next_copy_score = _clamp01(float(args.next_copy_score))
    order = str(args.order)
    seq_index = 0
    task_order = str(args.task_order)
    task_index = 0

    budget_left: int | None = None
    if int(args.budget_steps) > 0:
        budget_left = int(args.budget_steps)

    tutor_rng = random.Random(int(args.seed))
    recent_prompts: List[str] = []
    recent_outputs: List[str] = []
    history_window = 24

    focus = [ch for ch in str(args.focus) if ch.strip()]
    for ch in focus:
        if ch not in targets:
            raise SystemExit(f"--focus includes {ch!r} not present in --targets")

    cmd: List[str] = [
        sys.executable,
        "-u",
        "session.py",
        "--text-path",
        args.text_path,
        "--d-model",
        str(int(args.d_model)),
        "--n-layers",
        str(int(args.n_layers)),
        "--k-settle",
        str(k_settle),
        "--seed",
        str(int(args.seed)),
    ]
    if args.checkpoint:
        cmd += ["--checkpoint", str(args.checkpoint)]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None

    try:
        _send(proc, {"cmd": "reset"})
        _print_help()

        round_i = 0
        while True:
            round_i += 1
            sel = focus if focus else targets
            if order == "sequential":
                target = sel[seq_index % len(sel)]
                seq_index += 1
            else:
                target = tutor_rng.choice(sel)

            if len(tasks) == 1:
                task = tasks[0]
            elif task_order == "cycle":
                task = tasks[task_index % len(tasks)]
                task_index += 1
            else:
                task = tutor_rng.choice(tasks)

            alphabet_tag = "".join(targets)
            seq = target
            prompt = target
            expected = target
            if task == "copy2":
                seq = target + _succ(target, targets)
                prompt = seq
                expected = seq
            elif task == "next":
                prompt = f"N:{alphabet_tag}:{seq}:n"
                expected = _succ(seq[-1], targets)
            elif task == "next2":
                seq = target + _succ(target, targets)
                prompt = f"N:{alphabet_tag}:{seq}:n"
                expected = _succ(seq[-1], targets)

            gen_len = int(len(expected))
            if task.startswith("next"):
                gen_len = max(int(quiz_len), int(gen_len))

            if reset_each_round:
                _send(proc, {"cmd": "reset"})

            _send(proc, {"cmd": "ingest", "text": f"T:{prompt} S:", "k_settle": k_settle})
            out = _send(
                proc,
                {
                    "cmd": "generate",
                    "max_new_tokens": int(gen_len),
                    "temperature": temperature,
                    "k_settle": k_settle,
                },
            )
            got = str(out.get("text", ""))
            stats = out.get("stats", {})
            if not isinstance(stats, dict):
                stats = {}

            recent_prompts.append(prompt)
            recent_outputs.append(got)
            recent_prompts = recent_prompts[-history_window:]
            recent_outputs = recent_outputs[-history_window:]

            base_score, base_msg = _grade_task(
                expected,
                got,
                task=task,
                seq=seq,
                alphabet=targets,
                copy_continue_score=copy_continue_score,
                next_copy_score=next_copy_score,
            )
            score, sentiment = _grade_with_stats(
                expected,
                got,
                task=task,
                seq=seq,
                alphabet=targets,
                copy_continue_score=copy_continue_score,
                next_copy_score=next_copy_score,
                stats=stats,
                recent_prompts=recent_prompts,
                recent_outputs=recent_outputs,
            )

            entropy = _safe_float(stats.get("logits_entropy"))
            deltas = stats.get("delta_per_k")
            d_str = f"delta={deltas}" if isinstance(deltas, list) and deltas else "delta=?"
            e_str = "H=?" if math.isnan(entropy) else f"H={entropy:.2f}"

            budget_str = "unlimited" if budget_left is None else str(budget_left)
            focus_str = "" if not focus else f" | focus={''.join(focus)!r}"
            print(f"\nRound {round_i} | targets={''.join(targets)!r}{focus_str} | k={k_settle} | budget={budget_str}")
            print(f"  tasks={','.join(tasks)} | task_order={task_order}")
            print(f"  lr={lr:g} | weight_decay={weight_decay:g} | grad_clip={grad_clip:g}")
            print(f"  copycont={copy_continue_score:.2f} | nextcopy={next_copy_score:.2f} | rehearsal_mult={rehearsal_mult}")
            print(f"  w_copy={w_copy} | w_copy2={w_copy2} | w_next={w_next} | w_next2={w_next2}")
            print(f"  task={task} | prompt={prompt!r} | expected={expected!r} | student={got!r} (gen_len={gen_len})")
            print(f"  base_score={base_score:.2f} ({base_msg})")
            print(f"  score={score:.2f} | {e_str} | {d_str}")
            print(f"  sentiment: {sentiment}")

            suggested = 0
            if not (base_score >= 1.0 and score >= 0.9):
                suggested = _steps_from_score(score, min_steps=int(args.min_steps), max_steps=int(args.max_steps))
                suggested = max(1, int(suggested))
            if budget_left is not None:
                suggested = min(int(suggested), int(budget_left))

            replay_str = "on" if replay else "off"
            print(
                f"Suggested learn steps: {suggested} (k_learn={k_settle_learn}, detach_state={detach_state}, replay={replay_str})"
            )

            advance_round = False
            while True:
                try:
                    raw = input("Action> ").strip()
                except EOFError:
                    raw = "quit"

                if raw == "":
                    steps = suggested
                    break
                if raw in {"q", "quit", "exit"}:
                    _send(proc, {"cmd": "exit"})
                    return
                if raw in {"help", "h", "?"}:
                    _print_help()
                    continue
                if raw in {"0", "skip", "s"}:
                    steps = 0
                    break
                if raw == "stats":
                    print(_send(proc, {"cmd": "stats"}))
                    continue
                if raw.startswith("drill "):
                    _, val = raw.split(" ", 1)
                    n = int(val)
                    if n <= 0:
                        print("Usage: drill <int> (must be >= 1)")
                        continue
                    if budget_left is not None:
                        n = min(n, budget_left)
                    if budget_left is not None and budget_left <= 0:
                        print("Budget exhausted.")
                        continue
                    drill_examples: List[Dict[str, str]] = []
                    alpha_tag = "".join(targets)
                    for ch in targets:
                        if "copy" in tasks:
                            drill_examples.extend([{"prompt": ch, "answer": ch}] * max(w_copy, 0))
                        if "copy2" in tasks:
                            seq2 = ch + _succ(ch, targets)
                            drill_examples.extend([{"prompt": seq2, "answer": seq2}] * max(w_copy2, 0))
                        if "next" in tasks:
                            drill_examples.extend(
                                [{"prompt": f"N:{alpha_tag}:{ch}:n", "answer": _succ(ch, targets)}] * max(w_next, 0)
                            )
                        if "next2" in tasks:
                            seq2 = ch + _succ(ch, targets)
                            drill_examples.extend(
                                [
                                    {"prompt": f"N:{alpha_tag}:{seq2}:n", "answer": _succ(seq2[-1], targets)}
                                ]
                                * max(w_next2, 0)
                            )

                    resp = _send(
                        proc,
                        {
                            "cmd": "learn",
                            "examples": drill_examples,
                            "steps": int(n),
                            "lr": float(lr),
                            "weight_decay": float(weight_decay),
                            "grad_clip": float(grad_clip),
                            "k_settle": int(k_settle_learn),
                            "detach_state": bool(detach_state),
                            "loss_mode": "answer_only",
                            "seed": int(args.seed) + round_i,
                        },
                    )
                    train = resp.get("stats", {}).get("train", {})
                    print(f"Drill applied: {n} steps | train={train}")
                    if budget_left is not None:
                        budget_left -= int(n)
                    # Re-quiz the same target next round (especially useful in sequential mode).
                    if order == "sequential":
                        seq_index -= 1
                    advance_round = True
                    break
                if raw == "exam":
                    print("\nExam (retention):")
                    alpha_tag = "".join(targets)
                    for ch in targets:
                        for tsk in tasks:
                            if tsk == "copy":
                                seq2 = ch
                                prompt2 = ch
                                expected2 = ch
                            elif tsk == "copy2":
                                seq2 = ch + _succ(ch, targets)
                                prompt2 = seq2
                                expected2 = seq2
                            elif tsk == "next":
                                seq2 = ch
                                prompt2 = f"N:{alpha_tag}:{ch}:n"
                                expected2 = _succ(ch, targets)
                            elif tsk == "next2":
                                seq2 = ch + _succ(ch, targets)
                                prompt2 = f"N:{alpha_tag}:{seq2}:n"
                                expected2 = _succ(seq2[-1], targets)
                            else:
                                raise RuntimeError(f"Unknown task {tsk!r}")

                            gen2 = int(len(expected2))
                            if tsk.startswith("next"):
                                gen2 = max(int(quiz_len), int(gen2))
                            _send(proc, {"cmd": "reset"})
                            _send(proc, {"cmd": "ingest", "text": f"T:{prompt2} S:", "k_settle": k_settle})
                            out2 = _send(
                                proc,
                                {
                                    "cmd": "generate",
                                    "max_new_tokens": int(gen2),
                                    "temperature": temperature,
                                    "k_settle": k_settle,
                                },
                            )
                            got2 = str(out2.get("text", ""))
                            stats2 = out2.get("stats", {})
                            if not isinstance(stats2, dict):
                                stats2 = {}
                            s2, msg2 = _grade_with_stats(
                                expected2,
                                got2,
                                task=tsk,
                                seq=seq2,
                                alphabet=targets,
                                copy_continue_score=copy_continue_score,
                                next_copy_score=next_copy_score,
                                stats=stats2,
                                recent_prompts=recent_prompts,
                                recent_outputs=recent_outputs,
                            )
                            print(f"  {tsk}:{prompt2!r} -> {got2!r} (exp {expected2!r}) | score={s2:.2f} | {msg2}")
                    continue
                if raw.startswith("targets "):
                    _, val = raw.split(" ", 1)
                    targets = [ch for ch in val if ch.strip()]
                    if not targets:
                        print("Need at least 1 target.")
                        continue
                    if any(t in {"next", "next2", "copy2"} for t in tasks) and len(targets) < 2:
                        print("Need at least 2 targets for copy2/next/next2; keeping previous targets.")
                        continue
                    focus = [ch for ch in focus if ch in targets]
                    seq_index = 0
                    print(f"targets = {''.join(targets)!r}")
                    continue
                if raw.startswith("add "):
                    _, val = raw.split(" ", 1)
                    add_chars = [ch for ch in val if ch.strip()]
                    for ch in add_chars:
                        if ch not in targets:
                            targets.append(ch)
                    focus = [ch for ch in focus if ch in targets]
                    seq_index = 0
                    print(f"targets = {''.join(targets)!r}")
                    continue
                if raw.startswith("focus "):
                    _, val = raw.split(" ", 1)
                    val = val.strip()
                    if val.lower() in {"off", "none", "all", "*"}:
                        focus = []
                        print("focus = off")
                        continue
                    new_focus = [ch for ch in val if ch.strip()]
                    missing = [ch for ch in new_focus if ch not in targets]
                    if missing:
                        print(f"focus contains characters not in targets: {missing!r}")
                        continue
                    focus = new_focus
                    seq_index = 0
                    print(f"focus = {''.join(focus)!r}")
                    continue
                if raw.startswith("tasks "):
                    _, val = raw.split(" ", 1)
                    raw_tasks = [t.strip().lower() for t in val.split(",") if t.strip()]
                    new_tasks: List[str] = []
                    for t in raw_tasks:
                        tt = aliases.get(t)
                        if tt is None:
                            print("Usage: tasks copy,copy2,next,next2")
                            new_tasks = []
                            break
                        if tt not in new_tasks:
                            new_tasks.append(tt)
                    if not new_tasks:
                        continue
                    if any(t in {"next", "next2", "copy2"} for t in new_tasks) and len(targets) < 2:
                        print("Need at least 2 targets for copy2/next/next2.")
                        continue
                    tasks = new_tasks
                    task_index = 0
                    print(f"tasks = {','.join(tasks)}")
                    continue
                if raw.startswith("taskorder "):
                    _, val = raw.split(" ", 1)
                    val = val.strip().lower()
                    if val in {"rand", "random"}:
                        task_order = "random"
                    elif val in {"cycle", "seq", "sequential"}:
                        task_order = "cycle"
                        task_index = 0
                    else:
                        print("Usage: taskorder rand|cycle")
                        continue
                    print(f"task_order = {task_order}")
                    continue
                if raw.startswith("order "):
                    _, val = raw.split(" ", 1)
                    val = val.strip().lower()
                    if val in {"rand", "random"}:
                        order = "random"
                    elif val in {"seq", "sequential"}:
                        order = "sequential"
                        seq_index = 0
                    else:
                        print("Usage: order rand|seq")
                        continue
                    print(f"order = {order}")
                    continue
                if raw.startswith("k "):
                    _, val = raw.split(" ", 1)
                    k_settle = int(val)
                    print(f"k_settle = {k_settle}")
                    continue
                if raw.startswith("gen "):
                    _, val = raw.split(" ", 1)
                    quiz_len = int(val)
                    if quiz_len < 1:
                        print("gen must be >= 1")
                        continue
                    print(f"quiz_len = {quiz_len}")
                    continue
                if raw.startswith("kl "):
                    _, val = raw.split(" ", 1)
                    k_settle_learn = int(val)
                    print(f"k_settle_learn = {k_settle_learn}")
                    continue
                if raw.startswith("lr "):
                    _, val = raw.split(" ", 1)
                    lr = float(val)
                    print(f"lr = {lr:g}")
                    continue
                if raw.startswith("rehearsal "):
                    _, val = raw.split(" ", 1)
                    rehearsal_mult = int(val)
                    if rehearsal_mult < 1:
                        print("rehearsal must be >= 1")
                        continue
                    print(f"rehearsal_mult = {rehearsal_mult}")
                    continue
                if raw.startswith("copycont "):
                    _, val = raw.split(" ", 1)
                    copy_continue_score = _clamp01(float(val))
                    print(f"copy_continue_score = {copy_continue_score:g}")
                    continue
                if raw.startswith("nextcopy "):
                    _, val = raw.split(" ", 1)
                    next_copy_score = _clamp01(float(val))
                    print(f"next_copy_score = {next_copy_score:g}")
                    continue
                if raw.startswith("wcopy "):
                    _, val = raw.split(" ", 1)
                    w_copy = int(val)
                    if w_copy < 0:
                        print("wcopy must be >= 0")
                        continue
                    print(f"w_copy = {w_copy}")
                    continue
                if raw.startswith("wcopy2 "):
                    _, val = raw.split(" ", 1)
                    w_copy2 = int(val)
                    if w_copy2 < 0:
                        print("wcopy2 must be >= 0")
                        continue
                    print(f"w_copy2 = {w_copy2}")
                    continue
                if raw.startswith("wnext "):
                    _, val = raw.split(" ", 1)
                    w_next = int(val)
                    if w_next < 0:
                        print("wnext must be >= 0")
                        continue
                    print(f"w_next = {w_next}")
                    continue
                if raw.startswith("wnext2 "):
                    _, val = raw.split(" ", 1)
                    w_next2 = int(val)
                    if w_next2 < 0:
                        print("wnext2 must be >= 0")
                        continue
                    print(f"w_next2 = {w_next2}")
                    continue
                if raw.startswith("wd "):
                    _, val = raw.split(" ", 1)
                    weight_decay = float(val)
                    print(f"weight_decay = {weight_decay:g}")
                    continue
                if raw.startswith("gc "):
                    _, val = raw.split(" ", 1)
                    grad_clip = float(val)
                    print(f"grad_clip = {grad_clip:g}")
                    continue
                if raw.startswith("detach "):
                    _, val = raw.split(" ", 1)
                    val = val.strip().lower()
                    if val in {"on", "true", "1", "yes"}:
                        detach_state = True
                    elif val in {"off", "false", "0", "no"}:
                        detach_state = False
                    else:
                        print("Usage: detach on|off")
                        continue
                    print(f"detach_state = {detach_state}")
                    continue
                if raw.startswith("reset "):
                    _, val = raw.split(" ", 1)
                    val = val.strip().lower()
                    if val in {"on", "true", "1", "yes"}:
                        reset_each_round = True
                    elif val in {"off", "false", "0", "no"}:
                        reset_each_round = False
                    else:
                        print("Usage: reset on|off")
                        continue
                    print(f"reset_each_round = {reset_each_round}")
                    continue
                if raw.startswith("replay "):
                    _, val = raw.split(" ", 1)
                    val = val.strip().lower()
                    if val in {"on", "true", "1", "yes"}:
                        replay = True
                    elif val in {"off", "false", "0", "no"}:
                        replay = False
                    else:
                        print("Usage: replay on|off")
                        continue
                    print(f"replay = {replay}")
                    continue
                if raw.startswith("save "):
                    _, path = raw.split(" ", 1)
                    path = path.strip()
                    if not path:
                        print("Usage: save <path>")
                        continue
                    print(_send(proc, {"cmd": "save", "path": path}))
                    continue
                if raw.startswith("load "):
                    _, path = raw.split(" ", 1)
                    path = path.strip()
                    if not path:
                        print("Usage: load <path>")
                        continue
                    print(_send(proc, {"cmd": "load", "path": path}))
                    continue

                # Try to parse as an integer step count.
                try:
                    steps = int(raw)
                    if steps < 0:
                        raise ValueError
                    if budget_left is not None:
                        steps = min(steps, budget_left)
                    break
                except ValueError:
                    # Fall back: show the raw command that failed so we can debug typos.
                    print(f"Unknown command: {shlex.quote(raw)} (type 'help')")
                    continue

            if advance_round:
                continue

            if steps <= 0:
                print("No learning this round.")
                continue

            if budget_left is not None and budget_left <= 0:
                print("Budget exhausted. Use 'save' then restart with a new budget, or set --budget-steps 0.")
                continue

            current_example = {"prompt": prompt, "answer": expected}
            rehearsal_examples: List[Dict[str, str]] = []
            if replay:
                for ch in targets:
                    if "copy" in tasks:
                        rehearsal_examples.extend([{"prompt": ch, "answer": ch}] * max(w_copy, 0))
                    if "copy2" in tasks:
                        seq2 = ch + _succ(ch, targets)
                        rehearsal_examples.extend([{"prompt": seq2, "answer": seq2}] * max(w_copy2, 0))
                    if "next" in tasks:
                        rehearsal_examples.extend(
                            [{"prompt": f"N:{alphabet_tag}:{ch}:n", "answer": _succ(ch, targets)}] * max(w_next, 0)
                        )
                    if "next2" in tasks:
                        seq2 = ch + _succ(ch, targets)
                        rehearsal_examples.extend(
                            [
                                {"prompt": f"N:{alphabet_tag}:{seq2}:n", "answer": _succ(seq2[-1], targets)}
                            ]
                            * max(w_next2, 0)
                        )

            learn_resp = _send(
                proc,
                {
                    "cmd": "learn",
                    "examples": (
                        [current_example]
                        if not replay
                        else (
                            rehearsal_examples * int(rehearsal_mult)
                            + [current_example] * max(len(rehearsal_examples), 1)
                        )
                    ),
                    "steps": int(steps),
                    "lr": float(lr),
                    "weight_decay": float(weight_decay),
                    "grad_clip": float(grad_clip),
                    "k_settle": int(k_settle_learn),
                    "detach_state": bool(detach_state),
                    "loss_mode": "answer_only",
                    "seed": int(args.seed) + round_i,
                },
            )
            train = learn_resp.get("stats", {}).get("train", {})
            print(f"Applied learn steps: {steps} | train={train}")
            if budget_left is not None:
                budget_left -= int(steps)
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
