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


def _grade_task(expected: str, got: str, *, task: str, seq: str, alphabet: List[str]) -> Tuple[float, str]:
    """
    task:
      - copy: expect to echo seq (currently 1-char)
      - next / next2: expect successor of last char in seq
    """
    if not got:
        return 0.0, "No answer yet — let's try again."

    g0 = got[0]
    if g0 == expected:
        return 1.0, "Perfect!"
    if g0.lower() == expected.lower():
        return 0.9, "Good job (right letter, wrong case)."

    # Task-specific partial credit (kindergarten: “reasonable mistake” still earns points).
    if task.startswith("next"):
        last = seq[-1] if seq else ""
        if last and (g0 == last or g0.lower() == last.lower()):
            return 0.5, "Nice try (you copied the last letter instead of predicting the next)."
        # Off-by-one around the target alphabet cycle.
        try:
            prev = alphabet[(alphabet.index(expected) - 1) % len(alphabet)]
            nxt = alphabet[(alphabet.index(expected) + 1) % len(alphabet)]
        except ValueError:
            prev = ""
            nxt = ""
        if g0 in {prev, nxt} or g0.lower() in {prev.lower(), nxt.lower()}:
            return 0.35, "Close (off by one in the alphabet cycle)."

    if task == "copy":
        # If we accidentally predicted the successor, that's a “close” mistake once we start mixing tasks.
        if expected in alphabet:
            s = _succ(expected, alphabet)
            if g0 == s or g0.lower() == s.lower():
                return 0.35, "Close (you predicted the next letter instead of copying)."

    if g0.isalpha():
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
    stats: Dict[str, Any],
    recent_prompts: List[str],
    recent_outputs: List[str],
) -> Tuple[float, str]:
    base_score, base_msg = _grade_task(expected, got, task=task, seq=seq, alphabet=alphabet)
    score = float(base_score)
    reasons = [base_msg]

    # --- Confidence shaping (entropy) ---
    entropy = _safe_float(stats.get("logits_entropy"))
    if not math.isnan(entropy) and base_score < 0.9:
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
        "--order",
        choices=["random", "sequential"],
        default="random",
        help="Target selection order (random or sequential cycle).",
    )
    p.add_argument(
        "--tasks",
        default="copy",
        help="Comma-separated tasks: copy,next,next2 (next2 uses 2-char prompts like 'AB' -> 'C').",
    )
    p.add_argument(
        "--task-order",
        choices=["cycle", "random"],
        default="cycle",
        help="How to pick tasks when multiple are enabled (cycle is slower/more predictable).",
    )
    p.add_argument("--temperature", type=float, default=0.0, help="0.0 = greedy decode (recommended for grading).")

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
    p.add_argument("--budget-steps", type=int, default=0, help="0 = unlimited; otherwise total learn-step budget.")
    p.add_argument(
        "--replay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When learning, mix in other targets as rehearsal (recommended for multi-letter).",
    )
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
        "  order rand|seq    set target selection order\n"
        "  tasks copy,next   set enabled tasks (copy/next/next2)\n"
        "  taskorder rand|cycle  set task selection order\n"
        "  k <int>           set quiz k_settle\n"
        "  kl <int>          set learn k_settle\n"
        "  lr <float>        set learning rate\n"
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
    aliases = {"c": "copy", "copy": "copy", "n": "next", "next": "next", "next2": "next2", "predict": "next"}
    tasks: List[str] = []
    for t in tasks_raw:
        tt = aliases.get(t)
        if tt is None:
            raise SystemExit(f"Unknown task {t!r}. Use copy,next,next2.")
        if tt not in tasks:
            tasks.append(tt)
    if not tasks:
        raise SystemExit("--tasks must include at least one task")
    if any(t.startswith("next") for t in tasks) and len(targets) < 2:
        raise SystemExit("Need at least 2 --targets for next/next2 tasks.")

    k_settle = int(args.k_settle)
    k_settle_learn = int(args.k_settle_learn)
    lr = float(args.lr)
    weight_decay = float(args.weight_decay)
    grad_clip = float(args.grad_clip)
    temperature = float(args.temperature)
    reset_each_round = bool(args.reset_each_round)
    detach_state = bool(args.detach_state)
    replay = bool(args.replay)
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
            if order == "sequential":
                target = targets[seq_index % len(targets)]
                seq_index += 1
            else:
                target = tutor_rng.choice(targets)

            if len(tasks) == 1:
                task = tasks[0]
            elif task_order == "cycle":
                task = tasks[task_index % len(tasks)]
                task_index += 1
            else:
                task = tutor_rng.choice(tasks)

            seq = target
            prompt = target
            expected = target
            if task == "next":
                prompt = f"N:{seq}"
                expected = _succ(seq[-1], targets)
            elif task == "next2":
                seq = target + _succ(target, targets)
                prompt = f"N:{seq}"
                expected = _succ(seq[-1], targets)

            if reset_each_round:
                _send(proc, {"cmd": "reset"})

            _send(proc, {"cmd": "ingest", "text": f"T:{prompt} S:", "k_settle": k_settle})
            out = _send(
                proc,
                {
                    "cmd": "generate",
                    "max_new_tokens": 1,
                    "temperature": temperature,
                    "k_settle": k_settle,
                },
            )
            got = str(out.get("text", ""))
            stats = out.get("stats", {})
            if not isinstance(stats, dict):
                stats = {}

            g0 = got[:1]
            recent_prompts.append(prompt)
            recent_outputs.append(g0)
            recent_prompts = recent_prompts[-history_window:]
            recent_outputs = recent_outputs[-history_window:]

            base_score, base_msg = _grade_task(expected, got, task=task, seq=seq, alphabet=targets)
            score, sentiment = _grade_with_stats(
                expected,
                got,
                task=task,
                seq=seq,
                alphabet=targets,
                stats=stats,
                recent_prompts=recent_prompts,
                recent_outputs=recent_outputs,
            )

            entropy = _safe_float(stats.get("logits_entropy"))
            deltas = stats.get("delta_per_k")
            d_str = f"delta={deltas}" if isinstance(deltas, list) and deltas else "delta=?"
            e_str = "H=?" if math.isnan(entropy) else f"H={entropy:.2f}"

            budget_str = "unlimited" if budget_left is None else str(budget_left)
            print(f"\nRound {round_i} | targets={''.join(targets)!r} | k={k_settle} | budget={budget_str}")
            print(f"  tasks={','.join(tasks)} | task_order={task_order}")
            print(f"  lr={lr:g} | weight_decay={weight_decay:g} | grad_clip={grad_clip:g}")
            print(f"  task={task} | prompt={prompt!r} | expected={expected!r} | student={got!r}")
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
                    for ch in targets:
                        if "copy" in tasks:
                            drill_examples.append({"prompt": ch, "answer": ch})
                        if "next" in tasks:
                            drill_examples.append({"prompt": f"N:{ch}", "answer": _succ(ch, targets)})
                        if "next2" in tasks:
                            seq2 = ch + _succ(ch, targets)
                            drill_examples.append({"prompt": f"N:{seq2}", "answer": _succ(seq2[-1], targets)})

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
                    for ch in targets:
                        for tsk in tasks:
                            if tsk == "copy":
                                seq2 = ch
                                prompt2 = ch
                                expected2 = ch
                            elif tsk == "next":
                                seq2 = ch
                                prompt2 = f"N:{ch}"
                                expected2 = _succ(ch, targets)
                            else:
                                seq2 = ch + _succ(ch, targets)
                                prompt2 = f"N:{seq2}"
                                expected2 = _succ(seq2[-1], targets)

                            _send(proc, {"cmd": "reset"})
                            _send(proc, {"cmd": "ingest", "text": f"T:{prompt2} S:", "k_settle": k_settle})
                            out2 = _send(
                                proc,
                                {
                                    "cmd": "generate",
                                    "max_new_tokens": 1,
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
                    if any(t.startswith("next") for t in tasks) and len(targets) < 2:
                        print("Need at least 2 targets for next/next2; keeping previous targets.")
                        continue
                    seq_index = 0
                    print(f"targets = {''.join(targets)!r}")
                    continue
                if raw.startswith("add "):
                    _, val = raw.split(" ", 1)
                    add_chars = [ch for ch in val if ch.strip()]
                    for ch in add_chars:
                        if ch not in targets:
                            targets.append(ch)
                    seq_index = 0
                    print(f"targets = {''.join(targets)!r}")
                    continue
                if raw.startswith("tasks "):
                    _, val = raw.split(" ", 1)
                    raw_tasks = [t.strip().lower() for t in val.split(",") if t.strip()]
                    new_tasks: List[str] = []
                    for t in raw_tasks:
                        tt = aliases.get(t)
                        if tt is None:
                            print("Usage: tasks copy,next,next2")
                            new_tasks = []
                            break
                        if tt not in new_tasks:
                            new_tasks.append(tt)
                    if not new_tasks:
                        continue
                    if any(t.startswith("next") for t in new_tasks) and len(targets) < 2:
                        print("Need at least 2 targets for next/next2.")
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
                        rehearsal_examples.append({"prompt": ch, "answer": ch})
                    if "next" in tasks:
                        rehearsal_examples.append({"prompt": f"N:{ch}", "answer": _succ(ch, targets)})
                    if "next2" in tasks:
                        seq2 = ch + _succ(ch, targets)
                        rehearsal_examples.append({"prompt": f"N:{seq2}", "answer": _succ(seq2[-1], targets)})

            learn_resp = _send(
                proc,
                {
                    "cmd": "learn",
                    "examples": ([current_example] if not replay else (rehearsal_examples + [current_example] * max(len(rehearsal_examples), 1))),
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
