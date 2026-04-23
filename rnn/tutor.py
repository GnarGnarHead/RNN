from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple


CANON_ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
VALID_TASKS = ("copy", "copy2", "next", "next2")
TASK_ALIASES: Mapping[str, str] = {
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


@dataclass(frozen=True)
class Lesson:
    task: str
    target: str
    seq: str
    prompt: str
    expected: str

    def example(self) -> Dict[str, str]:
        return {"prompt": self.prompt, "answer": self.expected}


def normalize_tasks(tasks: str | Sequence[str]) -> List[str]:
    if isinstance(tasks, str):
        raw_tasks = [t.strip().lower() for t in tasks.split(",") if t.strip()]
    else:
        raw_tasks = [str(t).strip().lower() for t in tasks if str(t).strip()]

    out: List[str] = []
    for task in raw_tasks:
        normalized = TASK_ALIASES.get(task)
        if normalized is None:
            valid = ",".join(VALID_TASKS)
            raise ValueError(f"Unknown task {task!r}. Use {valid}.")
        if normalized not in out:
            out.append(normalized)
    if not out:
        raise ValueError("At least one task is required.")
    return out


def succ(ch: str, alphabet: Sequence[str]) -> str:
    if ch not in alphabet:
        raise ValueError(f"Character {ch!r} not in targets/alphabet")
    i = list(alphabet).index(ch)
    return alphabet[(i + 1) % len(alphabet)]


def build_lesson(task: str, target: str, alphabet: Sequence[str]) -> Lesson:
    task = normalize_tasks([task])[0]
    if target not in alphabet:
        raise ValueError(f"Target {target!r} not in alphabet.")
    if task in {"copy2", "next", "next2"} and len(alphabet) < 2:
        raise ValueError(f"Task {task!r} requires at least two targets.")

    alphabet_tag = "".join(alphabet)
    seq = target
    prompt = target
    expected = target

    if task == "copy2":
        seq = target + succ(target, alphabet)
        prompt = seq
        expected = seq
    elif task == "next":
        prompt = f"N:{alphabet_tag}:{seq}:n"
        expected = succ(seq[-1], alphabet)
    elif task == "next2":
        seq = target + succ(target, alphabet)
        prompt = f"N:{alphabet_tag}:{seq}:n"
        expected = succ(seq[-1], alphabet)

    return Lesson(task=task, target=target, seq=seq, prompt=prompt, expected=expected)


def task_examples(
    targets: Sequence[str],
    tasks: Sequence[str],
    *,
    weights: Mapping[str, int] | None = None,
) -> List[Dict[str, str]]:
    weights = weights or {}
    examples: List[Dict[str, str]] = []
    for ch in targets:
        for task in normalize_tasks(tasks):
            weight = int(weights.get(task, 1))
            if weight <= 0:
                continue
            examples.extend([build_lesson(task, ch, targets).example()] * weight)
    return examples


def select_target_task(
    targets: Sequence[str],
    tasks: Sequence[str],
    *,
    order: str,
    task_order: str,
    seq_index: int,
    task_index: int,
    rng: random.Random,
) -> Tuple[str, str, int, int]:
    """
    Pick a target/task pair for the next tutor round.

    Sequential target order plus cyclic task order is scheduled as a flattened
    cross product. That covers every pair before repeating and avoids starving
    combinations such as copy:H when focus=GH and tasks=copy,copy2.
    """
    if not targets:
        raise ValueError("targets must not be empty")
    normalized_tasks = normalize_tasks(tasks)
    if order not in {"random", "sequential"}:
        raise ValueError("order must be 'random' or 'sequential'")
    if task_order not in {"random", "cycle"}:
        raise ValueError("task_order must be 'random' or 'cycle'")

    if order == "sequential" and task_order == "cycle":
        flat = seq_index % (len(targets) * len(normalized_tasks))
        target = targets[flat // len(normalized_tasks)]
        task = normalized_tasks[flat % len(normalized_tasks)]
        return target, task, seq_index + 1, task_index + 1

    if order == "sequential":
        target = targets[seq_index % len(targets)]
        seq_index += 1
    else:
        target = rng.choice(list(targets))

    if len(normalized_tasks) == 1:
        task = normalized_tasks[0]
    elif task_order == "cycle":
        task = normalized_tasks[task_index % len(normalized_tasks)]
        task_index += 1
    else:
        task = rng.choice(normalized_tasks)

    return target, task, seq_index, task_index


def _canon(ch: str) -> str | None:
    if not ch:
        return None
    c = ch[0].upper()
    if c in CANON_ALPHA:
        return c
    return None


def _global_succ(ch: str) -> str | None:
    c = _canon(ch)
    if c is None:
        return None
    i = CANON_ALPHA.index(c)
    return CANON_ALPHA[(i + 1) % len(CANON_ALPHA)]


def grade_task(
    expected: str,
    got: str,
    *,
    task: str,
    seq: str,
    alphabet: Sequence[str],
    copy_continue_score: float,
    next_copy_score: float,
) -> Tuple[float, str]:
    if not got:
        return 0.0, "No answer yet - let's try again."
    if "\n" in got or "\r" in got:
        return 0.05, "Oops, a newline - let's stick to letters for now."

    last = seq[-1] if seq else ""
    next1 = ""
    next2 = ""
    if last:
        try:
            next1 = succ(last, alphabet)
            next2 = next1 + succ(next1, alphabet)
        except Exception:
            next1 = ""
            next2 = ""

    g_next1 = _global_succ(last) or ""
    g_next2 = ""
    if g_next1:
        g_next2 = g_next1 + (_global_succ(g_next1) or "")

    g_lo = got.lower()
    e_lo = expected.lower()

    if got == expected:
        return 1.0, "Perfect!"
    if g_lo == e_lo:
        return 0.9, "Good job (right letter, wrong case)."

    if expected and g_lo.startswith(e_lo):
        if len(expected) == 1 and len(got) >= 2:
            e0 = expected[0]
            e_canon = next((a for a in alphabet if a.lower() == e0.lower()), None)
            if e_canon is not None:
                try:
                    e_succ = succ(e_canon, alphabet)
                except Exception:
                    e_succ = ""
                if e_succ and got[1].lower() == e_succ.lower():
                    return 1.0, "Perfect (and you kept going)!"
        return 0.85, "Good (right start)."

    if task in {"copy", "copy2"}:
        if next1 and got and got[0].lower() == next1.lower():
            if g_lo == next1.lower():
                return copy_continue_score, "Good (you continued to the next letter)."
            if next2 and g_lo.startswith(next2.lower()):
                return min(
                    copy_continue_score + 0.1, 0.95
                ), "Great (you continued two letters)!"
            return max(0.0, copy_continue_score - 0.05), (
                "Good (you started continuing to the next letter)."
            )
        if g_next1 and got and got[0].lower() == g_next1.lower():
            if g_lo == g_next1.lower():
                return copy_continue_score, "Good (you continued to a new letter)."
            if g_next2 and g_lo.startswith(g_next2.lower()):
                return min(
                    copy_continue_score + 0.1, 0.95
                ), "Great (you continued two new letters)!"
            return max(0.0, copy_continue_score - 0.05), (
                "Good (you started continuing to a new letter)."
            )
        if expected and got and got[0].lower() == expected[0].lower():
            return 0.55, "Nice start (first letter right)."

    if task.startswith("next"):
        if seq and (g_lo == seq.lower() or g_lo == seq[-1].lower()):
            return (
                next_copy_score,
                "Nice try (you copied instead of predicting the next).",
            )
        if g_next1 and got and got[0].lower() == g_next1.lower():
            if expected and expected[0].lower() != g_next1.lower():
                if g_next2 and g_lo.startswith(g_next2.lower()):
                    return 0.9, "Great (you predicted a new letter and kept going)!"
                return (
                    0.8,
                    "Good (you predicted a new letter beyond the current wrap-around).",
                )
        try:
            prev = alphabet[(alphabet.index(expected[0]) - 1) % len(alphabet)]
            nxt = alphabet[(alphabet.index(expected[0]) + 1) % len(alphabet)]
        except Exception:
            prev = ""
            nxt = ""
        g0 = got[0]
        if prev and (g0 == prev or g0.lower() == prev.lower()):
            return 0.35, "Close (off by one: previous letter)."
        if nxt and (g0 == nxt or g0.lower() == nxt.lower()):
            return 0.35, "Close (off by one: next letter)."

    if got and got[0].isalpha():
        return 0.2, "Nice try (wrong letter, but you answered)."
    return 0.1, "You answered!"


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def steps_from_score(score: float, *, min_steps: int, max_steps: int) -> int:
    s = clamp01(score)
    if max_steps < min_steps:
        raise ValueError("max_steps must be >= min_steps")
    steps_f = min_steps + (max_steps - min_steps) * (1.0 - s)
    return int(round(steps_f))


def grade_with_stats(
    expected: str,
    got: str,
    *,
    task: str,
    seq: str,
    alphabet: Sequence[str],
    copy_continue_score: float,
    next_copy_score: float,
    stats: Mapping[str, Any],
    recent_prompts: Sequence[str],
    recent_outputs: Sequence[str],
) -> Tuple[float, str]:
    base_score, base_msg = grade_task(
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

    entropy = safe_float(stats.get("logits_entropy"))
    if not math.isnan(entropy) and base_score < 0.6:
        if entropy < 1.0:
            score *= 0.5
            reasons.append("Overconfident wrong answer (low entropy).")
        elif entropy > 3.5:
            score = min(1.0, score + 0.05)
            reasons.append("Good effort (uncertain, not collapsed).")

    deltas_any = stats.get("delta_per_k")
    if isinstance(deltas_any, list) and deltas_any:
        deltas = [safe_float(d) for d in deltas_any]
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

    if base_score < 0.9 and len(set(recent_prompts)) >= 2 and len(recent_outputs) >= 4:
        distinct_out = len(set(recent_outputs))
        if distinct_out == 1:
            score *= 0.5
            reasons.append(
                "Looks like mode collapse (same output for different prompts)."
            )
        else:
            counts: Dict[str, int] = {}
            for ch in recent_outputs:
                counts[ch] = counts.get(ch, 0) + 1
            dominant = max(counts.values())
            frac = dominant / max(len(recent_outputs), 1)
            if frac >= 0.9:
                score *= 0.8
                reasons.append("Low output diversity (repetition).")

    return clamp01(score), " ".join(reasons)
