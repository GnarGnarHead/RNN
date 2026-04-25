from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rnn.model import ModelCfg, SettleCharLM  # noqa: E402
from rnn.session import SettleSession  # noqa: E402
from rnn.tutor import (  # noqa: E402
    build_lesson,
    grade_with_stats,
    maintenance_examples,
    normalize_tasks,
)
from rnn.vocab import Vocab  # noqa: E402


@dataclass(frozen=True)
class Candidate:
    round_i: int
    try_i: int
    steps: int
    lr: float
    focus_mult: int


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _model_cfg_from_checkpoint(ckpt: Any, args: argparse.Namespace) -> ModelCfg:
    if isinstance(ckpt, dict) and isinstance(ckpt.get("model_cfg"), dict):
        valid = set(ModelCfg.__dataclass_fields__.keys())
        return ModelCfg(**{k: v for k, v in ckpt["model_cfg"].items() if k in valid})
    return ModelCfg(d_model=int(args.d_model), n_layers=int(args.n_layers))


def _checkpoint_to_session(
    path: Path, *, args: argparse.Namespace, vocab: Vocab, device: torch.device
) -> SettleSession:
    ckpt = torch.load(path, map_location=device)
    cfg = _model_cfg_from_checkpoint(ckpt, args)
    model = SettleCharLM(vocab.size, cfg).to(device)
    sess = SettleSession(model, vocab, device=device)
    if isinstance(ckpt, dict) and "model" in ckpt and "vocab_chars" in ckpt:
        sess.load_checkpoint_dict(ckpt, load_optimizer=True)
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt)
    else:
        raise ValueError(f"Unsupported checkpoint format: {path}")
    sess.reset()
    return sess


def _save_session(sess: SettleSession, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sess.checkpoint_dict(include_optimizer=True), path)


def _lesson_key(item: Dict[str, Any]) -> str:
    return f"{item['task']}::{item['prompt']}::{item['expected']}"


def _stats_float(stats: Dict[str, Any], key: str) -> float | None:
    val = stats.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None


def _fmt_optional_float(value: Any, *, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _min_margin(trace: list[Dict[str, Any]]) -> float | None:
    margins = [x.get("margin") for x in trace if x.get("margin") is not None]
    if not margins:
        return None
    return float(min(margins))


def _failure_brief(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "target": item.get("target"),
        "task": item.get("task"),
        "prompt": item.get("prompt"),
        "expected": item.get("expected"),
        "got": item.get("got"),
        "score": item.get("score"),
        "min_margin": item.get("min_margin"),
    }


def failure_transition(
    before: Sequence[Dict[str, Any]], after: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    before_failures = {_lesson_key(r): r for r in before if not r["passed"]}
    after_failures = {_lesson_key(r): r for r in after if not r["passed"]}

    resolved = [
        _failure_brief(before_failures[key])
        for key in sorted(before_failures.keys() - after_failures.keys())
    ]
    new = [
        _failure_brief(after_failures[key])
        for key in sorted(after_failures.keys() - before_failures.keys())
    ]
    persistent = [
        {
            "before": _failure_brief(before_failures[key]),
            "after": _failure_brief(after_failures[key]),
        }
        for key in sorted(before_failures.keys() & after_failures.keys())
    ]
    return {
        "resolved": resolved,
        "new": new,
        "persistent": persistent,
        "counts": {
            "resolved": len(resolved),
            "new": len(new),
            "persistent": len(persistent),
        },
    }


def exam(
    sess: SettleSession,
    *,
    targets: Sequence[str],
    tasks: Sequence[str],
    k_settle: int | None,
    temperature: float,
    quiz_len: int,
    restep_generate: bool,
    top_k: int,
) -> list[Dict[str, Any]]:
    results: list[Dict[str, Any]] = []
    for target in targets:
        for task in tasks:
            lesson = build_lesson(task, target, targets)
            gen_len = len(lesson.expected)
            if task.startswith("next"):
                gen_len = max(int(quiz_len), gen_len)

            sess.reset()
            sess.ingest(f"T:{lesson.prompt} S:", k_settle=k_settle)
            got, trace = sess.generate_with_trace(
                gen_len,
                expected=lesson.expected,
                top_k=top_k,
                temperature=float(temperature),
                k_settle=k_settle,
                restep_last_token=restep_generate,
            )
            stats = sess.stats()
            score, message = grade_with_stats(
                lesson.expected,
                got,
                task=task,
                seq=lesson.seq,
                alphabet=targets,
                copy_continue_score=0.8,
                next_copy_score=0.65,
                stats=stats,
                recent_prompts=[],
                recent_outputs=[],
            )
            results.append(
                {
                    "target": target,
                    "task": task,
                    "seq": lesson.seq,
                    "prompt": lesson.prompt,
                    "expected": lesson.expected,
                    "got": got,
                    "passed": got == lesson.expected,
                    "score": float(score),
                    "message": message,
                    "stats": stats,
                    "trace": trace,
                    "min_margin": _min_margin(trace),
                }
            )
    return results


def summarize(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    score_sum = sum(float(r["score"]) for r in results)
    entropies = [
        x
        for r in results
        if (x := _stats_float(r.get("stats", {}), "logits_entropy")) is not None
    ]
    state_norms = [
        x
        for r in results
        if (x := _stats_float(r.get("stats", {}), "state_norm")) is not None
    ]
    margins = [
        float(x)
        for r in results
        if (x := r.get("min_margin")) is not None
    ]
    weakest = sorted(
        [
            {
                "target": r["target"],
                "task": r["task"],
                "prompt": r["prompt"],
                "expected": r["expected"],
                "got": r["got"],
                "passed": r["passed"],
                "min_margin": r.get("min_margin"),
            }
            for r in results
            if r.get("min_margin") is not None
        ],
        key=lambda x: float(x["min_margin"]),
    )[:5]
    failures = [
        {
            "target": r["target"],
            "task": r["task"],
            "prompt": r["prompt"],
            "expected": r["expected"],
            "got": r["got"],
            "score": r["score"],
            "min_margin": r.get("min_margin"),
        }
        for r in results
        if not r["passed"]
    ]
    return {
        "passed": passed,
        "total": total,
        "failures": total - passed,
        "score_sum": round(score_sum, 6),
        "avg_score": round(score_sum / max(total, 1), 6),
        "avg_entropy": round(sum(entropies) / len(entropies), 6) if entropies else None,
        "avg_state_norm": round(sum(state_norms) / len(state_norms), 6)
        if state_norms
        else None,
        "avg_margin": round(sum(margins) / len(margins), 6) if margins else None,
        "min_margin": round(min(margins), 6) if margins else None,
        "weakest_items": weakest,
        "failure_items": failures,
    }


def regressions(
    before: Sequence[Dict[str, Any]], after: Sequence[Dict[str, Any]]
) -> list[Dict[str, Any]]:
    before_passed = {_lesson_key(r): r for r in before if r["passed"]}
    after_by_key = {_lesson_key(r): r for r in after}
    out = []
    for key, old in before_passed.items():
        new = after_by_key.get(key)
        if new is None or not new["passed"]:
            out.append(
                {
                    "target": old["target"],
                    "task": old["task"],
                    "prompt": old["prompt"],
                    "expected": old["expected"],
                    "got": None if new is None else new["got"],
                }
            )
    return out


def _example_text(example: Dict[str, str], *, align_restep: bool) -> str:
    answer_prefix = ":" if align_restep else ""
    return f"T:{example['prompt']} S:{answer_prefix}{example['answer']}"


def make_examples(
    *,
    targets: Sequence[str],
    tasks: Sequence[str],
    failures: Sequence[Dict[str, Any]],
    focus_mult: int,
    failure_repeats: int,
    align_restep: bool,
    weights: Dict[str, int],
    radius: int,
    include_confusions: bool,
    include_mimicry: bool,
) -> list[str]:
    examples = maintenance_examples(
        targets,
        tasks,
        weights=weights,
        failures=failures,
        focus_weight=max(int(focus_mult), 0),
        radius=radius,
        include_confusions=include_confusions,
        include_mimicry=include_mimicry,
    )
    texts = [_example_text(e, align_restep=align_restep) for e in examples]
    repeats = max(int(failure_repeats), 0)
    if repeats > 0:
        for failure in failures:
            prompt = str(failure.get("prompt", ""))
            expected = str(failure.get("expected", ""))
            if prompt and expected:
                answer_prefix = ":" if align_restep else ""
                texts.extend([f"T:{prompt} S:{answer_prefix}{expected}"] * repeats)
    return texts


def _parse_int_list(raw: str) -> list[int]:
    return [int(x) for x in raw.replace(",", " ").split() if x.strip()]


def _parse_float_list(raw: str) -> list[float]:
    return [float(x) for x in raw.replace(",", " ").split() if x.strip()]


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Promotion-gated tutor runner for small retention experiments."
    )
    p.add_argument("--text-path", default="input.txt")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", default="runs/guarded_h")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument("--targets", default="ABCDEFGH")
    p.add_argument("--tasks", default="copy,copy2,next,next2")
    p.add_argument("--k-settle", type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--quiz-len", type=int, default=1)
    p.add_argument(
        "--restep-generate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Re-step the last ingested token before sampling (legacy-compatible).",
    )
    p.add_argument(
        "--trace-top-k",
        type=int,
        default=5,
        help="Number of top logits to include in each exam item trace.",
    )

    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--steps-candidates", default="20,40,70")
    p.add_argument("--lr-candidates", default="0.00005,0.0001,0.00015")
    p.add_argument("--focus-mults", default="4,8,16")
    p.add_argument(
        "--require-clean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only promote candidates that pass the full requested exam.",
    )
    p.add_argument(
        "--save-best-rejected",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save the best non-promoted candidate as best_rejected.pt for diagnostics.",
    )
    p.add_argument("--maintenance-radius", type=int, default=2)
    p.add_argument(
        "--failure-repeats",
        type=int,
        default=0,
        help="Repeat exact failed lessons in each candidate training bundle.",
    )
    p.add_argument(
        "--align-restep-training",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train with S::answer examples to match --restep-generate exams.",
    )
    p.add_argument(
        "--maintenance-confusions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include confused outputs in local maintenance neighborhoods.",
    )
    p.add_argument(
        "--maintenance-mimicry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include copy lessons in focused maintenance bundles.",
    )
    p.add_argument("--min-score-delta", type=float, default=1e-6)
    p.add_argument(
        "--accept-score-ties",
        action="store_true",
        help="Accept equal-pass candidates when score_sum improves.",
    )
    p.add_argument(
        "--allow-regressions",
        action="store_true",
        help="Accept score improvements even if previously passed exam items regress.",
    )
    p.add_argument(
        "--repair-regressions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When a candidate improves but forgets old items, run small repair updates before deciding.",
    )
    p.add_argument("--repair-attempts", type=int, default=2)
    p.add_argument("--repair-steps-candidates", default="10,20")
    p.add_argument("--repair-lr-scale", type=float, default=0.5)
    p.add_argument("--repair-focus-mult", type=int, default=16)

    p.add_argument("--w-copy", type=int, default=2)
    p.add_argument("--w-copy2", type=int, default=2)
    p.add_argument("--w-next", type=int, default=1)
    p.add_argument("--w-next2", type=int, default=1)

    # Fallbacks for raw state_dict checkpoints that do not carry model_cfg.
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--n-layers", type=int, default=2)
    return p.parse_args(argv)


def log_event(log_f: Any, event: str, payload: Dict[str, Any]) -> None:
    obj = {"event": event, "time": time.time(), **payload}
    log_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    log_f.flush()


def _summary_better(
    candidate: Dict[str, Any],
    incumbent: Dict[str, Any] | None,
    *,
    min_score_delta: float,
) -> bool:
    if incumbent is None:
        return True
    if candidate["passed"] != incumbent["passed"]:
        return int(candidate["passed"]) > int(incumbent["passed"])
    return float(candidate["score_sum"]) > float(incumbent["score_sum"]) + min_score_delta


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(f"Missing checkpoint: {checkpoint}")
    if not Path(args.text_path).exists():
        raise SystemExit(f"Missing text path: {args.text_path}")

    torch.manual_seed(int(args.seed))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "events.jsonl"
    best_path = out_dir / "best.pt"
    best_rejected_path = out_dir / "best_rejected.pt"
    if not best_path.exists():
        shutil.copy2(checkpoint, best_path)

    targets = [ch for ch in str(args.targets) if ch.strip()]
    tasks = normalize_tasks(args.tasks)
    weights = {
        "copy": int(args.w_copy),
        "copy2": int(args.w_copy2),
        "next": int(args.w_next),
        "next2": int(args.w_next2),
    }
    steps_candidates = _parse_int_list(args.steps_candidates)
    lr_candidates = _parse_float_list(args.lr_candidates)
    focus_mults = _parse_int_list(args.focus_mults)
    repair_steps_candidates = _parse_int_list(args.repair_steps_candidates)
    if not steps_candidates or not lr_candidates or not focus_mults:
        raise SystemExit("Candidate lists must be non-empty.")
    if bool(args.repair_regressions) and not repair_steps_candidates:
        raise SystemExit("--repair-steps-candidates must be non-empty.")
    if int(args.maintenance_radius) < 0:
        raise SystemExit("--maintenance-radius must be >= 0.")

    device = torch.device(args.device)
    vocab = Vocab.from_text(load_text(args.text_path))

    with open(log_path, "a", encoding="utf-8") as log_f:
        sess = _checkpoint_to_session(best_path, args=args, vocab=vocab, device=device)
        best_results = exam(
            sess,
            targets=targets,
            tasks=tasks,
            k_settle=args.k_settle,
            temperature=float(args.temperature),
            quiz_len=int(args.quiz_len),
            restep_generate=bool(args.restep_generate),
            top_k=int(args.trace_top_k),
        )
        best_summary = summarize(best_results)
        log_event(
            log_f,
            "initial",
            {
                "checkpoint": str(best_path),
                "summary": best_summary,
                "args": vars(args),
            },
        )
        print(
            f"initial: {best_summary['passed']}/{best_summary['total']} "
            f"score={best_summary['score_sum']:.3f} "
            f"min_margin={_fmt_optional_float(best_summary['min_margin'])}"
        )

        if best_summary["failures"] == 0:
            print(f"already clean: {best_path}")
            return 0

        best_rejected_summary: Dict[str, Any] | None = None
        for round_i in range(1, int(args.rounds) + 1):
            failures = list(best_summary["failure_items"])
            accepted = False
            try_i = 0
            for steps in steps_candidates:
                for lr in lr_candidates:
                    for focus_mult in focus_mults:
                        try_i += 1
                        candidate = Candidate(
                            round_i, try_i, int(steps), float(lr), int(focus_mult)
                        )
                        sess = _checkpoint_to_session(
                            best_path, args=args, vocab=vocab, device=device
                        )
                        examples = make_examples(
                            targets=targets,
                            tasks=tasks,
                            failures=failures,
                            focus_mult=candidate.focus_mult,
                            failure_repeats=int(args.failure_repeats),
                            align_restep=bool(args.align_restep_training),
                            weights=weights,
                            radius=int(args.maintenance_radius),
                            include_confusions=bool(args.maintenance_confusions),
                            include_mimicry=bool(args.maintenance_mimicry),
                        )
                        sess.reset()
                        train_stats = sess.learn(
                            examples,
                            steps=candidate.steps,
                            k_settle=args.k_settle,
                            lr=candidate.lr,
                            weight_decay=0.0,
                            grad_clip=1.0,
                            detach_state=False,
                            reset_state_each_example=True,
                            seed=int(args.seed) + round_i * 1000 + try_i,
                            loss_mode="answer_only",
                        )
                        results = exam(
                            sess,
                            targets=targets,
                            tasks=tasks,
                            k_settle=args.k_settle,
                            temperature=float(args.temperature),
                            quiz_len=int(args.quiz_len),
                            restep_generate=bool(args.restep_generate),
                            top_k=int(args.trace_top_k),
                        )
                        summary = summarize(results)
                        regressed = (
                            []
                            if args.allow_regressions
                            else regressions(best_results, results)
                        )
                        score_tie_improved = (
                            bool(args.accept_score_ties)
                            and summary["passed"] == best_summary["passed"]
                            and summary["score_sum"]
                            > best_summary["score_sum"] + float(args.min_score_delta)
                        )
                        improved = (
                            summary["passed"] > best_summary["passed"]
                            or score_tie_improved
                        )
                        repair_log = []
                        if (
                            improved
                            and regressed
                            and bool(args.repair_regressions)
                            and not bool(args.allow_regressions)
                        ):
                            for repair_i in range(1, int(args.repair_attempts) + 1):
                                fixed_this_round = False
                                for repair_steps in repair_steps_candidates:
                                    repair_examples = make_examples(
                                        targets=targets,
                                        tasks=tasks,
                                        failures=regressed,
                                        focus_mult=int(args.repair_focus_mult),
                                        failure_repeats=int(args.failure_repeats),
                                        align_restep=bool(args.align_restep_training),
                                        weights=weights,
                                        radius=int(args.maintenance_radius),
                                        include_confusions=bool(
                                            args.maintenance_confusions
                                        ),
                                        include_mimicry=bool(args.maintenance_mimicry),
                                    )
                                    repair_train = sess.learn(
                                        repair_examples,
                                        steps=int(repair_steps),
                                        k_settle=args.k_settle,
                                        lr=candidate.lr * float(args.repair_lr_scale),
                                        weight_decay=0.0,
                                        grad_clip=1.0,
                                        detach_state=False,
                                        reset_state_each_example=True,
                                        seed=(
                                            int(args.seed)
                                            + round_i * 1000
                                            + try_i * 100
                                            + repair_i
                                            + int(repair_steps)
                                        ),
                                        loss_mode="answer_only",
                                    )
                                    repaired_results = exam(
                                        sess,
                                        targets=targets,
                                        tasks=tasks,
                                        k_settle=args.k_settle,
                                        temperature=float(args.temperature),
                                        quiz_len=int(args.quiz_len),
                                        restep_generate=bool(args.restep_generate),
                                        top_k=int(args.trace_top_k),
                                    )
                                    repaired_summary = summarize(repaired_results)
                                    repaired_regressed = regressions(
                                        best_results, repaired_results
                                    )
                                    repair_log.append(
                                        {
                                            "repair_attempt": repair_i,
                                            "repair_steps": int(repair_steps),
                                            "train": repair_train,
                                            "summary": repaired_summary,
                                            "regressions": repaired_regressed,
                                            "failure_transition": failure_transition(
                                                best_results, repaired_results
                                            ),
                                        }
                                    )
                                    results = repaired_results
                                    summary = repaired_summary
                                    regressed = repaired_regressed
                                    score_tie_improved = (
                                        bool(args.accept_score_ties)
                                        and summary["passed"] == best_summary["passed"]
                                        and summary["score_sum"]
                                        > best_summary["score_sum"]
                                        + float(args.min_score_delta)
                                    )
                                    improved = (
                                        summary["passed"] > best_summary["passed"]
                                        or score_tie_improved
                                    )
                                    if not regressed:
                                        fixed_this_round = True
                                        break
                                if not regressed or not fixed_this_round:
                                    break
                        clean = summary["failures"] == 0
                        if bool(args.require_clean):
                            decision = "accept" if clean and not regressed else "reject"
                        else:
                            decision = (
                                "accept" if improved and not regressed else "reject"
                            )
                        log_event(
                            log_f,
                            "candidate",
                            {
                                "round": round_i,
                                "try": try_i,
                                "candidate": candidate.__dict__,
                                "decision": decision,
                                "train": train_stats,
                                "summary": summary,
                                "regressions": regressed,
                                "failure_transition": failure_transition(
                                    best_results, results
                                ),
                                "repair": repair_log,
                                "improved": improved,
                                "clean": clean,
                            },
                        )
                        print(
                            f"round {round_i} try {try_i}: {decision} "
                            f"steps={steps} lr={lr:g} focus={focus_mult} "
                            f"pass={summary['passed']}/{summary['total']} "
                            f"score={summary['score_sum']:.3f} reg={len(regressed)} "
                            f"clean={clean} "
                            f"min_margin={_fmt_optional_float(summary['min_margin'])}"
                        )
                        if decision == "accept":
                            _save_session(sess, best_path)
                            best_results = results
                            best_summary = summary
                            accepted = True
                            break
                        if (
                            bool(args.save_best_rejected)
                            and not regressed
                            and _summary_better(
                                summary,
                                best_rejected_summary,
                                min_score_delta=float(args.min_score_delta),
                            )
                        ):
                            _save_session(sess, best_rejected_path)
                            best_rejected_summary = summary
                            log_event(
                                log_f,
                                "best_rejected",
                                {
                                    "round": round_i,
                                    "try": try_i,
                                    "checkpoint": str(best_rejected_path),
                                    "summary": summary,
                                    "candidate": candidate.__dict__,
                                },
                            )
                    if accepted:
                        break
                if accepted:
                    break

            if best_summary["failures"] == 0:
                log_event(log_f, "complete", {"checkpoint": str(best_path), "summary": best_summary})
                print(f"complete: {best_path}")
                return 0
            if not accepted:
                log_event(
                    log_f,
                    "stalled",
                    {"round": round_i, "checkpoint": str(best_path), "summary": best_summary},
                )
                print(f"stalled after round {round_i}: {best_path}")
                return 1

        log_event(log_f, "budget_exhausted", {"checkpoint": str(best_path), "summary": best_summary})
        print(f"budget exhausted: {best_path}")
        return 1 if best_summary["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
