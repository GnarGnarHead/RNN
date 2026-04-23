from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Iterable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rnn.model import ModelCfg, SettleCharLM  # noqa: E402
from rnn.session import SettleSession  # noqa: E402
from rnn.tutor import build_lesson, grade_with_stats, normalize_tasks  # noqa: E402
from rnn.vocab import Vocab  # noqa: E402


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a noninteractive retention exam for a tutor checkpoint."
    )
    p.add_argument(
        "--text-path",
        default="input.txt",
        help="Text file used to build the character vocab.",
    )
    p.add_argument("--checkpoint", required=True, help="Checkpoint to load.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument("--targets", default="ABCDEFG")
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
        "--jsonl", action="store_true", help="Print one JSON object per exam item."
    )

    # Fallbacks for raw state_dict checkpoints that do not carry model_cfg.
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--n-layers", type=int, default=2)
    return p.parse_args(argv)


def _cfg_from_checkpoint(ckpt: Any, args: argparse.Namespace) -> ModelCfg:
    if isinstance(ckpt, dict) and isinstance(ckpt.get("model_cfg"), dict):
        valid = {f.name for f in fields(ModelCfg)}
        return ModelCfg(**{k: v for k, v in ckpt["model_cfg"].items() if k in valid})
    return ModelCfg(d_model=int(args.d_model), n_layers=int(args.n_layers))


def _load_checkpoint(sess: SettleSession, model: SettleCharLM, ckpt: Any) -> None:
    if isinstance(ckpt, dict) and "model" in ckpt and "vocab_chars" in ckpt:
        sess.load_checkpoint_dict(ckpt, load_optimizer=False)
        return
    if isinstance(ckpt, dict):
        model.load_state_dict(ckpt)
        return
    raise ValueError("Unsupported checkpoint format.")


def _emit(result: Dict[str, Any], *, jsonl: bool) -> None:
    if jsonl:
        print(json.dumps(result, ensure_ascii=False))
        return

    status = "PASS" if result["passed"] else "FAIL"
    print(
        f"{status} {result['task']}:{result['prompt']!r} -> {result['got']!r} "
        f"(expected {result['expected']!r}, score={result['score']:.2f})"
    )


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    if not Path(args.text_path).exists():
        raise SystemExit(f"Missing text path: {args.text_path!r}")
    if not Path(args.checkpoint).exists():
        raise SystemExit(f"Missing checkpoint: {args.checkpoint!r}")

    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    text = load_text(args.text_path)
    vocab = Vocab.from_text(text)

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = _cfg_from_checkpoint(ckpt, args)
    model = SettleCharLM(vocab.size, cfg).to(device)
    sess = SettleSession(model, vocab, device=device)
    _load_checkpoint(sess, model, ckpt)

    targets = [ch for ch in str(args.targets) if ch.strip()]
    tasks = normalize_tasks(args.tasks)
    if not targets:
        raise SystemExit("--targets must include at least one character")
    if any(t in {"copy2", "next", "next2"} for t in tasks) and len(targets) < 2:
        raise SystemExit("copy2/next/next2 require at least two targets")

    failures = 0
    total = 0
    for ch in targets:
        for task in tasks:
            lesson = build_lesson(task, ch, targets)
            gen_len = len(lesson.expected)
            if task.startswith("next"):
                gen_len = max(int(args.quiz_len), gen_len)

            sess.reset()
            sess.ingest(f"T:{lesson.prompt} S:", k_settle=args.k_settle)
            got = sess.generate(
                gen_len,
                temperature=float(args.temperature),
                k_settle=args.k_settle,
                restep_last_token=bool(args.restep_generate),
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
            passed = got == lesson.expected
            result = {
                "passed": passed,
                "task": task,
                "prompt": lesson.prompt,
                "expected": lesson.expected,
                "got": got,
                "score": score,
                "message": message,
                "stats": stats,
            }
            _emit(result, jsonl=bool(args.jsonl))
            failures += 0 if passed else 1
            total += 1

    if args.jsonl:
        print(
            json.dumps(
                {
                    "summary": {
                        "passed": failures == 0,
                        "total": total,
                        "failures": failures,
                    }
                }
            )
        )
    else:
        print(f"\nExam summary: {total - failures}/{total} passed")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
