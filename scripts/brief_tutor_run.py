from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def _send(proc: subprocess.Popen[str], obj: Dict[str, Any]) -> Dict[str, Any]:
    proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("session.py exited unexpectedly")
    return json.loads(line)


def _grade_letter(expected: str, got: str) -> tuple[float, str]:
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tiny supervised 'kindergarten' tutor demo via session.py JSONL.")
    p.add_argument("--text-path", default="input.txt", help="Text used to build the vocab.")
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--k-settle", type=int, default=2)
    p.add_argument("--steps", type=int, default=400, help="Supervised learn steps.")
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if not Path(args.text_path).exists():
        raise SystemExit(f"Missing {args.text_path!r}.")

    cmd: List[str] = [
        sys.executable,
        "-u",
        "session.py",
        "--text-path",
        args.text_path,
        "--d-model",
        str(args.d_model),
        "--n-layers",
        str(args.n_layers),
        "--k-settle",
        str(args.k_settle),
        "--seed",
        "1337",
    ]

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
        print("Resetting session…")
        print(_send(proc, {"cmd": "reset"}))

        print("Teaching letters A/B (supervised)…")
        resp = _send(
            proc,
            {
                "cmd": "learn",
                "examples": [{"prompt": "A"}, {"prompt": "B"}],
                "steps": args.steps,
                "lr": args.lr,
                "k_settle": args.k_settle,
                "detach_state": False,
            },
        )
        print(resp)

        print("\nQuiz (retention): reset → prompt → generate 2 chars (letter + newline)")
        for ch in ["A", "B"]:
            _send(proc, {"cmd": "reset"})
            _send(proc, {"cmd": "ingest", "text": f"T:{ch}\nS:"})
            out = _send(proc, {"cmd": "generate", "max_new_tokens": 2, "temperature": 0.0})
            got = str(out.get("text", ""))
            score, sentiment = _grade_letter(ch, got)
            print(f"{ch!r} → {got!r} | score={score:.2f} | {sentiment}")

        _send(proc, {"cmd": "exit"})
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()

