from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

from rnn.model import ModelCfg, SettleCharLM
from rnn.session import SettleSession
from rnn.vocab import Vocab


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _emit(
    ok: bool,
    *,
    text: str | None = None,
    stats: Dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    obj: Dict[str, Any] = {"ok": ok}
    if text is not None:
        obj["text"] = text
    if stats is not None:
        obj["stats"] = stats
    if error is not None:
        obj["error"] = error
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Long-running JSONL session for the settle-loop char LM.")

    p.add_argument("--text-path", default="input.txt", help="Text file used to build the character vocab.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument("--d-model", type=int, default=ModelCfg.d_model)
    p.add_argument("--n-layers", type=int, default=ModelCfg.n_layers)
    p.add_argument("--k-settle", type=int, default=ModelCfg.k_settle)
    p.add_argument("--dropout", type=float, default=ModelCfg.dropout)

    p.add_argument("--no-state", action="store_true", help="Disable recurrent state across tokens.")
    p.add_argument("--state-alpha", type=float, default=ModelCfg.state_alpha)
    p.add_argument(
        "--detach-state",
        action=argparse.BooleanOptionalAction,
        default=ModelCfg.detach_state,
        help="Detach recurrent state each timestep (disables BPTT through time).",
    )
    p.add_argument(
        "--state-norm",
        action=argparse.BooleanOptionalAction,
        default=ModelCfg.state_norm,
        help="RMSNorm the recurrent state before injecting it into the current token.",
    )

    p.add_argument("--checkpoint", default="", help="Optional path to a torch state_dict to load.")

    return p.parse_args()


def _repeat_frac(text: str) -> float:
    if len(text) <= 1:
        return 0.0
    repeats = sum(1 for i in range(1, len(text)) if text[i] == text[i - 1])
    return repeats / (len(text) - 1)


def main() -> None:
    args = _parse_args()
    if not Path(args.text_path).exists():
        raise SystemExit(f"Missing '{args.text_path}'. Provide a text file to build the vocab.")

    torch.manual_seed(int(args.seed))
    text = load_text(args.text_path)
    vocab = Vocab.from_text(text)

    device = torch.device(args.device)
    cfg = ModelCfg(
        d_model=args.d_model,
        n_layers=args.n_layers,
        k_settle=args.k_settle,
        dropout=args.dropout,
        use_state=not args.no_state,
        state_alpha=args.state_alpha,
        detach_state=args.detach_state,
        state_norm=args.state_norm,
    )

    model = SettleCharLM(vocab.size, cfg).to(device)
    sess = SettleSession(model, vocab, device=device)

    if args.checkpoint:
        loaded = torch.load(args.checkpoint, map_location=device)
        # Back-compat: allow passing a raw state_dict.
        if isinstance(loaded, dict) and "model" in loaded and "vocab_chars" in loaded:
            sess.load_checkpoint_dict(loaded, load_optimizer=True)
        elif isinstance(loaded, dict):
            model.load_state_dict(loaded)
        else:
            raise ValueError("Unsupported checkpoint format.")

    def _lesson_to_text(obj: Any) -> str:
        if isinstance(obj, str):
            return obj
        if isinstance(obj, dict):
            if "text" in obj:
                return str(obj["text"])
            if "prompt" in obj:
                prompt = str(obj.get("prompt", ""))
                answer = str(obj.get("answer", prompt))
                # Call/response format tuned for next-token LM.
                #
                # Important: keep the "answer" as the final token (no trailing newline),
                # so early kindergarten-style "echo" tasks don't accidentally teach '\n'
                # as part of the response.
                #
                # Also: use a space separator instead of '\n' to keep the alphabet lesson
                # as minimal as possible.
                return f"T:{prompt} S:{answer}"
        raise ValueError("Example must be a string or an object with {text} or {prompt[,answer]}.")

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            if not isinstance(msg, dict):
                raise ValueError("command must be a JSON object")
            cmd = msg.get("cmd")
            if cmd == "reset":
                sess.reset()
                _emit(True, stats=sess.stats())
                continue
            if cmd == "stats":
                _emit(True, stats=sess.stats())
                continue
            if cmd == "ingest":
                text_in = msg.get("text", "")
                k_settle = msg.get("k_settle", None)
                sess.ingest(str(text_in), k_settle=k_settle)
                _emit(True, stats=sess.stats())
                continue
            if cmd == "generate":
                max_new = int(msg.get("max_new_tokens", 50))
                temperature = float(msg.get("temperature", 1.0))
                k_settle = msg.get("k_settle", None)
                out = sess.generate(max_new, temperature=temperature, k_settle=k_settle)
                stats = sess.stats()
                stats["repeat_frac"] = _repeat_frac(out)
                _emit(True, text=out, stats=stats)
                continue
            if cmd == "learn":
                examples_raw = msg.get("examples", None)
                if examples_raw is None:
                    # Convenience: allow single "text" or {prompt,answer}.
                    examples_raw = [msg]
                if not isinstance(examples_raw, list):
                    raise ValueError("'examples' must be a list")
                examples: List[str] = [_lesson_to_text(e) for e in examples_raw]

                steps = int(msg.get("steps", 50))
                k_settle = msg.get("k_settle", None)
                lr = float(msg.get("lr", 3e-4))
                weight_decay = float(msg.get("weight_decay", 0.1))
                grad_clip = float(msg.get("grad_clip", 1.0))
                detach_state = bool(msg.get("detach_state", args.detach_state))
                reset_state_each_example = bool(msg.get("reset_state_each_example", True))
                loss_mode = str(msg.get("loss_mode", "full"))
                seed = msg.get("seed", None)
                seed_i = int(seed) if seed is not None else None

                train_stats = sess.learn(
                    examples,
                    steps=steps,
                    k_settle=k_settle,
                    lr=lr,
                    weight_decay=weight_decay,
                    grad_clip=grad_clip,
                    detach_state=detach_state,
                    reset_state_each_example=reset_state_each_example,
                    seed=seed_i,
                    loss_mode=loss_mode,
                )
                stats = sess.stats()
                stats["train"] = train_stats
                _emit(True, stats=stats)
                continue
            if cmd == "save":
                path = msg.get("path", "")
                if not path:
                    raise ValueError("save requires 'path'")
                p = Path(str(path))
                p.parent.mkdir(parents=True, exist_ok=True)
                torch.save(sess.checkpoint_dict(include_optimizer=True), p)
                _emit(True, stats=sess.stats())
                continue
            if cmd == "load":
                path = msg.get("path", "")
                if not path:
                    raise ValueError("load requires 'path'")
                p = Path(str(path))
                ckpt = torch.load(p, map_location=device)
                if isinstance(ckpt, dict) and "model" in ckpt and "vocab_chars" in ckpt:
                    sess.load_checkpoint_dict(ckpt, load_optimizer=True)
                elif isinstance(ckpt, dict):
                    model.load_state_dict(ckpt)
                else:
                    raise ValueError("Unsupported checkpoint format.")
                _emit(True, stats=sess.stats())
                continue
            if cmd == "exit":
                _emit(True)
                break

            raise ValueError(f"unknown cmd: {cmd!r}")
        except Exception as e:
            _emit(False, error=str(e))


if __name__ == "__main__":
    main()
