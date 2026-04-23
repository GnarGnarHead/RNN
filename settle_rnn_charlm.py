from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rnn.model import ModelCfg, SettleCharLM
from rnn.vocab import Vocab


@dataclass
class Cfg:
    # io
    text_path: str = "input.txt"
    device: str = "cpu"
    seed: int = 1337

    # data
    seq_len: int = 128
    batch_size: int = 16

    # model
    d_model: int = 256
    n_layers: int = 4
    k_settle: int = 2
    dropout: float = 0.0
    use_state: bool = True
    state_alpha: float = 0.2
    detach_state: bool = True
    state_norm: bool = True

    # training
    steps: int = 8000
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    log_every: int = 200
    sample_every: int = 1000
    sample_len: int = 400
    temperature: float = 0.9
    start_text: str = "To be"
    restep_generate: bool = True


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def encode(text: str, vocab: Vocab) -> torch.Tensor:
    return torch.tensor(vocab.encode(text, strict=True), dtype=torch.long)


def get_batch(
    data: torch.Tensor, cfg: Cfg, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    if data.numel() <= cfg.seq_len + 1:
        raise ValueError(
            f"Text is too short for seq_len={cfg.seq_len}. Need at least {cfg.seq_len + 2} characters."
        )
    ix = torch.randint(0, data.numel() - cfg.seq_len - 1, (cfg.batch_size,))
    x = torch.stack([data[i : i + cfg.seq_len] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + cfg.seq_len + 1] for i in ix]).to(device)
    return x, y


def _sanitize_start_text(start: str, vocab: Vocab) -> str:
    if not start:
        return next(iter(vocab.stoi.keys()))
    ok = vocab.sanitize(start)
    if ok:
        return ok
    return next(iter(vocab.stoi.keys()))


@torch.no_grad()
def sample(
    model: SettleCharLM,
    start: str,
    vocab: Vocab,
    cfg: Cfg,
    length: int,
) -> str:
    model.eval()
    device = torch.device(cfg.device)

    start = _sanitize_start_text(start, vocab)
    start_ids = vocab.encode(start, strict=False)
    if not start_ids:
        start_ids = [next(iter(vocab.stoi.values()))]
        start = vocab.decode(start_ids)

    model.reset_state(batch_size=1)
    for tid in start_ids:
        model.step(torch.tensor([tid], device=device), k_settle=cfg.k_settle)

    gen_ids, _ = model.generate(
        length,
        temperature=cfg.temperature,
        k_settle=cfg.k_settle,
        restep_last_token=cfg.restep_generate,
    )
    return start + vocab.decode(gen_ids[0].tolist())


def _ensure_out_dir(out_dir: Path, run_name: str) -> Path:
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _run_name_for_k(base: str, k: int) -> str:
    suffix = f"k{k}"
    return f"{base}_{suffix}" if base else suffix


def train(cfg: Cfg, *, out_dir: Path | None = None, run_name: str = "") -> None:
    set_seed(cfg.seed)

    text = load_text(cfg.text_path)
    vocab = Vocab.from_text(text)
    data = encode(text, vocab)

    device = torch.device(cfg.device)
    model_cfg = ModelCfg(
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        k_settle=cfg.k_settle,
        dropout=cfg.dropout,
        use_state=cfg.use_state,
        state_alpha=cfg.state_alpha,
        detach_state=cfg.detach_state,
        state_norm=cfg.state_norm,
    )

    model = SettleCharLM(vocab.size, model_cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    run_dir: Path | None = None
    log_f = None
    if out_dir is not None:
        run_dir = _ensure_out_dir(out_dir, run_name or time.strftime("%Y%m%d-%H%M%S"))
        (run_dir / "config.json").write_text(
            json.dumps(asdict(cfg), indent=2) + "\n", encoding="utf-8"
        )
        log_f = open(run_dir / "log.jsonl", "a", encoding="utf-8")

    t0 = time.time()
    for step in range(1, cfg.steps + 1):
        model.train()
        xb, yb = get_batch(data, cfg, device)

        model.reset_state(batch_size=xb.shape[0])
        logits, stats = model.forward_sequence(xb, k_settle=cfg.k_settle)
        loss = F.cross_entropy(logits.reshape(-1, vocab.size), yb.reshape(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if step == 1 or step % cfg.log_every == 0:
            dt = time.time() - t0
            delta = stats["delta_per_k"].detach().cpu().tolist()
            delta_str = ", ".join(f"{v:.4f}" for v in delta)
            print(
                f"step {step:5d} | loss {loss.item():.4f} | dt {dt:.1f}s | delta[k]: [{delta_str}]"
            )
            if log_f is not None:
                log_f.write(
                    json.dumps(
                        {
                            "step": step,
                            "loss": float(loss.item()),
                            "dt_s": float(dt),
                            "delta_per_k": [float(v) for v in delta],
                            "state_norm": float(
                                stats["state_norm"].detach().cpu().item()
                            ),
                            "logits_entropy": float(
                                stats["logits_entropy"].detach().cpu().item()
                            ),
                        }
                    )
                    + "\n"
                )
                log_f.flush()
            t0 = time.time()

        if cfg.sample_every > 0 and step % cfg.sample_every == 0:
            s = sample(
                model, start=cfg.start_text, vocab=vocab, cfg=cfg, length=cfg.sample_len
            )
            print("\n--- SAMPLE ---")
            print(s)
            print("--------------\n")
            if run_dir is not None:
                (run_dir / f"sample_step{step}.txt").write_text(
                    s + "\n", encoding="utf-8"
                )

    if log_f is not None:
        log_f.close()


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CPU-only settle-before-decode char LM (no attention)."
    )

    p.add_argument("--text-path", default=Cfg.text_path)
    p.add_argument("--device", default=Cfg.device)
    p.add_argument("--seed", type=int, default=Cfg.seed)

    p.add_argument("--seq-len", type=int, default=Cfg.seq_len)
    p.add_argument("--batch-size", type=int, default=Cfg.batch_size)

    p.add_argument("--d-model", type=int, default=Cfg.d_model)
    p.add_argument("--n-layers", type=int, default=Cfg.n_layers)
    p.add_argument("--k-settle", type=int, default=Cfg.k_settle)
    p.add_argument("--dropout", type=float, default=Cfg.dropout)
    p.add_argument("--state-alpha", type=float, default=Cfg.state_alpha)

    p.add_argument(
        "--no-state", action="store_true", help="Disable recurrent state across tokens."
    )
    p.add_argument(
        "--detach-state",
        action=argparse.BooleanOptionalAction,
        default=Cfg.detach_state,
        help="Detach recurrent state each timestep (disables BPTT through time).",
    )
    p.add_argument(
        "--state-norm",
        action=argparse.BooleanOptionalAction,
        default=Cfg.state_norm,
        help="RMSNorm the recurrent state before injecting it into the current token.",
    )

    p.add_argument("--steps", type=int, default=Cfg.steps)
    p.add_argument("--lr", type=float, default=Cfg.lr)
    p.add_argument("--weight-decay", type=float, default=Cfg.weight_decay)
    p.add_argument("--grad-clip", type=float, default=Cfg.grad_clip)
    p.add_argument("--log-every", type=int, default=Cfg.log_every)
    p.add_argument("--sample-every", type=int, default=Cfg.sample_every)
    p.add_argument("--sample-len", type=int, default=Cfg.sample_len)
    p.add_argument("--temperature", type=float, default=Cfg.temperature)
    p.add_argument("--start-text", default=Cfg.start_text)
    p.add_argument(
        "--restep-generate",
        action=argparse.BooleanOptionalAction,
        default=Cfg.restep_generate,
        help="Re-step the last prompt token before sampling (legacy-compatible).",
    )

    p.add_argument(
        "--sweep-k",
        type=int,
        nargs="*",
        default=[],
        help="Run multiple trainings, one per K (e.g. --sweep-k 1 2 4 8).",
    )
    p.add_argument(
        "--out-dir",
        default="",
        help="Optional directory to write logs/samples (e.g. runs).",
    )
    p.add_argument(
        "--run-name", default="", help="Optional run name prefix (used with --out-dir)."
    )

    return p.parse_args(argv)


def main() -> None:
    args = _parse_args()

    if not Path(args.text_path).exists():
        raise SystemExit(
            f"Missing '{args.text_path}'. Put a text file there, or run:\n"
            f"  python3 scripts/download_tiny_shakespeare.py --out {args.text_path}\n"
        )

    cfg = Cfg(
        text_path=args.text_path,
        device=args.device,
        seed=args.seed,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        k_settle=args.k_settle,
        dropout=args.dropout,
        use_state=not args.no_state,
        state_alpha=args.state_alpha,
        detach_state=args.detach_state,
        state_norm=args.state_norm,
        steps=args.steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        log_every=args.log_every,
        sample_every=args.sample_every,
        sample_len=args.sample_len,
        temperature=args.temperature,
        start_text=args.start_text,
        restep_generate=args.restep_generate,
    )

    out_dir = Path(args.out_dir) if args.out_dir else None
    sweep = [int(k) for k in args.sweep_k if int(k) > 0]
    if sweep:
        for k in sweep:
            cfg_k = Cfg(**{**asdict(cfg), "k_settle": k})
            train(cfg_k, out_dir=out_dir, run_name=_run_name_for_k(args.run_name, k))
        return

    train(cfg, out_dir=out_dir, run_name=args.run_name)


if __name__ == "__main__":
    main()
