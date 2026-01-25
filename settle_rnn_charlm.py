from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    detach_state: bool = True   # stop-grad through time (BPTT off)
    state_norm: bool = True     # RMSNorm(state) before injection

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


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_vocab(text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode(text: str, stoi: Dict[str, int]) -> torch.Tensor:
    return torch.tensor([stoi[c] for c in text], dtype=torch.long)


def get_batch(data: torch.Tensor, cfg: Cfg, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if data.numel() <= cfg.seq_len + 1:
        raise ValueError(
            f"Text is too short for seq_len={cfg.seq_len}. Need at least {cfg.seq_len + 2} characters."
        )
    ix = torch.randint(0, data.numel() - cfg.seq_len - 1, (cfg.batch_size,))
    x = torch.stack([data[i : i + cfg.seq_len] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + cfg.seq_len + 1] for i in ix]).to(device)
    return x, y


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.scale


class MLPBlock(nn.Module):
    def __init__(self, d: int, dropout: float = 0.0):
        super().__init__()
        self.norm = RMSNorm(d)
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc1(h)
        h = F.gelu(h)
        h = self.fc2(h)
        h = self.drop(h)
        return x + h


class SettleCore(nn.Module):
    def __init__(self, d: int, n_layers: int, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList([MLPBlock(d, dropout) for _ in range(n_layers)])

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            h = b(h)
        return h


class SettleCharLM(nn.Module):
    def __init__(self, vocab_size: int, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        self.tok_emb = nn.Embedding(vocab_size, d)
        self.core = SettleCore(d, cfg.n_layers, cfg.dropout)
        self.out_norm = RMSNorm(d)
        self.lm_head = nn.Linear(d, vocab_size, bias=False)

        # gated injection of previous-token state
        self.state_gate = nn.Linear(d, d)
        self.state_norm = RMSNorm(d) if cfg.state_norm else None
        # damping inside settle loop (encourages contraction)
        self.settle_gate = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        x: (B, T) token ids
        returns logits: (B, T, V)
        """
        bsz, seq = x.shape
        d_model = self.cfg.d_model

        emb = self.tok_emb(x)  # (B, T, d)
        state = torch.zeros((bsz, d_model), device=emb.device, dtype=emb.dtype)

        logits_out = []
        deltas_accum = torch.zeros((self.cfg.k_settle,), device=emb.device, dtype=emb.dtype)

        for t in range(seq):
            h0 = emb[:, t, :]  # (B, d)

            if self.cfg.use_state:
                if self.cfg.detach_state:
                    state = state.detach()

                state_in = state
                if self.state_norm is not None:
                    state_in = self.state_norm(state_in)
                g = torch.sigmoid(self.state_gate(state_in))
                h = h0 + g * state_in
            else:
                h = h0

            prev = h
            for k in range(self.cfg.k_settle):
                h_candidate = self.core(h)
                dg = torch.sigmoid(self.settle_gate(h_candidate))
                h = prev + dg * (h_candidate - prev)

                with torch.no_grad():
                    deltas_accum[k] += (h - prev).abs().mean()

                prev = h

            if self.cfg.use_state:
                a = float(self.cfg.state_alpha)
                state = (1.0 - a) * state + a * h

            h_out = self.out_norm(h)
            logits_out.append(self.lm_head(h_out).unsqueeze(1))

        logits = torch.cat(logits_out, dim=1)
        stats = {"delta_per_k": deltas_accum / seq}
        return logits, stats


def _sanitize_start_text(start: str, stoi: Dict[str, int]) -> str:
    if not start:
        return next(iter(stoi.keys()))
    ok = [c for c in start if c in stoi]
    if ok:
        return "".join(ok)
    return next(iter(stoi.keys()))


@torch.no_grad()
def sample(
    model: SettleCharLM,
    start: str,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    cfg: Cfg,
    length: int,
) -> str:
    model.eval()
    device = torch.device(cfg.device)

    start = _sanitize_start_text(start, stoi)
    ids = torch.tensor([[stoi[c] for c in start]], dtype=torch.long, device=device)
    out = list(start)

    for _ in range(length):
        x = ids[:, -cfg.seq_len :]
        logits, _ = model(x)
        next_logits = logits[:, -1, :] / max(cfg.temperature, 1e-6)
        probs = F.softmax(next_logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, nxt], dim=1)
        out.append(itos[int(nxt.item())])

    return "".join(out)


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
    stoi, itos = build_vocab(text)
    data = encode(text, stoi)

    device = torch.device(cfg.device)
    vocab_size = len(stoi)

    model = SettleCharLM(vocab_size, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    run_dir: Path | None = None
    log_f = None
    if out_dir is not None:
        run_dir = _ensure_out_dir(out_dir, run_name or time.strftime("%Y%m%d-%H%M%S"))
        (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n", encoding="utf-8")
        log_f = open(run_dir / "log.jsonl", "a", encoding="utf-8")

    t0 = time.time()
    for step in range(1, cfg.steps + 1):
        model.train()
        xb, yb = get_batch(data, cfg, device)

        logits, stats = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), yb.reshape(-1))

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
                        }
                    )
                    + "\n"
                )
                log_f.flush()
            t0 = time.time()

        if cfg.sample_every > 0 and step % cfg.sample_every == 0:
            s = sample(model, start=cfg.start_text, stoi=stoi, itos=itos, cfg=cfg, length=cfg.sample_len)
            print("\n--- SAMPLE ---")
            print(s)
            print("--------------\n")
            if run_dir is not None:
                (run_dir / f"sample_step{step}.txt").write_text(s + "\n", encoding="utf-8")

    if log_f is not None:
        log_f.close()


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CPU-only settle-before-decode char LM (no attention).")

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

    p.add_argument("--no-state", action="store_true", help="Disable recurrent state across tokens.")
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
        "--sweep-k",
        type=int,
        nargs="*",
        default=[],
        help="Run multiple trainings, one per K (e.g. --sweep-k 1 2 4 8).",
    )
    p.add_argument("--out-dir", default="", help="Optional directory to write logs/samples (e.g. runs).")
    p.add_argument("--run-name", default="", help="Optional run name prefix (used with --out-dir).")

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
