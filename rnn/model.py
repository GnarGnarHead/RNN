from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelCfg:
    d_model: int = 256
    n_layers: int = 4
    k_settle: int = 2
    dropout: float = 0.0

    # recurrence
    use_state: bool = True
    state_alpha: float = 0.2
    detach_state: bool = True
    state_norm: bool = True

    # guardrail
    max_k_settle: int = 64


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
    """
    A minimal recurrent char LM with an inner "settle loop" per token.

    Important: state is persistent and must be managed explicitly via reset_state().
    forward()/forward_sequence() do NOT reset state.
    """

    def __init__(self, vocab_size: int, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        self.tok_emb = nn.Embedding(vocab_size, d)
        self.core = SettleCore(d, cfg.n_layers, cfg.dropout)
        self.out_norm = RMSNorm(d)
        self.lm_head = nn.Linear(d, vocab_size, bias=False)

        self.state_gate = nn.Linear(d, d)
        self.state_norm = RMSNorm(d) if cfg.state_norm else None
        self.settle_gate = nn.Linear(d, d)

        self.state: torch.Tensor | None = None
        self.last_token_id: torch.Tensor | None = None

    def reset_state(self, batch_size: int = 1) -> None:
        p = next(self.parameters())
        self.state = torch.zeros((batch_size, self.cfg.d_model), device=p.device, dtype=p.dtype)
        self.last_token_id = None

    def _k_settle(self, k_settle: int | None) -> int:
        k = int(self.cfg.k_settle if k_settle is None else k_settle)
        if k <= 0:
            raise ValueError(f"k_settle must be >= 1, got {k}")
        if k > int(self.cfg.max_k_settle):
            raise ValueError(f"k_settle too large (max {self.cfg.max_k_settle}), got {k}")
        return k

    def step(
        self,
        token_id: torch.Tensor,
        *,
        k_settle: int | None = None,
        detach_state: bool | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        token_id: (B,) long
        returns logits: (B, V) for predicting the *next* token.
        Updates internal recurrent state.
        """
        if self.state is None:
            raise RuntimeError("State is not initialized. Call reset_state(batch_size=...) first.")
        if token_id.ndim != 1:
            raise ValueError(f"token_id must have shape (B,), got {tuple(token_id.shape)}")
        if token_id.dtype != torch.long:
            raise ValueError(f"token_id must be torch.long, got {token_id.dtype}")
        if token_id.device != self.state.device:
            raise ValueError("token_id device does not match state device. Move inputs or reset_state().")
        if token_id.shape[0] != self.state.shape[0]:
            raise ValueError(
                f"token batch size {token_id.shape[0]} does not match state batch size {self.state.shape[0]}"
            )

        k = self._k_settle(k_settle)

        state = self.state
        detach = self.cfg.detach_state if detach_state is None else bool(detach_state)

        if self.cfg.use_state and detach:
            state = state.detach()

        emb = self.tok_emb(token_id)  # (B, d)

        if self.cfg.use_state:
            state_in = state
            if self.state_norm is not None:
                state_in = self.state_norm(state_in)
            g = torch.sigmoid(self.state_gate(state_in))
            h = emb + g * state_in
        else:
            h = emb

        prev = h
        deltas = torch.zeros((k,), device=h.device, dtype=h.dtype)
        for i in range(k):
            h_candidate = self.core(h)
            dg = torch.sigmoid(self.settle_gate(h_candidate))
            h = prev + dg * (h_candidate - prev)
            with torch.no_grad():
                deltas[i] = (h - prev).abs().mean()
            prev = h

        if self.cfg.use_state:
            a = float(self.cfg.state_alpha)
            if detach:
                with torch.no_grad():
                    self.state = (1.0 - a) * state + a * h
            else:
                self.state = (1.0 - a) * state + a * h
        else:
            self.state = state

        self.last_token_id = token_id

        h_out = self.out_norm(h)
        logits = self.lm_head(h_out)  # (B, V)

        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * (probs + 1e-12).log()).sum(dim=-1).mean()
            state_norm = self.state.pow(2).mean(dim=-1).sqrt().mean()

        stats = {
            "k_settle": torch.tensor(k, device=logits.device),
            "delta_per_k": deltas.detach(),
            "logits_entropy": entropy.detach(),
            "state_norm": state_norm.detach(),
        }
        return logits, stats

    def forward_sequence(
        self, x: torch.Tensor, *, k_settle: int | None = None, detach_state: bool | None = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        x: (B, T) token ids
        returns logits: (B, T, V)
        """
        if self.state is None:
            raise RuntimeError("State is not initialized. Call reset_state(batch_size=...) first.")
        if x.ndim != 2:
            raise ValueError(f"x must have shape (B, T), got {tuple(x.shape)}")
        if x.dtype != torch.long:
            raise ValueError(f"x must be torch.long, got {x.dtype}")

        bsz, seq = x.shape
        k = self._k_settle(k_settle)
        delta_accum = torch.zeros((k,), device=x.device, dtype=torch.float32)
        entropy_accum = torch.tensor(0.0, device=x.device)

        logits_out = []
        for t in range(seq):
            logits, stats = self.step(x[:, t], k_settle=k, detach_state=detach_state)
            logits_out.append(logits.unsqueeze(1))
            delta_accum += stats["delta_per_k"].to(delta_accum.dtype)
            entropy_accum += stats["logits_entropy"].to(entropy_accum.dtype)

        logits_seq = torch.cat(logits_out, dim=1)

        stats_out = {
            "k_settle": torch.tensor(k, device=x.device),
            "delta_per_k": (delta_accum / seq).detach(),
            "logits_entropy": (entropy_accum / seq).detach(),
            "state_norm": stats["state_norm"].detach(),
        }
        return logits_seq, stats_out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.forward_sequence(x)

    @torch.no_grad()
    def generate(
        self, max_new_tokens: int, *, temperature: float = 1.0, k_settle: int | None = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Generates tokens by repeatedly calling step() starting from last_token_id.
        Returns (token_ids, last_stats) where token_ids has shape (B, max_new_tokens).
        """
        if self.state is None:
            raise RuntimeError("State is not initialized. Call reset_state(batch_size=...) first.")
        if self.last_token_id is None:
            raise RuntimeError("No last_token_id. Call step() at least once before generate().")
        if max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")

        was_training = self.training
        self.eval()
        try:
            k = self._k_settle(k_settle)
            token = self.last_token_id
            out = []
            delta_accum = torch.zeros((k,), device=token.device, dtype=torch.float32)
            entropy_accum = torch.tensor(0.0, device=token.device)
            last_stats: Dict[str, torch.Tensor] = {}

            for _ in range(max_new_tokens):
                logits, last_stats = self.step(token, k_settle=k)
                delta_accum += last_stats["delta_per_k"].to(delta_accum.dtype)
                entropy_accum += last_stats["logits_entropy"].to(entropy_accum.dtype)
                next_logits = logits / max(float(temperature), 1e-6)
                probs = F.softmax(next_logits, dim=-1)
                token = torch.multinomial(probs, num_samples=1).squeeze(1)
                out.append(token)

            self.last_token_id = token
            stats_out = {
                "k_settle": torch.tensor(k, device=token.device),
                "delta_per_k": (delta_accum / max_new_tokens).detach(),
                "logits_entropy": (entropy_accum / max_new_tokens).detach(),
                "state_norm": last_stats["state_norm"].detach() if last_stats else torch.tensor(float("nan")),
            }
            return torch.stack(out, dim=1), stats_out
        finally:
            self.train(was_training)
