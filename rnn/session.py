from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from rnn.model import ModelCfg, SettleCharLM
from rnn.vocab import Vocab


@dataclass
class SessionStats:
    k_settle: int | None = None
    delta_per_k: list[float] | None = None
    state_norm: float | None = None
    logits_entropy: float | None = None
    total_ingested: int = 0
    total_generated: int = 0
    total_train_steps: int = 0
    total_train_tokens: int = 0
    last_train_loss: float | None = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "total_ingested": self.total_ingested,
            "total_generated": self.total_generated,
            "total_train_steps": self.total_train_steps,
            "total_train_tokens": self.total_train_tokens,
        }
        if self.k_settle is not None:
            out["k_settle"] = self.k_settle
        if self.delta_per_k is not None:
            out["delta_per_k"] = self.delta_per_k
        if self.state_norm is not None:
            out["state_norm"] = self.state_norm
        if self.logits_entropy is not None:
            out["logits_entropy"] = self.logits_entropy
        if self.last_train_loss is not None:
            out["last_train_loss"] = self.last_train_loss
        return out


class SettleSession:
    """
    Thin wrapper around a SettleCharLM + Vocab to provide an interactive,
    persistent recurrent session.

    The model state is persistent and must be initialized via reset().
    """

    def __init__(self, model: SettleCharLM, vocab: Vocab, *, device: torch.device):
        self.model = model
        self.vocab = vocab
        self.device = device
        self._stats = SessionStats()
        self._opt: torch.optim.Optimizer | None = None
        self._opt_cfg: Dict[str, float] | None = None

    def _require_ready(self) -> None:
        if self.model.state is None:
            raise RuntimeError(
                'Session state is not initialized. Send {"cmd":"reset"} first.'
            )

    def reset(self) -> None:
        self.model.reset_state(batch_size=1)
        # Keep training counters; only clear transient, per-turn stats.
        self._stats.k_settle = None
        self._stats.delta_per_k = None
        self._stats.state_norm = None
        self._stats.logits_entropy = None
        self._stats.total_ingested = 0
        self._stats.total_generated = 0

    def ingest(self, text: str, *, k_settle: int | None = None) -> None:
        self._require_ready()
        ids = self.vocab.encode(text, strict=True)
        if not ids:
            return

        k: int | None = None
        delta_accum: Optional[torch.Tensor] = None
        entropy_accum = torch.tensor(0.0, device=self.device)
        last_state_norm = None

        for tid in ids:
            token = torch.tensor([tid], device=self.device, dtype=torch.long)
            _, stats = self.model.step(token, k_settle=k_settle)
            if delta_accum is None:
                delta_accum = torch.zeros_like(
                    stats["delta_per_k"], dtype=torch.float32
                )
                k = int(stats["k_settle"].detach().cpu().item())
            delta_accum += stats["delta_per_k"].to(delta_accum.dtype)
            entropy_accum += stats["logits_entropy"].to(entropy_accum.dtype)
            last_state_norm = stats["state_norm"]

        if delta_accum is None or k is None:
            return

        self._stats.k_settle = k
        self._stats.delta_per_k = (delta_accum / len(ids)).detach().cpu().tolist()
        self._stats.logits_entropy = float(
            (entropy_accum / len(ids)).detach().cpu().item()
        )
        if last_state_norm is not None:
            self._stats.state_norm = float(last_state_norm.detach().cpu().item())
        self._stats.total_ingested += len(ids)

    def generate(
        self,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        k_settle: int | None = None,
        restep_last_token: bool = True,
    ) -> str:
        self._require_ready()
        ids, stats = self.model.generate(
            max_new_tokens,
            temperature=temperature,
            k_settle=k_settle,
            restep_last_token=restep_last_token,
        )
        self._stats.k_settle = int(stats["k_settle"].detach().cpu().item())
        self._stats.delta_per_k = stats["delta_per_k"].detach().cpu().tolist()
        self._stats.logits_entropy = float(
            stats["logits_entropy"].detach().cpu().item()
        )
        self._stats.state_norm = float(stats["state_norm"].detach().cpu().item())
        self._stats.total_generated += int(max_new_tokens)
        return self.vocab.decode(ids[0].tolist())

    def _logit_trace_item(
        self,
        logits: torch.Tensor,
        *,
        position: int,
        selected_id: int,
        expected_id: int | None,
        top_k: int,
    ) -> Dict[str, Any]:
        row = logits[0].detach()
        k = max(1, min(int(top_k), int(row.numel())))
        top_vals, top_ids = torch.topk(row, k=k)
        top = [
            {
                "char": self.vocab.decode([int(idx)]),
                "logit": float(val.detach().cpu().item()),
            }
            for val, idx in zip(top_vals, top_ids)
        ]

        selected_logit = float(row[int(selected_id)].detach().cpu().item())
        expected_char = None
        expected_logit = None
        best_alt_char = None
        best_alt_logit = None
        margin = None
        if expected_id is not None:
            expected_char = self.vocab.decode([int(expected_id)])
            expected_logit = float(row[int(expected_id)].detach().cpu().item())
            alt = row.clone()
            alt[int(expected_id)] = -torch.inf
            alt_logit, alt_id = torch.max(alt, dim=0)
            best_alt_char = self.vocab.decode([int(alt_id.detach().cpu().item())])
            best_alt_logit = float(alt_logit.detach().cpu().item())
            margin = expected_logit - best_alt_logit

        return {
            "position": int(position),
            "expected": expected_char,
            "expected_logit": expected_logit,
            "predicted": top[0]["char"],
            "predicted_logit": top[0]["logit"],
            "selected": self.vocab.decode([int(selected_id)]),
            "selected_logit": selected_logit,
            "best_alternative": best_alt_char,
            "best_alternative_logit": best_alt_logit,
            "margin": margin,
            "topk": top,
        }

    @torch.no_grad()
    def generate_with_trace(
        self,
        max_new_tokens: int,
        *,
        expected: str | None = None,
        top_k: int = 5,
        temperature: float = 1.0,
        k_settle: int | None = None,
        restep_last_token: bool = True,
    ) -> tuple[str, list[Dict[str, Any]]]:
        """
        Generate like generate(), while recording per-token logit diagnostics.

        The trace is captured from the logits used for sampling each generated
        token. This makes pass/fail exams inspectable without changing the
        generation path.
        """
        self._require_ready()
        expected_ids = (
            []
            if expected is None
            else self.vocab.encode(expected, strict=True)
        )

        ids_out: list[int] = []
        trace: list[Dict[str, Any]] = []
        was_training = self.model.training
        self.model.eval()
        try:
            k = self.model._k_settle(k_settle)
            token = self.model.last_token_id
            logits = self.model.last_logits
            if token is None:
                raise RuntimeError(
                    "No last_token_id. Call step() at least once before generate()."
                )
            if not restep_last_token and logits is None:
                raise RuntimeError(
                    "No cached logits. Call step() at least once before generate()."
                )
            if max_new_tokens <= 0:
                raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")

            delta_accum = torch.zeros((k,), device=token.device, dtype=torch.float32)
            entropy_accum = torch.tensor(0.0, device=token.device)
            last_stats: Dict[str, torch.Tensor] = {}

            for i in range(int(max_new_tokens)):
                if restep_last_token:
                    logits, last_stats = self.model.step(token, k_settle=k)
                    delta_accum += last_stats["delta_per_k"].to(delta_accum.dtype)
                    entropy_accum += last_stats["logits_entropy"].to(
                        entropy_accum.dtype
                    )
                elif logits is None:
                    raise RuntimeError("No cached logits available for generation.")

                temp = float(temperature)
                if temp <= 0.0:
                    token = torch.argmax(logits, dim=-1)
                else:
                    next_logits = logits / temp
                    probs = F.softmax(next_logits, dim=-1)
                    token = torch.multinomial(probs, num_samples=1).squeeze(1)

                selected_id = int(token[0].detach().cpu().item())
                expected_id = expected_ids[i] if i < len(expected_ids) else None
                trace.append(
                    self._logit_trace_item(
                        logits,
                        position=i,
                        selected_id=selected_id,
                        expected_id=expected_id,
                        top_k=top_k,
                    )
                )
                ids_out.append(selected_id)

                if not restep_last_token:
                    logits, last_stats = self.model.step(token, k_settle=k)
                    delta_accum += last_stats["delta_per_k"].to(delta_accum.dtype)
                    entropy_accum += last_stats["logits_entropy"].to(
                        entropy_accum.dtype
                    )

            self.model.last_token_id = token
            self._stats.k_settle = k
            self._stats.delta_per_k = (
                delta_accum / int(max_new_tokens)
            ).detach().cpu().tolist()
            self._stats.logits_entropy = float(
                (entropy_accum / int(max_new_tokens)).detach().cpu().item()
            )
            self._stats.state_norm = float(
                last_stats["state_norm"].detach().cpu().item()
            ) if last_stats else None
            self._stats.total_generated += int(max_new_tokens)
            return self.vocab.decode(ids_out), trace
        finally:
            self.model.train(was_training)

    def stats(self) -> Dict[str, Any]:
        out = self._stats.to_dict()
        out["ready"] = self.model.state is not None
        return out

    def _ensure_optimizer(
        self, *, lr: float, weight_decay: float
    ) -> torch.optim.Optimizer:
        if self._opt is None:
            self._opt = torch.optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
            self._opt_cfg = {"lr": float(lr), "weight_decay": float(weight_decay)}
            return self._opt

        # Update hyperparams in-place.
        for pg in self._opt.param_groups:
            pg["lr"] = float(lr)
            pg["weight_decay"] = float(weight_decay)
        self._opt_cfg = {"lr": float(lr), "weight_decay": float(weight_decay)}
        return self._opt

    def learn(
        self,
        examples: list[str],
        *,
        steps: int = 50,
        k_settle: int | None = None,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        grad_clip: float = 1.0,
        detach_state: bool = False,
        reset_state_each_example: bool = True,
        seed: int | None = None,
        loss_mode: str = "full",
    ) -> Dict[str, Any]:
        """
        Supervised next-token learning on short text examples.

        Important: this updates model weights (semi-permanent).
        """
        self._require_ready()
        if not examples:
            raise ValueError("learn() requires at least 1 example")
        if steps <= 0:
            raise ValueError(f"steps must be >= 1, got {steps}")
        if loss_mode not in {"full", "answer_only"}:
            raise ValueError(
                f"loss_mode must be 'full' or 'answer_only', got {loss_mode!r}"
            )

        opt = self._ensure_optimizer(lr=lr, weight_decay=weight_decay)
        rng = random.Random(seed)

        # Preserve interactive session state. Training will reset/unroll state internally.
        saved_state = self.model.state.detach().clone()
        saved_last_token_id = (
            None
            if self.model.last_token_id is None
            else self.model.last_token_id.detach().clone()
        )
        saved_last_logits = (
            None
            if self.model.last_logits is None
            else self.model.last_logits.detach().clone()
        )
        saved_last_step_stats = self.model.last_step_stats

        total_loss = 0.0
        total_tokens = 0
        vocab_size = self.vocab.size

        self.model.train()
        try:
            for _ in range(int(steps)):
                text = rng.choice(examples)
                ids = self.vocab.encode(text, strict=True)
                if len(ids) < 2:
                    continue

                x_ids = ids[:-1]
                y_ids = ids[1:]
                x = torch.tensor([x_ids], device=self.device, dtype=torch.long)
                y = torch.tensor([y_ids], device=self.device, dtype=torch.long)

                if reset_state_each_example:
                    self.model.reset_state(batch_size=1)

                logits, _ = self.model.forward_sequence(
                    x, k_settle=k_settle, detach_state=detach_state
                )

                if loss_mode == "answer_only":
                    marker = " S:"
                    m = text.rfind(marker)
                    if m == -1:
                        raise ValueError(
                            "answer_only loss_mode requires examples containing ' S:'."
                        )
                    ans_start = m + len(marker)
                    # y positions correspond to original indices 1..len(text)-1.
                    # Keep loss only for tokens at original indices >= ans_start.
                    cut = max(int(ans_start) - 1, 0)
                    y_masked = y.clone()
                    if cut > 0:
                        y_masked[:, :cut] = -100
                    loss = F.cross_entropy(
                        logits.reshape(-1, vocab_size),
                        y_masked.reshape(-1),
                        ignore_index=-100,
                    )
                    total_tokens += int((y_masked != -100).sum().detach().cpu().item())
                else:
                    loss = F.cross_entropy(
                        logits.reshape(-1, vocab_size), y.reshape(-1)
                    )
                    total_tokens += len(y_ids)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), float(grad_clip)
                )
                opt.step()

                total_loss += float(loss.detach().cpu().item())
        finally:
            # Restore interactive state.
            self.model.state = saved_state
            self.model.last_token_id = saved_last_token_id
            self.model.last_logits = saved_last_logits
            self.model.last_step_stats = saved_last_step_stats

        if total_tokens == 0:
            raise RuntimeError("No trainable tokens produced (are examples too short?)")

        avg_loss = total_loss / max(int(steps), 1)
        self._stats.total_train_steps += int(steps)
        self._stats.total_train_tokens += int(total_tokens)
        self._stats.last_train_loss = float(avg_loss)

        return {
            "avg_loss": float(avg_loss),
            "steps": int(steps),
            "tokens": int(total_tokens),
        }

    def checkpoint_dict(self, *, include_optimizer: bool = True) -> Dict[str, Any]:
        chars = [self.vocab.itos[i] for i in range(self.vocab.size)]
        ckpt: Dict[str, Any] = {
            "format": "rnn-session-v1",
            "model_cfg": asdict(self.model.cfg)
            if isinstance(self.model.cfg, ModelCfg)
            else {},
            "vocab_chars": chars,
            "model": self.model.state_dict(),
            "stats": self._stats.to_dict(),
        }
        if include_optimizer and self._opt is not None:
            ckpt["optimizer"] = self._opt.state_dict()
            ckpt["optimizer_cfg"] = self._opt_cfg or {}
        return ckpt

    def load_checkpoint_dict(
        self, ckpt: Dict[str, Any], *, load_optimizer: bool = True
    ) -> None:
        vocab_chars = ckpt.get("vocab_chars")
        if vocab_chars is not None:
            expected = [self.vocab.itos[i] for i in range(self.vocab.size)]
            if list(vocab_chars) != expected:
                raise ValueError(
                    "Checkpoint vocab does not match current session vocab (different text-path?)."
                )

        model_state = ckpt.get("model")
        if not isinstance(model_state, dict):
            raise ValueError("Checkpoint missing 'model' state_dict.")
        self.model.load_state_dict(model_state)
        self.model.last_logits = None
        self.model.last_step_stats = None

        if load_optimizer and "optimizer" in ckpt:
            opt_cfg = ckpt.get("optimizer_cfg") or {}
            lr = float(opt_cfg.get("lr", 3e-4))
            weight_decay = float(opt_cfg.get("weight_decay", 0.1))
            opt = self._ensure_optimizer(lr=lr, weight_decay=weight_decay)
            opt.load_state_dict(ckpt["optimizer"])

        stats = ckpt.get("stats")
        if isinstance(stats, dict):
            # Restore training counters only; transient stats reset is explicit via reset().
            self._stats.total_train_steps = int(
                stats.get("total_train_steps", self._stats.total_train_steps)
            )
            self._stats.total_train_tokens = int(
                stats.get("total_train_tokens", self._stats.total_train_tokens)
            )
            last_loss = stats.get("last_train_loss", None)
            if last_loss is not None:
                self._stats.last_train_loss = float(last_loss)
