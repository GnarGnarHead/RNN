## Brief: CPU-only “settle-before-decode” prototype (no attention)

### Goal

Test whether a **shallow, wide, recurrent “settling loop”** learns anything useful and whether increasing **internal iterations K** improves coherence/stability instead of causing collapse.

This is not about SOTA language modeling. It’s about **dynamics**:

* convergence vs oscillation
* bland attractor collapse vs meaningful refinement
* tradeoff: depth vs iterations

---

## Experiment design

### Model (no attention)

* Token embedding → recurrent core → output head
* Core is **L layers** (suggest 4) of:

  * RMSNorm
  * Linear → GELU → Linear (MLP)
  * Residual
* **Recurrence**: apply the core **K times** before producing logits.

Key idea:

* The hidden state is iteratively refined:

  * `h_{t,0} = embed(x_t) + state_from_prev_token`
  * `h_{t,k+1} = Core(h_{t,k})`
  * logits from `h_{t,K}`

You keep a **token-level state** across time (like an RNN), but with **iterative settling** at each step.

### Data (easy)

Character-level corpus:

* Tiny Shakespeare (or any plain text file you have).

### What to measure

1. **Training loss** vs steps
2. **State convergence** per token:

   * `delta_k = mean(|h_{k+1} - h_k|)` across the batch
   * You want `delta_k` to decrease with k (contraction-ish behavior)
3. **Generation quality**:

   * Watch for repetition / mode collapse as K increases

### Suggested sweep

Run a few short training runs with identical settings except K:

* K ∈ {1, 2, 4, 8}
  Keep everything else fixed. Compare:
* loss curves
* convergence deltas
* generated samples

---

## Practical settings for a weak CPU laptop

* vocab: unique characters in text
* seq_len: 128
* batch_size: 16 (or 8 if slow)
* width: 256
* layers L: 4
* K: start at 2
* optimizer: AdamW, lr=3e-4
* train steps: 5k–20k (enough to see behavior)
* clip_grad_norm: 1.0

---

## Single-file PyTorch template (CPU)

Save as `settle_rnn_charlm.py` and run with Python 3.

```python
import math
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Config
# -----------------------------

@dataclass
class Cfg:
    text_path: str = "input.txt"
    device: str = "cpu"
    seed: int = 1337

    # data
    seq_len: int = 128
    batch_size: int = 16

    # model
    d_model: int = 256
    n_layers: int = 4
    k_settle: int = 2          # INTERNAL ITERATIONS (sweep this)
    dropout: float = 0.0
    use_state: bool = True     # carry state across tokens

    # training
    steps: int = 8000
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    log_every: int = 200
    sample_every: int = 1000
    sample_len: int = 400
    temperature: float = 0.9

# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def build_vocab(text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos

def encode(text: str, stoi: Dict[str, int]) -> torch.Tensor:
    return torch.tensor([stoi[c] for c in text], dtype=torch.long)

def get_batch(data: torch.Tensor, cfg: Cfg) -> Tuple[torch.Tensor, torch.Tensor]:
    # random contiguous chunks
    ix = torch.randint(0, data.numel() - cfg.seq_len - 1, (cfg.batch_size,))
    x = torch.stack([data[i:i+cfg.seq_len] for i in ix])
    y = torch.stack([data[i+1:i+cfg.seq_len+1] for i in ix])
    return x, y

# -----------------------------
# Model
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x):
        # x: (..., d)
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.scale

class MLPBlock(nn.Module):
    def __init__(self, d: int, dropout: float = 0.0):
        super().__init__()
        self.norm = RMSNorm(d)
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        h = self.fc1(h)
        h = F.gelu(h)
        h = self.fc2(h)
        h = self.drop(h)
        return x + h  # residual

class SettleCore(nn.Module):
    def __init__(self, d: int, n_layers: int, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList([MLPBlock(d, dropout) for _ in range(n_layers)])

    def forward(self, h):
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

        # learned gate controlling how much previous-token state influences current token
        self.state_gate = nn.Linear(d, d)

        # learned damping inside settle loop (prevents runaway)
        self.settle_gate = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        x: (B, T) token ids
        returns logits: (B, T, V)
        """
        B, T = x.shape
        d = self.cfg.d_model

        # token embeddings
        emb = self.tok_emb(x)  # (B, T, d)

        # recurrent state carried across time
        state = torch.zeros((B, d), device=emb.device, dtype=emb.dtype)

        logits_out = []
        # track convergence deltas averaged over sequence
        deltas_accum = torch.zeros((self.cfg.k_settle,), device=emb.device)

        for t in range(T):
            h0 = emb[:, t, :]  # (B, d)

            if self.cfg.use_state:
                # gated injection of previous state
                g = torch.sigmoid(self.state_gate(state))
                h = h0 + g * state
            else:
                h = h0

            # settle loop
            prev = h
            for k in range(self.cfg.k_settle):
                h = self.core(h)

                # mild damping gate to encourage contraction
                dg = torch.sigmoid(self.settle_gate(h))
                h = prev + dg * (h - prev)

                # convergence delta
                deltas_accum[k] += (h - prev).abs().mean()
                prev = h

            # update state from settled h (EMA-style)
            if self.cfg.use_state:
                alpha = 0.2
                state = (1.0 - alpha) * state + alpha * h

            # logits for this timestep
            h_out = self.out_norm(h)
            logits = self.lm_head(h_out)  # (B, V)
            logits_out.append(logits.unsqueeze(1))

        logits_out = torch.cat(logits_out, dim=1)  # (B, T, V)
        stats = {
            "delta_per_k": deltas_accum / T
        }
        return logits_out, stats

@torch.no_grad()
def sample(model: SettleCharLM, start: str, stoi, itos, cfg: Cfg, length: int = 400) -> str:
    model.eval()
    device = cfg.device
    ids = torch.tensor([[stoi[c] for c in start]], dtype=torch.long, device=device)
    out = list(start)

    for _ in range(length):
        # feed last seq_len tokens
        x = ids[:, -cfg.seq_len:]
        logits, _ = model(x)
        next_logits = logits[:, -1, :] / max(cfg.temperature, 1e-6)
        probs = F.softmax(next_logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)  # (1,1)
        ids = torch.cat([ids, nxt], dim=1)
        out.append(itos[int(nxt.item())])

    return "".join(out)

# -----------------------------
# Train loop
# -----------------------------

def main():
    cfg = Cfg()
    set_seed(cfg.seed)

    text = load_text(cfg.text_path)
    stoi, itos = build_vocab(text)
    data = encode(text, stoi)

    device = torch.device(cfg.device)
    vocab_size = len(stoi)

    model = SettleCharLM(vocab_size, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    t0 = time.time()
    for step in range(1, cfg.steps + 1):
        model.train()
        xb, yb = get_batch(data, cfg)
        xb, yb = xb.to(device), yb.to(device)

        logits, stats = model(xb)
        loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if step % cfg.log_every == 0 or step == 1:
            dt = time.time() - t0
            delta = stats["delta_per_k"].detach().cpu().tolist()
            delta_str = ", ".join([f"{v:.4f}" for v in delta])
            print(f"step {step:5d} | loss {loss.item():.4f} | dt {dt:.1f}s | delta[k]: [{delta_str}]")
            t0 = time.time()

        if step % cfg.sample_every == 0:
            s = sample(model, start="To be", stoi=stoi, itos=itos, cfg=cfg, length=cfg.sample_len)
            print("\n--- SAMPLE ---")
            print(s)
            print("--------------\n")

if __name__ == "__main__":
    main()
```

---

## How to run

1. Put a text file named `input.txt` in the same folder.
2. Install torch (CPU build).
3. Run:

```bash
python settle_rnn_charlm.py
```

Then sweep K:

* Edit `k_settle` in the config (2, 4, 8)
* Compare `delta[k]` and samples

---

## What “good” looks like

* `delta[k]` decreases with k (e.g., 0.18 → 0.11 → 0.07 → 0.05)
* samples become less chaotic with larger K *without* collapsing into repetition
* loss improves modestly with K (not guaranteed, but often)

If instead you see:

* `delta[k]` growing → instability
* fast repetition loops → attractor collapse
* no change at all with K → the loop is being ignored / too damped

Those outcomes are still useful; they tell you what to change next (gate init, damping strength, EMA alpha, width, etc.).

If you run it and paste ~20 lines of logs + one sample, I can tell you which failure mode you hit and the next two tweaks that are most likely to move it.
