from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class Vocab:
    stoi: Dict[str, int]
    itos: Dict[int, str]

    @classmethod
    def from_text(cls, text: str) -> "Vocab":
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for ch, i in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    @property
    def size(self) -> int:
        return len(self.stoi)

    def sanitize(self, text: str) -> str:
        return "".join(c for c in text if c in self.stoi)

    def encode(self, text: str, *, strict: bool = True) -> List[int]:
        if strict:
            missing = sorted({c for c in text if c not in self.stoi})
            if missing:
                preview = "".join(missing[:20])
                raise ValueError(f"Text contains OOV characters (showing up to 20): {preview!r}")
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos[int(i)] for i in ids)

