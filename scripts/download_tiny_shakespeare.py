from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlopen


DEFAULT_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download Tiny Shakespeare as a plain text file."
    )
    p.add_argument(
        "--out", default="input.txt", help="Output path (default: input.txt)"
    )
    p.add_argument("--url", default=DEFAULT_URL, help="Source URL")
    p.add_argument(
        "--force", action="store_true", help="Overwrite if the output file exists"
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        raise SystemExit(
            f"Refusing to overwrite existing file: {out_path} (use --force)"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(args.url) as r:
        data = r.read()
    out_path.write_bytes(data)
    print(f"Wrote {len(data):,} bytes to {out_path}")


if __name__ == "__main__":
    main()
