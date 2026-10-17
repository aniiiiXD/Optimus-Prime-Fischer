"""Fetch exactly the paper author's FooPlayer file for repeatable testing."""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

COMMIT = "b1f2f1c6d43afd8d5ce5ca71542b78d37173b2a1"
SHA256 = "b01c69386a7a362365085e4b6866a003c93068e6911d79a958c999ce72c7c692"
SOURCE_RELATIVE = Path("supplementary/Evolutions/GPT Evolution 2/foo_player.py")
TARGET = Path(__file__).resolve().parent / ".source_cache" / SOURCE_RELATIVE
URL = f"https://raw.githubusercontent.com/nbelle1/strategy-game-agents/{COMMIT}/{quote(str(SOURCE_RELATIVE))}"


def main():
    data = urlopen(URL, timeout=30).read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SHA256:
        raise ValueError(f"Published source checksum mismatch: {digest}")
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_bytes(data)
    print(f"Fetched {TARGET} ({digest})")


if __name__ == "__main__":
    main()
