"""Fetch GargantuaX's calibrated Gemini alpha map for 96×96 watermarks.

Idempotent: skips download if the file already exists and matches the expected
size. Source license: MIT (GargantuaX/gemini-watermark-remover).
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import structlog

log = structlog.get_logger()

SOURCE_URL = "https://raw.githubusercontent.com/GargantuaX/gemini-watermark-remover/main/src/assets/bg_96.png"
EXPECTED_SIZE = 8165  # bytes (verified against GargantuaX main as of 2026-06-03)
DEFAULT_DEST = Path("data/gemini_96_alpha.png")


def fetch(dest: Path = DEFAULT_DEST, *, force: bool = False) -> Path:
    """Download the alpha map PNG to *dest* unless it is already present and valid."""
    if not force and dest.is_file() and dest.stat().st_size == EXPECTED_SIZE:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = httpx.get(SOURCE_URL, follow_redirects=True, timeout=30.0)
    response.raise_for_status()
    dest.write_bytes(response.content)
    return dest


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DEST
    path = fetch(target)
    log.info("alpha_map_fetched", path=str(path), size_bytes=path.stat().st_size)
