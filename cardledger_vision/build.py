"""Builds a game's index from its catalog: the entry point of `cardledger-db generate`, which makes the published
indexes."""

import logging
import time
from datetime import UTC, datetime
from pathlib import Path

from . import catalog, images
from .index import Entry, Header, encode

logger = logging.getLogger("cardledger_vision")


def index_key(slug: str) -> str:
    """The index's path, the same in an output folder and in the bucket."""
    return f"vision/v1/{slug}.bin"


def build_index(slug: str, catalog_bytes: bytes, sha256: str, image_root: Path, cache_dir: Path,
                confirm_only: bool = False) -> bytes:
    """The index of the game's cards; a card whose image can't be read enters without a hash.

    `catalog_bytes` is the game's catalog file and `sha256` its digest, kept in the header. The rows' image paths
    are relative to `image_root`; the hashes are cached under `cache_dir`."""
    start = time.monotonic()
    products = sorted(catalog.load(catalog_bytes), key=lambda p: p.id)
    # The normal image is read only for the products whose large image failed.
    large = images.local_hashes([image_root / p.image_large for p in products if p.image_large], cache_dir)
    retry = [image_root / p.image for p in products
             if p.image and (not p.image_large or large[image_root / p.image_large] is None)]
    normal = images.local_hashes(retry, cache_dir)
    entries = []
    for p in products:
        h = large[image_root / p.image_large] if p.image_large else None
        if h is None and p.image:
            h = normal[image_root / p.image]
        entries.append(Entry(p.id, p.name, p.set_code, p.number, h))
    header = Header(slug, datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), sha256, confirm_only)
    logger.info("%s: entries=%d with_hash=%d seconds=%.0f", slug, len(entries),
                sum(e.hash is not None for e in entries), time.monotonic() - start)
    return encode(header, entries)
