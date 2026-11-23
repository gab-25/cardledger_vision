"""Builds a game's index from its catalog: the entry point for the programs that make the indexes
(`cardledger-vision build-index` and `cardledger-db generate`)."""

import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from . import catalog, images
from .index import Entry, Header, encode

logger = logging.getLogger("cardledger_vision")


def index_key(slug: str) -> str:
    """The index's path, the same in an output folder and in the bucket."""
    return f"vision/v1/{slug}.bin"


def build_index(slug: str, catalog_bytes: bytes, sha256: str, client: httpx.Client, cache_dir: Path,
                confirm_only: bool = False, host_interval: float = images.MIN_INTERVAL) -> bytes:
    """The index of the game's single cards; a card whose image can't be had enters without a hash.

    `catalog_bytes` is the game's catalog file and `sha256` its digest, kept in the header. The images and their
    hashes are cached under `cache_dir`."""
    start = time.monotonic()
    products = sorted(catalog.load(catalog_bytes), key=lambda p: p.id_product)
    # Large images are fetched only for the products whose normal image failed.
    primary = images.fetch_hashes([p.image_url for p in products if p.image_url], cache_dir, client, host_interval)
    retry = [p.image_large_url for p in products if p.image_large_url and primary.get(p.image_url) is None]
    fallback = images.fetch_hashes(retry, cache_dir, client, host_interval)
    entries = []
    for p in products:
        h = primary.get(p.image_url) if p.image_url else None
        if h is None and p.image_large_url:
            h = fallback.get(p.image_large_url)
        entries.append(Entry(p.id_product, p.name, p.expansion_code, p.number, h))
    header = Header(slug, datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), sha256, confirm_only)
    logger.info("%s: entries=%d with_hash=%d seconds=%.0f", slug, len(entries),
                sum(e.hash is not None for e in entries), time.monotonic() - start)
    return encode(header, entries)
