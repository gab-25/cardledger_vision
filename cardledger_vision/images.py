"""Downloads catalog images politely: few at once, a pause between requests to a host, retries, a disk cache."""

import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

import httpx

MAX_PARALLEL = 8
MIN_INTERVAL = 0.1  # seconds between two requests to the same host
ATTEMPTS = 3

sleep = time.sleep  # patched by the tests
clock = time.monotonic

_lock = threading.Lock()
_next_slot: dict[str, float] = {}


def _wait_turn(host: str) -> None:
    """Reserves the host's next request slot, then waits for it outside the lock."""
    with _lock:
        now = clock()
        slot = max(now, _next_slot.get(host, now))
        _next_slot[host] = slot + MIN_INTERVAL
    if slot > now:
        sleep(slot - now)


def _download(url: str, client: httpx.Client) -> bytes | None:
    host = urlsplit(url).netloc
    for attempt in range(ATTEMPTS):
        _wait_turn(host)
        try:
            response = client.get(url)
            if response.status_code == 200:
                return response.content
            if response.status_code < 500 and response.status_code != 429:
                return None  # 404 and the like will not get better
        except httpx.HTTPError:
            pass
    return None


def _one(url: str, cache_dir: Path | None, client: httpx.Client) -> bytes | None:
    cached = None
    if cache_dir is not None:
        cached = cache_dir / "images" / hashlib.sha1(url.encode()).hexdigest()
        if cached.is_file():
            return cached.read_bytes()
    data = _download(url, client)
    if data is not None and cached is not None:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(data)
    return data


def fetch(urls: list[str], cache_dir: Path | None, client: httpx.Client) -> dict[str, bytes | None]:
    """{url: image bytes, or None if it could not be downloaded}."""
    unique = list(dict.fromkeys(urls))
    with ThreadPoolExecutor(MAX_PARALLEL) as pool:
        return dict(zip(unique, pool.map(lambda url: _one(url, cache_dir, client), unique)))
