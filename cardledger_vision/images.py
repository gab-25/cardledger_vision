"""Downloads catalog images politely: few at once, a pause between requests to a host, retries with backoff,
a disk cache of the images that decode."""

import hashlib
import logging
import os
import struct
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import httpx
import numpy as np

from .phash import CARD_SIZE, hash_card

MAX_PARALLEL = 8
MIN_INTERVAL = 0.1  # default seconds between two requests to the same host
ATTEMPTS = 3
BACKOFF = (0.5, 1, 2)  # seconds to wait after a failed attempt, when another one follows
MAX_RETRY_AFTER = 60  # a longer Retry-After is capped
BATCH = 1000  # URLs submitted to the workers at once
BACKGROUND = 128  # grey under the transparent pixels of a PNG
HASHES = "hashes"  # cache subfolder of the image hashes; rename it if the hash recipe changes

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_next_slot: dict[str, float] = {}
_cancel = threading.Event()  # set while an interrupted download pool winds down: the workers stop early


def sleep(seconds: float) -> None:  # patched by the tests
    """Waits, but returns at once when the downloads are cancelled."""
    _cancel.wait(seconds)


clock = time.monotonic


@contextmanager
def _pool() -> Iterator[ThreadPoolExecutor]:
    """The download workers. On Ctrl+C (or any error) the queued URLs are dropped and the running workers give up
    at their next wait or attempt, so the interruption doesn't wait for the whole batch."""
    pool = ThreadPoolExecutor(MAX_PARALLEL)
    try:
        yield pool
    except BaseException:
        _cancel.set()
        try:
            pool.shutdown(wait=True, cancel_futures=True)
        finally:
            _cancel.clear()
        raise
    pool.shutdown(wait=True)


def _wait_turn(host: str, interval: float) -> None:
    """Reserves the host's next request slot, then waits for it outside the lock."""
    with _lock:
        now = clock()
        slot = max(now, _next_slot.get(host, now))
        _next_slot[host] = slot + interval
    if slot > now:
        sleep(slot - now)


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = parsedate_to_datetime(value).timestamp() - time.time()
        except (TypeError, ValueError):
            return None
    return min(max(seconds, 0), MAX_RETRY_AFTER)


def _download(url: str, client: httpx.Client, interval: float) -> bytes | None:
    host = urlsplit(url).netloc
    for attempt in range(ATTEMPTS):
        _wait_turn(host, interval)
        if _cancel.is_set():
            return None
        wait = BACKOFF[attempt]
        try:
            response = client.get(url)
            if response.status_code == 200:
                return response.content
            if response.status_code < 500 and response.status_code != 429:
                return None  # 404 and the like will not get better
            if response.status_code in (429, 503):
                wait = _retry_after(response) or wait
        except httpx.HTTPError as exc:
            logger.debug("%s: %s", url, exc)
        if attempt + 1 < ATTEMPTS:
            sleep(wait)
    return None


def decode(data: bytes) -> np.ndarray | None:
    """The upright BGR card in the image bytes, or None if they don't decode.

    Transparent pixels are laid on mid-grey; a landscape image is turned 90 degrees, so it compares with the
    portrait output of `detect` (a 180 degree turn is handled by the recognition).
    """
    buf = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 3 and image.shape[2] == 4:
        if image.dtype == np.uint16:
            image = (image / 257).round().astype(np.uint8)
        elif image.dtype != np.uint8:
            return None
        alpha = image[..., 3:].astype(np.float32) / 255
        image = (image[..., :3].astype(np.float32) * alpha + BACKGROUND * (1 - alpha)).round().astype(np.uint8)
    else:
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # grey, 16-bit and EXIF orientation handled by OpenCV
        if image is None:
            return None
    if image.shape[1] > image.shape[0]:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


def card_hash(image: np.ndarray) -> bytes:
    """The hash of a decoded catalog image (see `decode`), as the index stores it."""
    return hash_card(cv2.resize(image, CARD_SIZE, interpolation=cv2.INTER_AREA))


def _one(url: str, cache_dir: Path | None, client: httpx.Client, interval: float) -> tuple[bytes, np.ndarray] | None:
    """The image's bytes and decoded card, from the cache or the network; None if they can't be had."""
    cached = None
    if cache_dir is not None:
        cached = cache_dir / "images" / hashlib.sha1(url.encode()).hexdigest()
        if cached.is_file():
            data = cached.read_bytes()
            image = decode(data)
            if image is not None:
                return data, image
            cached.unlink(missing_ok=True)  # corrupt: download it again
    data = _download(url, client, interval)
    if data is None:
        return None
    image = decode(data)
    if image is None:
        logger.debug("%s: not an image", url)
        return None
    if cached is not None:
        cached.parent.mkdir(parents=True, exist_ok=True)
        tmp = cached.with_name(f"{cached.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, cached)
    return data, image


def _safe(url: str, work, cache_dir: Path | None, client: httpx.Client, interval: float):
    """`work(bytes, image)` on the URL's image; any failure on one URL gives None and the others go on."""
    try:
        got = _one(url, cache_dir, client, interval)
        return None if got is None else work(*got)
    except Exception as exc:  # httpx.InvalidURL, cv2.error, OSError...: skip this image only
        logger.debug("%s: %s: %s", url, type(exc).__name__, exc)
        return None


def _stamp(path: Path) -> bytes | None:
    """Size and modification time of the cached image, which a cached hash must match."""
    try:
        st = path.stat()
    except OSError:
        return None
    return struct.pack("<QQ", st.st_size, st.st_mtime_ns)


def _cached_hash(url: str, cache_dir: Path | None, client: httpx.Client, interval: float) -> bytes | None:
    """The hash of the URL's image. With a cache, the hash is kept next to the image (cache/hashes/<sha1>, with the
    image file's size and mtime), so a rebuild doesn't decode the image again; an image file that changed or went
    is fetched and hashed again."""
    if cache_dir is None:
        return _safe(url, lambda data, image: card_hash(image), None, client, interval)
    name = hashlib.sha1(url.encode()).hexdigest()
    image_path, hash_path = cache_dir / "images" / name, cache_dir / HASHES / name
    try:
        saved = hash_path.read_bytes()
    except OSError:
        saved = b""
    stamp = _stamp(image_path)
    if stamp is not None and len(saved) == len(stamp) + 32 and saved.startswith(stamp):
        return saved[len(stamp):]
    h = _safe(url, lambda data, image: card_hash(image), cache_dir, client, interval)
    stamp = _stamp(image_path)
    if h is not None and stamp is not None:
        try:
            hash_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = hash_path.with_name(f"{name}.{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_bytes(stamp + h)
            os.replace(tmp, hash_path)
        except OSError as exc:  # the hash is still good, only not kept
            logger.debug("%s: %s", hash_path, exc)
    return h


def _hash_batch(urls: list[str], pool: ThreadPoolExecutor, cache_dir: Path | None, client: httpx.Client,
                interval: float) -> list[bytes | None]:
    return list(pool.map(lambda url: _cached_hash(url, cache_dir, client, interval), urls))


def fetch_hashes(urls: list[str], cache_dir: Path | None, client: httpx.Client,
                 host_interval: float = MIN_INTERVAL) -> dict[str, bytes | None]:
    """{url: hash of its image, or None}. Each image is hashed in the worker as soon as it is downloaded and only
    the 32-byte hash is kept; the URLs go to the workers in batches of BATCH."""
    unique = list(dict.fromkeys(urls))
    out: dict[str, bytes | None] = {}
    with _pool() as pool:
        for start in range(0, len(unique), BATCH):
            batch = unique[start:start + BATCH]
            out.update(zip(batch, _hash_batch(batch, pool, cache_dir, client, host_interval)))
    return out


def fetch(urls: list[str], cache_dir: Path | None, client: httpx.Client,
          host_interval: float = MIN_INTERVAL) -> dict[str, bytes | None]:
    """{url: image bytes that decode, or None if they could not be had}."""
    unique = list(dict.fromkeys(urls))

    def work(data: bytes, image: np.ndarray) -> bytes:
        return data

    with _pool() as pool:
        return dict(zip(unique, pool.map(lambda url: _safe(url, work, cache_dir, client, host_interval), unique)))
