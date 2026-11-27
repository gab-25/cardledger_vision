"""Hashes the catalog's images from local files, on every core, with a disk cache of the hashes."""

import logging
import os
import struct
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

from .phash import CARD_SIZE, hash_card

WORKERS = os.cpu_count() or 1
BACKGROUND = 128  # grey under the transparent pixels of a PNG
HASHES = "hashes"  # cache subfolder of the image hashes; rename it if the hash recipe changes

logger = logging.getLogger(__name__)


@contextmanager
def _pool() -> Iterator[ThreadPoolExecutor]:
    """The hashing workers. On Ctrl+C (or any error) the queued files are dropped, so the interruption waits only
    for the files being decoded."""
    pool = ThreadPoolExecutor(WORKERS)
    try:
        yield pool
    except BaseException:
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    pool.shutdown(wait=True)


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


def _stamp(path: Path) -> bytes:
    """Size and modification time of the image file, which a cached hash must match."""
    st = path.stat()
    return struct.pack("<QQ", st.st_size, st.st_mtime_ns)


def _hash(path: Path) -> bytes | None:
    try:
        image = decode(path.read_bytes())
    except (OSError, cv2.error) as exc:  # missing, a folder, unreadable, or the decoder failed
        logger.debug("%s: %s: %s", path, type(exc).__name__, exc)
        return None
    if image is None:
        logger.debug("%s: not an image", path)
        return None
    return card_hash(image)


def _cached_hash(path: Path, cache_dir: Path) -> bytes | None:
    """The hash of the image file. It is kept in cache/hashes/<file name> with the file's size and mtime, so a
    rebuild doesn't decode the image again; a file that changed is hashed again."""
    try:
        stamp = _stamp(path)
    except OSError:
        return None
    saved_path = cache_dir / HASHES / path.name
    try:
        saved = saved_path.read_bytes()
    except OSError:
        saved = b""
    if len(saved) == len(stamp) + 32 and saved.startswith(stamp):
        return saved[len(stamp):]
    h = _hash(path)
    if h is not None:
        try:
            saved_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = saved_path.with_name(f"{saved_path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_bytes(stamp + h)
            os.replace(tmp, saved_path)
        except OSError as exc:  # the hash is still good, only not kept
            logger.debug("%s: %s", saved_path, exc)
    return h


def local_hashes(paths: list[Path], cache_dir: Path) -> dict[Path, bytes | None]:
    """{path: hash of its image, or None if the file is missing or doesn't decode}."""
    unique = list(dict.fromkeys(paths))
    with _pool() as pool:
        return dict(zip(unique, pool.map(lambda path: _cached_hash(path, cache_dir), unique)))
