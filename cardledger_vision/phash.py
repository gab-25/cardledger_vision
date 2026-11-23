import cv2
import numpy as np

CARD_SIZE = (488, 680)  # (width, height), 63x88 proportion
_N = 64
_K = 16


def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    c = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2.0 / n)
    c[0] /= np.sqrt(2.0)
    return c


_C = _dct_matrix(_N)


def hash_card(card: np.ndarray) -> bytes:
    """256-bit pHash of a straightened BGR card image of any size."""
    b, g, r = card[..., 0], card[..., 1], card[..., 2]
    gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
    small = cv2.resize(gray, (_N, _N), interpolation=cv2.INTER_AREA).astype(np.float64)
    low = (_C @ small @ _C.T)[:_K, :_K]
    bits = (low > np.median(low)).reshape(-1)
    return np.packbits(bits).tobytes()


def hamming(a: bytes, b: bytes) -> int:
    return int.from_bytes(a, "big").__xor__(int.from_bytes(b, "big")).bit_count()


def hamming_many(table: np.ndarray, h: bytes) -> np.ndarray:
    return np.bitwise_count(table ^ np.frombuffer(h, np.uint8)).sum(axis=1)
