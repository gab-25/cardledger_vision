"""256-bit perceptual hash of a straightened card, the one used for the catalog images and the photos.

Exact recipe, for a port (for example to JavaScript) that must give the same bits:

1. Input: a BGR image (OpenCV channel order) of the straightened card, any size; the catalog images are resized
   to CARD_SIZE (488x680, INTER_AREA) first, the photos come out of `detect` at that size.
2. Gray, BT.601 weights, in float: gray = 0.299 * R + 0.587 * G + 0.114 * B (no rounding), stored as float32.
3. Resize the gray image to 64x64 with OpenCV's INTER_AREA (area averaging), then work in float64.
4. Orthonormal 2D DCT-II: Y = C @ X @ C.T, where C[k, i] = sqrt(2/64) * cos(pi * (2i + 1) * k / 128) for k > 0
   and C[0, i] = sqrt(1/64) (row k = frequency, column i = sample); rows of X are image rows.
5. Keep the top-left 16x16 block of Y, DC coefficient Y[0, 0] included: 256 coefficients.
6. Threshold at their median (numpy's: the mean of the 128th and 129th smallest): bit = coefficient > median,
   strictly.
7. Pack the 256 bits row-major (Y[0, 0], Y[0, 1], ... Y[0, 15], Y[1, 0], ...) into 32 bytes, most significant bit
   first (numpy.packbits): bit 0 is the 0x80 bit of byte 0.

Distance between two hashes: the Hamming distance of the 256 bits.
"""

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
