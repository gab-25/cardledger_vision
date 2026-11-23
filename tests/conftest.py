import cv2
import numpy as np
import pytest


@pytest.fixture
def card_image():
    def make(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        h, w = 680, 488
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        img = np.zeros((h, w, 3), np.float32)
        for c in range(3):
            gx, gy = rng.uniform(-1, 1, 2)
            img[..., c] = rng.uniform(40, 200) + gx * xx / w * 80 + gy * yy / h * 80
        img = np.clip(img, 0, 255).astype(np.uint8)
        for _ in range(12):
            color = tuple(int(v) for v in rng.integers(0, 256, 3))
            x0, y0 = int(rng.integers(0, w - 60)), int(rng.integers(0, h - 60))
            x1, y1 = x0 + int(rng.integers(40, 250)), y0 + int(rng.integers(40, 300))
            if rng.random() < 0.5:
                cv2.rectangle(img, (x0, y0), (min(x1, w - 1), min(y1, h - 1)), color, -1)
            else:
                cv2.circle(img, (x0 + 30, y0 + 30), int(rng.integers(30, 150)), color, -1)
        return img

    return make
