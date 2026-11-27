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


@pytest.fixture
def photo_of():
    def make(card: np.ndarray, angle: float, bg_seed: int) -> np.ndarray:
        rng = np.random.default_rng(bg_seed)
        h, w = 2000, 1500
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        bg = np.zeros((h, w, 3), np.float32)
        for c in range(3):
            gx, gy = rng.uniform(-1, 1, 2)
            bg[..., c] = rng.uniform(90, 150) + gx * xx / w * 40 + gy * yy / h * 40
        canvas = np.clip(bg, 0, 255).astype(np.uint8)

        ch = int(h * 0.7)
        cw = int(ch * card.shape[1] / card.shape[0])
        small = cv2.resize(card, (cw, ch), interpolation=cv2.INTER_AREA)
        # A thin dark frame, like the border of a real card, keeps its edge visible
        # against any background.
        small = cv2.copyMakeBorder(small, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=(15, 15, 15))
        ch, cw = small.shape[:2]
        # Rotate on a square layer large enough for any angle, with an alpha mask.
        side = int(np.hypot(cw, ch)) + 4
        layer = np.zeros((side, side, 3), np.uint8)
        mask = np.zeros((side, side), np.uint8)
        x0, y0 = (side - cw) // 2, (side - ch) // 2
        layer[y0:y0 + ch, x0:x0 + cw] = small
        mask[y0:y0 + ch, x0:x0 + cw] = 255
        m = cv2.getRotationMatrix2D((side / 2, side / 2), angle, 1.0)
        layer = cv2.warpAffine(layer, m, (side, side), flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(mask, m, (side, side), flags=cv2.INTER_LINEAR)

        ox, oy = (w - side) // 2, (h - side) // 2
        sx0, sy0 = max(ox, 0), max(oy, 0)
        sx1, sy1 = min(ox + side, w), min(oy + side, h)
        lx0, ly0 = sx0 - ox, sy0 - oy
        lx1, ly1 = lx0 + (sx1 - sx0), ly0 + (sy1 - sy0)
        region = canvas[sy0:sy1, sx0:sx1].astype(np.float32)
        a = (mask[ly0:ly1, lx0:lx1].astype(np.float32) / 255)[..., None]
        blended = layer[ly0:ly1, lx0:lx1] * a + region * (1 - a)
        canvas[sy0:sy1, sx0:sx1] = blended.astype(np.uint8)
        return canvas

    return make


@pytest.fixture
def write_card_image(card_image):
    """Writes the synthetic card `seed` at `path`, in the format of its extension (WebP lossless)."""
    def write(path, seed: int = 1, image: np.ndarray | None = None):
        path.parent.mkdir(parents=True, exist_ok=True)
        ok, data = cv2.imencode(path.suffix, card_image(seed) if image is None else image,
                                [cv2.IMWRITE_WEBP_QUALITY, 101])
        assert ok
        path.write_bytes(data.tobytes())
        return path

    return write
