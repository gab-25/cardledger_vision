"""Synthetic benchmark: photos made from the catalog images, recognized against an index."""

import random
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np

from .catalog import CatalogProduct
from .index import Index
from .match import recognize

CANVAS_W, CANVAS_H = 1500, 2000
OUTCOMES = ("ok", "wrong_certain", "in_candidates", "not_found", "no_card")
MODES = ("no_lock", "lock")


def _gradient(rng: random.Random, np_rng: np.random.Generator, lo: float, hi: float, amp: float) -> np.ndarray:
    yy, xx = np.mgrid[0:CANVAS_H, 0:CANVAS_W].astype(np.float32)
    out = np.zeros((CANVAS_H, CANVAS_W, 3), np.float32)
    for c in range(3):
        gx, gy = rng.uniform(-1, 1), rng.uniform(-1, 1)
        out[..., c] = rng.uniform(lo, hi) + gx * xx / CANVAS_W * amp + gy * yy / CANVAS_H * amp
    return out


def synth_photo(card: np.ndarray, rng: random.Random) -> bytes:
    """A JPEG of the card as a phone would shoot it; deterministic given `rng`."""
    np_rng = np.random.default_rng(rng.randrange(2**32))

    # Background: gradient plus blurred noise.
    bg = _gradient(rng, np_rng, 70, 170, 60)
    noise = np_rng.normal(0, 25, (CANVAS_H // 20, CANVAS_W // 20, 3)).astype(np.float32)
    noise = cv2.resize(noise, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_CUBIC)
    canvas = np.clip(cv2.GaussianBlur(bg + noise, (0, 0), 15), 0, 255).astype(np.uint8)

    # Card: 62-85% of the canvas height, shifted up to 80 px, corners moved up to 7%, rotated up to 10 degrees.
    ch = rng.uniform(0.62, 0.85) * CANVAS_H
    cw = ch * card.shape[1] / card.shape[0]
    cx = CANVAS_W / 2 + rng.uniform(-80, 80)
    cy = CANVAS_H / 2 + rng.uniform(-80, 80)
    angle = np.radians(rng.uniform(-10, 10))
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    dst = []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        jx, jy = rng.uniform(-0.07, 0.07) * cw, rng.uniform(-0.07, 0.07) * ch
        dst.append(rot @ np.array([sx * cw / 2 + jx, sy * ch / 2 + jy]) + (cx, cy))
    h, w = card.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    m = cv2.getPerspectiveTransform(src, np.float32(dst))
    warped = cv2.warpPerspective(card, m, (CANVAS_W, CANVAS_H), flags=cv2.INTER_AREA)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), m, (CANVAS_W, CANVAS_H), flags=cv2.INTER_LINEAR)
    a = (mask.astype(np.float32) / 255)[..., None]
    img = warped.astype(np.float32) * a + canvas.astype(np.float32) * (1 - a)

    # Light: brightness, a gradient, sometimes an elliptical glare.
    img = img * rng.uniform(0.75, 1.15) + rng.uniform(-20, 20)
    img += _gradient(rng, np_rng, 0, 0, 40)  # zero base: only the slopes act
    if rng.random() < 0.35:
        glare = np.zeros((CANVAS_H, CANVAS_W), np.float32)
        axes = (int(rng.uniform(0.1, 0.3) * CANVAS_W), int(rng.uniform(0.05, 0.15) * CANVAS_H))
        cv2.ellipse(glare, (int(cx + rng.uniform(-200, 200)), int(cy + rng.uniform(-300, 300))), axes,
                    rng.uniform(0, 180), 0, 360, 1.0, -1)
        glare = cv2.GaussianBlur(glare, (0, 0), 40) * rng.uniform(60, 140)
        img += glare[..., None]
    img = np.clip(img, 0, 255).astype(np.uint8)

    # Optics: blur, sometimes motion blur, sensor noise, JPEG.
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.8))
    if rng.random() < 0.20:
        k = rng.randrange(5, 16)
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1 / k
        rm = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1)
        kernel = cv2.warpAffine(kernel, rm, (k, k))
        img = cv2.filter2D(img, -1, kernel / max(kernel.sum(), 1e-6))
    img = np.clip(img.astype(np.float32) + np_rng.normal(0, rng.uniform(2, 7), img.shape), 0, 255).astype(np.uint8)
    ok, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randrange(70, 91)])
    return jpeg.tobytes()


def _url(product: CatalogProduct, images: dict) -> str | None:
    for url in (product.image_large_url, product.image_url):
        if url and images.get(url):
            return url
    return None


def sample(products: list[CatalogProduct], images: dict, size: int, seed: int) -> list[CatalogProduct]:
    """Half at random, a quarter from names with 6+ printings, a quarter from names with 2-5; only products with an image."""
    pool = sorted((p for p in products if _url(p, images)), key=lambda p: p.id_product)
    size = min(size, len(pool))
    rng = random.Random(seed)
    printings = Counter(p.name.casefold() for p in pool)
    many = [p for p in pool if printings[p.name.casefold()] >= 6]
    few = [p for p in pool if 2 <= printings[p.name.casefold()] <= 5]
    chosen: dict[int, CatalogProduct] = {}

    def take(source: list[CatalogProduct], n: int) -> None:
        free = [p for p in source if p.id_product not in chosen]
        for p in rng.sample(free, min(n, len(free))):
            chosen[p.id_product] = p

    quarter = size // 4
    take(many, quarter)
    take(few, quarter)
    take(pool, size - len(chosen))  # the random half, plus whatever the strata lacked
    return list(chosen.values())


def _percent(part: int, total: int) -> float:
    return 100 * part / total if total else 0.0


@dataclass
class Report:
    total: int = 0
    counts: dict[str, dict[str, int]] = field(default_factory=lambda: {m: dict.fromkeys(OUTCOMES, 0) for m in MODES})
    times_ms: dict[str, list[float]] = field(default_factory=lambda: {m: [] for m in MODES})

    def table(self) -> str:
        lines = [f"photos: {self.total}", f"{'outcome':<15}" + "".join(f"{m:>18}" for m in MODES)]
        for outcome in OUTCOMES:
            row = f"{outcome:<15}"
            for mode in MODES:
                n = self.counts[mode][outcome]
                row += f"{n:>9} {_percent(n, self.total):>6.1f}% "
            lines.append(row)
        for label, fn in (("median ms", statistics.median),
                          ("p90 ms", lambda t: statistics.quantiles(t, n=10, method="inclusive")[-1]
                           if len(t) > 1 else t[0])):
            lines.append(f"{label:<15}" + "".join(
                f"{fn(self.times_ms[m]) if self.times_ms[m] else 0:>18.1f}" for m in MODES))
        return "\n".join(lines)


def run(index: Index, products: list[CatalogProduct], images: dict, size: int, seed: int) -> Report:
    report = Report()
    rng = random.Random(seed)
    for product in sample(products, images, size, seed):
        card = cv2.imdecode(np.frombuffer(images[_url(product, images)], np.uint8), cv2.IMREAD_COLOR)
        if card is None:
            continue
        photo = synth_photo(card, rng)
        report.total += 1
        for mode, lock in (("no_lock", None), ("lock", product.expansion_code or None)):
            start = time.perf_counter()
            result = recognize(photo, index, lock)
            report.times_ms[mode].append((time.perf_counter() - start) * 1000)
            if result.outcome == "ok":
                outcome = "ok" if result.product_id == product.id_product else "wrong_certain"
            elif result.outcome == "no_card":
                outcome = "no_card"
            elif product.id_product in result.candidates:
                outcome = "in_candidates"
            else:
                outcome = "not_found"
            report.counts[mode][outcome] += 1
    return report
