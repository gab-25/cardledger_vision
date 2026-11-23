"""Synthetic benchmark: photos made from the catalog images, recognized against an index."""

import logging
import os
import random
import statistics
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import cv2
import numpy as np

from .catalog import CatalogProduct
from .images import decode
from .index import Index
from .match import drop_row, recognize

CANVAS_W, CANVAS_H = 1500, 2000
OUTCOMES = ("ok", "wrong_certain", "in_candidates", "not_found", "no_card")
MODES = ("no_lock", "lock", "unindexed")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Shot:
    """Every random draw of one synthetic photo, in the order `random.Random` gives them; the pixels follow from
    these alone, so the photos can be rendered in parallel once the draws are made in turn."""
    np_seed: int
    background: tuple  # (gx, gy, base) per channel
    height: float  # card height, fraction of the canvas height
    shift: tuple[float, float]
    angle: float  # degrees
    corners: tuple  # (jx, jy) per corner, fraction of the card size
    gain: float
    offset: float
    light: tuple  # (gx, gy, base) per channel
    glare: tuple | None  # (axis x, axis y, centre dx, centre dy, angle, strength)
    blur: float
    motion: tuple[int, float] | None  # (kernel size, angle)
    noise: float
    quality: int


def _gradient_draws(rng: random.Random, lo: float, hi: float) -> tuple:
    out = []
    for _ in range(3):
        gx, gy = rng.uniform(-1, 1), rng.uniform(-1, 1)
        out.append((gx, gy, rng.uniform(lo, hi)))
    return tuple(out)


def _draw(rng: random.Random) -> _Shot:
    np_seed = rng.randrange(2**32)
    background = _gradient_draws(rng, 70, 170)
    height = rng.uniform(0.62, 0.85)
    shift = (rng.uniform(-80, 80), rng.uniform(-80, 80))
    angle = rng.uniform(-10, 10)
    corners = tuple((rng.uniform(-0.07, 0.07), rng.uniform(-0.07, 0.07)) for _ in range(4))
    gain, offset = rng.uniform(0.75, 1.15), rng.uniform(-20, 20)
    light = _gradient_draws(rng, 0, 0)  # zero base: only the slopes act
    glare = None
    if rng.random() < 0.35:
        glare = (rng.uniform(0.1, 0.3), rng.uniform(0.05, 0.15), rng.uniform(-200, 200), rng.uniform(-300, 300),
                 rng.uniform(0, 180), rng.uniform(60, 140))
    blur = rng.uniform(0.3, 1.8)
    motion = (rng.randrange(5, 16), rng.uniform(0, 180)) if rng.random() < 0.20 else None
    noise = rng.uniform(2, 7)
    quality = rng.randrange(70, 91)
    return _Shot(np_seed, background, height, shift, angle, corners, gain, offset, light, glare, blur, motion, noise,
                 quality)


def _gradient(draws: tuple, amp: float) -> np.ndarray:
    # One row and one column, broadcast: the same values as full coordinate grids, without building them.
    xx = np.arange(CANVAS_W, dtype=np.float32)[None, :]
    yy = np.arange(CANVAS_H, dtype=np.float32)[:, None]
    out = np.empty((CANVAS_H, CANVAS_W, 3), np.float32)
    for c, (gx, gy, base) in enumerate(draws):
        out[..., c] = (base + gx * xx / CANVAS_W * amp) + gy * yy / CANVAS_H * amp
    return out


def _render(card: np.ndarray, shot: _Shot) -> bytes:
    np_rng = np.random.default_rng(shot.np_seed)

    # Background: gradient plus blurred noise.
    bg = _gradient(shot.background, 60)
    noise = np_rng.normal(0, 25, (CANVAS_H // 20, CANVAS_W // 20, 3)).astype(np.float32)
    noise = cv2.resize(noise, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_CUBIC)
    canvas = np.clip(cv2.GaussianBlur(bg + noise, (0, 0), 15), 0, 255).astype(np.uint8)

    # Card: 62-85% of the canvas height, shifted up to 80 px, corners moved up to 7%, rotated up to 10 degrees.
    ch = shot.height * CANVAS_H
    cw = ch * card.shape[1] / card.shape[0]
    cx = CANVAS_W / 2 + shot.shift[0]
    cy = CANVAS_H / 2 + shot.shift[1]
    angle = np.radians(shot.angle)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    dst = []
    for (sx, sy), (jx, jy) in zip(((-1, -1), (1, -1), (1, 1), (-1, 1)), shot.corners):
        dst.append(rot @ np.array([sx * cw / 2 + jx * cw, sy * ch / 2 + jy * ch]) + (cx, cy))
    h, w = card.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    m = cv2.getPerspectiveTransform(src, np.float32(dst))
    warped = cv2.warpPerspective(card, m, (CANVAS_W, CANVAS_H), flags=cv2.INTER_AREA)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), m, (CANVAS_W, CANVAS_H), flags=cv2.INTER_LINEAR)
    a = (mask.astype(np.float32) / 255)[..., None]
    img = warped.astype(np.float32) * a + canvas.astype(np.float32) * (1 - a)

    # Light: brightness, a gradient, sometimes an elliptical glare.
    img = img * shot.gain + shot.offset
    img += _gradient(shot.light, 40)
    if shot.glare is not None:
        ax, ay, dx, dy, glare_angle, strength = shot.glare
        glare = np.zeros((CANVAS_H, CANVAS_W), np.float32)
        cv2.ellipse(glare, (int(cx + dx), int(cy + dy)), (int(ax * CANVAS_W), int(ay * CANVAS_H)), glare_angle,
                    0, 360, 1.0, -1)
        glare = cv2.GaussianBlur(glare, (0, 0), 40) * strength
        img += glare[..., None]
    img = np.clip(img, 0, 255).astype(np.uint8)

    # Optics: blur, sometimes motion blur, sensor noise, JPEG.
    img = cv2.GaussianBlur(img, (0, 0), shot.blur)
    if shot.motion is not None:
        k, motion_angle = shot.motion
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1 / k
        rm = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), motion_angle, 1)
        kernel = cv2.warpAffine(kernel, rm, (k, k))
        img = cv2.filter2D(img, -1, kernel / max(kernel.sum(), 1e-6))
    img = np.clip(img.astype(np.float32) + np_rng.normal(0, shot.noise, img.shape), 0, 255).astype(np.uint8)
    ok, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, shot.quality])
    return jpeg.tobytes()


def synth_photo(card: np.ndarray, rng: random.Random) -> bytes:
    """A JPEG of the card as a phone would shoot it; deterministic given `rng`."""
    return _render(card, _draw(rng))


def _urls(product: CatalogProduct) -> list[str]:
    """The product's image URLs, the one the bench prefers first."""
    return [u for u in (product.image_large_url, product.image_url) if u]


def sample(products: list[CatalogProduct], size: int, seed: int) -> list[CatalogProduct]:
    """Half at random, a quarter from names with 6+ printings, a quarter from names with 2-5; only products with an
    image URL."""
    pool = sorted((p for p in products if _urls(p)), key=lambda p: p.id_product)
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


def collect(products: list[CatalogProduct], size: int, seed: int,
            fetch: Callable[[list[str]], dict[str, bytes | None]]) -> tuple[list[tuple[CatalogProduct, bytes]], int]:
    """The sampled products with their image bytes, and how many sampled products had to be skipped.

    Only the sampled images are fetched (the large one, else the normal one) with `fetch`, which answers None for
    an image it can't download or decode; such a product is replaced by another drawn at random from the products not tried yet.
    """
    pool = sorted((p for p in products if _urls(p)), key=lambda p: p.id_product)
    rng = random.Random(seed + 1)
    todo = sample(pool, size, seed)
    tried = {p.id_product for p in todo}
    got: list[tuple[CatalogProduct, bytes]] = []
    skipped = 0
    while todo:
        first = fetch([_urls(p)[0] for p in todo])
        second = fetch([_urls(p)[1] for p in todo if len(_urls(p)) > 1 and first.get(_urls(p)[0]) is None])
        failed = 0
        for p in todo:
            data = next((d for u in _urls(p) if (d := first.get(u) or second.get(u))), None)
            if data is not None:  # `fetch` gives only bytes that decode
                got.append((p, data))
            else:
                failed += 1
        skipped += failed
        free = [p for p in pool if p.id_product not in tried]
        todo = rng.sample(free, min(failed, len(free)))
        tried.update(p.id_product for p in todo)
    if skipped:
        logger.info("bench: skipped %d sampled products whose image could not be fetched or decoded", skipped)
    return got, skipped


def _percent(part: int, total: int) -> float:
    return 100 * part / total if total else 0.0


@dataclass
class Report:
    total: int = 0
    skipped: int = 0
    counts: dict[str, dict[str, int]] = field(default_factory=lambda: {m: dict.fromkeys(OUTCOMES, 0) for m in MODES})
    times_ms: dict[str, list[float]] = field(default_factory=lambda: {m: [] for m in MODES})

    def table(self) -> str:
        lines = [f"photos: {self.total}  skipped images: {self.skipped}",
                 f"{'outcome':<15}" + "".join(f"{m:>18}" for m in MODES)]
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
        lines.append("unindexed: the card's own entry removed from the index, so any ok is a wrong_certain")
        return "\n".join(lines)


def _outcome(result, product: CatalogProduct) -> str:
    if result.outcome == "ok":
        return "ok" if result.product_id == product.id_product else "wrong_certain"
    if result.outcome == "no_card":
        return "no_card"
    return "in_candidates" if product.id_product in result.candidates else "not_found"


def _without(index: Index, product_id: int) -> Index:
    for row in reversed(np.flatnonzero(index.ids == product_id)):
        index = drop_row(index, int(row))
    return index


def _photos(samples: list[tuple[CatalogProduct, bytes]], seed: int) -> list[tuple[CatalogProduct, bytes] | None]:
    """The synthetic photo of each sample, None where its image doesn't decode. The random draws are made in
    sample order, so the photos are the same as one by one; the pixels are rendered on every core."""
    with ThreadPoolExecutor(os.cpu_count()) as pool:
        decodes = list(pool.map(lambda s: decode(s[1]) is not None, samples))
        rng = random.Random(seed)
        shots = [_draw(rng) if ok else None for ok in decodes]

        def render(item):
            (product, data), shot = item
            return None if shot is None else (product, _render(decode(data), shot))

        return list(pool.map(render, zip(samples, shots)))


def run(index: Index, samples: list[tuple[CatalogProduct, bytes]], seed: int, skipped: int = 0) -> Report:
    """Recognizes a synthetic photo of each sample without lock, with the lock on its set, and with its own entry
    removed from the index (no lock). The photos are all made first, so nothing else runs while `recognize` is
    timed."""
    report = Report(skipped=skipped)
    for item in _photos(samples, seed):
        if item is None:
            report.skipped += 1
            continue
        product, photo = item
        report.total += 1
        for mode in MODES:
            lock = product.expansion_code or None if mode == "lock" else None
            target = _without(index, product.id_product) if mode == "unindexed" else index
            start = time.perf_counter()
            result = recognize(photo, target, lock)
            report.times_ms[mode].append((time.perf_counter() - start) * 1000)
            report.counts[mode][_outcome(result, product)] += 1
    return report
