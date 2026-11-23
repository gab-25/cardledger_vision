import cv2
import numpy as np

from cardledger_vision.phash import CARD_SIZE

_WORK_HEIGHT = 1000
_MIN_AREA_RATIO = 0.15
_CARD_RATIO = 63 / 88  # short side / long side
_RATIO_TOLERANCE = 0.25  # a quad whose side ratio is further than 25% from a card's is not a card


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order four points clockwise (image coordinates) around their centre, so
    the result is never mirrored whatever the rotation; the starting corner is
    arbitrary."""
    c = pts.mean(axis=0)
    return pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]


def detect(photo: np.ndarray) -> np.ndarray | None:
    """Find the largest card-like shape in a BGR photo and return it dewarped
    to CARD_SIZE, portrait. A 180 degree flip is not corrected here."""
    if photo is None or photo.ndim != 3 or photo.shape[0] < 2 or photo.shape[1] < 2:
        return None
    scale = _WORK_HEIGHT / photo.shape[0]
    small = cv2.resize(photo, (max(1, round(photo.shape[1] * scale)), _WORK_HEIGHT))
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(gray, 30, 100), np.ones((5, 5), np.uint8))
    # Close gaps in weak edges, fill the outlines, then undo the dilation so
    # the shape's border sits on the card's edge.
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    outlines, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not outlines:
        return None
    filled = np.zeros_like(edges)
    cv2.drawContours(filled, outlines, -1, 255, cv2.FILLED)
    filled = cv2.erode(filled, np.ones((13, 13), np.uint8))
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < _MIN_AREA_RATIO * small.shape[0] * small.shape[1]:
        return None

    approx = cv2.approxPolyDP(best, 0.02 * cv2.arcLength(best, True), True)
    if len(approx) == 4:
        quad = approx.reshape(4, 2).astype(np.float32)
    else:
        quad = cv2.boxPoints(cv2.minAreaRect(best)).astype(np.float32)
    quad = _order_corners(quad)
    if not cv2.isContourConvex(quad.reshape(-1, 1, 2)):
        return None
    sides = np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1)
    width, height = (sides[0] + sides[2]) / 2, (sides[1] + sides[3]) / 2  # averaged opposite sides
    ratio = min(width, height) / max(width, height)
    if abs(ratio - _CARD_RATIO) > _RATIO_TOLERANCE * _CARD_RATIO:
        return None
    if width > height:
        quad = np.roll(quad, -1, axis=0)

    w, h = CARD_SIZE
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    m = cv2.getPerspectiveTransform(quad / scale, dst)
    return cv2.warpPerspective(photo, m, (w, h))
