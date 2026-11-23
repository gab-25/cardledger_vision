"""Recognition: nearest search in an index and the decision rules."""

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from cardledger_vision.detect import detect
from cardledger_vision.index import Index
from cardledger_vision.phash import hamming_many, hash_card

MAX_CANDIDATES = 5


@dataclass(frozen=True)
class Result:
    outcome: Literal["ok", "ambiguous", "not_found", "no_card"]
    product_id: int | None
    candidates: list[int]


def _name_key(name: str) -> str:
    return name.split(" // ", 1)[0].casefold()


def _norm_number(number: str) -> str:
    return number.strip().lstrip("0").upper() or ("0" if number.strip() else "")


@dataclass(frozen=True)
class _Keys:
    names: list[str]
    numbers: list[str]


def _keys(index: Index) -> _Keys:
    cached = getattr(index, "_match_keys", None)
    if cached is None:
        cached = _Keys([_name_key(n) for n in index.names], [_norm_number(n) for n in index.numbers])
        index._match_keys = cached
    return cached


def _distances(h: bytes, index: Index) -> np.ndarray:
    """Hamming distance to every entry, infinite for entries without a hash."""
    d = np.full(len(index), np.inf)
    if len(index):
        d[index.has_hash] = hamming_many(index.hashes[index.has_hash], h)
    return d


def _ids(index: Index, rows) -> list[int]:
    out: list[int] = []
    for i in rows:
        pid = int(index.ids[i])
        if pid not in out:
            out.append(pid)
        if len(out) == MAX_CANDIDATES:
            break
    return out


def recognize_hash(h: bytes, index: Index, lock: str | None = None) -> Result:
    t = index.header.thresholds
    keys = _keys(index)
    d = _distances(h, index)
    order = [int(i) for i in np.argsort(d, kind="stable")]
    hashed = [i for i in order if np.isfinite(d[i])]

    # Rule 1: nothing close enough.
    if not hashed or d[hashed[0]] > t.identity_max:
        return Result("not_found", None, _ids(index, hashed))

    # Rule 2: is the name certain?
    best = hashed[0]
    name = keys.names[best]
    others = [i for i in hashed if keys.names[i] != name]
    other = d[others[0]] if others else np.inf
    identity_sure = other - d[best] >= t.name_gap

    # Rule 3: printings of that name, narrowed by the lock if it matches any.
    same_name = [i for i in order if keys.names[i] == name]
    printings = same_name
    locked = False
    if lock is not None:
        in_lock = [i for i in same_name if index.sets[i] == lock]
        if in_lock:
            printings, locked = in_lock, True

    # Rule 4: one printing clearly nearer than the next.
    pick = None
    first = printings[0]
    if np.isfinite(d[first]) and (len(printings) == 1 or d[printings[1]] - d[first] >= t.printing_gap):
        pick = first

    # Rule 5: a twin (same number in another set, or the same image) blocks certainty.
    if pick is not None and not locked:
        for i in same_name:
            if i == pick:
                continue
            same_number = keys.numbers[i] == keys.numbers[pick] and index.sets[i] != index.sets[pick]
            same_image = bool(index.has_hash[i]) and np.array_equal(index.hashes[i], index.hashes[pick])
            if same_number or same_image:
                pick = None
                break

    # Rule 6: name not certain.
    if not identity_sure:
        next_names: list[int] = []
        seen: set[str] = set()
        for i in others:
            if keys.names[i] not in seen:
                seen.add(keys.names[i])
                next_names.append(i)
            if len(next_names) == 2:
                break
        return Result("ambiguous", None, _ids(index, printings[:3] + next_names))

    if pick is not None:
        candidates = _ids(index, [pick] + [i for i in printings if i != pick])
        # Rule 7: a confirm-only index never answers "ok".
        if index.header.confirm_only:
            return Result("ambiguous", None, candidates)
        # Rule 8.
        return Result("ok", int(index.ids[pick]), candidates)
    return Result("ambiguous", None, _ids(index, printings))


def recognize(photo: bytes, index: Index, lock: str | None = None) -> Result:
    """Recognize the card in a JPEG/PNG photo."""
    no_card = Result("no_card", None, [])
    try:
        image = cv2.imdecode(np.frombuffer(photo, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return no_card
    if image is None:
        return no_card
    card = detect(image)
    if card is None:
        return no_card
    h = hash_card(card)
    if index.has_hash.any():
        flipped = hash_card(cv2.rotate(card, cv2.ROTATE_180))
        if _distances(flipped, index).min() < _distances(h, index).min():
            h = flipped
    return recognize_hash(h, index, lock)
