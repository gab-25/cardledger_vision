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
    return name.strip().split(" // ", 1)[0].strip().casefold()


def _norm_number(number: str) -> str:
    return number.strip().lstrip("0").upper() or ("0" if number.strip() else "")


def _set_key(set_code: str) -> str:
    return set_code.strip().casefold()


def _codes(values: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    """An integer id per distinct value, and the value -> id map."""
    ids: dict[str, int] = {}
    return np.fromiter((ids.setdefault(v, len(ids)) for v in values), np.int32, len(values)), ids


@dataclass(frozen=True)
class _Keys:
    names: np.ndarray  # name key id per entry
    numbers: np.ndarray  # normalized number id per entry
    sets: np.ndarray  # set key id per entry
    set_ids: dict[str, int]
    hashed: int  # entries with a hash


def _keys(index: Index) -> _Keys:
    cached = getattr(index, "_match_keys", None)
    if cached is None:
        names, _ = _codes([_name_key(n) for n in index.names])
        numbers, _ = _codes([_norm_number(n) for n in index.numbers])
        sets, set_ids = _codes([_set_key(s) for s in index.sets])
        cached = _Keys(names, numbers, sets, set_ids, int(index.has_hash.sum()))
        index._match_keys = cached
    return cached


def _distances(h: bytes, index: Index) -> np.ndarray:
    """Hamming distance to every entry, infinite for entries without a hash."""
    if not len(index):
        return np.empty(0)
    return np.where(index.has_hash, hamming_many(index.hashes, h), np.inf)


def _sorted(rows: np.ndarray, d: np.ndarray) -> list[int]:
    """`rows` by distance, ties by position (as a stable sort would)."""
    return [int(i) for i in rows[np.lexsort((rows, d[rows]))]]


def _nearest(d: np.ndarray, k: int) -> list[int]:
    """Up to `k` finite rows of `d`, nearest first."""
    if not len(d) or k <= 0:
        return []
    k = min(k, len(d))
    kth = np.partition(d, k - 1)[k - 1]
    rows = np.flatnonzero((d <= kth) & np.isfinite(d))
    return _sorted(rows, d)[:k]


def _ids(index: Index, rows) -> list[int]:
    out: list[int] = []
    for i in rows:
        pid = int(index.ids[i])
        if pid not in out:
            out.append(pid)
        if len(out) == MAX_CANDIDATES:
            break
    return out


def _next_names(d: np.ndarray, names: np.ndarray, count: int) -> list[int]:
    """The nearest row of each of the `count` nearest names in `d` (rows of other names already set to inf)."""
    k = 64
    while True:
        rows = _nearest(d, k)
        out: list[int] = []
        seen: set[int] = set()
        for i in rows:
            if int(names[i]) not in seen:
                seen.add(int(names[i]))
                out.append(i)
                if len(out) == count:
                    return out
        if len(rows) < k:  # every finite row seen
            return out
        k *= 4


def recognize_hash(h: bytes, index: Index, lock: str | None = None) -> Result:
    """Recognize a card from its hash (see `hash_card`)."""
    return _decide(_distances(h, index), index, lock)


def _decide(d: np.ndarray, index: Index, lock: str | None) -> Result:
    """The decision rules on the distances `d` to every entry of `index` (inf where there is no hash)."""
    t = index.header.thresholds
    keys = _keys(index)

    # Rule 1: nothing close enough.
    if not keys.hashed:
        return Result("not_found", None, [])
    best = int(np.argmin(d))
    if d[best] > t.identity_max:
        return Result("not_found", None, _ids(index, _nearest(d, MAX_CANDIDATES)))

    # Rule 2: is the name certain?
    name = keys.names[best]
    same = keys.names == name
    d_others = np.where(same, np.inf, d)
    identity_sure = d_others.min() - d[best] >= t.name_gap

    # Rule 3: printings of that name, narrowed by the lock if it matches any (case-insensitive).
    same_name = np.flatnonzero(same)
    printings = _sorted(same_name, d)
    outside: list[int] = []
    locked = False
    lock_id = keys.set_ids.get(_set_key(lock)) if lock is not None else None
    if lock_id is not None:
        in_lock = [i for i in printings if keys.sets[i] == lock_id]
        if in_lock:
            outside = [i for i in printings if keys.sets[i] != lock_id]
            printings, locked = in_lock, True

    # Rule 4: one printing clearly nearer than every other, all of which have an image: a printing without an
    # image could be the card, so it blocks certainty (without a lock: any printing of the name, in any set).
    pick = None
    first, rest = printings[0], printings[1:]
    if (np.isfinite(d[first]) and all(index.has_hash[i] for i in rest)
            and (not rest or d[rest[0]] - d[first] >= t.printing_gap)):
        pick = first

    # Lock guard: a printing outside the lock clearly nearer than the pick means the lock is likely wrong.
    if pick is not None and locked and outside and d[pick] - d[outside[0]] >= t.printing_gap:
        pick = None
    printings = printings + outside  # candidates: in-lock printings first, then the others of the name

    # Rule 5: a twin (same number in another set, or the same image) blocks certainty.
    if pick is not None and not locked:
        for i in printings:
            if i == pick:
                continue
            same_number = keys.numbers[i] == keys.numbers[pick] and keys.sets[i] != keys.sets[pick]
            same_image = bool(index.has_hash[i]) and np.array_equal(index.hashes[i], index.hashes[pick])
            if same_number or same_image:
                pick = None
                break

    # Rule 6: name not certain.
    if not identity_sure:
        return Result("ambiguous", None, _ids(index, printings[:3] + _next_names(d_others, keys.names, 2)))

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
    d = _distances(hash_card(card), index)
    if index.has_hash.any():
        flipped = _distances(hash_card(cv2.rotate(card, cv2.ROTATE_180)), index)
        if flipped.min() < d.min():
            d = flipped
    return _decide(d, index, lock)


def drop_row(index: Index, row: int) -> Index:
    """`index` without its entry at `row` (the bench's card missing from the index), sharing the cached keys."""
    keys = _keys(index)
    out = Index()
    out.header = index.header
    out.ids = np.delete(index.ids, row)
    out.hashes = np.delete(index.hashes, row, axis=0)
    out.has_hash = np.delete(index.has_hash, row)
    out.names = index.names[:row] + index.names[row + 1:]
    out.sets = index.sets[:row] + index.sets[row + 1:]
    out.numbers = index.numbers[:row] + index.numbers[row + 1:]
    out._match_keys = _Keys(np.delete(keys.names, row), np.delete(keys.numbers, row), np.delete(keys.sets, row),
                            keys.set_ids, keys.hashed - int(index.has_hash[row]))
    return out
