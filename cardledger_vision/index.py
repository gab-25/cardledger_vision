"""Versioned binary index format (little-endian, readable from JavaScript).

Byte layout (offsets from the start of the file):

    0               4 bytes   magic "CLVI"
    4               u16       format version (1)
    6               u32       header_len
    10              header_len bytes: header JSON, utf-8, padded with trailing spaces (0x20) so that
                    10 + header_len + 4 is a multiple of 8
    10+header_len   u32       n, the number of entries
    A = 14+header_len  (multiple of 8)
                    n x u32   product_id
    A + 4n          n x u8    has_hash (1 = the entry has a hash, 0 = no image)
    A + 5n          (-5n mod 8) zero bytes, so that the hashes start at a multiple of 8
    B = A + 5n + pad  (multiple of 8)
                    n x 32 bytes  hash (see phash.hash_card; all zeros when has_hash is 0)
    B + 32n         u32       strings_len
    B + 32n + 4     strings_len bytes: JSON {"names": [...], "sets": [...], "numbers": [...]}, n strings each
    end of file (no trailing bytes)

The alignment lets a JavaScript reader view the ids as a Uint32Array and the hashes as a Uint8Array or
BigUint64Array directly on the file's ArrayBuffer. The header JSON holds "game", "built_at", "catalog_sha256",
"confirm_only" and "thresholds" ({"identity_max", "name_gap", "printing_gap"}); readers ignore unknown keys and
use the defaults for missing optional ones.
"""

import json
import struct
from dataclasses import asdict, dataclass, fields

import numpy as np

MAGIC = b"CLVI"
FORMAT_VERSION = 1
HASH_BYTES = 32


class InvalidIndex(ValueError):
    """The data is not a valid index."""


@dataclass(frozen=True)
class Entry:
    product_id: int
    name: str
    set_code: str
    number: str
    hash: bytes | None


@dataclass(frozen=True)
class Thresholds:
    identity_max: int = 100
    name_gap: int = 6
    printing_gap: int = 8


@dataclass(frozen=True)
class Header:
    game: str
    built_at: str
    catalog_sha256: str
    confirm_only: bool = False
    thresholds: Thresholds = Thresholds()


def _json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


_PREFIX = 10  # magic, version, header_len


def _pad(n: int) -> int:
    """Zero bytes after has_hash, so that the hashes start 8-aligned (the ids block is)."""
    return -(5 * n) % 8


def encode(header: Header, entries: list[Entry]) -> bytes:
    n = len(entries)
    head = _json(asdict(header))
    head += b" " * (-(_PREFIX + len(head) + 4) % 8)  # the ids block starts 8-aligned
    for e in entries:
        if e.hash is not None and len(e.hash) != HASH_BYTES:
            raise ValueError(f"product {e.product_id}: hash must be {HASH_BYTES} bytes, not {len(e.hash)}")
    has_hash = bytes(1 if e.hash is not None else 0 for e in entries)
    hashes = b"".join(e.hash if e.hash is not None else bytes(HASH_BYTES) for e in entries)
    strings = _json(
        {
            "names": [e.name for e in entries],
            "sets": [e.set_code for e in entries],
            "numbers": [e.number for e in entries],
        }
    )
    return b"".join(
        [
            MAGIC,
            struct.pack("<HI", FORMAT_VERSION, len(head)),
            head,
            struct.pack("<I", n),
            struct.pack(f"<{n}I", *(e.product_id for e in entries)),
            has_hash,
            bytes(_pad(n)),
            hashes,
            struct.pack("<I", len(strings)),
            strings,
        ]
    )


class _Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def take(self, size: int) -> bytes:
        if size < 0 or self.pos + size > len(self.data):
            raise InvalidIndex("truncated index")
        chunk = self.data[self.pos : self.pos + size]
        self.pos += size
        return chunk

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]


def _loads(raw: bytes, what: str):
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise InvalidIndex(f"invalid {what} JSON") from exc


def _known(cls, raw) -> dict:
    """The keys of `raw` that are fields of the dataclass `cls` (unknown keys come from newer builders)."""
    if not isinstance(raw, dict):
        raise InvalidIndex(f"invalid {cls.__name__.lower()}")
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in raw.items() if k in names}


def _header(raw) -> Header:
    known = _known(Header, raw)
    try:
        if "thresholds" in known:
            known["thresholds"] = Thresholds(**_known(Thresholds, known["thresholds"]))
        return Header(**known)
    except TypeError as exc:  # a required key is missing
        raise InvalidIndex("invalid header") from exc


class Index:
    header: Header
    ids: np.ndarray
    hashes: np.ndarray
    has_hash: np.ndarray
    names: list[str]
    sets: list[str]
    numbers: list[str]

    def __len__(self) -> int:
        return len(self.ids)

    @classmethod
    def load(cls, data: bytes) -> "Index":
        r = _Reader(bytes(data))
        if r.take(4) != MAGIC:
            raise InvalidIndex("bad magic")
        (version,) = struct.unpack("<H", r.take(2))
        if version != FORMAT_VERSION:
            raise InvalidIndex(f"unsupported format version {version}")
        header = _header(_loads(r.take(r.u32()), "header"))

        n = r.u32()
        ids = np.frombuffer(r.take(n * 4), dtype="<u4").astype(np.uint32)
        has_hash = np.frombuffer(r.take(n), dtype=np.uint8).astype(bool)
        r.take(_pad(n))
        hashes = np.frombuffer(r.take(n * HASH_BYTES), dtype=np.uint8).reshape(n, HASH_BYTES)
        strings = _loads(r.take(r.u32()), "strings")
        if r.pos != len(r.data):
            raise InvalidIndex("trailing data")
        try:
            names, sets, numbers = strings["names"], strings["sets"], strings["numbers"]
        except (TypeError, KeyError) as exc:
            raise InvalidIndex("invalid strings block") from exc
        for arr in (names, sets, numbers):
            if not isinstance(arr, list) or len(arr) != n or not all(isinstance(s, str) for s in arr):
                raise InvalidIndex("string arrays do not match the entry count")

        index = cls()
        index.header, index.ids, index.has_hash, index.hashes = header, ids, has_hash, hashes
        index.names, index.sets, index.numbers = names, sets, numbers
        return index
