"""Versioned binary index format (little-endian, readable from JavaScript).

Layout:
    "CLVI" | u16 format=1 | u32 header_len | header JSON (utf-8)
    u32 n | n x u32 product_id | n x u8 has_hash (padded to a multiple of 4)
    | n x 32 byte hash (zeros if absent)
    u32 strings_len | JSON {"names": [...], "sets": [...], "numbers": [...]}
"""

import json
import struct
from dataclasses import asdict, dataclass

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


def _pad(n: int) -> int:
    return -n % 4


def encode(header: Header, entries: list[Entry]) -> bytes:
    n = len(entries)
    head = _json(asdict(header))
    has_hash = bytes(1 if e.hash is not None else 0 for e in entries)
    hashes = b"".join(e.hash if e.hash is not None else bytes(HASH_BYTES) for e in entries)
    if len(hashes) != n * HASH_BYTES:
        raise ValueError(f"every hash must be {HASH_BYTES} bytes")
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
        raw_header = _loads(r.take(r.u32()), "header")
        try:
            raw_header["thresholds"] = Thresholds(**raw_header["thresholds"])
            header = Header(**raw_header)
        except (TypeError, KeyError) as exc:
            raise InvalidIndex("invalid header") from exc

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
