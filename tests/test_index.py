import json
import struct

import pytest

from cardledger_vision.index import Entry, Header, Index, InvalidIndex, Thresholds, encode


def test_round_trip_keeps_string_ids():
    entries = [Entry("pokemon:sv01:1", "A", "sv01", "1", bytes(32)), Entry("pokemon:sv01:2", "B", "sv01", "2", None)]
    index = Index.load(encode(Header("pokemon", "t", "s"), entries))
    assert index.ids == ["pokemon:sv01:1", "pokemon:sv01:2"]
    assert list(index.has_hash) == [True, False]


def test_roundtrip():
    entries = [Entry("mtg:lea:161", "Bolt", "lea", "161", bytes(range(32))),
               Entry("mtg:plst:LEA-161", "Bolt", "plst", "LEA-161", None)]
    header = Header("mtg", "2026-11-23T10:00:00Z", "ab" * 32, confirm_only=True, thresholds=Thresholds(90, 5, 7))
    idx = Index.load(encode(header, entries))
    assert idx.header == header and len(idx) == 2
    assert idx.ids == ["mtg:lea:161", "mtg:plst:LEA-161"] and list(idx.has_hash) == [True, False]
    assert bytes(idx.hashes[0]) == bytes(range(32)) and idx.numbers == ["161", "LEA-161"]
    assert bytes(idx.hashes[1]) == bytes(32)
    assert idx.names == ["Bolt", "Bolt"] and idx.sets == ["lea", "plst"]


def test_empty_index():
    idx = Index.load(encode(Header("mtg", "t", "x"), []))
    assert len(idx) == 0 and idx.hashes.shape == (0, 32)


def test_unicode_and_padding():
    entries = [Entry(f"m:s:{i}", f"Limûl{i}", "s", str(i), bytes([i]) * 32) for i in range(1, 6)]
    idx = Index.load(encode(Header("mtg", "t", "x"), entries))
    assert idx.names[0] == "Limûl1" and bytes(idx.hashes[4]) == bytes([5]) * 32


def test_encode_is_deterministic():
    e = [Entry("m:s:1", "A", "s", "1", bytes(32))]
    h = Header("mtg", "t", "x")
    assert encode(h, e) == encode(h, e)


@pytest.mark.parametrize("data", [b"", b"XXXX" + bytes(20), b"CLVI\x02\x00" + bytes(20)])
def test_rejects_bad_data(data):
    with pytest.raises(InvalidIndex):
        Index.load(data)


def test_rejects_truncated_and_trailing():
    data = encode(Header("mtg", "t", "x"), [Entry("m:s:1", "A", "s", "1", bytes(32))])
    for cut in range(len(data)):
        with pytest.raises(InvalidIndex):
            Index.load(data[:cut])
    with pytest.raises(InvalidIndex):
        Index.load(data + b"\x00")


GOOD_STRINGS = {"ids": ["m:s:1"], "names": ["A"], "sets": ["s"], "numbers": ["1"]}


def _with_strings(strings: dict) -> bytes:
    data = encode(Header("mtg", "t", "x"), [Entry("m:s:1", "A", "s", "1", bytes(32))])
    good = json.dumps(GOOD_STRINGS, sort_keys=True, separators=(",", ":")).encode()
    start = data.index(good) - 4
    raw = json.dumps(strings).encode()
    return data[:start] + struct.pack("<I", len(raw)) + raw


@pytest.mark.parametrize("change", [
    {"names": ["A", "B"]}, {"ids": ["m:s:1", "m:s:2"]}, {"ids": [1]}, {"ids": None}])
def test_rejects_bad_string_arrays(change):
    assert Index.load(_with_strings(GOOD_STRINGS)).ids == ["m:s:1"]
    with pytest.raises(InvalidIndex):
        Index.load(_with_strings(GOOD_STRINGS | change))
    with pytest.raises(InvalidIndex):
        Index.load(_with_strings({k: v for k, v in GOOD_STRINGS.items() if k != "ids"}))


@pytest.mark.parametrize("n", [0, 1, 3, 1000])
def test_has_hash_and_hashes_blocks_are_8_aligned(n):
    for game in ("mtg", "pokemon", "x" * 7):  # different header lengths
        data = encode(Header(game, "t", "x"), [Entry(f"m:s:{i}", "A", "s", "1", bytes(32)) for i in range(n)])
        (header_len,) = struct.unpack("<I", data[6:10])
        has_hash_at = 10 + header_len + 4
        hashes_at = has_hash_at + n + (-n % 8)
        assert has_hash_at % 8 == 0 and hashes_at % 8 == 0
        assert data[has_hash_at:has_hash_at + n] == b"\x01" * n
        assert data[hashes_at:hashes_at + 32 * n] == bytes(32 * n)
        assert len(Index.load(data)) == n


def _with_header(raw_header: dict) -> bytes:
    data = encode(Header("mtg", "t", "x"), [Entry("m:s:1", "A", "s", "1", bytes(32))])
    (header_len,) = struct.unpack("<I", data[6:10])
    head = json.dumps(raw_header).encode()
    head += b" " * (-(14 + len(head)) % 8)
    return data[:6] + struct.pack("<I", len(head)) + head + data[10 + header_len:]


def test_unknown_header_keys_are_ignored_and_defaults_fill_missing_ones():
    raw = {"game": "mtg", "built_at": "t", "catalog_sha256": "x", "future": 1,
           "thresholds": {"identity_max": 90, "later": True}}
    header = Index.load(_with_header(raw)).header
    assert header == Header("mtg", "t", "x", False, Thresholds(identity_max=90))
    assert Index.load(_with_header({"game": "mtg", "built_at": "t", "catalog_sha256": "x"})).header == Header("mtg", "t", "x")
    with pytest.raises(InvalidIndex):
        Index.load(_with_header({"game": "mtg", "built_at": "t"}))
    with pytest.raises(InvalidIndex):
        Index.load(_with_header({"game": "mtg", "built_at": "t", "catalog_sha256": "x", "thresholds": 3}))


def test_encode_rejects_a_hash_of_the_wrong_length():
    # 31 + 33 bytes add up to two hashes: each one must be checked on its own.
    with pytest.raises(ValueError):
        encode(Header("mtg", "t", "x"), [Entry("m:s:1", "A", "s", "1", bytes(31)), Entry("m:s:2", "B", "s", "2", bytes(33))])


def test_invalid_index_is_exported():
    import cardledger_vision

    assert cardledger_vision.InvalidIndex is InvalidIndex
