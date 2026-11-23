import json
import struct

import pytest

from cardledger_vision.index import Entry, Header, Index, InvalidIndex, Thresholds, encode


def test_roundtrip():
    entries = [Entry(10, "Bolt", "lea", "161", bytes(range(32))), Entry(11, "Bolt", "plst", "LEA-161", None)]
    header = Header("mtg", "2026-11-23T10:00:00Z", "ab" * 32, confirm_only=True, thresholds=Thresholds(90, 5, 7))
    idx = Index.load(encode(header, entries))
    assert idx.header == header and len(idx) == 2
    assert list(idx.ids) == [10, 11] and list(idx.has_hash) == [True, False]
    assert bytes(idx.hashes[0]) == bytes(range(32)) and idx.numbers == ["161", "LEA-161"]
    assert bytes(idx.hashes[1]) == bytes(32)
    assert idx.names == ["Bolt", "Bolt"] and idx.sets == ["lea", "plst"]


def test_empty_index():
    idx = Index.load(encode(Header("mtg", "t", "x"), []))
    assert len(idx) == 0 and idx.hashes.shape == (0, 32)


def test_unicode_and_padding():
    entries = [Entry(i, f"Limûl{i}", "s", str(i), bytes([i]) * 32) for i in range(1, 6)]
    idx = Index.load(encode(Header("mtg", "t", "x"), entries))
    assert idx.names[0] == "Limûl1" and bytes(idx.hashes[4]) == bytes([5]) * 32


def test_encode_is_deterministic():
    e = [Entry(1, "A", "s", "1", bytes(32))]
    h = Header("mtg", "t", "x")
    assert encode(h, e) == encode(h, e)


@pytest.mark.parametrize("data", [b"", b"XXXX" + bytes(20), b"CLVI\x02\x00" + bytes(20)])
def test_rejects_bad_data(data):
    with pytest.raises(InvalidIndex):
        Index.load(data)


def test_rejects_truncated_and_trailing():
    data = encode(Header("mtg", "t", "x"), [Entry(1, "A", "s", "1", bytes(32))])
    for cut in range(len(data)):
        with pytest.raises(InvalidIndex):
            Index.load(data[:cut])
    with pytest.raises(InvalidIndex):
        Index.load(data + b"\x00")


def test_rejects_string_length_mismatch():
    data = bytearray(encode(Header("mtg", "t", "x"), [Entry(1, "A", "s", "1", bytes(32))]))
    good = json.dumps({"names": ["A"], "sets": ["s"], "numbers": ["1"]}, sort_keys=True, separators=(",", ":")).encode()
    bad = json.dumps({"names": ["A", "B"], "sets": ["s"], "numbers": ["1"]}, sort_keys=True, separators=(",", ":")).encode()
    start = bytes(data).index(good) - 4
    data[start:] = struct.pack("<I", len(bad)) + bad
    with pytest.raises(InvalidIndex):
        Index.load(bytes(data))
