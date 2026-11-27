import hashlib
import json

import cv2

import cardledger_vision
from cardledger_vision import build_index, catalog
from cardledger_vision.index import Index
from cardledger_vision.phash import CARD_SIZE, hash_card


def row(id, image=None, image_large=None, kind="card", name=None, set_code="s", number="1"):
    return {"id": id, "kind": kind, "name": name or f"Card {id}", "set_code": set_code, "number": number,
            "rarity": "common", "image": image, "image_large": image_large}


def catalog_bytes(rows):
    return json.dumps({"sets": [], "rows": rows}).encode()


def build(slug, rows, tmp_path, **kw):
    raw = catalog_bytes(rows)
    sha = hashlib.sha256(raw).hexdigest()
    return build_index(slug, raw, sha, tmp_path, tmp_path / "cache", **kw), sha


def expected_hash(card):
    return hash_card(cv2.resize(card, CARD_SIZE, interpolation=cv2.INTER_AREA))


def test_loads_only_card_rows_of_the_catalog():
    data = json.dumps({"sets": [], "rows": [
        {"id": "mtg:lea:161", "kind": "card", "name": "Bolt", "set_code": "lea", "number": "161",
         "image": "images/v1/mtg/a-s.webp", "image_large": "images/v1/mtg/a-l.webp"},
        {"id": "mtg:sealed:x", "kind": "sealed", "name": "Box", "set_code": "lea", "number": "",
         "image": "images/v1/mtg/b-s.webp", "image_large": "images/v1/mtg/b-l.webp"},
    ]}).encode()
    assert [p.id for p in catalog.load(data)] == ["mtg:lea:161"]
    (p,) = catalog.load(data)
    assert (p.name, p.set_code, p.number, p.image, p.image_large) == (
        "Bolt", "lea", "161", "images/v1/mtg/a-s.webp", "images/v1/mtg/a-l.webp")


def test_build_reads_local_images(tmp_path, write_card_image):
    write_card_image(tmp_path / "images/v1/x/a-l.webp")
    data = catalog_bytes([row("x:s:1", image_large="images/v1/x/a-l.webp"),
                          row("x:s:2", image_large="images/v1/x/gone-l.webp")])
    index = Index.load(build_index("x", data, "sha", tmp_path, tmp_path / "cache"))
    assert index.ids == ["x:s:1", "x:s:2"] and list(index.has_hash) == [True, False]


def test_entries_are_sorted_by_id_and_carry_the_row_fields(tmp_path):
    rows = [row("x:b:2", set_code="b", number="2", name="Two"), row("x:a:10", set_code="a", number="10", name="Ten")]
    index = Index.load(build("x", rows, tmp_path)[0])
    assert index.ids == ["x:a:10", "x:b:2"]
    assert index.names == ["Ten", "Two"] and index.sets == ["a", "b"] and index.numbers == ["10", "2"]


def test_only_card_rows_enter_the_index(tmp_path):
    data, _ = build("mtg", [row("m:s:1"), row("m:s:2"), row("m:sealed:3", kind="sealed")], tmp_path)
    assert Index.load(data).ids == ["m:s:1", "m:s:2"]


def test_hash_is_the_large_image_s(tmp_path, write_card_image, card_image):
    write_card_image(tmp_path / "l.png", seed=1)
    write_card_image(tmp_path / "s.png", seed=2)
    data, _ = build("mtg", [row("m:s:1", image="s.png", image_large="l.png")], tmp_path)
    assert Index.load(data).hashes[0].tobytes() == expected_hash(card_image(1))


def test_normal_image_is_the_fallback(tmp_path, write_card_image, card_image):
    write_card_image(tmp_path / "s.png", seed=2)
    (tmp_path / "broken-l.png").write_bytes(b"\x89PNG\r\n\x1a\nbroken")
    rows = [row("m:s:1", image="s.png", image_large="gone-l.png"),
            row("m:s:2", image="s.png", image_large="broken-l.png"),
            row("m:s:3", image="s.png")]
    index = Index.load(build("mtg", rows, tmp_path)[0])
    assert [h.tobytes() for h in index.hashes] == [expected_hash(card_image(2))] * 3


def test_row_without_a_usable_image_enters_without_hash(tmp_path, write_card_image):
    write_card_image(tmp_path / "ok.png")
    (tmp_path / "bad.png").write_bytes(b"<html>not an image</html>")
    rows = [row("m:s:1", image_large="ok.png"), row("m:s:2", image="bad.png", image_large="gone.png"), row("m:s:3")]
    index = Index.load(build("mtg", rows, tmp_path)[0])
    assert len(index) == 3 and list(index.has_hash) == [True, False, False]


def test_header_carries_game_sha_and_flag(tmp_path):
    data, sha = build("pokemon", [row("p:s:1")], tmp_path, confirm_only=True)
    header = Index.load(data).header
    assert header.game == "pokemon"
    assert header.catalog_sha256 == sha
    assert header.confirm_only is True


def test_index_key_is_the_published_path():
    assert cardledger_vision.index_key("mtg") == "vision/v1/mtg.bin"
