import random

import cv2

from cardledger_vision import cli
from cardledger_vision.bench import run, sample, synth_photo
from cardledger_vision.catalog import CatalogProduct
from cardledger_vision.index import Entry, Header, Index, encode
from cardledger_vision.phash import CARD_SIZE, hash_card


def test_synth_photo_is_deterministic(card_image):
    a = synth_photo(card_image(1), random.Random(5))
    assert a == synth_photo(card_image(1), random.Random(5))
    assert a != synth_photo(card_image(1), random.Random(6))
    assert cv2.imdecode(__import__("numpy").frombuffer(a, "uint8"), cv2.IMREAD_COLOR).shape == (2000, 1500, 3)


def make(card_image, n=3):
    products, images, entries = [], {}, []
    for i in range(1, n + 1):
        url = f"https://img.test/large/{i}.png"
        products.append(CatalogProduct(i, f"Card {i}", "set", str(i), None, url))
        ok, png = cv2.imencode(".png", card_image(i))
        images[url] = png.tobytes()
        h = hash_card(cv2.resize(card_image(i), CARD_SIZE, interpolation=cv2.INTER_AREA))
        entries.append(Entry(i, f"Card {i}", "set", str(i), h))
    index = Index.load(encode(Header("mtg", "2026-01-01T00:00:00Z", "x", False), entries))
    return index, products, images


def test_run_counts_outcomes(card_image):
    index, products, images = make(card_image)
    report = run(index, products, images, size=3, seed=1)
    assert report.total == 3 and report.counts["no_lock"]["wrong_certain"] == 0
    assert report.counts["lock"]["wrong_certain"] == 0
    for mode in ("no_lock", "lock"):
        assert sum(report.counts[mode].values()) == 3
    table = report.table()
    assert "wrong_certain" in table and "%" in table and "p90" in table


def test_sample_is_capped_unique_and_skips_missing_images():
    products = [CatalogProduct(i, f"Name {i % 10}", "s", str(i), None, f"u{i}") for i in range(100)]
    images = {f"u{i}": b"x" for i in range(100) if i % 2 == 0}
    picked = sample(products, images, 1000, 3)
    assert len(picked) == 50 and len({p.id_product for p in picked}) == 50
    assert [p.id_product for p in sample(products, images, 20, 3)] == [p.id_product for p in sample(products, images, 20, 3)]
    assert len(sample(products, images, 20, 3)) == 20


def test_bench_command(card_image, tmp_path, monkeypatch, capsys):
    import json
    index, products, images = make(card_image)
    (tmp_path / "mtg.bin").write_bytes(encode(index.header, [
        Entry(int(index.ids[i]), index.names[i], index.sets[i], index.numbers[i], index.hashes[i].tobytes())
        for i in range(len(index))]))
    cat = tmp_path / "c.json"
    cat.write_text(json.dumps({"expansions": [], "products": [
        {"id_product": p.id_product, "name": p.name, "expansion_code": "set", "number": p.number,
         "image_url": None, "image_large_url": p.image_large_url, "category_name": "Magic Singles"}
        for p in products]}))
    import httpx

    def handler(request):
        return httpx.Response(200, content=images[str(request.url)])

    monkeypatch.setattr(cli, "make_client", lambda: httpx.Client(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr("cardledger_vision.images.sleep", lambda s: None)
    code = cli.main(["bench", "--game", "mtg", "--index", str(tmp_path / "mtg.bin"), "--catalog", str(cat),
                     "--cache", str(tmp_path / "cache"), "--size", "3"])
    assert code == 0
    assert "wrong_certain" in capsys.readouterr().out
