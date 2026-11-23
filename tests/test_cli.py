import hashlib
import io
import json

import cv2
import httpx
import pytest

from cardledger_vision import cli, images
from cardledger_vision.index import Index
from cardledger_vision.phash import CARD_SIZE, hash_card


@pytest.fixture(autouse=True)
def sleeps(monkeypatch):
    """No test waits for the politeness delay: the waits are recorded instead."""
    waits: list[float] = []
    monkeypatch.setattr(images, "sleep", waits.append)
    return waits


def product(i, category="Magic Singles", image=True, large=False):
    return {
        "id_product": i, "name": f"Card {i}", "expansion_code": "set", "number": str(i), "rarity": "common",
        "image_url": f"https://img.test/{i}.png" if image else None,
        "image_large_url": f"https://img.test/large/{i}.png" if large else None,
        "category_id": 1, "category_name": category,
    }


def catalog_bytes(products):
    return json.dumps({"expansions": [], "products": products}).encode()


class Server:
    """Serves a PNG of a different card per URL; the URLs in `missing` answer 404."""

    def __init__(self, card_image, missing=()):
        self.card_image, self.missing, self.requests = card_image, set(missing), []

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append(url)
        if url in self.missing:
            return httpx.Response(404)
        seed = int(url.rsplit("/", 1)[1].split(".")[0])
        ok, png = cv2.imencode(".png", self.card_image(seed))
        return httpx.Response(200, content=png.tobytes())

    def client(self):
        return httpx.Client(transport=httpx.MockTransport(self.handler))


def build(slug, products, server, tmp_path, **kw):
    raw = catalog_bytes(products)
    sha = hashlib.sha256(raw).hexdigest()
    with server.client() as client:
        return cli.build_index(slug, raw, sha, client, tmp_path / "cache", **kw), sha


def test_only_singles_enter_the_index(card_image, tmp_path):
    products = [product(1), product(2, "Magic Single"), product(3, "Booster Boxes")]
    data, _ = build("mtg", products, Server(card_image), tmp_path)
    assert len(Index.load(data)) == 2


def test_hashes_match_hash_card(card_image, tmp_path):
    data, _ = build("mtg", [product(1)], Server(card_image), tmp_path)
    index = Index.load(data)
    expected = hash_card(cv2.resize(card_image(1), CARD_SIZE))
    assert index.hashes[0].tobytes() == expected


def test_failed_image_gives_entry_without_hash(card_image, tmp_path):
    server = Server(card_image, missing={"https://img.test/2.png"})
    data, _ = build("mtg", [product(1), product(2), product(3, image=False)], server, tmp_path)
    index = Index.load(data)
    assert len(index) == 3
    assert list(index.has_hash) == [True, False, False]


def test_large_image_is_the_fallback(card_image, tmp_path):
    server = Server(card_image, missing={"https://img.test/1.png"})
    data, _ = build("mtg", [product(1, large=True)], server, tmp_path)
    assert list(Index.load(data).has_hash) == [True]


def test_header_carries_game_sha_and_flag(card_image, tmp_path):
    data, sha = build("pokemon", [product(1)], Server(card_image), tmp_path, confirm_only=True)
    header = Index.load(data).header
    assert header.game == "pokemon"
    assert header.catalog_sha256 == sha
    assert header.confirm_only is True


def test_images_are_cached(card_image, tmp_path):
    server = Server(card_image)
    first, _ = build("mtg", [product(1), product(2)], server, tmp_path)
    assert len(server.requests) == 2
    second, _ = build("mtg", [product(1), product(2)], server, tmp_path)
    assert len(server.requests) == 2
    assert Index.load(first).hashes.tobytes() == Index.load(second).hashes.tobytes()


def test_fetch_retries_three_times_and_waits_between_requests(sleeps):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = images.fetch(["https://img.test/a.png"], None, client)
    assert result == {"https://img.test/a.png": None}
    assert len(calls) == 3


class FakeS3:
    def __init__(self, objects=None):
        self.objects, self.puts, self.gets = objects or {}, [], []

    def get_object(self, Bucket, Key):
        self.gets.append((Bucket, Key))
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, **kw):
        self.puts.append(kw)


def test_catalog_read_from_s3_and_file(tmp_path):
    from cardledger_vision import catalog

    raw = catalog_bytes([product(1)])
    s3 = FakeS3({"catalog/v1/mtg.json": raw})
    monkey = pytest.MonkeyPatch()
    monkey.setenv("CARDLEDGER_S3_BUCKET", "bkt")
    try:
        assert catalog.read("mtg", None, s3) == (raw, hashlib.sha256(raw).hexdigest())
    finally:
        monkey.undo()
    assert s3.gets == [("bkt", "catalog/v1/mtg.json")]
    path = tmp_path / "c.json"
    path.write_bytes(raw)
    assert catalog.read("mtg", path, None)[0] == raw


def test_upload_puts_vision_key(tmp_path, monkeypatch):
    monkeypatch.setenv("CARDLEDGER_S3_BUCKET", "bkt")
    key = tmp_path / "vision/v1/mtg.bin"
    key.parent.mkdir(parents=True)
    key.write_bytes(b"data")
    s3 = FakeS3()
    assert cli.upload(["mtg"], tmp_path, s3) == ["vision/v1/mtg.bin"]
    assert s3.puts[0]["Key"] == "vision/v1/mtg.bin"
    assert s3.puts[0]["Bucket"] == "bkt" and s3.puts[0]["Body"] == b"data"


def test_upload_without_bucket_fails(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("CARDLEDGER_S3_BUCKET", raising=False)
    monkeypatch.chdir(tmp_path)  # no .env here
    assert cli.main(["upload", "--game", "mtg", "--out", str(tmp_path)]) != 0
    assert "CARDLEDGER_S3_BUCKET" in capsys.readouterr().out


def test_build_index_command_writes_file(card_image, tmp_path, monkeypatch):
    path = tmp_path / "c.json"
    path.write_bytes(catalog_bytes([product(1)]))
    server = Server(card_image)
    monkeypatch.setattr(cli, "make_client", server.client)
    code = cli.main(["build-index", "--game", "mtg", "--catalog", str(path), "--out", str(tmp_path / "out"),
                     "--cache", str(tmp_path / "cache")])
    assert code == 0
    assert len(Index.load((tmp_path / "out/vision/v1/mtg.bin").read_bytes())) == 1
