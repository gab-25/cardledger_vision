import hashlib
import json

import cv2
import httpx
import numpy as np
import pytest

import cardledger_vision
from cardledger_vision import images
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

    def __init__(self, card_image, missing=(), bodies=None):
        self.card_image, self.missing, self.requests = card_image, set(missing), []
        self.bodies = bodies or {}  # url -> the bytes to answer instead of the card

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append(url)
        if url in self.missing:
            return httpx.Response(404)
        if url in self.bodies:
            return httpx.Response(200, content=self.bodies[url])
        seed = int(url.rsplit("/", 1)[1].split(".")[0])
        ok, png = cv2.imencode(".png", self.card_image(seed))
        return httpx.Response(200, content=png.tobytes())

    def client(self):
        return httpx.Client(transport=httpx.MockTransport(self.handler))


def build(slug, products, server, tmp_path, **kw):
    raw = catalog_bytes(products)
    sha = hashlib.sha256(raw).hexdigest()
    with server.client() as client:
        return cardledger_vision.build_index(slug, raw, sha, client, tmp_path / "cache", **kw), sha


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


def test_bad_url_or_undecodable_image_leaves_the_entry_without_hash(card_image, tmp_path, monkeypatch):
    bad = [product(1), product(2), product(3), product(4)]
    bad[0]["image_url"] = "https://img.test/\x00x.png"  # httpx.InvalidURL
    bad[1]["image_url"] = "not a url"
    server = Server(card_image, bodies={"https://img.test/3.png": b"\x89PNG\r\n\x1a\nbroken"})
    real, png4 = cv2.imdecode, cv2.imencode(".png", card_image(4))[1].tobytes()

    def imdecode(buf, flags):
        if np.asarray(buf).tobytes() == png4:
            raise cv2.error("decoder failure")
        return real(buf, flags)

    monkeypatch.setattr(cv2, "imdecode", imdecode)
    data, _ = build("mtg", bad, server, tmp_path)
    assert list(Index.load(data).has_hash) == [False, False, False, False]


def test_only_decodable_images_are_cached(card_image, tmp_path):
    server = Server(card_image, bodies={"https://img.test/2.png": b"<html>rate limited</html>"})
    build("mtg", [product(1), product(2)], server, tmp_path)
    assert len(list((tmp_path / "cache/images").iterdir())) == 1
    build("mtg", [product(1), product(2)], server, tmp_path)
    assert server.requests.count("https://img.test/2.png") == 2  # not cached, asked again
    assert server.requests.count("https://img.test/1.png") == 1


def test_corrupt_cached_file_is_fetched_again(card_image, tmp_path):
    server = Server(card_image)
    build("mtg", [product(1)], server, tmp_path)
    (cached,) = (tmp_path / "cache/images").iterdir()
    cached.write_bytes(b"truncated")
    data, _ = build("mtg", [product(1)], server, tmp_path)
    assert server.requests.count("https://img.test/1.png") == 2
    assert list(Index.load(data).has_hash) == [True]
    assert cv2.imdecode(np.frombuffer(cached.read_bytes(), np.uint8), cv2.IMREAD_COLOR) is not None
    assert [p.name for p in cached.parent.iterdir()] == [cached.name]  # no temporary file left


def test_large_image_is_asked_only_when_the_primary_fails(card_image, tmp_path):
    server = Server(card_image, missing={"https://img.test/2.png"})
    build("mtg", [product(1, large=True), product(2, large=True)], server, tmp_path)
    assert "https://img.test/large/1.png" not in server.requests
    assert "https://img.test/large/2.png" in server.requests


def test_build_keeps_only_hashes(card_image, tmp_path, monkeypatch):
    # The images are hashed in the workers: build_index never sees image bytes.
    monkeypatch.setattr(images, "fetch", lambda *a, **k: pytest.fail("build_index must not collect image bytes"))
    data, _ = build("mtg", [product(1), product(2)], Server(card_image), tmp_path)
    assert list(Index.load(data).has_hash) == [True, True]


def test_hashing_is_submitted_in_bounded_batches(card_image, tmp_path, monkeypatch):
    monkeypatch.setattr(images, "BATCH", 2)
    seen = []
    real = images._hash_batch

    def spy(urls, *a, **k):
        seen.append(len(urls))
        return real(urls, *a, **k)

    monkeypatch.setattr(images, "_hash_batch", spy)
    data, _ = build("mtg", [product(i) for i in range(1, 6)], Server(card_image), tmp_path)
    assert seen == [2, 2, 1] and list(Index.load(data).has_hash) == [True] * 5


def test_backoff_and_retry_after(sleeps):
    answers = iter([httpx.Response(429, headers={"Retry-After": "3"}), httpx.Response(503), httpx.Response(503)])

    with httpx.Client(transport=httpx.MockTransport(lambda request: next(answers))) as client:
        assert images.fetch(["https://backoff-a.test/b.png"], None, client, host_interval=0) == {"https://backoff-a.test/b.png": None}
    assert sleeps == [3, 1]  # Retry-After, then the second backoff step (a host of its own: no throttle wait)


def test_backoff_without_retry_after(sleeps):
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(503))) as client:
        images.fetch(["https://backoff-b.test/c.png"], None, client, host_interval=0)
    assert sleeps == [0.5, 1]


def test_landscape_catalog_image_is_turned_portrait(card_image, tmp_path):
    card = card_image(1)
    ok, png = cv2.imencode(".png", cv2.rotate(card, cv2.ROTATE_90_COUNTERCLOCKWISE))
    server = Server(card_image, bodies={"https://img.test/1.png": png.tobytes()})
    data, _ = build("mtg", [product(1)], server, tmp_path)
    expected = hash_card(cv2.resize(card, CARD_SIZE, interpolation=cv2.INTER_AREA))
    assert Index.load(data).hashes[0].tobytes() == expected


def test_transparent_png_is_composited_on_grey(card_image, tmp_path):
    card = card_image(1)
    bgra = cv2.cvtColor(card, cv2.COLOR_BGR2BGRA)
    bgra[:200, :, 3] = 0  # the top is transparent; its colour channels are junk
    bgra[:200, :, :3] = 0
    ok, png = cv2.imencode(".png", bgra)
    server = Server(card_image, bodies={"https://img.test/1.png": png.tobytes()})
    data, _ = build("mtg", [product(1)], server, tmp_path)
    grey = card.copy()
    grey[:200] = 128
    expected = hash_card(cv2.resize(grey, CARD_SIZE, interpolation=cv2.INTER_AREA))
    assert Index.load(data).hashes[0].tobytes() == expected


def test_ctrl_c_drops_the_queued_downloads(tmp_path):
    """The interrupt reaches the main thread from the first URL; the busy workers give up and the queue is dropped."""
    requests = []

    def handler(request):
        requests.append(str(request.url))
        if str(request.url).endswith("/0.png"):
            raise KeyboardInterrupt
        assert images._cancel.wait(5), "the pool was not cancelled"
        return httpx.Response(503)  # would be retried, were it not cancelled

    urls = [f"https://cancel.test/{i}.png" for i in range(200)]
    with httpx.Client(transport=httpx.MockTransport(handler)) as client, pytest.raises(KeyboardInterrupt):
        images.fetch_hashes(urls, tmp_path / "cache", client, host_interval=0)
    assert len(requests) <= images.MAX_PARALLEL + 1  # each busy worker made one attempt, the queue none
    assert not images._cancel.is_set()  # the next fetch works again


def test_rebuild_reuses_cached_hashes_without_decoding(card_image, tmp_path, monkeypatch):
    server = Server(card_image)
    first, _ = build("mtg", [product(1), product(2)], server, tmp_path)
    assert len(list((tmp_path / "cache" / images.HASHES).iterdir())) == 2
    decoded = []
    real = images.decode
    monkeypatch.setattr(images, "decode", lambda data: decoded.append(1) or real(data))
    second, _ = build("mtg", [product(1), product(2)], server, tmp_path)
    assert decoded == [] and len(server.requests) == 2
    assert Index.load(first).hashes.tobytes() == Index.load(second).hashes.tobytes()


def test_index_key_is_the_published_path():
    assert cardledger_vision.index_key("mtg") == "vision/v1/mtg.bin"
