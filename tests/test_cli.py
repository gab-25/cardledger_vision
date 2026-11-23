import hashlib
import io
import json

import cv2
import httpx
import numpy as np
import pytest

from cardledger_vision import cli, images
from cardledger_vision.index import Entry, Header, Index, encode
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


def write_index(tmp_path, slug, game=None):
    data = encode(Header(game or slug, "t", "x"), [Entry(1, "A", "s", "1", bytes(32))])
    key = tmp_path / f"vision/v1/{slug}.bin"
    key.parent.mkdir(parents=True, exist_ok=True)
    key.write_bytes(data)
    return data


def test_upload_puts_vision_key(tmp_path, monkeypatch):
    monkeypatch.setenv("CARDLEDGER_S3_BUCKET", "bkt")
    data = write_index(tmp_path, "mtg")
    s3 = FakeS3()
    assert cli.upload(["mtg"], tmp_path, s3) == ["vision/v1/mtg.bin"]
    assert s3.puts[0]["Key"] == "vision/v1/mtg.bin"
    assert s3.puts[0]["Bucket"] == "bkt" and s3.puts[0]["Body"] == data


@pytest.mark.parametrize("content", ["garbage", "other game"])
def test_upload_refuses_an_invalid_index_or_another_games(tmp_path, monkeypatch, content):
    monkeypatch.setenv("CARDLEDGER_S3_BUCKET", "bkt")
    write_index(tmp_path, "mtg")
    if content == "garbage":
        (tmp_path / "vision/v1/pokemon.bin").write_bytes(b"data")
    else:
        write_index(tmp_path, "pokemon", game="lorcana")
    s3 = FakeS3()
    with pytest.raises(cli.CliError, match="pokemon"):
        cli.upload(["mtg", "pokemon"], tmp_path, s3)
    assert s3.puts == []  # nothing is uploaded when one file is wrong


def test_catalog_s3_error_is_a_cli_error(tmp_path, monkeypatch, capsys):
    from botocore.exceptions import ClientError

    class Failing:
        def get_object(self, Bucket, Key):
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": "missing"}}, "GetObject")

    monkeypatch.setenv("CARDLEDGER_S3_BUCKET", "bkt")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "make_s3", Failing)
    assert cli.main(["build-index", "--game", "mtg", "--out", str(tmp_path / "out")]) == 1
    assert "catalog/v1/mtg.json" in capsys.readouterr().out


def test_build_index_writes_atomically(card_image, tmp_path, monkeypatch):
    path = tmp_path / "c.json"
    path.write_bytes(catalog_bytes([product(1)]))
    out = tmp_path / "out/vision/v1/mtg.bin"
    out.parent.mkdir(parents=True)
    out.write_bytes(b"previous")
    monkeypatch.setattr(cli, "make_client", Server(card_image).client)

    def fail(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(cli.os, "replace", fail)
    with pytest.raises(OSError):
        cli.main(["build-index", "--game", "mtg", "--catalog", str(path), "--out", str(tmp_path / "out"),
                  "--cache", str(tmp_path / "cache")])
    assert out.read_bytes() == b"previous"


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


def test_host_interval_option(card_image, tmp_path, monkeypatch):
    path = tmp_path / "c.json"
    path.write_bytes(catalog_bytes([product(1)]))
    got = {}
    monkeypatch.setattr(cli, "make_client", Server(card_image).client)
    real = cli.build_index

    def spy(*a, **k):
        got.update(k)
        return real(*a, **k)

    monkeypatch.setattr(cli, "build_index", spy)
    cli.main(["build-index", "--game", "mtg", "--catalog", str(path), "--out", str(tmp_path / "out"),
              "--cache", str(tmp_path / "cache"), "--host-interval", "0.5"])
    assert got["host_interval"] == 0.5


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


class Bucket:
    """An S3 client serving one catalog per game; the games in `missing` have none."""

    def __init__(self, missing=()):
        self.missing, self.keys = set(missing), []

    def get_object(self, Bucket, Key):
        from botocore.exceptions import ClientError

        self.keys.append(Key)
        if any(Key == f"catalog/v1/{slug}.json" for slug in self.missing):
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": "missing"}}, "GetObject")
        return {"Body": io.BytesIO(catalog_bytes([product(1)]))}


def build_all(card_image, tmp_path, monkeypatch, bucket, argv):
    monkeypatch.setenv("CARDLEDGER_S3_BUCKET", "bkt")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "make_s3", lambda: bucket)
    monkeypatch.setattr(cli, "make_client", Server(card_image).client)
    return cli.main(["build-index", *argv, "--out", str(tmp_path / "out"), "--cache", str(tmp_path / "cache")])


@pytest.mark.parametrize("argv", [[], ["--game", "all"]])
def test_build_index_without_game_builds_every_game(card_image, tmp_path, monkeypatch, argv):
    bucket = Bucket()
    assert build_all(card_image, tmp_path, monkeypatch, bucket, argv) == 0
    assert bucket.keys == [f"catalog/v1/{slug}.json" for slug in cli.GAMES]
    for slug in cli.GAMES:
        assert Index.load((tmp_path / f"out/vision/v1/{slug}.bin").read_bytes()).header.game == slug


def test_a_failing_game_does_not_stop_the_others(card_image, tmp_path, monkeypatch, capsys):
    assert build_all(card_image, tmp_path, monkeypatch, Bucket(missing={"lorcana"}), []) == 1
    assert "build-index failed for: lorcana" in capsys.readouterr().out
    assert not (tmp_path / "out/vision/v1/lorcana.bin").exists()
    assert all((tmp_path / f"out/vision/v1/{slug}.bin").is_file() for slug in cli.GAMES if slug != "lorcana")


def test_catalog_without_game_is_refused(tmp_path, monkeypatch, capsys):
    path = tmp_path / "c.json"
    path.write_bytes(catalog_bytes([product(1)]))
    assert cli.main(["build-index", "--catalog", str(path), "--out", str(tmp_path / "out")]) == 1
    assert "--game" in capsys.readouterr().out


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


def test_ctrl_c_exits_without_traceback(tmp_path, monkeypatch, capsys):
    path = tmp_path / "c.json"
    path.write_bytes(catalog_bytes([product(1)]))

    def interrupted(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "build_index", interrupted)
    assert cli.main(["build-index", "--game", "mtg", "--catalog", str(path), "--out", str(tmp_path / "out")]) == 130
    assert "interrupted" in capsys.readouterr().out
    assert not (tmp_path / "out/vision/v1/mtg.bin").exists()


@pytest.mark.parametrize("argv", [[], ["--game", "all"]])
def test_upload_without_game_uploads_every_game(tmp_path, monkeypatch, argv):
    monkeypatch.setenv("CARDLEDGER_S3_BUCKET", "bkt")
    monkeypatch.chdir(tmp_path)
    for slug in cli.GAMES:
        write_index(tmp_path, slug)
    s3 = FakeS3()
    monkeypatch.setattr(cli, "make_s3", lambda: s3)
    assert cli.main(["upload", *argv, "--out", str(tmp_path)]) == 0
    assert [put["Key"] for put in s3.puts] == [cli.index_key(slug) for slug in cli.GAMES]


def test_upload_without_game_uploads_nothing_when_one_is_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CARDLEDGER_S3_BUCKET", "bkt")
    monkeypatch.chdir(tmp_path)
    for slug in cli.GAMES[1:]:
        write_index(tmp_path, slug)
    s3 = FakeS3()
    monkeypatch.setattr(cli, "make_s3", lambda: s3)
    assert cli.main(["upload", "--out", str(tmp_path)]) == 1
    assert cli.GAMES[0] in capsys.readouterr().out and s3.puts == []


def test_bench_without_game_benches_every_game(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "make_s3", lambda: Bucket(missing={"lorcana"}))
    benched = []

    def bench_game(slug, args, s3):
        if slug == "lorcana":
            raise cli.CliError("no catalog")
        benched.append((slug, args.index or args.out / cli.index_key(slug)))

    monkeypatch.setattr(cli, "_bench_game", bench_game)
    assert cli.main(["bench", "--out", str(tmp_path / "out")]) == 1
    assert benched == [(slug, tmp_path / "out" / cli.index_key(slug)) for slug in cli.GAMES if slug != "lorcana"]
    assert "bench failed for: lorcana" in capsys.readouterr().out


@pytest.mark.parametrize("option", ["--index", "--catalog"])
def test_bench_one_games_file_without_game_is_refused(tmp_path, monkeypatch, capsys, option):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["bench", option, str(tmp_path / "x")]) == 1
    assert "--game" in capsys.readouterr().out
