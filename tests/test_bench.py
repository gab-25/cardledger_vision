import random

import cv2

from cardledger_vision.bench import collect, run, sample, synth_photo
from cardledger_vision.catalog import CatalogProduct
from cardledger_vision.index import Entry, Header, Index, encode
from cardledger_vision.match import recognize_hash
from cardledger_vision.phash import CARD_SIZE, hash_card


def test_synth_photo_is_deterministic(card_image):
    a = synth_photo(card_image(1), random.Random(5))
    assert a == synth_photo(card_image(1), random.Random(5))
    assert a != synth_photo(card_image(1), random.Random(6))
    assert cv2.imdecode(__import__("numpy").frombuffer(a, "uint8"), cv2.IMREAD_COLOR).shape == (2000, 1500, 3)


def framed(card_image, i):
    """A test card with a dark frame, like a real card's border, so detect finds it on the synthetic backgrounds."""
    return cv2.copyMakeBorder(card_image(i), 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=(15, 15, 15))


def make(card_image, n=3, extra=(), confirm_only=False):
    products, images, entries = [], {}, []
    for i in range(1, n + 1):
        url = f"https://img.test/large/{i}.png"
        products.append(CatalogProduct(i, f"Card {i}", "set", str(i), None, url))
        ok, png = cv2.imencode(".png", framed(card_image, i))
        images[url] = png.tobytes()
        h = hash_card(cv2.resize(framed(card_image, i), CARD_SIZE, interpolation=cv2.INTER_AREA))
        entries.append(Entry(i, f"Card {i}", "set", str(i), h))
    entries += list(extra)
    index = Index.load(encode(Header("mtg", "2026-01-01T00:00:00Z", "x", confirm_only), entries))
    return index, products, images


def samples_of(products, images):
    return [(p, images[p.image_large_url]) for p in products]


def test_run_counts_outcomes(card_image):
    index, products, images = make(card_image)
    report = run(index, samples_of(products, images), seed=1)
    assert report.total == 3 and report.counts["no_lock"]["wrong_certain"] == 0
    assert report.counts["no_lock"]["ok"] + report.counts["no_lock"]["in_candidates"] == 3  # the photos are read
    assert report.counts["lock"]["wrong_certain"] == 0
    for mode in ("no_lock", "lock", "unindexed"):
        assert sum(report.counts[mode].values()) == 3
    assert report.counts["unindexed"]["ok"] == 0
    table = report.table()
    assert "wrong_certain" in table and "%" in table and "p90" in table and "unindexed" in table


def test_a_confirm_only_index_is_measured_as_if_it_were_not(card_image):
    index, products, images = make(card_image)
    certain = run(index, samples_of(products, images), seed=1)
    confirm_only, _, _ = make(card_image, confirm_only=True)
    report = run(confirm_only, samples_of(products, images), seed=1)
    assert report.counts == certain.counts and report.counts["no_lock"]["ok"] > 0
    assert report.ranks == certain.ranks
    assert confirm_only.header.confirm_only  # the index passed in is left as it was
    assert "confirm-only index" in report.table() and "confirm-only index" not in certain.table()


def test_ranks_place_the_right_card_among_the_candidates(card_image):
    index, products, images = make(card_image)
    report = run(index, samples_of(products, images), seed=1)
    for mode in ("no_lock", "lock"):
        found = report.counts[mode]["ok"] + report.counts[mode]["in_candidates"]
        assert sum(report.ranks[mode].values()) == found
        assert report.ranks[mode]["1st"] >= report.counts[mode]["ok"]  # an ok answer is its first candidate
    assert sum(report.ranks["unindexed"].values()) == 0  # the card is not in the index
    assert "right card among the candidates" in report.table() and "2nd-3rd" in report.table()


def test_unindexed_mode_counts_a_certain_answer_as_wrong(card_image):
    # Product 9 has card 1's image under another name: with card 1 in the index the two names tie, without it
    # product 9 wins with certainty, which is a wrong certain answer for a card missing from the index.
    twin = Entry(9, "Other", "set", "9", hash_card(cv2.resize(framed(card_image, 1), CARD_SIZE,
                                                               interpolation=cv2.INTER_AREA)))
    index, products, images = make(card_image, n=1, extra=[twin])
    report = run(index, samples_of(products, images), seed=1)
    assert report.counts["no_lock"] == {"ok": 0, "wrong_certain": 0, "in_candidates": 1, "not_found": 0, "no_card": 0}
    assert report.counts["unindexed"]["wrong_certain"] == 1


def test_drop_row_is_the_index_without_that_entry(card_image):
    from cardledger_vision.match import drop_row

    entries = [Entry(i, f"N{i % 2}", f"s{i}", str(i), hash_card(card_image(i)) if i != 2 else None) for i in range(1, 5)]
    full = Index.load(encode(Header("mtg", "t", "x"), entries))
    without = Index.load(encode(Header("mtg", "t", "x"), entries[:2] + entries[3:]))
    recognize_hash(entries[0].hash, full)  # caches the match keys of the full index
    dropped = drop_row(full, 2)
    assert list(dropped.ids) == list(without.ids) and dropped.names == without.names
    for i in (1, 3, 4):
        h = entries[i - 1].hash
        assert recognize_hash(h, dropped) == recognize_hash(h, without)
        assert recognize_hash(h, dropped, lock="s1") == recognize_hash(h, without, lock="s1")


def test_sample_is_capped_unique_and_only_from_products_with_an_image():
    products = [CatalogProduct(i, f"Name {i % 10}", "s", str(i), None, f"u{i}" if i % 2 == 0 else None)
                for i in range(100)]
    picked = sample(products, 1000, 3)
    assert len(picked) == 50 and len({p.id_product for p in picked}) == 50
    assert all(p.image_large_url for p in picked)
    assert [p.id_product for p in sample(products, 20, 3)] == [p.id_product for p in sample(products, 20, 3)]
    assert len(sample(products, 20, 3)) == 20


def test_collect_fetches_only_the_sample_and_replacements():
    products = [CatalogProduct(i, f"Name {i}", "s", str(i), f"p{i}", f"l{i}" if i % 3 else None) for i in range(1000)]
    broken = {f"l{i}" for i in range(0, 1000, 3)} | {f"p{i}" for i in range(0, 1000, 9)} | {f"l{i}" for i in range(1, 1000, 7)}
    asked: list[str] = []

    def fetch(urls):
        asked.extend(urls)
        return {u: None if u in broken else b"img" for u in urls}

    got, skipped = collect(products, 10, 5, fetch)
    assert len(got) == 10 and len({p.id_product for p, _ in got}) == 10
    assert len(asked) < 40  # never the whole catalog
    touched = {int(u[1:]) for u in asked}
    assert skipped == len(touched) - len(got) > 0
    for p, _ in got:  # the large image when it works, else the normal one
        assert (p.image_large_url in asked and p.image_large_url not in broken) or p.image_url not in broken
    assert collect(products, 10, 5, fetch)[0] == got  # deterministic


def test_parallel_photos_match_the_serial_ones(card_image):
    from cardledger_vision.bench import _photos

    ok, png = cv2.imencode(".png", framed(card_image, 1))
    ok, png2 = cv2.imencode(".png", framed(card_image, 2))
    samples = [("a", png.tobytes()), ("bad", b"not an image"), ("b", png2.tobytes())]
    rng = random.Random(3)
    expect = [("a", synth_photo(framed(card_image, 1), rng)), None, ("b", synth_photo(framed(card_image, 2), rng))]
    assert _photos(samples, 3) == expect
