import cv2
import numpy as np
import pytest

from cardledger_vision import Entry, Header, Index, Result, encode, hash_card, recognize, recognize_hash

H: list[bytes] = []


@pytest.fixture(autouse=True)
def _hashes(card_image):
    # Six hashes far from each other (>= 80 apart, Task 1), computed once per module.
    if not H:
        H.extend(hash_card(card_image(i)) for i in range(6))


def idx(*entries, confirm_only=False):
    return Index.load(encode(Header("mtg", "t", "x", confirm_only=confirm_only), list(entries)))


def near(h, k):
    bits = np.unpackbits(np.frombuffer(h, np.uint8))
    bits[:k] ^= 1
    return np.packbits(bits).tobytes()


def test_unique_card_is_ok():
    r = recognize_hash(near(H[0], 3), idx(Entry(1, "Bolt", "lea", "1", H[0]), Entry(2, "Wrath", "lea", "2", H[1])))
    assert r == Result("ok", 1, [1])


def test_far_from_everything_is_not_found():
    assert recognize_hash(H[5], idx(Entry(1, "Bolt", "lea", "1", H[0]))).outcome == "not_found"


def test_close_names_are_ambiguous():
    r = recognize_hash(H[0], idx(Entry(1, "Bolt", "a", "1", H[0]), Entry(2, "Shock", "b", "1", near(H[0], 4))))
    assert r.outcome == "ambiguous" and set(r.candidates) == {1, 2}


def test_reprint_same_art_needs_lock():
    e = [Entry(1, "Bolt", "m10", "146", H[0]), Entry(2, "Bolt", "m11", "149", near(H[0], 2))]
    assert recognize_hash(H[0], idx(*e)).outcome == "ambiguous"
    # Locked candidates: the in-lock printings, then the other printings of the name.
    assert recognize_hash(H[0], idx(*e), lock="m11") == Result("ok", 2, [2, 1])


def test_lock_without_that_name_is_ignored():
    r = recognize_hash(H[0], idx(Entry(1, "Bolt", "lea", "1", H[0])), lock="neo")
    assert r == Result("ok", 1, [1])


def test_promo_twin_with_same_number_is_never_certain_without_lock():
    e = [Entry(1, "Shadow", "neo", "86", H[0]), Entry(2, "Shadow", "pneo", "086", near(H[0], 30))]
    assert recognize_hash(H[0], idx(*e)).outcome == "ambiguous"


def test_twin_without_image_blocks_certainty():
    # An imageless printing of the same name blocks certainty without a lock, whatever its set and number.
    e = [Entry(1, "Dragon", "lob", "LOB-001", H[0]), Entry(2, "Dragon", "sdk", "SDK-001", None)]
    assert recognize_hash(H[0], idx(*e)).outcome == "ambiguous"
    # With the lock on the hashed printing's set, the imageless one elsewhere no longer matters.
    assert recognize_hash(H[0], idx(*e), lock="lob") == Result("ok", 1, [1, 2])


def test_imageless_printing_in_the_same_set_and_number_blocks_certainty():
    e = [Entry(1, "Dragon", "lob", "1", H[0]), Entry(2, "Dragon", "lob", "1", None)]
    assert recognize_hash(H[0], idx(*e)).outcome == "ambiguous"
    assert recognize_hash(H[0], idx(*e), lock="lob").outcome == "ambiguous"


def test_lock_guard_when_a_printing_outside_the_lock_is_much_nearer():
    e = [Entry(1, "Bolt", "m10", "1", H[1]), Entry(2, "Bolt", "m11", "2", H[0])]
    r = recognize_hash(H[0], idx(*e), lock="m10")
    assert r.outcome == "ambiguous" and r.product_id is None
    assert set(r.candidates) == {1, 2} and r.candidates.index(2) <= 1


def test_locked_candidates_are_in_lock_first_then_out_of_lock():
    e = [Entry(1, "Bolt", "m10", "1", near(H[0], 30)), Entry(2, "Bolt", "m11", "2", near(H[0], 3)),
         Entry(3, "Bolt", "m10", "9", near(H[0], 12)), Entry(4, "Bolt", "m12", "4", near(H[0], 1))]
    r = recognize_hash(H[0], idx(*e), lock="m10")
    assert r.candidates == [3, 1, 4, 2]


def test_lock_is_case_insensitive_and_names_are_stripped():
    e = [Entry(1, "Bolt", "M10", "146", H[0]), Entry(2, " bolt ", "m11", "149", near(H[0], 2))]
    assert recognize_hash(H[0], idx(*e)).outcome == "ambiguous"  # same name once stripped
    assert recognize_hash(H[0], idx(*e), lock="m10") == Result("ok", 1, [1, 2])


def test_identical_hash_in_two_sets_is_never_certain():
    e = [Entry(1, "Elf", "a", "1", H[0]), Entry(2, "Elf", "b", "7", H[0])]
    assert recognize_hash(H[0], idx(*e)).outcome == "ambiguous"


def test_confirm_only_never_ok():
    r = recognize_hash(H[0], idx(Entry(1, "Bolt", "lea", "1", H[0]), confirm_only=True))
    assert r == Result("ambiguous", None, [1])


def test_index_without_hashes_is_not_found():
    assert recognize_hash(H[0], idx(Entry(1, "Bolt", "lea", "1", None))).outcome == "not_found"


def test_garbage_bytes_is_no_card():
    assert recognize(b"not a jpeg", idx(Entry(1, "Bolt", "lea", "1", H[0]))).outcome == "no_card"


def test_photo_end_to_end_and_upside_down(card_image, photo_of):
    card = card_image(4)
    index = idx(Entry(7, "Bolt", "lea", "1", hash_card(card)), Entry(8, "Wrath", "lea", "2", H[1]))
    for angle in (5, 185):
        ok, buf = cv2.imencode(".jpg", photo_of(card, angle=angle, bg_seed=3))
        assert recognize(buf.tobytes(), index) == Result("ok", 7, [7])


# Extra cases from the controller's rulings.


def test_empty_index_is_not_found_without_candidates():
    assert recognize_hash(H[0], idx()) == Result("not_found", None, [])


def test_not_found_lists_the_nearest_hashed_entries():
    e = [Entry(i + 1, f"C{i}", "s", str(i), H[i]) for i in range(1, 5)] + [Entry(9, "X", "s", "9", None)]
    r = recognize_hash(H[0], idx(*e))
    assert r.outcome == "not_found" and r.product_id is None
    assert sorted(r.candidates) == [2, 3, 4, 5]


def test_no_card_in_a_blank_photo():
    ok, buf = cv2.imencode(".png", np.full((400, 300, 3), 128, np.uint8))
    assert recognize(buf.tobytes(), idx(Entry(1, "Bolt", "lea", "1", H[0]))) == Result("no_card", None, [])


def test_double_faced_name_and_number_normalization():
    # "A // B" keys on "a"; "007" and "7" are the same number, so the twin blocks certainty.
    e = [Entry(1, "Fire // Ice", "apc", "007", H[0]), Entry(2, "FIRE", "pap", "7 ", near(H[0], 40))]
    r = recognize_hash(H[0], idx(*e))
    assert r.outcome == "ambiguous" and r.candidates == [1, 2]


def test_ok_with_several_printings_puts_pick_first():
    e = [Entry(1, "Bolt", "a", "1", near(H[0], 20)), Entry(2, "Bolt", "b", "2", H[0])]
    assert recognize_hash(H[0], idx(*e)) == Result("ok", 2, [2, 1])


def big_index(n=100_000, seed=0):
    rng = np.random.default_rng(seed)
    hashes = rng.integers(0, 256, (n, 32), dtype=np.uint8)
    entries = [Entry(i + 1, f"Card {i % 30_000}", f"s{i % 400}", str(i % 300),
                     hashes[i].tobytes() if i % 10 else None) for i in range(n)]
    return Index.load(encode(Header("mtg", "t", "x"), entries)), hashes


def test_recognize_hash_is_fast_on_100k_entries():
    import statistics
    import time

    index, hashes = big_index()
    recognize_hash(hashes[1].tobytes(), index)  # builds the cached keys
    times = []
    for k in range(1, 31):
        h = near(hashes[k * 7 + 1].tobytes(), 5)
        start = time.perf_counter()
        recognize_hash(h, index, lock="s8" if k % 2 else None)
        times.append((time.perf_counter() - start) * 1000)
    median = statistics.median(times)
    print(f"recognize_hash median on 100k entries: {median:.1f} ms")
    assert median < 40
