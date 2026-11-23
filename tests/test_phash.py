import cv2
import numpy as np

from cardledger_vision.phash import hamming, hamming_many, hash_card


def test_hash_is_32_bytes_and_stable(card_image):
    img = card_image(1)
    assert len(hash_card(img)) == 32 and hash_card(img) == hash_card(img.copy())


def test_same_card_resized_and_jpeg_is_close(card_image):
    img = card_image(1)
    small = cv2.resize(img, (146, 204))
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
    assert hamming(hash_card(img), hash_card(cv2.imdecode(buf, cv2.IMREAD_COLOR))) <= 20


def test_different_cards_are_far(card_image):
    assert hamming(hash_card(card_image(1)), hash_card(card_image(2))) >= 80


def test_hamming_many_matches_hamming(card_image):
    hs = [hash_card(card_image(i)) for i in range(5)]
    table = np.frombuffer(b"".join(hs), np.uint8).reshape(5, 32)
    assert list(hamming_many(table, hs[0])) == [hamming(hs[0], h) for h in hs]
