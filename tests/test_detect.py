import cv2
import numpy as np

from cardledger_vision.detect import detect
from cardledger_vision.phash import hamming, hash_card


def test_detects_rotated_card(card_image, photo_of):
    card = card_image(3)
    out = detect(photo_of(card, angle=8, bg_seed=1))
    assert out.shape[:2] == (680, 488)
    assert hamming(hash_card(out), hash_card(card)) <= 24


def test_landscape_card_is_returned_portrait(card_image, photo_of):
    out = detect(photo_of(card_image(3), angle=90, bg_seed=2))
    assert out.shape[:2] == (680, 488)


def test_empty_table_is_none():
    assert detect(np.full((2000, 1500, 3), 128, np.uint8)) is None


def test_small_blob_is_none():
    img = np.full((2000, 1500, 3), 128, np.uint8)
    cv2.rectangle(img, (700, 900), (800, 1000), (0, 0, 0), -1)
    assert detect(img) is None
