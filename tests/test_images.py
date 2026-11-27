import os
import time

import cv2
import numpy as np
import pytest

from cardledger_vision import images
from cardledger_vision.phash import CARD_SIZE, hash_card


def expected_hash(card):
    return hash_card(cv2.resize(card, CARD_SIZE, interpolation=cv2.INTER_AREA))


def test_local_hashes_hash_each_file(tmp_path, write_card_image, card_image):
    a = write_card_image(tmp_path / "a.webp", seed=1)
    b = write_card_image(tmp_path / "b.png", seed=2)
    assert images.local_hashes([a, b], tmp_path / "cache") == {
        a: expected_hash(card_image(1)), b: expected_hash(card_image(2))}


def test_missing_or_undecodable_file_is_none(tmp_path, write_card_image, monkeypatch, card_image):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"\x89PNG\r\n\x1a\nbroken")
    raising = write_card_image(tmp_path / "raising.png", seed=4)
    real = cv2.imdecode

    def imdecode(buf, flags):
        if np.asarray(buf).tobytes() == raising.read_bytes():
            raise cv2.error("decoder failure")
        return real(buf, flags)

    monkeypatch.setattr(cv2, "imdecode", imdecode)
    gone, folder = tmp_path / "gone.png", tmp_path
    assert images.local_hashes([gone, bad, raising, folder], tmp_path / "cache") == dict.fromkeys(
        [gone, bad, raising, folder])


def test_hashes_are_cached_by_file_name_and_reused_without_decoding(tmp_path, write_card_image, monkeypatch):
    paths = [write_card_image(tmp_path / f"{i}.png", seed=i) for i in (1, 2)]
    first = images.local_hashes(paths, tmp_path / "cache")
    assert sorted(p.name for p in (tmp_path / "cache" / images.HASHES).iterdir()) == ["1.png", "2.png"]
    decoded = []
    real = images.decode
    monkeypatch.setattr(images, "decode", lambda data: decoded.append(1) or real(data))
    assert images.local_hashes(paths, tmp_path / "cache") == first
    assert decoded == []


def test_a_changed_file_is_hashed_again(tmp_path, write_card_image, card_image):
    path = write_card_image(tmp_path / "a.png", seed=1)
    images.local_hashes([path], tmp_path / "cache")
    write_card_image(path, seed=3)
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))  # a later mtime, whatever the clock's resolution
    assert images.local_hashes([path], tmp_path / "cache") == {path: expected_hash(card_image(3))}
    assert [p.name for p in (tmp_path / "cache" / images.HASHES).iterdir()] == ["a.png"]  # no temporary file left


def test_landscape_image_is_turned_portrait(tmp_path, write_card_image, card_image):
    path = write_card_image(tmp_path / "a.png", image=cv2.rotate(card_image(1), cv2.ROTATE_90_COUNTERCLOCKWISE))
    assert images.local_hashes([path], tmp_path / "cache")[path] == expected_hash(card_image(1))


def test_transparent_png_is_composited_on_grey(tmp_path, write_card_image, card_image):
    card = card_image(1)
    bgra = cv2.cvtColor(card, cv2.COLOR_BGR2BGRA)
    bgra[:200, :, 3] = 0  # the top is transparent; its colour channels are junk
    bgra[:200, :, :3] = 0
    path = write_card_image(tmp_path / "a.png", image=bgra)
    grey = card.copy()
    grey[:200] = 128
    assert images.local_hashes([path], tmp_path / "cache")[path] == expected_hash(grey)


def test_ctrl_c_drops_the_queued_files(tmp_path, monkeypatch):
    """The interrupt reaches the main thread from the first file; the queued files are never decoded."""
    paths = []
    for i in range(200):
        paths.append(tmp_path / f"{i}.png")
        paths[-1].write_bytes(str(i).encode())
    calls = []

    def decode(data):
        calls.append(data)
        if data == b"0":
            raise KeyboardInterrupt
        time.sleep(0.2)  # the busy workers are still at it when the main thread drops the queue

    monkeypatch.setattr(images, "decode", decode)
    with pytest.raises(KeyboardInterrupt):
        images.local_hashes(paths, tmp_path / "cache")
    assert len(calls) <= images.WORKERS + 1
