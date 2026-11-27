# cardledger_vision

Python library that recognizes trading cards in photos by perceptual hash (256-bit pHash). It has no command line and
no access to the bucket or the network: it gets catalogs as bytes, reads their images from a local folder and returns
indexes as bytes.

Commands: `uv sync`, `uv run pytest -q`.

## Building an index

```python
from cardledger_vision import build_index, index_key

index = build_index("mtg", catalog_bytes, catalog_sha256, image_root, cache_dir, confirm_only=False)
# its path in an output folder and in the bucket: index_key("mtg"), i.e. vision/v1/mtg.bin
```

From the catalog JSON (`{"sets": [...], "rows": [...]}`) only the rows with `kind` `"card"` enter the index, and of
each only `id`, `name`, `set_code`, `number`, `image` and `image_large` are read. The images are paths relative to
`image_root` (the local copy of the bucket root): the large one is hashed, the normal one when the large one is
missing or doesn't decode; a card with neither enters without a hash. The hashes are cached in
`cache_dir/hashes/<file name>`, together with the image file's size and mtime.

The index (`FORMAT_VERSION` 1, little-endian) is: magic `CLVI`, version, a header JSON (game, build time, catalog
sha256, confirm-only flag, thresholds), the entry count n, n `has_hash` bytes, padding, n 32-byte hashes (8-aligned),
then a strings JSON with n `ids` (the catalog rows' ids, what `Result.product_id` and `Result.candidates` answer),
`names`, `sets` and `numbers`. The byte layout is in the docstring of `cardledger_vision/index.py`.

The published indexes are built by `cardledger-db generate` in
[cardledger_database](https://github.com/gab-25/cardledger_database), right after each catalog, and uploaded with it by
`cardledger-db upload`. That repository pins a release tag of this one (`v0.2.0` now) in its `pyproject.toml`: after
changing the hash, the index format or the thresholds here, make a new release, move the `tag` there, run
`uv lock --upgrade-package cardledger-vision` and rebuild the indexes with `cardledger-db index`.

## Measuring an index

`bench` makes synthetic photos (rotation, perspective, light, glare, blur, noise, JPEG) of a seeded sample of the
catalog images (read from the local image root, like `build_index`) and recognizes each without a set lock and with
the card's own set as lock. The table counts `ok`, `wrong_certain` (answered `ok` with the wrong card),
`in_candidates`, `not_found` and `no_card`, then where the right card is among the candidates (1st, 2nd-3rd, 4th+:
what the user picks from on a confirm-only index), with the median and p90 time of `recognize`. A confirm-only index is measured as if it were not, so `ok` and
`wrong_certain` still say how often a certain answer would be right.

```python
from pathlib import Path

from cardledger_vision import Index, bench, catalog

root = Path("out")  # the local copy of the bucket: catalog, images and indexes
products = catalog.load((root / "catalog/v1/mtg.json").read_bytes())
index = Index.load((root / "vision/v1/mtg.bin").read_bytes())
samples, skipped = bench.collect(products, 200, 42, root)
print(bench.run(index, samples, 42, skipped).table())
```

Confirm-only rule: a game whose index is `confirm_only` never answers `ok` without confirmation. In
cardledger_database every game is confirm-only for now; a game can drop it once its `wrong_certain` is at most 2%.
