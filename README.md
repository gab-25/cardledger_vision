# cardledger_vision

Python library that recognizes trading cards in photos by perceptual hash (256-bit pHash). It has no command line and
no access to the bucket: it gets catalogs as bytes and returns indexes as bytes.

Commands: `uv sync`, `uv run pytest -q`.

## Building an index

```python
from cardledger_vision import build_index, index_key

index = build_index("mtg", catalog_bytes, catalog_sha256, httpx_client, cache_dir, confirm_only=False)
# its path in an output folder and in the bucket: index_key("mtg"), i.e. vision/v1/mtg.bin
```

The published indexes are built by `cardledger-db generate` in
[cardledger_database](https://github.com/gab-25/cardledger_database), right after each catalog, and uploaded with it by
`cardledger-db upload`. That repository pins a release tag of this one (`v0.1.0` now) in its `pyproject.toml`: after
changing the hash, the index format or the thresholds here, make a new release, move the `tag` there, run
`uv lock --upgrade-package cardledger-vision` and rebuild the indexes with `cardledger-db index`.

## Measuring an index

`bench` makes synthetic photos (rotation, perspective, light, glare, blur, noise, JPEG) of a seeded sample of the
catalog images and recognizes each without a set lock and with the product's own set as lock. The table counts
`ok`, `wrong_certain` (answered `ok` with the wrong product), `in_candidates`, `not_found` and `no_card`, then where
the right card is among the candidates (1st, 2nd-3rd, 4th+: what the user picks from on a confirm-only index), with
the median and p90 time of `recognize`. A confirm-only index is measured as if it were not, so `ok` and
`wrong_certain` still say how often a certain answer would be right.

```python
from pathlib import Path

import httpx

from cardledger_vision import Index, bench, catalog, images

products = catalog.load(Path("catalog/v1/mtg.json").read_bytes())
index = Index.load(Path("vision/v1/mtg.bin").read_bytes())
with httpx.Client(follow_redirects=True) as client:
    samples, skipped = bench.collect(products, 200, 42, lambda urls: images.fetch(urls, Path(".cache"), client))
print(bench.run(index, samples, 42, skipped).table())
```

Confirm-only rule: a game whose index is `confirm_only` never answers `ok` without confirmation. In
cardledger_database every game is confirm-only for now; a game can drop it once its `wrong_certain` is at most 2%.
