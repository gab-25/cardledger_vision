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
`cardledger-db upload`. That repository follows this one's `main`, locked to one commit in its `uv.lock`: after
changing the hash, the index format or the thresholds here, run `uv lock --upgrade-package cardledger-vision` there
and rebuild the indexes with `cardledger-db index`.

## Measuring an index

`bench` makes synthetic photos (rotation, perspective, light, glare, blur, noise, JPEG) of a seeded sample of the
catalog images and recognizes each without a set lock and with the product's own set as lock. The table counts
`ok`, `wrong_certain` (answered `ok` with the wrong product), `in_candidates`, `not_found` and `no_card`, with the
median and p90 time of `recognize`.

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

Confirm-only rule: if `wrong_certain` is above 2% for a game, mark the game `confirm_only` in cardledger_database, so
its index never answers `ok` without confirmation.
