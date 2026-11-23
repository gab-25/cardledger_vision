# cardledger_vision

Python library that recognizes trading cards in photos by perceptual hash (256-bit pHash).

Commands: `uv sync`, `uv run pytest -q`.

## Who builds the published indexes

The indexes in the bucket are built by `cardledger-db generate` in
[cardledger_database](https://github.com/gab-25/cardledger_database), right after each catalog, with this library's
`build_index`; `cardledger-db upload` uploads the catalog and its index together. It follows this repository's `main`,
locked to one commit in its `uv.lock`: after changing the hash, the index format or the thresholds here, run
`uv lock --upgrade-package cardledger-vision` there and rebuild the indexes.
The commands below stay for development: build an index, measure it with `bench`, try a change.

```python
from cardledger_vision import build_index, index_key

index = build_index("mtg", catalog_bytes, catalog_sha256, httpx_client, cache_dir, confirm_only=False)
# write it to out / index_key("mtg"), i.e. out/vision/v1/mtg.bin
```

## Bucket settings

The commands read the bucket settings from the environment or from a `.env` file in the working directory
(`cp .env.example .env` and fill it in); a variable already set in the environment wins over `.env`.

## Build an index and measure it

```bash
uv run cardledger-vision build-index                      # every game (same as --game all)
uv run cardledger-vision build-index --game mtg           # reads catalog/v1/mtg.json from the bucket, writes out/vision/v1/mtg.bin
uv run cardledger-vision upload                           # every game's index in out/ (or --game mtg)
uv run cardledger-vision bench                            # every game's index in out/ (or --game mtg)
uv run cardledger-vision bench --game mtg --index out/vision/v1/mtg.bin   # --catalog PATH, --size 200, --seed 42
```

`bench` makes synthetic photos (rotation, perspective, light, glare, blur, noise, JPEG) of a seeded sample of the
catalog images and recognizes each without a set lock and with the product's own set as lock. The table counts
`ok`, `wrong_certain` (answered `ok` with the wrong product), `in_candidates`, `not_found` and `no_card`, with the
median and p90 time of `recognize`.

Confirm-only rule: if `wrong_certain` is above 2% for a game, rebuild that game's index with
`build-index --game <slug> --confirm-only`, so it never answers `ok` without confirmation.

Every command works on all the games when `--game` is left out (same as `--game all`). `build-index` and `bench`
go through the games in turn: a game that fails (e.g. its catalog is missing from the bucket) is logged and skipped,
and the command exits with 1 naming it. `upload` checks every index first and uploads nothing if one is missing or
invalid. `--catalog` and `--index` are one game's file, so they need `--game`.
