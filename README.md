# cardledger_vision

Python library that recognizes trading cards in photos by perceptual hash (256-bit pHash).

Commands: `uv sync`, `uv run pytest -q`.

## Bucket settings

The commands read the bucket settings from the environment or from a `.env` file in the working directory
(`cp .env.example .env` and fill it in); a variable already set in the environment wins over `.env`.

## Build an index and measure it

```bash
uv run cardledger-vision build-index                      # every game (same as --game all)
uv run cardledger-vision build-index --game mtg           # reads catalog/v1/mtg.json from the bucket, writes out/vision/v1/mtg.bin
uv run cardledger-vision upload --game mtg
uv run cardledger-vision bench --game mtg --index out/vision/v1/mtg.bin   # --catalog PATH, --size 200, --seed 42
```

`bench` makes synthetic photos (rotation, perspective, light, glare, blur, noise, JPEG) of a seeded sample of the
catalog images and recognizes each without a set lock and with the product's own set as lock. The table counts
`ok`, `wrong_certain` (answered `ok` with the wrong product), `in_candidates`, `not_found` and `no_card`, with the
median and p90 time of `recognize`.

Confirm-only rule: if `wrong_certain` is above 2% for a game, rebuild that game's index with
`build-index --game <slug> --confirm-only`, so it never answers `ok` without confirmation.

Without `--game`, `build-index` builds every game in turn: a game that fails (e.g. its catalog is missing from the
bucket) is logged and skipped, and the command exits with 1 naming it. `--catalog` is one game's file, so it needs
`--game`.
