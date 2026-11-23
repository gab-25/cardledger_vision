"""`cardledger-vision build-index` builds a game's index from its catalog; `upload` copies it to the bucket."""

import argparse
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

from . import bench, catalog, images
from .index import Entry, Header, Index, InvalidIndex, encode

GAMES = ("mtg", "pokemon", "lorcana", "onepiece", "yugioh", "riftbound")

logger = logging.getLogger("cardledger_vision")


class CliError(Exception):
    """A problem to show to the user without a traceback."""


def index_key(slug: str) -> str:
    return f"vision/v1/{slug}.bin"


def make_client() -> httpx.Client:
    return httpx.Client(timeout=30, follow_redirects=True, headers={"User-Agent": "cardledger-vision"})


def make_s3():
    import boto3

    required = ("CARDLEDGER_S3_BUCKET", "CARDLEDGER_S3_ACCESS_KEY_ID", "CARDLEDGER_S3_SECRET_ACCESS_KEY",
                "CARDLEDGER_S3_REGION")
    unset = [name for name in required if not os.environ.get(name)]
    if unset:
        raise CliError(f"{', '.join(unset)} not set")
    return boto3.client("s3", endpoint_url=os.environ.get("CARDLEDGER_S3_ENDPOINT") or None,
                        aws_access_key_id=os.environ["CARDLEDGER_S3_ACCESS_KEY_ID"],
                        aws_secret_access_key=os.environ["CARDLEDGER_S3_SECRET_ACCESS_KEY"],
                        region_name=os.environ["CARDLEDGER_S3_REGION"])


def build_index(slug: str, catalog_bytes: bytes, sha256: str, client: httpx.Client, cache_dir: Path,
                confirm_only: bool = False, host_interval: float = images.MIN_INTERVAL) -> bytes:
    """The index of the game's single cards; a card whose image can't be had enters without a hash."""
    start = time.monotonic()
    products = sorted(catalog.load(catalog_bytes), key=lambda p: p.id_product)
    # Large images are fetched only for the products whose normal image failed.
    primary = images.fetch_hashes([p.image_url for p in products if p.image_url], cache_dir, client, host_interval)
    retry = [p.image_large_url for p in products if p.image_large_url and primary.get(p.image_url) is None]
    fallback = images.fetch_hashes(retry, cache_dir, client, host_interval)
    entries = []
    for p in products:
        h = primary.get(p.image_url) if p.image_url else None
        if h is None and p.image_large_url:
            h = fallback.get(p.image_large_url)
        entries.append(Entry(p.id_product, p.name, p.expansion_code, p.number, h))
    header = Header(slug, datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), sha256, confirm_only)
    logger.info("%s: entries=%d with_hash=%d seconds=%.0f", slug, len(entries),
                sum(e.hash is not None for e in entries), time.monotonic() - start)
    return encode(header, entries)


def upload(slugs: list[str], out: Path, s3=None) -> list[str]:
    """Copies the games' indexes from `out` to the bucket under the same keys; returns the keys uploaded."""
    bucket = os.environ.get("CARDLEDGER_S3_BUCKET")
    if not bucket:
        raise CliError("CARDLEDGER_S3_BUCKET is not set")
    files = {index_key(slug): out / index_key(slug) for slug in slugs}
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise CliError(f"nothing to upload, run build-index first: {', '.join(missing)}")
    contents = {}
    for (key, path), slug in zip(files.items(), slugs):
        data = path.read_bytes()
        try:
            game = Index.load(data).header.game
        except InvalidIndex as exc:
            raise CliError(f"{path} is not a valid index: {exc}") from exc
        if game != slug:
            raise CliError(f"{path} is the index of {game!r}, not {slug!r}")
        contents[key] = data
    if s3 is None:
        s3 = make_s3()
    for key, data in contents.items():
        s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType="application/octet-stream")
        logger.info("uploaded s3://%s/%s", bucket, key)
    return list(files)


def _build(args) -> None:
    s3 = None if args.catalog else make_s3()
    try:
        data, sha = catalog.read(args.game, args.catalog, s3)
    except RuntimeError as exc:
        raise CliError(str(exc)) from exc
    with make_client() as client:
        index = build_index(args.game, data, sha, client, args.cache, args.confirm_only,
                            host_interval=args.host_interval)
    path = args.out / index_key(args.game)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")  # a reader never sees a half-written index
    try:
        tmp.write_bytes(index)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    logger.info("%s: written %s", args.game, path)


def _bench(args) -> None:
    try:
        data, _ = catalog.read(args.game, args.catalog, None if args.catalog else make_s3())
        index = Index.load(args.index.read_bytes())
    except (RuntimeError, OSError, ValueError) as exc:
        raise CliError(str(exc)) from exc
    products = catalog.load(data)
    with make_client() as client:
        samples, skipped = bench.collect(products, args.size, args.seed,
                                         lambda urls: images.fetch(urls, args.cache, client, args.host_interval))
    print(bench.run(index, samples, args.seed, skipped).table())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cardledger-vision")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help in (("build-index", "build a game's index from its catalog"),
                       ("upload", "upload a game's index to the S3 bucket")):
        command = commands.add_parser(name, help=help)
        command.add_argument("--game", required=True, choices=GAMES)
        command.add_argument("--out", type=Path, default=Path("out"), help="output folder (default: out)")
        if name == "build-index":
            command.add_argument("--catalog", type=Path, help="local catalog file (default: read it from the bucket)")
            command.add_argument("--cache", type=Path, default=Path(".cache"), help="cache folder (default: .cache)")
            command.add_argument("--confirm-only", action="store_true",
                                 help="mark the index as never enough to accept a card without confirmation")
            command.add_argument("--host-interval", type=float, default=images.MIN_INTERVAL,
                                 help=f"seconds between two requests to an image host (default: {images.MIN_INTERVAL})")
    command = commands.add_parser("bench", help="measure recognition on synthetic photos of the catalog")
    command.add_argument("--game", required=True, choices=GAMES)
    command.add_argument("--index", required=True, type=Path, help="the game's index file")
    command.add_argument("--catalog", type=Path, help="local catalog file (default: read it from the bucket)")
    command.add_argument("--cache", type=Path, default=Path(".cache"), help="cache folder (default: .cache)")
    command.add_argument("--size", type=int, default=200, help="photos to make (default: 200)")
    command.add_argument("--seed", type=int, default=42, help="sample and photo seed (default: 42)")
    command.add_argument("--host-interval", type=float, default=images.MIN_INTERVAL,
                         help=f"seconds between two requests to an image host (default: {images.MIN_INTERVAL})")
    args = parser.parse_args(argv)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        load_dotenv(Path(".env"))  # the working directory's .env, without overriding the environment
        if args.command == "build-index":
            _build(args)
        elif args.command == "bench":
            _bench(args)
        else:
            upload([args.game], args.out)
        return 0
    except CliError as exc:
        logger.error("%s", exc)
        return 1
    finally:
        logger.removeHandler(handler)


def run() -> None:
    sys.exit(main())
