"""`cardledger-vision build-index` builds a game's index from its catalog; `upload` copies it to the bucket.

Every command works on all the games unless `--game` names one."""

import argparse
import logging
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

from . import bench, catalog, images
from .build import build_index, index_key
from .index import Index, InvalidIndex

GAMES = ("mtg", "pokemon", "lorcana", "onepiece", "yugioh", "riftbound")

logger = logging.getLogger("cardledger_vision")


class CliError(Exception):
    """A problem to show to the user without a traceback."""


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


def _slugs(args) -> tuple[str, ...]:
    """The games of `--game`: every game when it is left out or is `all`."""
    return GAMES if args.game in (None, "all") else (args.game,)


def _each_game(command: str, slugs, run) -> None:
    """Runs `run(slug)` for each game; one game's failure doesn't stop the others."""
    if len(slugs) == 1:
        run(slugs[0])
        return
    failed = []
    for slug in slugs:
        try:
            run(slug)
        except CliError as exc:
            logger.error("%s: %s", slug, exc)
            failed.append(slug)
    if failed:
        raise CliError(f"{command} failed for: {', '.join(failed)}")


def _build(args) -> None:
    """Builds the index of `--game`, or of every game when it is left out or is `all`."""
    slugs = _slugs(args)
    if args.catalog and len(slugs) > 1:
        raise CliError("--catalog is one game's catalog: pass --game too")
    s3 = None if args.catalog else make_s3()
    _each_game("build-index", slugs, lambda slug: _build_game(slug, args, s3))


def _build_game(slug: str, args, s3) -> None:
    try:
        data, sha = catalog.read(slug, args.catalog, s3)
    except RuntimeError as exc:
        raise CliError(str(exc)) from exc
    with make_client() as client:
        index = build_index(slug, data, sha, client, args.cache, args.confirm_only,
                            host_interval=args.host_interval)
    path = args.out / index_key(slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")  # a reader never sees a half-written index
    try:
        tmp.write_bytes(index)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    logger.info("%s: written %s", slug, path)


def _bench(args) -> None:
    """Benches the index of `--game`, or of every game when it is left out or is `all`."""
    slugs = _slugs(args)
    if (args.catalog or args.index) and len(slugs) > 1:
        raise CliError("--catalog and --index are one game's files: pass --game too")
    s3 = None if args.catalog else make_s3()
    _each_game("bench", slugs, lambda slug: _bench_game(slug, args, s3))


def _bench_game(slug: str, args, s3) -> None:
    index_path = args.index or args.out / index_key(slug)
    try:
        data, _ = catalog.read(slug, args.catalog, s3)
        index = Index.load(index_path.read_bytes())
    except (RuntimeError, OSError, ValueError) as exc:
        raise CliError(str(exc)) from exc
    products = catalog.load(data)
    with make_client() as client:
        samples, skipped = bench.collect(products, args.size, args.seed,
                                         lambda urls: images.fetch(urls, args.cache, client, args.host_interval))
    print(f"== {slug} ==")
    print(bench.run(index, samples, args.seed, skipped).table())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cardledger-vision")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help in (("build-index", "build a game's index from its catalog"),
                       ("upload", "upload a game's index to the S3 bucket")):
        command = commands.add_parser(name, help=help)
        command.add_argument("--game", choices=(*GAMES, "all"), help="the game (default: all)")
        command.add_argument("--out", type=Path, default=Path("out"), help="output folder (default: out)")
        if name == "build-index":
            command.add_argument("--catalog", type=Path,
                                 help="local catalog file of --game (default: read it from the bucket)")
            command.add_argument("--cache", type=Path, default=Path(".cache"), help="cache folder (default: .cache)")
            command.add_argument("--confirm-only", action="store_true",
                                 help="mark the index as never enough to accept a card without confirmation")
            command.add_argument("--host-interval", type=float, default=images.MIN_INTERVAL,
                                 help=f"seconds between two requests to an image host (default: {images.MIN_INTERVAL})")
    command = commands.add_parser("bench", help="measure recognition on synthetic photos of the catalog")
    command.add_argument("--game", choices=(*GAMES, "all"), help="the game (default: all)")
    command.add_argument("--index", type=Path, help="index file of --game (default: the one in --out)")
    command.add_argument("--out", type=Path, default=Path("out"), help="folder of the indexes (default: out)")
    command.add_argument("--catalog", type=Path,
                         help="local catalog file of --game (default: read it from the bucket)")
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
            upload(list(_slugs(args)), args.out)
        return 0
    except CliError as exc:
        logger.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        logger.error("interrupted: no index written for the game in progress (the image cache is kept)")
        return 130
    finally:
        logger.removeHandler(handler)


def run() -> None:
    sys.exit(main())
