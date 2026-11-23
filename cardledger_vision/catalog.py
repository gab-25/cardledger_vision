"""Reads a game's catalog (the JSON file of cardledger_database) and keeps its single cards."""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CatalogProduct:
    id_product: int
    name: str
    expansion_code: str
    number: str
    image_url: str | None
    image_large_url: str | None


def catalog_key(slug: str) -> str:
    return f"catalog/v1/{slug}.json"


def is_single(category_name: str | None) -> bool:
    return (category_name or "").strip().lower().endswith(("single", "singles"))


def load(data: bytes) -> list[CatalogProduct]:
    """The single cards of the catalog, in file order (sealed products and accessories are left out)."""
    return [
        CatalogProduct(
            id_product=p["id_product"],
            name=p["name"],
            expansion_code=p.get("expansion_code") or "",
            number=p.get("number") or "",
            image_url=p.get("image_url") or None,
            image_large_url=p.get("image_large_url") or None,
        )
        for p in json.loads(data)["products"]
        if is_single(p.get("category_name"))
    ]


def read(slug: str, path: Path | None, s3) -> tuple[bytes, str]:
    """The catalog bytes and their sha256: from `path` if given, else from the bucket.

    A missing bucket setting or an S3 error raises RuntimeError, which the CLI shows as a CliError."""
    if path is not None:
        data = path.read_bytes()
    else:
        bucket = os.environ.get("CARDLEDGER_S3_BUCKET")
        if not bucket:
            raise RuntimeError("CARDLEDGER_S3_BUCKET is not set")
        from botocore.exceptions import BotoCoreError, ClientError

        key = catalog_key(slug)
        try:
            data = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        except (BotoCoreError, ClientError) as exc:
            raise RuntimeError(f"cannot read s3://{bucket}/{key}: {exc}") from exc
    return data, hashlib.sha256(data).hexdigest()
