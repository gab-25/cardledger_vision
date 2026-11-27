"""Reads a game's catalog (the JSON file of cardledger_database) and keeps its cards."""

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class CatalogProduct:
    id: str
    name: str
    set_code: str
    number: str
    image: str | None  # path relative to the image root (the bucket root)
    image_large: str | None


def load(data: bytes) -> list[CatalogProduct]:
    """The cards of the catalog (rows of kind "card"), in file order; sealed products are left out."""
    return [
        CatalogProduct(
            id=r["id"],
            name=r["name"],
            set_code=r.get("set_code") or "",
            number=r.get("number") or "",
            image=r.get("image") or None,
            image_large=r.get("image_large") or None,
        )
        for r in json.loads(data)["rows"]
        if r.get("kind") == "card"
    ]
