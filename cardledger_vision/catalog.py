"""Reads a game's catalog (the JSON file of cardledger_database) and keeps its single cards."""

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class CatalogProduct:
    id_product: int
    name: str
    expansion_code: str
    number: str
    image_url: str | None
    image_large_url: str | None


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
