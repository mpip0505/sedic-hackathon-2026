"""Shared fixtures for converter tests — a synthetic schema + image helper."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from PIL import Image

# A self-contained schema mirroring the real class IDs, plus test-only mappings
# so the tests never depend on (or drift with) configs/schema.yaml.
_SCHEMA = {
    "classes": {
        0: "container_ship",
        1: "tanker",
        2: "cargo",
        3: "passenger_ferry",
        4: "yacht",
        5: "speedboat",
        6: "fishing_boat",
        7: "military_vessel",
    },
    "mappings": {
        "voc_test": {
            "warship": "military_vessel",
            "cargo_ship": "cargo",
            "drop_me": None,          # explicit drop
            # "mystery" is intentionally absent -> unmapped
        },
        "dota_test": {
            "ship": "cargo",
            "plane": None,            # explicit drop
        },
        "cls_test": {
            "*": "military_vessel",   # wildcard: every folder -> military
            "not_a_ship": None,       # except this one
        },
    },
}


@pytest.fixture
def schema_path(tmp_path: Path) -> Path:
    path = tmp_path / "schema.yaml"
    path.write_text(yaml.safe_dump(_SCHEMA), encoding="utf-8")
    return path


@pytest.fixture
def make_image():
    def _make(path: Path, w: int = 100, h: int = 100) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (w, h), color=(10, 20, 30)).save(path)
        return path

    return _make
