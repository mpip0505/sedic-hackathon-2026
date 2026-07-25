"""Fixtures for data-pipeline tests: a synthetic schema, images, and interim sets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

_SCHEMA = {
    "classes": {
        0: "container_ship", 1: "tanker", 2: "cargo", 3: "passenger_ferry",
        4: "yacht", 5: "speedboat", 6: "fishing_boat", 7: "military_vessel",
    },
    "groups": {"civilian": [0, 1, 2, 3, 4], "small_craft": [4, 5, 6], "military": [7]},
    "domains": {
        "seaships": "surface",
        "hrsc2016": "aerial",
        "shiprsimagenet": "aerial",
    },
}


@pytest.fixture
def schema_path(tmp_path: Path) -> Path:
    path = tmp_path / "schema.yaml"
    path.write_text(yaml.safe_dump(_SCHEMA), encoding="utf-8")
    return path


def _make_image(path: Path, seed: int, w: int = 64, h: int = 48) -> None:
    """Distinct-per-seed textured image (so perceptual hashes differ)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    grad = np.tile(np.linspace(0, 255, w), (h, 1)) * ((seed % 5 + 1) / 5)
    noise = rng.integers(0, 90, size=(h, w))
    base = ((grad + noise) % 255).astype(np.uint8)
    arr = np.stack([base, np.roll(base, seed % 7, axis=1), base[::-1]], axis=-1)
    Image.fromarray(arr).save(path)


@pytest.fixture
def make_image():
    return _make_image


def _write_label(path: Path, class_ids: list[int]) -> None:
    """One centered full-image box per class id (valid YOLO)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{cid} 0.500000 0.500000 0.900000 0.900000" for cid in class_ids]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def build_interim(tmp_path: Path):
    """Build a synthetic data/interim tree; return (interim_root, expected_kept).

    Includes a cross-dataset duplicate (same pixels in hrsc2016 + shiprsimagenet)
    that dedup must collapse to one image.
    """
    interim = tmp_path / "interim"

    def img(ds: str, stem: str, seed: int) -> None:
        _make_image(interim / ds / "images" / f"{stem}.png", seed)

    def lbl(ds: str, stem: str, classes: list[int]) -> None:
        _write_label(interim / ds / "labels" / f"{stem}.txt", classes)

    # seaships (surface): civilian + small craft.
    for i in range(8):
        img("seaships", f"s{i}", 1000 + i)
        lbl("seaships", f"s{i}", [2, 6] if i % 2 == 0 else [2])

    # hrsc2016 (aerial): military.
    for i in range(8):
        img("hrsc2016", f"h{i}", 2000 + i)
        lbl("hrsc2016", f"h{i}", [7])

    # shiprsimagenet (aerial): military, one of which duplicates hrsc h0.
    for i in range(6):
        img("shiprsimagenet", f"r{i}", 3000 + i)
        lbl("shiprsimagenet", f"r{i}", [7])
    _make_image(interim / "shiprsimagenet" / "images" / "dup.png", 2000)  # == h0
    _write_label(interim / "shiprsimagenet" / "labels" / "dup.txt", [7])

    total = 8 + 8 + 6 + 1
    expected_kept = total - 1  # the duplicate is dropped
    return interim, expected_kept
