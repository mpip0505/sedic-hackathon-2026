"""Tests for yolo2yolo (passthrough remap)."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.data.converters import yolo2yolo


def _build_source(src: Path, make_image) -> None:
    # Source taxonomy (index = source class id).
    (src).mkdir(parents=True, exist_ok=True)
    (src / "data.yaml").write_text(
        yaml.safe_dump({"names": ["warship", "cargo_ship", "drop_me", "mystery"]}),
        encoding="utf-8",
    )
    # Roboflow-style split dirs; merge flattens them anyway.
    for split in ("train", "valid"):
        (src / split / "images").mkdir(parents=True)
        (src / split / "labels").mkdir(parents=True)

    make_image(src / "train" / "images" / "a.jpg", 100, 100)
    (src / "train" / "labels" / "a.txt").write_text(
        "0 0.500000 0.500000 0.200000 0.300000\n"   # warship -> military_vessel(7)
        "1 0.100000 0.100000 0.050000 0.050000\n"   # cargo_ship -> cargo(2)
        "2 0.900000 0.900000 0.100000 0.100000\n"   # drop_me -> null (dropped)
        "3 0.400000 0.400000 0.200000 0.200000\n"   # mystery -> unmapped (dropped)
        "9 0.400000 0.400000 0.200000 0.200000\n",  # out-of-range id (skipped)
        encoding="utf-8",
    )
    make_image(src / "valid" / "images" / "b.jpg", 100, 100)
    (src / "valid" / "labels" / "b.txt").write_text(
        "0 0.250000 0.250000 0.100000 0.100000\n", encoding="utf-8"
    )


def test_yolo_passthrough_remap(tmp_path: Path, schema_path: Path, make_image):
    src = tmp_path / "src"
    dst = tmp_path / "out"
    _build_source(src, make_image)

    stats = yolo2yolo.convert("voc_test", src, dst, schema_path)

    # a: 2 kept boxes; b: 1 kept box -> 2 images, 3 boxes.
    assert stats.images_converted == 2
    assert stats.boxes_written == 3
    assert stats.per_class[7] == 2       # military_vessel (a + b)
    assert stats.per_class[2] == 1       # cargo (a)
    assert stats.dropped_classes["drop_me"] == 1
    assert stats.unmapped_classes["mystery"] == 1

    # Coordinates pass through unchanged; only the class id is remapped.
    lines = (dst / "labels" / "a.txt").read_text().strip().splitlines()
    assert lines == [
        "7 0.500000 0.500000 0.200000 0.300000",
        "2 0.100000 0.100000 0.050000 0.050000",
    ]
    assert (dst / "images" / "a.jpg").is_file()
    assert (dst / "labels" / "b.txt").read_text().strip() == \
        "7 0.250000 0.250000 0.100000 0.100000"


def test_yolo_polygon_enveloped_to_hbb(tmp_path: Path, schema_path: Path, make_image):
    src = tmp_path / "src"
    dst = tmp_path / "out"
    src.mkdir(parents=True)
    (src / "data.yaml").write_text(yaml.safe_dump({"names": ["warship"]}),
                                   encoding="utf-8")
    (src / "images").mkdir()
    (src / "labels").mkdir()
    make_image(src / "images" / "p.jpg", 100, 100)
    # Closed polygon (5 points) with envelope x:[0.2,0.6] y:[0.3,0.7]
    # -> center (0.4,0.5), size (0.4,0.4).
    (src / "labels" / "p.txt").write_text(
        "0 0.4 0.3 0.6 0.5 0.4 0.7 0.2 0.5 0.4 0.3\n", encoding="utf-8")

    stats = yolo2yolo.convert("voc_test", src, dst, schema_path)
    assert stats.boxes_written == 1
    line = (dst / "labels" / "p.txt").read_text().strip()
    assert line == "7 0.400000 0.500000 0.400000 0.400000"
