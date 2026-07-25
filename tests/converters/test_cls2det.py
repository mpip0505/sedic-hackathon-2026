"""Tests for cls2det."""

from __future__ import annotations

from pathlib import Path

from src.data.converters import cls2det


def test_cls_conversion_wildcard(tmp_path: Path, schema_path: Path, make_image):
    src = tmp_path / "fgsc"
    dst = tmp_path / "out"

    # Two fine-grained warship folders (wildcard -> military_vessel) and one
    # explicitly-dropped folder.
    make_image(src / "Arleigh_Burke" / "1.jpg", 64, 48)
    make_image(src / "Arleigh_Burke" / "2.jpg", 64, 48)
    make_image(src / "Nimitz" / "1.jpg", 120, 90)     # same chip name, diff folder
    make_image(src / "not_a_ship" / "1.jpg", 30, 30)

    stats = cls2det.convert("cls_test", src, dst, schema_path)

    assert stats.images_converted == 3
    assert stats.boxes_written == 3
    assert stats.per_class[7] == 3               # military_vessel via wildcard
    assert stats.dropped_classes["not_a_ship"] == 1
    assert stats.images_no_labels == 1

    # Full-image box: centered, full width/height.
    line = (dst / "labels" / "Arleigh_Burke__1.txt").read_text().strip()
    cid, cx, cy, bw, bh = line.split()
    assert cid == "7"
    assert float(cx) == 0.5 and float(cy) == 0.5
    assert float(bw) == 1.0 and float(bh) == 1.0

    # Folder-prefixed stems prevent collisions between identically named chips.
    assert (dst / "images" / "Arleigh_Burke__1.jpg").is_file()
    assert (dst / "images" / "Nimitz__1.jpg").is_file()
