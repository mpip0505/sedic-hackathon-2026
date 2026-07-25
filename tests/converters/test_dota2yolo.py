"""Tests for dota2yolo."""

from __future__ import annotations

from pathlib import Path

from src.data.converters import dota2yolo


def test_dota_conversion(tmp_path: Path, schema_path: Path, make_image):
    src = tmp_path / "dota"
    dst = tmp_path / "out"
    labels = src / "labelTxt"
    images = src / "images"
    labels.mkdir(parents=True)
    images.mkdir(parents=True)

    make_image(images / "img1.png", 100, 100)
    # Two metadata lines, one rotated ship polygon, one dropped 'plane',
    # one malformed line (should be skipped, not crash).
    (labels / "img1.txt").write_text(
        "imagesource:GoogleEarth\n"
        "gsd:0.5\n"
        # A diamond polygon whose envelope is (20,10)-(60,50):
        "40 10 60 30 40 50 20 30 ship 0\n"
        "0 0 10 0 10 10 0 10 plane 0\n"
        "garbage line here\n",
        encoding="utf-8",
    )

    # Image with only a dropped class -> no labels written.
    make_image(images / "img2.png", 80, 80)
    (labels / "img2.txt").write_text("0 0 10 0 10 10 0 10 plane 0\n", encoding="utf-8")

    # Label file with no matching image -> skipped.
    (labels / "orphan.txt").write_text("0 0 10 0 10 10 0 10 ship 0\n", encoding="utf-8")

    stats = dota2yolo.convert("dota_test", src, dst, schema_path)

    assert stats.images_converted == 1
    assert stats.boxes_written == 1
    assert stats.per_class[2] == 1               # cargo (ship -> cargo)
    assert stats.dropped_classes["plane"] == 2   # img1 + img2
    assert stats.images_no_labels == 1           # img2
    assert stats.images_skipped == 1             # orphan.txt

    line = (dst / "labels" / "img1.txt").read_text().strip()
    cid, cx, cy, bw, bh = line.split()
    assert cid == "2"
    # Envelope (20,10)-(60,50): center (40,30) -> (0.4, 0.3), size (40,40) -> 0.4.
    assert float(cx) == 0.4 and float(cy) == 0.3
    assert float(bw) == 0.4 and float(bh) == 0.4
