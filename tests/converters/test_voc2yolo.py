"""Tests for voc2yolo."""

from __future__ import annotations

from pathlib import Path

from src.data.converters import voc2yolo


def _write_xml(path: Path, filename: str, w: int, h: int, objects) -> None:
    obj_xml = "".join(
        f"""
        <object>
          <name>{name}</name>
          <bndbox>
            <xmin>{x1}</xmin><ymin>{y1}</ymin>
            <xmax>{x2}</xmax><ymax>{y2}</ymax>
          </bndbox>
        </object>"""
        for name, x1, y1, x2, y2 in objects
    )
    path.write_text(
        f"""<annotation>
          <filename>{filename}</filename>
          <size><width>{w}</width><height>{h}</height><depth>3</depth></size>
          {obj_xml}
        </annotation>""",
        encoding="utf-8",
    )


def _build(src: Path, make_image):
    ann = src / "Annotations"
    img = src / "JPEGImages"
    ann.mkdir(parents=True)
    img.mkdir(parents=True)

    # Well-formed: one mapped (warship), one mapped (cargo_ship), one dropped,
    # one unmapped (mystery), plus a box that runs off the image edge (clamped).
    make_image(img / "a.jpg", 100, 100)
    _write_xml(
        ann / "a.xml", "a.jpg", 100, 100,
        [
            ("warship", 10, 10, 50, 60),
            ("cargo_ship", 20, 20, 40, 40),
            ("drop_me", 0, 0, 10, 10),
            ("mystery", 0, 0, 10, 10),
            ("warship", 80, 80, 500, 500),   # exceeds bounds -> clamped to 100
        ],
    )

    # Degenerate box (zero area) -> dropped, leaving no labels for this image.
    make_image(img / "b.jpg", 100, 100)
    _write_xml(ann / "b.xml", "b.jpg", 100, 100, [("warship", 30, 30, 30, 80)])

    # Malformed XML -> skipped, must not crash.
    (ann / "c.xml").write_text("<annotation><not-closed>", encoding="utf-8")
    make_image(img / "c.jpg", 100, 100)


def test_voc_conversion(tmp_path: Path, schema_path: Path, make_image):
    src = tmp_path / "voc"
    dst = tmp_path / "out"
    _build(src, make_image)

    stats = voc2yolo.convert("voc_test", src, dst, schema_path)

    # a.xml yields 3 boxes (2 warship + 1 cargo); b/c produce nothing usable.
    assert stats.images_converted == 1
    assert stats.boxes_written == 3
    assert stats.per_class[7] == 2       # military_vessel
    assert stats.per_class[2] == 1       # cargo
    assert stats.dropped_classes["drop_me"] == 1
    assert stats.unmapped_classes["mystery"] == 1
    assert stats.images_no_labels == 1   # b.xml (degenerate only)
    assert stats.boxes_degenerate == 1
    assert stats.images_skipped == 1     # c.xml malformed

    label = (dst / "labels" / "a.txt").read_text().strip().splitlines()
    assert len(label) == 3
    assert (dst / "images" / "a.jpg").is_file()

    # Every value normalized within [0, 1] (clamping worked).
    for line in label:
        cid, *coords = line.split()
        assert cid in {"7", "2"}
        assert all(0.0 <= float(v) <= 1.0 for v in coords)


def test_voc_clamped_box_value(tmp_path: Path, schema_path: Path, make_image):
    src = tmp_path / "voc"
    dst = tmp_path / "out"
    ann = src / "Annotations"
    img = src / "JPEGImages"
    ann.mkdir(parents=True)
    img.mkdir(parents=True)
    make_image(img / "x.jpg", 100, 100)
    # Box from (50,50) to (200,200) clamps to (50,50)-(100,100):
    # center (75,75) -> 0.75, size 50 -> 0.5.
    _write_xml(ann / "x.xml", "x.jpg", 100, 100, [("warship", 50, 50, 200, 200)])

    voc2yolo.convert("voc_test", src, dst, schema_path)
    line = (dst / "labels" / "x.txt").read_text().strip()
    cid, cx, cy, bw, bh = line.split()
    assert cid == "7"
    assert float(cx) == 0.75 and float(cy) == 0.75
    assert float(bw) == 0.5 and float(bh) == 0.5
