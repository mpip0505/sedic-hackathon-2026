"""voc2yolo.py — Pascal VOC XML -> YOLO txt.

Handles SeaShips and ShipRSImageNet (both ship in VOC layout: an Annotations/
dir of `*.xml` alongside a JPEGImages/ dir). Class names are translated through
configs/schema.yaml; nothing is hardcoded.

    python -m src.data.converters.voc2yolo --dataset seaships
    python -m src.data.converters.voc2yolo --dataset shiprsimagenet \\
        --src data/raw/shiprsimagenet --dst data/interim/shiprsimagenet
"""

from __future__ import annotations

import argparse
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

from . import _common
from ._common import ClassMapper, ConversionStats, OutputWriter

logger = logging.getLogger(__name__)


def _image_for(xml_root: ET.Element, xml_path: Path, img_dir: Path) -> Path | None:
    """Locate the image referenced by a VOC annotation."""
    filename = (xml_root.findtext("filename") or "").strip()
    if filename:
        candidate = img_dir / filename
        if candidate.is_file():
            return candidate
    # Fall back to matching the annotation stem against known extensions.
    return _common.find_image(img_dir, xml_path.stem)


def _size_from_xml(xml_root: ET.Element) -> tuple[int, int] | None:
    size = xml_root.find("size")
    if size is None:
        return None
    try:
        w = int(float(size.findtext("width")))   # type: ignore[arg-type]
        h = int(float(size.findtext("height")))   # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return (w, h) if w > 0 and h > 0 else None


def convert(
    dataset: str,
    src: Path,
    dst: Path,
    schema_path: Path,
    ann_dir: Path | None = None,
    img_dir: Path | None = None,
) -> ConversionStats:
    """Convert a VOC dataset under `src` into YOLO layout under `dst`."""
    mapper = ClassMapper(dataset, schema_path)
    stats = ConversionStats()
    ann_dir = ann_dir or (src / "Annotations")
    img_dir = img_dir or (src / "JPEGImages")

    if not ann_dir.is_dir():
        raise FileNotFoundError(f"annotations dir not found: {ann_dir}")

    writer = OutputWriter(dst)

    for xml_path in sorted(ann_dir.glob("*.xml")):
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as exc:
            stats.note_skip(xml_path, f"malformed XML ({exc})")
            continue

        image_path = _image_for(root, xml_path, img_dir)
        if image_path is None:
            stats.note_skip(xml_path, "no matching image file")
            continue

        size = _size_from_xml(root) or _common.read_image_size(image_path)
        if size is None:
            stats.note_skip(xml_path, "no usable image size")
            continue
        w, h = size

        lines: list[str] = []
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip()
            res = mapper.resolve(name)
            stats.record(name, res)
            if res.class_id is None:
                continue

            box = obj.find("bndbox")
            if box is None:
                continue
            try:
                x1 = float(box.findtext("xmin"))   # type: ignore[arg-type]
                y1 = float(box.findtext("ymin"))   # type: ignore[arg-type]
                x2 = float(box.findtext("xmax"))   # type: ignore[arg-type]
                y2 = float(box.findtext("ymax"))   # type: ignore[arg-type]
            except (TypeError, ValueError):
                logger.warning("bad bndbox in %s (object %r)", xml_path, name)
                continue

            clamped = _common.clamp_box(x1, y1, x2, y2, w, h)
            if clamped is None:
                stats.boxes_degenerate += 1
                continue
            lines.append(_common.to_yolo_line(res.class_id, *clamped, w, h))
            stats.count_written(res.class_id)

        if not lines:
            stats.images_no_labels += 1
            continue

        writer.write(image_path, lines)
        stats.images_converted += 1
        stats.boxes_written += len(lines)

    stats.log_summary(mapper)
    return stats


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        prog="python -m src.data.converters.voc2yolo",
        description="Pascal VOC XML -> YOLO txt (SeaShips, ShipRSImageNet).",
    )
    _common.add_common_args(parser)
    parser.add_argument("--ann-dir", type=Path, default=None,
                        help="annotations dir (default: <src>/Annotations)")
    parser.add_argument("--img-dir", type=Path, default=None,
                        help="images dir (default: <src>/JPEGImages)")
    args = parser.parse_args(argv)

    src, dst = _common.resolve_io(args)
    convert(args.dataset, src, dst, args.schema, args.ann_dir, args.img_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
