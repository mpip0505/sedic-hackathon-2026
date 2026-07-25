"""dota2yolo.py — DOTA oriented-box txt -> YOLO horizontal boxes.

DOTA labels are one `labelTxt/<stem>.txt` per `images/<stem>.<ext>`. Each object
line is 8 polygon coordinates plus a category and difficulty flag:

    x1 y1 x2 y2 x3 y3 x4 y4 category difficult

The first one or two lines may be metadata (``imagesource:...``, ``gsd:...``);
they have too few tokens and are skipped. Each oriented polygon is converted to
a horizontal box by taking the min/max envelope of its four corners.

    python -m src.data.converters.dota2yolo --dataset dota
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from . import _common
from ._common import ClassMapper, ConversionStats, OutputWriter

logger = logging.getLogger(__name__)

# 8 coords + category (+ optional difficulty) => at least 9 tokens.
_MIN_TOKENS = 9


def _parse_line(line: str) -> tuple[list[float], str] | None:
    """Parse one DOTA object line into (8 coords, category). None if not a box."""
    parts = line.split()
    if len(parts) < _MIN_TOKENS:
        return None  # metadata or blank line
    try:
        coords = [float(v) for v in parts[:8]]
    except ValueError:
        return None  # not a coordinate line
    category = parts[8]
    return coords, category


def convert(
    dataset: str,
    src: Path,
    dst: Path,
    schema_path: Path,
    label_dir: Path | None = None,
    img_dir: Path | None = None,
) -> ConversionStats:
    """Convert a DOTA dataset under `src` into YOLO layout under `dst`."""
    mapper = ClassMapper(dataset, schema_path)
    stats = ConversionStats()
    label_dir = label_dir or (src / "labelTxt")
    img_dir = img_dir or (src / "images")

    if not label_dir.is_dir():
        raise FileNotFoundError(f"label dir not found: {label_dir}")

    writer = OutputWriter(dst)

    for txt_path in sorted(label_dir.glob("*.txt")):
        image_path = _common.find_image(img_dir, txt_path.stem)
        if image_path is None:
            stats.note_skip(txt_path, "no matching image file")
            continue

        size = _common.read_image_size(image_path)
        if size is None:
            stats.note_skip(txt_path, "unreadable image")
            continue
        w, h = size

        try:
            raw_lines = txt_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            stats.note_skip(txt_path, f"unreadable label file ({exc})")
            continue

        lines: list[str] = []
        for raw in raw_lines:
            parsed = _parse_line(raw)
            if parsed is None:
                continue
            coords, category = parsed
            res = mapper.resolve(category)
            stats.record(category, res)
            if res.class_id is None:
                continue

            xs = coords[0::2]
            ys = coords[1::2]
            clamped = _common.clamp_box(min(xs), min(ys), max(xs), max(ys), w, h)
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
        prog="python -m src.data.converters.dota2yolo",
        description="DOTA oriented-box txt -> YOLO horizontal boxes (min/max envelope).",
    )
    _common.add_common_args(parser)
    parser.add_argument("--label-dir", type=Path, default=None,
                        help="label dir (default: <src>/labelTxt)")
    parser.add_argument("--img-dir", type=Path, default=None,
                        help="images dir (default: <src>/images)")
    args = parser.parse_args(argv)

    src, dst = _common.resolve_io(args)
    convert(args.dataset, src, dst, args.schema, args.label_dir, args.img_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
