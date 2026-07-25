"""cls2det.py — classification folder layout -> single full-image YOLO box.

For classification datasets (FGSC-23 style) the source is one directory per
class, each holding cropped chips of that class:

    <src>/<native_class_name>/<chip>.jpg

Each chip becomes one detection image with a single box covering the whole
image, labelled with the schema class its folder maps to. This is how
fine-grained classification chips are folded into the coarse detector's
"military_vessel" (etc.) via schema.yaml — commonly with a ``"*"`` wildcard.

    python -m src.data.converters.cls2det --dataset roboflow_military_ships \\
        --src data/raw/fgsc23 --dst data/interim/fgsc23
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from . import _common
from ._common import ClassMapper, ConversionStats, OutputWriter

logger = logging.getLogger(__name__)


def _iter_images(class_dir: Path):
    for path in sorted(class_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in _common.IMAGE_EXTS:
            yield path


def convert(
    dataset: str,
    src: Path,
    dst: Path,
    schema_path: Path,
) -> ConversionStats:
    """Convert a classification-folder dataset into YOLO layout."""
    mapper = ClassMapper(dataset, schema_path)
    stats = ConversionStats()

    if not src.is_dir():
        raise FileNotFoundError(f"source dir not found: {src}")

    writer = OutputWriter(dst)

    for class_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        native = class_dir.name
        res = mapper.resolve(native)
        # Resolve once per folder, but tally per chip so counts reflect volume.
        for image_path in _iter_images(class_dir):
            stats.record(native, res)
            if res.class_id is None:
                stats.images_no_labels += 1
                continue

            size = _common.read_image_size(image_path)
            if size is None:
                stats.note_skip(image_path, "unreadable image")
                continue
            w, h = size

            clamped = _common.clamp_box(0, 0, w, h, w, h)
            if clamped is None:
                stats.boxes_degenerate += 1
                continue

            line = _common.to_yolo_line(res.class_id, *clamped, w, h)
            # Prefix with the folder name so identical chip names don't collide.
            writer.write(image_path, [line], stem=f"{native}__{image_path.stem}")
            stats.images_converted += 1
            stats.boxes_written += 1
            stats.count_written(res.class_id)

    stats.log_summary(mapper)
    return stats


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        prog="python -m src.data.converters.cls2det",
        description="Classification folders (one dir per class) -> full-image YOLO boxes.",
    )
    _common.add_common_args(parser)
    args = parser.parse_args(argv)

    src, dst = _common.resolve_io(args)
    convert(args.dataset, src, dst, args.schema)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
