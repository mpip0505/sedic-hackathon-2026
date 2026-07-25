"""yolo2yolo.py — YOLO passthrough remapper.

For datasets that already ship in YOLO format but with their OWN class taxonomy
(e.g. Roboflow exports), this rewrites each label's class ID from the source
taxonomy into our schema taxonomy, via configs/schema.yaml.

Horizontal boxes (class cx cy w h) pass through unchanged. Polygon / oriented
labels (YOLO-seg or OBB, class + >=3 points) are collapsed to their horizontal
min/max envelope — horizontal boxes first, per the project plan. Some Roboflow
exports (ShipRSImageNet, military_ships) are polygon-format, so this matters.

Source class names come from the dataset's own `data.yaml` `names:` list (index
= source class ID). Source names mapped to null are dropped; the `"*"` wildcard
is honoured. The source train/valid/test split is flattened — merge.py does its
own split.

    python -m src.data.converters.yolo2yolo --dataset shiprsimagenet
    python -m src.data.converters.yolo2yolo --dataset military_ships \\
        --src data/raw/military_ships --dst data/interim/military_ships
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from . import _common
from ._common import ClassMapper, ConversionStats, OutputWriter

logger = logging.getLogger(__name__)


def _to_hbb(coords: list[str]) -> list[str] | None:
    """Normalize a label's geometry to a horizontal box (cx cy w h) tokens.

    - 4 coords -> already a horizontal box; passthrough unchanged.
    - >=6 even coords -> a polygon / oriented box (YOLO-seg or OBB); collapse to
      the min/max envelope (horizontal boxes first, per the project plan).
    Returns None for a degenerate (zero-area) box; raises ValueError if the
    token count is not a valid geometry.
    """
    n = len(coords)
    if n == 4:
        [float(v) for v in coords]  # validate numeric; keep original strings
        return coords
    if n >= 6 and n % 2 == 0:
        vals = [float(v) for v in coords]
        xs, ys = vals[0::2], vals[1::2]
        x1, x2 = min(max(min(xs), 0.0), 1.0), min(max(max(xs), 0.0), 1.0)
        y1, y2 = min(max(min(ys), 0.0), 1.0), min(max(max(ys), 0.0), 1.0)
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            return None
        return [f"{(x1 + x2) / 2:.6f}", f"{(y1 + y2) / 2:.6f}",
                f"{x2 - x1:.6f}", f"{y2 - y1:.6f}"]
    raise ValueError(f"unexpected coordinate count {n}")


def _load_source_names(names_yaml: Path) -> list[str]:
    """Load the source dataset's own class-name list (index = source class ID)."""
    with open(names_yaml, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    names = data.get("names")
    if names is None:
        raise SystemExit(f"{names_yaml} has no `names:` block")
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return list(names)


def convert(
    dataset: str,
    src: Path,
    dst: Path,
    schema_path: Path,
    names_yaml: Path | None = None,
) -> ConversionStats:
    """Remap a YOLO-format dataset under `src` into schema IDs under `dst`."""
    mapper = ClassMapper(dataset, schema_path)
    source_names = _load_source_names(names_yaml or (src / "data.yaml"))
    stats = ConversionStats()

    # Every `labels/` dir under src (handles train/valid/test and flat layouts).
    label_dirs = sorted({p for p in src.rglob("labels") if p.is_dir()})
    if not label_dirs:
        raise FileNotFoundError(f"no labels/ directory found under {src}")

    writer = OutputWriter(dst)

    for labels_dir in label_dirs:
        images_dir = labels_dir.parent / "images"
        for label_path in sorted(labels_dir.glob("*.txt")):
            image_path = _common.find_image(images_dir, label_path.stem)
            if image_path is None:
                stats.note_skip(label_path, "no matching image file")
                continue

            out_lines: list[str] = []
            for raw in label_path.read_text(encoding="utf-8").splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                parts = raw.split()
                try:
                    sid = int(float(parts[0]))
                except (ValueError, IndexError):
                    logger.warning("bad line in %s: %r", label_path, raw)
                    continue
                if not 0 <= sid < len(source_names):
                    logger.warning("class id %d out of range in %s", sid, label_path)
                    continue

                name = source_names[sid]
                res = mapper.resolve(name)
                stats.record(name, res)
                if res.class_id is None:
                    continue
                # Remap the class ID; passthrough HBB or envelope polygons -> HBB.
                try:
                    coords = _to_hbb(parts[1:])
                except ValueError:
                    logger.warning("malformed geometry in %s: %r", label_path, raw)
                    continue
                if coords is None:
                    stats.boxes_degenerate += 1
                    continue
                out_lines.append(" ".join([str(res.class_id), *coords]))
                stats.count_written(res.class_id)

            if not out_lines:
                stats.images_no_labels += 1
                continue

            writer.write(image_path, out_lines)
            stats.images_converted += 1
            stats.boxes_written += len(out_lines)

    stats.log_summary(mapper)
    return stats


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        prog="python -m src.data.converters.yolo2yolo",
        description="YOLO passthrough remap: source taxonomy -> schema class IDs.",
    )
    _common.add_common_args(parser)
    parser.add_argument("--names-yaml", type=Path, default=None,
                        help="source class-name yaml (default: <src>/data.yaml)")
    args = parser.parse_args(argv)

    src, dst = _common.resolve_io(args)
    convert(args.dataset, src, dst, args.schema, args.names_yaml)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
