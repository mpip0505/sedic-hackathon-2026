"""Shared helpers for the dataset converters.

Every converter reads its source->schema class mapping from configs/schema.yaml
via `ClassMapper` — no class names are hardcoded anywhere. Converters produce
YOLO-format output (one `labels/<stem>.txt` per copied `images/<stem>.<ext>`),
each label line being ``class_id cx cy w h`` normalized to [0, 1].
"""

from __future__ import annotations

import logging
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from PIL import Image

logger = logging.getLogger(__name__)

# src/data/converters/_common.py -> parents[3] is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCHEMA_PATH = _REPO_ROOT / "configs" / "schema.yaml"
DEFAULT_RAW_ROOT = _REPO_ROOT / "data" / "raw"
DEFAULT_INTERIM_ROOT = _REPO_ROOT / "data" / "interim"

WILDCARD = "*"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# Sentinel distinguishing "key absent from mapping" from "value is null (drop)".
_MISSING = object()

# Resolution reasons.
MAPPED = "mapped"
DROPPED = "dropped"      # native class explicitly mapped to null
UNMAPPED = "unmapped"    # no rule and no wildcard — unexpected, should be logged


@dataclass(frozen=True)
class Resolution:
    """Outcome of resolving a native class name against the schema mapping."""

    class_id: int | None
    reason: str  # MAPPED | DROPPED | UNMAPPED


# ---------------------------------------------------------------------------
# Schema access
# ---------------------------------------------------------------------------
def load_schema(schema_path: Path | str = DEFAULT_SCHEMA_PATH) -> dict:
    """Load and return the parsed schema.yaml."""
    with open(schema_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class ClassMapper:
    """Resolves a dataset's native class names into schema class IDs.

    Reads `classes` and `mappings.<dataset>` from schema.yaml. Values may be a
    schema class name, `null` (drop the annotation), or the wildcard key ``"*"``
    which catches any native name not listed explicitly.
    """

    def __init__(self, dataset: str, schema_path: Path | str = DEFAULT_SCHEMA_PATH):
        schema = load_schema(schema_path)
        classes: dict[int, str] = schema["classes"]
        self.id_to_name: dict[int, str] = dict(classes)
        self.name_to_id: dict[str, int] = {name: cid for cid, name in classes.items()}

        mappings = schema.get("mappings") or {}
        if dataset not in mappings:
            raise KeyError(
                f"No mapping for dataset {dataset!r} in {schema_path}. "
                f"Known datasets: {sorted(mappings)}"
            )
        self.dataset = dataset
        self._map: dict[str, str | None] = mappings[dataset] or {}
        self._wildcard = self._map.get(WILDCARD, _MISSING)

        # Fail fast on config errors: every non-null target must be a real class.
        for native, target in self._map.items():
            if target is not None and target not in self.name_to_id:
                raise ValueError(
                    f"Mapping {dataset}.{native!r} -> {target!r} is not a schema "
                    f"class. Valid classes: {sorted(self.name_to_id)}"
                )

    def resolve(self, native_name: str) -> Resolution:
        """Resolve a native class name to a `Resolution`."""
        target = self._map.get(native_name, _MISSING)
        if target is _MISSING:
            target = self._wildcard
        if target is _MISSING:
            return Resolution(None, UNMAPPED)
        if target is None:
            return Resolution(None, DROPPED)
        return Resolution(self.name_to_id[target], MAPPED)


# ---------------------------------------------------------------------------
# Conversion bookkeeping
# ---------------------------------------------------------------------------
@dataclass
class ConversionStats:
    """Accumulates counts across a conversion run and logs a final summary."""

    images_converted: int = 0
    images_skipped: int = 0        # malformed / unreadable / missing image
    images_no_labels: int = 0      # parsed fine, but no box survived
    boxes_written: int = 0
    boxes_degenerate: int = 0      # zero width/height after clamping
    per_class: Counter = field(default_factory=Counter)     # class_id -> count
    dropped_classes: Counter = field(default_factory=Counter)   # native -> count
    unmapped_classes: Counter = field(default_factory=Counter)  # native -> count
    skipped_files: list[str] = field(default_factory=list)

    def record(self, native_name: str, res: Resolution) -> None:
        """Tally a resolved annotation's disposition (dropped / unmapped).

        Mapped classes are counted in `per_class` only when a box is actually
        written (see `count_written`), so degenerate boxes don't inflate totals.
        """
        if res.reason == DROPPED:
            self.dropped_classes[native_name] += 1
        elif res.reason == UNMAPPED:
            self.unmapped_classes[native_name] += 1

    def count_written(self, class_id: int) -> None:
        """Record that one box was written for `class_id`."""
        self.per_class[class_id] += 1

    def note_skip(self, path: Path, reason: str) -> None:
        """Log and record a file that could not be converted."""
        logger.warning("skipping %s: %s", path, reason)
        self.skipped_files.append(str(path))
        self.images_skipped += 1

    def log_summary(self, mapper: ClassMapper) -> None:
        """Emit the end-of-run summary via the logging module."""
        logger.info("=== conversion summary: %s ===", mapper.dataset)
        logger.info("images converted : %d", self.images_converted)
        logger.info("images skipped   : %d (malformed/missing)", self.images_skipped)
        logger.info("images no labels : %d (all boxes dropped)", self.images_no_labels)
        logger.info("boxes written    : %d", self.boxes_written)
        logger.info("boxes degenerate : %d (dropped)", self.boxes_degenerate)
        logger.info("per-class counts :")
        for cid in sorted(self.per_class):
            logger.info("    %-16s %d", mapper.id_to_name[cid], self.per_class[cid])
        if self.dropped_classes:
            logger.info("dropped (mapped to null):")
            for name, n in sorted(self.dropped_classes.items()):
                logger.info("    %-16s %d", name, n)
        if self.unmapped_classes:
            logger.warning("UNMAPPED native classes (no rule, no wildcard):")
            for name, n in sorted(self.unmapped_classes.items()):
                logger.warning("    %-16s %d", name, n)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def clamp_box(
    x1: float, y1: float, x2: float, y2: float, w: int, h: int
) -> tuple[float, float, float, float] | None:
    """Order + clamp a box to image bounds. Return None if degenerate."""
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = min(max(x1, 0.0), w)
    x2 = min(max(x2, 0.0), w)
    y1 = min(max(y1, 0.0), h)
    y2 = min(max(y2, 0.0), h)
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None
    return x1, y1, x2, y2


def to_yolo_line(
    class_id: int, x1: float, y1: float, x2: float, y2: float, w: int, h: int
) -> str:
    """Format one YOLO label line (normalized cx cy bw bh)."""
    cx = ((x1 + x2) / 2) / w
    cy = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def find_image(img_dir: Path, stem: str) -> Path | None:
    """Find an image file for `stem` in `img_dir`, trying known extensions."""
    for ext in IMAGE_EXTS:
        for candidate in (img_dir / f"{stem}{ext}", img_dir / f"{stem}{ext.upper()}"):
            if candidate.is_file():
                return candidate
    return None


def read_image_size(path: Path) -> tuple[int, int] | None:
    """Return (width, height) for an image, or None if it can't be read."""
    try:
        with Image.open(path) as im:
            return im.width, im.height
    except Exception as exc:  # noqa: BLE001 - malformed images must not crash
        logger.warning("cannot read image %s: %s", path, exc)
        return None


class OutputWriter:
    """Writes the YOLO `images/` + `labels/` layout, keeping stems unique."""

    def __init__(self, dst: Path):
        self.images_dir = dst / "images"
        self.labels_dir = dst / "labels"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self._used: set[str] = set()

    def _unique_stem(self, stem: str) -> str:
        candidate = stem
        i = 1
        while candidate in self._used:
            candidate = f"{stem}_{i}"
            i += 1
        self._used.add(candidate)
        return candidate

    def write(self, image_path: Path, lines: list[str], stem: str | None = None) -> None:
        """Copy an image and write its label file with matching stem."""
        stem = self._unique_stem(stem or image_path.stem)
        shutil.copy2(image_path, self.images_dir / f"{stem}{image_path.suffix}")
        (self.labels_dir / f"{stem}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )


def add_common_args(parser, dataset_default: str | None = None) -> None:
    """Register the CLI arguments shared by every converter."""
    parser.add_argument(
        "--dataset", default=dataset_default, required=dataset_default is None,
        help="schema.yaml mappings key selecting the source->schema mapping",
    )
    parser.add_argument(
        "--src", type=Path, default=None,
        help="source dataset root (default: data/raw/<dataset>)",
    )
    parser.add_argument(
        "--dst", type=Path, default=None,
        help="output root (default: data/interim/<dataset>)",
    )
    parser.add_argument(
        "--schema", type=Path, default=DEFAULT_SCHEMA_PATH,
        help="path to schema.yaml (default: configs/schema.yaml)",
    )


def resolve_io(args) -> tuple[Path, Path]:
    """Fill in default src/dst from the dataset name."""
    src = args.src or (DEFAULT_RAW_ROOT / args.dataset)
    dst = args.dst or (DEFAULT_INTERIM_ROOT / args.dataset)
    return src, dst
