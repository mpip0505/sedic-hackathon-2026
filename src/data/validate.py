"""validate.py — sanity-check data/processed before training.

Checks, per split (train/val/test):
  - every image has a matching label file (and vice-versa: no orphan labels);
  - no empty label files;
  - every label line is well-formed: 5 fields, class id within schema range,
    and cx/cy/w/h all in [0, 1] with positive width/height;
and across splits:
  - no train/val (or train/test, val/test) leakage — no image appears, or has a
    perceptual near-duplicate, in two splits at once.

Prints a report and exits nonzero if anything fails.

    python -m src.data.validate
    python -m src.data.validate --processed data/processed --leak-threshold 5
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

from . import phash, schema_utils

logger = logging.getLogger(__name__)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
SPLITS = ("train", "val", "test")


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    checked_images: int = 0
    checked_labels: int = 0
    checked_boxes: int = 0

    def fail(self, msg: str) -> None:
        self.errors.append(msg)

    @property
    def ok(self) -> bool:
        return not self.errors


def _find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def _validate_label_file(path: Path, nc: int, report: Report) -> None:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        report.fail(f"empty label file: {path}")
        return
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        report.checked_boxes += 1
        parts = line.split()
        if len(parts) != 5:
            report.fail(f"{path}:{lineno} expected 5 fields, got {len(parts)}")
            continue
        try:
            cid = int(float(parts[0]))
            cx, cy, bw, bh = (float(v) for v in parts[1:])
        except ValueError:
            report.fail(f"{path}:{lineno} non-numeric field: {line!r}")
            continue
        if not 0 <= cid < nc:
            report.fail(f"{path}:{lineno} class id {cid} out of range [0,{nc})")
        for name, val in (("cx", cx), ("cy", cy), ("w", bw), ("h", bh)):
            if not 0.0 <= val <= 1.0:
                report.fail(f"{path}:{lineno} {name}={val} outside [0,1]")
        if bw <= 0 or bh <= 0:
            report.fail(f"{path}:{lineno} non-positive box size w={bw} h={bh}")


def _check_split(processed_root: Path, split: str, nc: int, report: Report) -> None:
    images_dir = processed_root / "images" / split
    labels_dir = processed_root / "labels" / split
    if not images_dir.is_dir() or not labels_dir.is_dir():
        report.fail(f"missing split directory for {split}")
        return

    image_stems = {p.stem for p in images_dir.iterdir()
                   if p.suffix.lower() in IMAGE_EXTS}
    label_stems = {p.stem for p in labels_dir.glob("*.txt")}

    for stem in sorted(image_stems - label_stems):
        report.fail(f"[{split}] image without label: {stem}")
    for stem in sorted(label_stems - image_stems):
        report.fail(f"[{split}] label without image: {stem}")

    report.checked_images += len(image_stems)
    for label_path in sorted(labels_dir.glob("*.txt")):
        report.checked_labels += 1
        _validate_label_file(label_path, nc, report)


def _check_leakage(processed_root: Path, threshold: int, report: Report) -> None:
    """Assert no split shares an image (by stem) or a near-duplicate with another."""
    stems: dict[str, set[str]] = {}
    hashes: dict[str, list[int]] = {}
    paths: dict[str, list[str]] = {}
    for split in SPLITS:
        images_dir = processed_root / "images" / split
        if not images_dir.is_dir():
            continue
        stems[split] = set()
        hashes[split] = []
        paths[split] = []
        for p in sorted(images_dir.iterdir()):
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            stems[split].add(p.stem)
            h = phash.phash(p)
            if h is not None:
                hashes[split].append(h)
                paths[split].append(p.name)

    for a in range(len(SPLITS)):
        for b in range(a + 1, len(SPLITS)):
            sa, sb = SPLITS[a], SPLITS[b]
            if sa not in stems or sb not in stems:
                continue
            for shared in sorted(stems[sa] & stems[sb]):
                report.fail(f"LEAKAGE: {shared} appears in both {sa} and {sb}")
            for i, j, dist in phash.cross_group_near_duplicates(
                hashes[sa], hashes[sb], threshold
            ):
                report.fail(
                    f"LEAKAGE: near-duplicate {paths[sa][i]} ({sa}) ~= "
                    f"{paths[sb][j]} ({sb}) hamming={dist}"
                )


def validate(processed_root: Path, schema_path: Path,
             leak_threshold: int | None = None) -> Report:
    schema = schema_utils.load_schema(schema_path)
    nc = schema_utils.num_classes(schema)
    # Leakage uses the SAME "duplicate" definition as dedup, so we never flag
    # pairs that merge deliberately kept as distinct (would always false-alarm
    # when dedup_threshold < leak_threshold).
    if leak_threshold is None:
        leak_threshold = schema_utils.dedup_config(schema)["threshold"]
    report = Report()
    for split in SPLITS:
        _check_split(processed_root, split, nc, report)
    _check_leakage(processed_root, leak_threshold, report)
    return report


def _print_report(report: Report) -> None:
    logger.info("=== validate report ===")
    logger.info("images checked : %d", report.checked_images)
    logger.info("labels checked : %d", report.checked_labels)
    logger.info("boxes checked  : %d", report.checked_boxes)
    if report.ok:
        logger.info("RESULT: PASS — no problems found")
        return
    logger.error("RESULT: FAIL — %d problem(s):", len(report.errors))
    for msg in report.errors[:50]:
        logger.error("  - %s", msg)
    if len(report.errors) > 50:
        logger.error("  ... and %d more", len(report.errors) - 50)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(
        prog="python -m src.data.validate",
        description="Sanity-check data/processed; exit nonzero on failure.",
    )
    p.add_argument("--processed", type=Path, default=schema_utils.DEFAULT_PROCESSED_ROOT)
    p.add_argument("--schema", type=Path, default=schema_utils.DEFAULT_SCHEMA_PATH)
    p.add_argument("--leak-threshold", type=int, default=None,
                   help="max hamming distance treated as cross-split leakage "
                        "(default: schema `dedup.threshold`, so it matches dedup)")
    args = p.parse_args(argv)

    report = validate(args.processed, args.schema, args.leak_threshold)
    _print_report(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
