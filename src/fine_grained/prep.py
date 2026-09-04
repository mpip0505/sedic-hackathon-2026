"""prep.py — build classifier-ready crops for the RMN-vs-Foreign classifier.

Converts the two raw halves in data/raw/fine_grained/ into a common flat
per-class crop layout under `processed_dir`/{train,val}/{malaysian_rmn,foreign}/:

  - malaysian_rmn: already single-ship photos (class subfolders) -> copied as-is.
  - foreign: Roboflow YOLO detection format -> each surviving warship box is
    cropped out (with context padding) into its own image. Native classes in
    `foreign_exclude_classes` (junk: non-military / hull-number-text boxes)
    are skipped.

Split is stratified per source subtype (RMN hull class / foreign native
class), seeded. The foreign TRAIN pool is then capped at
`balance.max_foreign_ratio` times the RMN TRAIN count so the huge foreign:RMN
imbalance (~5000:360) doesn't dominate training on its own.

    python -m src.fine_grained.prep --config configs/fine_grained.yaml
    python -m src.fine_grained.prep --config configs/fine_grained.yaml --limit-per-class 10 --clean

This is entirely separate from the main detector pipeline (src/data/*) — it
does not read or write configs/schema.yaml or data/processed/{images,labels}.
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image

logger = logging.getLogger(__name__)

# src/fine_grained/prep.py -> parents[2] is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "fine_grained.yaml"

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

RMN_LABEL = "malaysian_rmn"
FOREIGN_LABEL = "foreign"


def _repo_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if not isinstance(config, dict):
        raise SystemExit(f"config {path} did not parse to a mapping")
    return config


# ---------------------------------------------------------------------------
# Source item: one image (RMN) or one crop-to-be (foreign box), tagged with
# its source subtype so the split can stratify on it.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RmnItem:
    image_path: Path
    subtype: str  # e.g. kasturi_corvette


@dataclass(frozen=True)
class ForeignItem:
    image_path: Path
    box: tuple[float, float, float, float]  # normalized cx, cy, w, h
    native_class: str
    box_idx: int


# ---------------------------------------------------------------------------
# Gathering
# ---------------------------------------------------------------------------
def gather_rmn_items(rmn_root: Path, limit_per_class: int | None = None) -> list[RmnItem]:
    items: list[RmnItem] = []
    for subtype_dir in sorted(p for p in rmn_root.iterdir() if p.is_dir()):
        subtype = subtype_dir.name
        images = sorted(
            p for p in subtype_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS
        )
        if limit_per_class is not None:
            images = images[:limit_per_class]
        for img in images:
            items.append(RmnItem(image_path=img, subtype=subtype))
        logger.info("rmn subtype %-18s %d images", subtype, len(images))
    return items


def gather_foreign_items(
    foreign_root: Path, exclude_classes: set[str], limit_per_class: int | None = None,
) -> list[ForeignItem]:
    data_yaml_candidates = list(foreign_root.parent.glob("data.yaml"))
    if not data_yaml_candidates:
        raise SystemExit(f"no data.yaml found near {foreign_root}")
    with open(data_yaml_candidates[0], "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    class_names: list[str] = list(data["names"])

    images_dir = foreign_root / "images"
    labels_dir = foreign_root / "labels"

    per_class_count: Counter = Counter()
    items: list[ForeignItem] = []
    # Group by native class first so --limit-per-class caps evenly, then flatten.
    by_class: dict[str, list[ForeignItem]] = defaultdict(list)

    for label_file in sorted(labels_dir.glob("*.txt")):
        img_path = None
        for ext in IMAGE_EXTS:
            candidate = images_dir / f"{label_file.stem}{ext}"
            if candidate.is_file():
                img_path = candidate
                break
        if img_path is None:
            logger.warning("no image found for label %s; skipping", label_file)
            continue

        for box_idx, line in enumerate(label_file.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cid = int(parts[0])
            native_class = class_names[cid]
            if native_class in exclude_classes:
                continue
            cx, cy, w, h = (float(v) for v in parts[1:5])
            by_class[native_class].append(
                ForeignItem(image_path=img_path, box=(cx, cy, w, h),
                            native_class=native_class, box_idx=box_idx)
            )

    for native_class, class_items in sorted(by_class.items()):
        if limit_per_class is not None:
            class_items = class_items[:limit_per_class]
        items.extend(class_items)
        per_class_count[native_class] = len(class_items)
        logger.info("foreign native class %-20s %d boxes", native_class, len(class_items))

    return items


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------
def crop_box(
    img: Image.Image, box: tuple[float, float, float, float],
    pad_frac: float, min_size: int,
) -> Image.Image | None:
    """Crop a normalized (cx, cy, w, h) box out of `img`, with context padding."""
    cx, cy, w, h = box
    iw, ih = img.size
    bw, bh = w * iw, h * ih
    px, py = bw * pad_frac, bh * pad_frac

    x1 = (cx * iw) - bw / 2 - px
    x2 = (cx * iw) + bw / 2 + px
    y1 = (cy * ih) - bh / 2 - py
    y2 = (cy * ih) + bh / 2 + py

    x1, x2 = max(0, int(x1)), min(iw, int(x2))
    y1, y2 = max(0, int(y1)), min(ih, int(y2))

    if x2 - x1 < min_size or y2 - y1 < min_size:
        return None
    return img.crop((x1, y1, x2, y2))


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
def stratified_split(
    items_by_subtype: dict[str, list], val_frac: float, rng: random.Random,
) -> tuple[list, list]:
    """Split each subtype's items independently so both splits cover every subtype."""
    train: list = []
    val: list = []
    for subtype_items in items_by_subtype.values():
        shuffled = list(subtype_items)
        rng.shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * val_frac)) if len(shuffled) > 1 else 0
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])
    return train, val


def _group_by(items: list, key) -> dict[str, list]:
    grouped: dict[str, list] = defaultdict(list)
    for item in items:
        grouped[key(item)].append(item)
    return grouped


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def _unique_dest(dst_dir: Path, stem: str, suffix: str) -> Path:
    candidate = dst_dir / f"{stem}{suffix}"
    i = 1
    while candidate.exists():
        candidate = dst_dir / f"{stem}_{i}{suffix}"
        i += 1
    return candidate


def write_rmn_items(items: list[RmnItem], dst_dir: Path) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for item in items:
        stem = f"{item.subtype}__{item.image_path.stem}"
        dest = _unique_dest(dst_dir, stem, item.image_path.suffix.lower())
        shutil.copy2(item.image_path, dest)
        written += 1
    return written


def write_foreign_items(items: list[ForeignItem], dst_dir: Path, pad_frac: float, min_size: int) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for item in items:
        try:
            with Image.open(item.image_path) as img:
                img = img.convert("RGB")
                crop = crop_box(img, item.box, pad_frac, min_size)
                if crop is None:
                    continue
                slug = item.native_class.lower().replace(" ", "_")
                stem = f"{slug}__{item.image_path.stem}__{item.box_idx}"
                dest = _unique_dest(dst_dir, stem, ".jpg")
                crop.save(dest, "JPEG", quality=95)
                written += 1
        except Exception as exc:  # noqa: BLE001 - malformed images must not crash the run
            logger.warning("failed to crop %s box %d: %s", item.image_path, item.box_idx, exc)
    return written


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run(config: dict, limit_per_class: int | None = None, clean: bool = False) -> dict[str, dict[str, int]]:
    seed = config.get("seed", 42)
    val_frac = config.get("val_frac", 0.15)
    rng = random.Random(seed)

    raw_cfg = config["raw"]
    crop_cfg = config.get("crop", {})
    balance_cfg = config.get("balance", {})
    pad_frac = crop_cfg.get("pad_frac", 0.12)
    min_size = crop_cfg.get("min_size", 24)
    max_foreign_ratio = balance_cfg.get("max_foreign_ratio", 3.0)

    rmn_root = _repo_path(raw_cfg["rmn_root"])
    foreign_root = _repo_path(raw_cfg["foreign_root"])
    exclude_classes = set(raw_cfg.get("foreign_exclude_classes", []))

    processed_dir = _repo_path(config["processed_dir"])
    if clean and processed_dir.is_dir():
        logger.info("--clean: removing %s", processed_dir)
        shutil.rmtree(processed_dir)

    logger.info("=== gathering RMN items from %s ===", rmn_root)
    rmn_items = gather_rmn_items(rmn_root, limit_per_class)
    logger.info("=== gathering foreign items from %s (excluding %s) ===",
                foreign_root, sorted(exclude_classes))
    foreign_items = gather_foreign_items(foreign_root, exclude_classes, limit_per_class)

    rmn_train, rmn_val = stratified_split(
        _group_by(rmn_items, lambda it: it.subtype), val_frac, rng
    )
    foreign_train, foreign_val = stratified_split(
        _group_by(foreign_items, lambda it: it.native_class), val_frac, rng
    )

    # Cap the foreign TRAIN pool relative to RMN TRAIN count. Val is left
    # un-capped (natural distribution) so reported val metrics aren't
    # artificially easy, but still stratified/seeded like everything else.
    cap = int(len(rmn_train) * max_foreign_ratio)
    if cap and len(foreign_train) > cap:
        logger.info("capping foreign TRAIN pool: %d -> %d (max_foreign_ratio=%.1f)",
                    len(foreign_train), cap, max_foreign_ratio)
        rng.shuffle(foreign_train)
        foreign_train = foreign_train[:cap]

    counts: dict[str, dict[str, int]] = {"train": {}, "val": {}}

    counts["train"][RMN_LABEL] = write_rmn_items(rmn_train, processed_dir / "train" / RMN_LABEL)
    counts["val"][RMN_LABEL] = write_rmn_items(rmn_val, processed_dir / "val" / RMN_LABEL)
    counts["train"][FOREIGN_LABEL] = write_foreign_items(
        foreign_train, processed_dir / "train" / FOREIGN_LABEL, pad_frac, min_size
    )
    counts["val"][FOREIGN_LABEL] = write_foreign_items(
        foreign_val, processed_dir / "val" / FOREIGN_LABEL, pad_frac, min_size
    )

    logger.info("=== final crop counts ===")
    for split in ("train", "val"):
        for label in (RMN_LABEL, FOREIGN_LABEL):
            logger.info("  %-5s %-14s %d", split, label, counts[split][label])
    logger.info("output: %s", processed_dir)
    return counts


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(
        prog="python -m src.fine_grained.prep",
        description="Build classifier-ready crops for the RMN-vs-Foreign classifier.",
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    p.add_argument("--limit-per-class", type=int, default=None,
                   help="cap source images/boxes considered per subtype/native-class "
                        "(smoke testing)")
    p.add_argument("--clean", action="store_true",
                   help="remove processed_dir before writing")
    args = p.parse_args(argv)

    config = load_config(args.config)
    run(config, limit_per_class=args.limit_per_class, clean=args.clean)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
