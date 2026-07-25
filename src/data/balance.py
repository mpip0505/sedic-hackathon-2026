"""balance.py — oversample + copy-paste augment rare classes (military first).

Boosts recall on scarce classes — the >90% military-recall gate above all — by
generating synthetic TRAIN images: military instances are cropped from real
train images and pasted onto other maritime train backgrounds of the SAME
domain, with matching YOLO labels added.

TRAIN SPLIT ONLY. This never reads or writes val/ or test/ — augmenting the
evaluation sets would make reported recall dishonest.

    python -m src.data.balance                      # military copy-paste on train
    python -m src.data.balance --copies 400 --max-paste 3 --clean
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from . import schema_utils

logger = logging.getLogger(__name__)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
AUG_PREFIX = "augcp_"       # copy-paste synthetic images
OS_PREFIX = "augos_"        # plain oversampled (duplicated) images


@dataclass
class TrainImage:
    stem: str
    domain: str
    image_path: Path
    label_path: Path


@dataclass
class Instance:
    """A military box cropped from a real train image."""

    domain: str
    image_path: Path
    box: tuple[float, float, float, float]   # pixel x1,y1,x2,y2


def _find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def _read_manifest_domains(processed_root: Path) -> dict[str, str]:
    manifest = processed_root / "manifest.csv"
    domains: dict[str, str] = {}
    if not manifest.is_file():
        logger.warning("no manifest.csv; defaulting all domains to %s",
                       schema_utils.SURFACE)
        return domains
    with open(manifest, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["split"] == "train":
                domains[row["stem"]] = row["domain"]
    return domains


def _yolo_to_pixels(line: str, w: int, h: int) -> tuple[int, float, float, float, float]:
    cid, cx, cy, bw, bh = line.split()[:5]
    cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
    return int(float(cid)), cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2


def _pixels_to_yolo(cid: int, x1: float, y1: float, x2: float, y2: float,
                    w: int, h: int) -> str:
    return (f"{cid} {((x1 + x2) / 2) / w:.6f} {((y1 + y2) / 2) / h:.6f} "
            f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _clean_previous(train_images: Path, train_labels: Path) -> None:
    removed = 0
    for d, pat in ((train_images, IMAGE_EXTS), (train_labels, (".txt",))):
        for p in d.iterdir():
            if p.name.startswith((AUG_PREFIX, OS_PREFIX)) and p.suffix in pat:
                p.unlink()
                removed += 1
    logger.info("cleaned %d previously-generated file(s)", removed)


def _load_train(processed_root: Path) -> tuple[list[TrainImage], dict[str, str]]:
    images_dir = processed_root / "images" / "train"
    labels_dir = processed_root / "labels" / "train"
    if not labels_dir.is_dir():
        raise SystemExit(f"no train labels at {labels_dir}; run merge first")
    domains = _read_manifest_domains(processed_root)

    train: list[TrainImage] = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        if label_path.stem.startswith((AUG_PREFIX, OS_PREFIX)):
            continue  # never build on top of prior synthetic output
        image_path = _find_image(images_dir, label_path.stem)
        if image_path is None:
            continue
        train.append(TrainImage(
            stem=label_path.stem,
            domain=domains.get(label_path.stem, schema_utils.SURFACE),
            image_path=image_path,
            label_path=label_path,
        ))
    return train, domains


def _collect_military(train: list[TrainImage], military: set[int]) -> list[Instance]:
    instances: list[Instance] = []
    for ti in train:
        with Image.open(ti.image_path) as im:
            w, h = im.width, im.height
        for line in ti.label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            cid, x1, y1, x2, y2 = _yolo_to_pixels(line, w, h)
            if cid in military and x2 - x1 >= 4 and y2 - y1 >= 4:
                instances.append(Instance(ti.domain, ti.image_path, (x1, y1, x2, y2)))
    return instances


def copy_paste(
    processed_root: Path,
    schema_path: Path,
    copies: int | None = None,
    max_paste: int = 3,
    seed: int = 42,
    clean: bool = False,
) -> int:
    """Generate copy-paste-augmented train images. Returns count generated."""
    schema = schema_utils.load_schema(schema_path)
    military = schema_utils.military_class_ids(schema)
    mil_id = min(military) if military else None
    if mil_id is None:
        raise SystemExit("schema has no military group; nothing to augment")

    images_dir = processed_root / "images" / "train"
    labels_dir = processed_root / "labels" / "train"
    if clean:
        _clean_previous(images_dir, labels_dir)

    train, _ = _load_train(processed_root)
    instances = _collect_military(train, military)
    if not instances:
        logger.warning("no military instances found in train; nothing to do")
        return 0

    # Group instances and backgrounds by domain so we paste like onto like.
    inst_by_domain: dict[str, list[Instance]] = {}
    for inst in instances:
        inst_by_domain.setdefault(inst.domain, []).append(inst)
    bg_by_domain: dict[str, list[TrainImage]] = {}
    for ti in train:
        bg_by_domain.setdefault(ti.domain, []).append(ti)

    # Default: roughly double the military instance pool.
    if copies is None:
        copies = len(instances)

    rng = random.Random(seed)
    generated = 0
    for i in range(copies):
        domain = rng.choice([d for d in inst_by_domain if bg_by_domain.get(d)])
        bg = rng.choice(bg_by_domain[domain])
        with Image.open(bg.image_path) as im:
            canvas = im.convert("RGB")
        bw, bh = canvas.width, canvas.height

        existing = [
            _yolo_to_pixels(ln, bw, bh)[1:]
            for ln in bg.label_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        new_lines: list[str] = list(bg.label_path.read_text(encoding="utf-8").splitlines())

        n_paste = rng.randint(1, max_paste)
        pasted = 0
        for _ in range(n_paste):
            inst = rng.choice(inst_by_domain[domain])
            with Image.open(inst.image_path) as im:
                crop = im.convert("RGB").crop(tuple(map(int, inst.box)))
            if crop.width < 2 or crop.height < 2:
                continue
            if rng.random() < 0.5:
                crop = crop.transpose(Image.FLIP_LEFT_RIGHT)

            # Scale to a random fraction of the background width.
            target_w = rng.uniform(0.08, 0.30) * bw
            scale = target_w / crop.width
            new_w = max(2, int(crop.width * scale))
            new_h = max(2, int(crop.height * scale))
            if new_w >= bw or new_h >= bh:
                continue
            crop = crop.resize((new_w, new_h), Image.LANCZOS)

            placed = None
            for _try in range(20):
                px = rng.randint(0, bw - new_w)
                py = rng.randint(0, bh - new_h)
                cand = (px, py, px + new_w, py + new_h)
                if all(_iou(cand, e) < 0.3 for e in existing):
                    placed = cand
                    break
            if placed is None:
                continue

            canvas.paste(crop, (placed[0], placed[1]))
            existing.append(placed)
            new_lines.append(_pixels_to_yolo(mil_id, *placed, bw, bh))
            pasted += 1

        if pasted == 0:
            continue

        stem = f"{AUG_PREFIX}{i:05d}"
        canvas.save(images_dir / f"{stem}.jpg")
        (labels_dir / f"{stem}.txt").write_text("\n".join(new_lines) + "\n",
                                                 encoding="utf-8")
        generated += 1

    logger.info("copy-paste: generated %d synthetic train image(s) from %d "
                "military instance(s)", generated, len(instances))
    return generated


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(
        prog="python -m src.data.balance",
        description="Copy-paste augment rare (military-first) classes in the TRAIN split.",
    )
    p.add_argument("--processed", type=Path, default=schema_utils.DEFAULT_PROCESSED_ROOT)
    p.add_argument("--schema", type=Path, default=schema_utils.DEFAULT_SCHEMA_PATH)
    p.add_argument("--copies", type=int, default=None,
                   help="number of synthetic images (default: ~= military instance count)")
    p.add_argument("--max-paste", type=int, default=3,
                   help="max military instances pasted per synthetic image")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clean", action="store_true",
                   help="remove previously generated augcp_/augos_ files first")
    args = p.parse_args(argv)

    copy_paste(args.processed, args.schema, copies=args.copies,
               max_paste=args.max_paste, seed=args.seed, clean=args.clean)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
