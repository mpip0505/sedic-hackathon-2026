"""balance.py — oversample + copy-paste augment rare classes (military first).

Boosts recall on scarce classes — the >90% military-recall gate above all — by
generating synthetic TRAIN images with matching YOLO labels. Two modes:

  1. same-domain copy-paste (`copy_paste`): military crops pasted onto other
     backgrounds of the SAME domain.
  2. cross-domain copy-paste (`cross_domain_copy_paste`): aerial military crops
     pasted onto SURFACE backgrounds — this manufactures surface/frontal-view
     military examples to close the surface-military gap (we have 0 real ones),
     which the multi-angle competition requirement needs.

TRAIN SPLIT ONLY. Neither mode reads or writes val/ or test/ — augmenting the
evaluation sets would make reported recall dishonest. Synthetic cross-domain
images are tagged `surface_synth` in domains.json so their effect can be
reported separately and disabled if they hurt.

Parameters are config-driven (a `balance:` block in the train config); CLI flags
override.

    python -m src.data.balance                       # config-driven (cross-domain)
    python -m src.data.balance --cross-copies 300 --clean
    python -m src.data.balance --same-domain          # also same-domain paste
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from . import phash, schema_utils

logger = logging.getLogger(__name__)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
AUG_PREFIX = "augcp_"       # same-domain copy-paste synthetic images
OS_PREFIX = "augos_"        # plain oversampled (duplicated) images
XD_PREFIX = "augxd_"        # cross-domain (aerial->surface) synthetic images
SYNTH_PREFIXES = (AUG_PREFIX, OS_PREFIX, XD_PREFIX)


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
            if p.name.startswith(SYNTH_PREFIXES) and p.suffix in pat:
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
        if label_path.stem.startswith(SYNTH_PREFIXES):
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


def _feather_mask(w: int, h: int, border_frac: float = 0.15) -> Image.Image:
    """Soft-edged alpha mask so pasted crops don't read as hard rectangles."""
    bx = max(1, int(w * border_frac))
    by = max(1, int(h * border_frac))
    ramp_x = np.clip(np.minimum(np.arange(w), np.arange(w)[::-1]) / bx, 0, 1)
    ramp_y = np.clip(np.minimum(np.arange(h), np.arange(h)[::-1]) / by, 0, 1)
    mask = np.minimum(ramp_x[None, :], ramp_y[:, None])
    return Image.fromarray((mask * 255).astype("uint8"), mode="L")


def update_domains_json(processed_root: Path, stem_to_domain: dict[str, str]) -> None:
    """Additively tag synthetic train images with a domain in domains.json."""
    path = processed_root / "domains.json"
    if not path.is_file():
        logger.warning("no domains.json; skipping synthetic domain tagging")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("train", {}).update(stem_to_domain)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("tagged %d synthetic image(s) in domains.json", len(stem_to_domain))


def military_counts_by_domain(processed_root: Path, schema_path: Path) -> Counter:
    """Count military instances in the TRAIN split, grouped by domain.

    Domain comes from domains.json (which now includes any `surface_synth`
    tags), falling back to manifest.csv then surface.
    """
    schema = schema_utils.load_schema(schema_path)
    military = schema_utils.military_class_ids(schema)
    dj = processed_root / "domains.json"
    train_dom = json.loads(dj.read_text()).get("train", {}) if dj.is_file() else {}
    manifest_dom = _read_manifest_domains(processed_root)

    labels_dir = processed_root / "labels" / "train"
    counts: Counter = Counter()
    for lp in sorted(labels_dir.glob("*.txt")):
        dom = train_dom.get(lp.stem) or manifest_dom.get(lp.stem) or schema_utils.SURFACE
        n = sum(1 for ln in lp.read_text(encoding="utf-8").splitlines()
                if ln.strip() and int(float(ln.split()[0])) in military)
        if n:
            counts[dom] += n
    return counts


def cross_domain_copy_paste(
    processed_root: Path,
    schema_path: Path,
    copies: int = 300,
    max_paste: int = 2,
    seed: int = 42,
    source_domain: str = "aerial",
    target_domain: str = "surface",
    synth_tag: str = "surface_synth",
    min_frac: float = 0.04,
    max_frac: float = 0.16,
    respect_horizon: bool = True,
    clean: bool = False,
    leak_threshold: int = 5,
) -> tuple[int, int]:
    """Paste source-domain military crops onto target-domain backgrounds.

    Manufactures synthetic surface-military TRAIN images (aerial military ->
    surface scenes). Returns (images_generated, military_boxes_pasted). TRAIN
    ONLY; val/test untouched.

    A small pasted object barely changes the background's perceptual hash, so a
    synthetic image can coincidentally near-duplicate a val/test image. Each
    candidate is hashed and DROPPED if it falls within `leak_threshold` of any
    val/test image — the gate must never see a synthetic near-copy of held-out
    data.
    """
    schema = schema_utils.load_schema(schema_path)
    military = schema_utils.military_class_ids(schema)
    mil_id = min(military) if military else None
    if mil_id is None:
        raise SystemExit("schema has no military group; nothing to augment")

    images_dir = processed_root / "images" / "train"
    labels_dir = processed_root / "labels" / "train"
    if clean:
        _clean_previous(images_dir, labels_dir)

    # Perceptual hashes of every held-out (val/test) image, to reject synthetic
    # images that would leak into evaluation.
    holdout_hashes: list[int] = []
    for split in ("val", "test"):
        for p in sorted((processed_root / "images" / split).glob("*")):
            if p.suffix.lower() in IMAGE_EXTS:
                h = phash.phash(p)
                if h is not None:
                    holdout_hashes.append(h)

    train, _ = _load_train(processed_root)
    src_instances = [i for i in _collect_military(train, military)
                     if i.domain == source_domain]
    backgrounds = [ti for ti in train if ti.domain == target_domain]
    if not src_instances:
        logger.warning("no %s military instances; nothing to paste", source_domain)
        return 0, 0
    if not backgrounds:
        logger.warning("no %s backgrounds; nothing to paste onto", target_domain)
        return 0, 0

    rng = random.Random(seed)
    generated = 0
    pasted_total = 0
    leak_skipped = 0
    synth_domains: dict[str, str] = {}

    for i in range(copies):
        bg = rng.choice(backgrounds)
        with Image.open(bg.image_path) as im:
            canvas = im.convert("RGB")
        bw, bh = canvas.width, canvas.height

        existing = [
            _yolo_to_pixels(ln, bw, bh)[1:]
            for ln in bg.label_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        new_lines: list[str] = list(bg.label_path.read_text(encoding="utf-8").splitlines())

        pasted = 0
        for _ in range(rng.randint(1, max_paste)):
            inst = rng.choice(src_instances)
            with Image.open(inst.image_path) as im:
                crop = im.convert("RGB").crop(tuple(map(int, inst.box)))
            if crop.width < 2 or crop.height < 2:
                continue
            if rng.random() < 0.5:
                crop = crop.transpose(Image.FLIP_LEFT_RIGHT)

            # Reasonable size for a ship in a surface scene: a modest fraction
            # of the frame width, aspect preserved.
            target_w = rng.uniform(min_frac, max_frac) * bw
            scale = target_w / crop.width
            new_w = max(2, int(crop.width * scale))
            new_h = max(2, int(crop.height * scale))
            if new_w >= bw or new_h >= bh:
                continue
            crop = crop.resize((new_w, new_h), Image.LANCZOS)

            # Respect the horizon: ships sit in the mid/lower band of a surface
            # scene, not in the sky or hard against the bottom edge.
            if respect_horizon:
                y_lo, y_hi = int(0.35 * bh), int(0.80 * bh) - new_h
            else:
                y_lo, y_hi = 0, bh - new_h
            if y_hi <= y_lo:
                y_lo, y_hi = 0, max(0, bh - new_h)

            placed = None
            for _try in range(25):
                px = rng.randint(0, bw - new_w)
                py = rng.randint(y_lo, max(y_lo, y_hi))
                cand = (px, py, px + new_w, py + new_h)
                if all(_iou(cand, e) < 0.3 for e in existing):
                    placed = cand
                    break
            if placed is None:
                continue

            canvas.paste(crop, (placed[0], placed[1]), _feather_mask(new_w, new_h))
            existing.append(placed)
            new_lines.append(_pixels_to_yolo(mil_id, *placed, bw, bh))
            pasted += 1

        if pasted == 0:
            continue

        # Reject if this synthetic image near-duplicates any val/test image.
        cand_hash = phash.phash_pixels(canvas)
        if any(phash.hamming(cand_hash, hh) <= leak_threshold for hh in holdout_hashes):
            leak_skipped += 1
            continue

        stem = f"{XD_PREFIX}{i:05d}"
        canvas.save(images_dir / f"{stem}.jpg")
        (labels_dir / f"{stem}.txt").write_text("\n".join(new_lines) + "\n",
                                                 encoding="utf-8")
        synth_domains[stem] = synth_tag
        generated += 1
        pasted_total += pasted

    update_domains_json(processed_root, synth_domains)
    logger.info("cross-domain: generated %d synthetic %s image(s) from %d %s "
                "military instance(s); %d military boxes pasted; %d dropped as "
                "val/test near-duplicates",
                generated, synth_tag, len(src_instances), source_domain,
                pasted_total, leak_skipped)
    return generated, pasted_total


def _prune_synth_domains(processed_root: Path) -> None:
    """Drop stale synthetic (augxd_/augcp_/augos_) entries from domains.json."""
    path = processed_root / "domains.json"
    if not path.is_file():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    train = data.get("train", {})
    pruned = {k: v for k, v in train.items() if not k.startswith(SYNTH_PREFIXES)}
    if len(pruned) != len(train):
        data["train"] = pruned
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")


def load_balance_config(config_path: Path | None) -> dict:
    """Read the `balance:` block from the train config (empty if absent)."""
    if config_path is None or not Path(config_path).is_file():
        return {}
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    return cfg.get("balance", {}) or {}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(
        prog="python -m src.data.balance",
        description="Copy-paste augment military (incl. cross-domain surface synth) in TRAIN.",
    )
    p.add_argument("--config", type=Path,
                   default=schema_utils.REPO_ROOT / "configs" / "train_baseline.yaml",
                   help="config file with a `balance:` block (default: train_baseline.yaml)")
    p.add_argument("--processed", type=Path, default=schema_utils.DEFAULT_PROCESSED_ROOT)
    p.add_argument("--schema", type=Path, default=schema_utils.DEFAULT_SCHEMA_PATH)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clean", action="store_true",
                   help="remove previously generated synthetic files first")
    p.add_argument("--same-domain", action="store_true",
                   help="also run same-domain copy-paste (config `balance.copies`)")
    p.add_argument("--cross-domain", dest="cross_domain", action="store_true", default=None,
                   help="force cross-domain paste on (overrides config)")
    p.add_argument("--no-cross-domain", dest="cross_domain", action="store_false",
                   help="force cross-domain paste off")
    p.add_argument("--cross-copies", type=int, default=None,
                   help="override number of cross-domain synthetic images")
    args = p.parse_args(argv)

    cfg = load_balance_config(args.config)
    xd = cfg.get("cross_domain", {}) or {}
    # Reject synthetic images within the project's duplicate threshold of any
    # val/test image, matching merge/validate's definition of a duplicate.
    leak_thr = schema_utils.dedup_config(schema_utils.load_schema(args.schema))["threshold"]

    # Clean first (shared across modes) so the "before" baseline is REAL data only.
    if args.clean:
        _clean_previous(args.processed / "images" / "train",
                        args.processed / "labels" / "train")
        _prune_synth_domains(args.processed)

    before = military_counts_by_domain(args.processed, args.schema)
    logger.info("military instances by domain (before): %s", dict(before))

    if args.same_domain:
        copy_paste(args.processed, args.schema,
                   copies=cfg.get("copies"), max_paste=cfg.get("max_paste", 3),
                   seed=args.seed, clean=False)

    enabled = args.cross_domain if args.cross_domain is not None else xd.get("enabled", True)
    if enabled:
        cross_domain_copy_paste(
            args.processed, args.schema,
            copies=args.cross_copies if args.cross_copies is not None
            else xd.get("copies", 300),
            max_paste=xd.get("max_paste", 2),
            seed=args.seed,
            source_domain=xd.get("source_domain", "aerial"),
            target_domain=xd.get("target_domain", "surface"),
            synth_tag=xd.get("synth_tag", "surface_synth"),
            min_frac=xd.get("min_frac", 0.04),
            max_frac=xd.get("max_frac", 0.16),
            respect_horizon=xd.get("respect_horizon", True),
            clean=False,
            leak_threshold=leak_thr,
        )

    after = military_counts_by_domain(args.processed, args.schema)
    logger.info("military instances by domain (after):  %s", dict(after))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
