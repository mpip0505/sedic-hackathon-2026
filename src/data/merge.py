"""merge.py — unify converted datasets into data/processed and split.

Reads every ``data/interim/<dataset>/`` (images/ + labels/ in YOLO layout,
class IDs already in the schema taxonomy), then:

  1. tags each image with its visual domain (surface|aerial) from schema.yaml;
  2. drops cross-dataset near-duplicates via perceptual hash BEFORE splitting,
     so a duplicate can never straddle train and val;
  3. produces a seeded, stratified 70/20/10 split balanced across BOTH class
     (rarest-class-first, so military_vessel stays represented) AND domain;
  4. writes data/processed/{images,labels}/{train,val,test} + a manifest;
  5. regenerates configs/data.yaml from configs/schema.yaml.

    python -m src.data.merge
    python -m src.data.merge --val 0.2 --test 0.1 --seed 42
    python -m src.data.merge --data-yaml-only   # just resync data.yaml
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from . import phash, schema_utils

logger = logging.getLogger(__name__)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
SPLITS = ("train", "val", "test")


@dataclass
class Item:
    dataset: str
    domain: str
    image_path: Path
    label_path: Path
    stem: str                      # unique output stem: "<dataset>__<origstem>"
    classes: frozenset[int]
    num_boxes: int
    hash: int | None = None
    split: str = ""


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------
def _image_for_label(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def _parse_label(path: Path) -> tuple[frozenset[int], int]:
    classes: set[int] = set()
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        classes.add(int(float(line.split()[0])))
        n += 1
    return frozenset(classes), n


def collect_items(interim_root: Path, schema: dict, default_domain: str) -> list[Item]:
    """Gather every labelled image across all interim datasets."""
    items: list[Item] = []
    for ds_dir in sorted(p for p in interim_root.iterdir() if p.is_dir()):
        dataset = ds_dir.name
        domain = schema_utils.domain_for(schema, dataset, default_domain)
        images_dir = ds_dir / "images"
        labels_dir = ds_dir / "labels"
        if not labels_dir.is_dir():
            logger.warning("no labels/ in %s — skipping dataset", ds_dir)
            continue

        for label_path in sorted(labels_dir.glob("*.txt")):
            image_path = _image_for_label(images_dir, label_path.stem)
            if image_path is None:
                logger.warning("label %s has no image — skipping", label_path)
                continue
            classes, n = _parse_label(label_path)
            if n == 0:
                logger.warning("empty label %s — skipping", label_path)
                continue
            items.append(
                Item(
                    dataset=dataset,
                    domain=domain,
                    image_path=image_path,
                    label_path=label_path,
                    stem=f"{dataset}__{label_path.stem}",
                    classes=classes,
                    num_boxes=n,
                )
            )
    logger.info("collected %d labelled images across %d datasets",
                len(items), len({it.dataset for it in items}))
    return items


# ---------------------------------------------------------------------------
# Deduplication (before splitting)
# ---------------------------------------------------------------------------
def deduplicate(
    items: list[Item], threshold: int, military: set[int]
) -> tuple[list[Item], list[tuple[Item, Item, int]]]:
    """Drop near-duplicate images, keeping one representative per cluster.

    Representative = most boxes, then prefers an image containing military
    (so we never drop the richer / military-bearing copy), then lexicographic
    stem for determinism. Returns (kept, dropped_log).
    """
    for it in items:
        it.hash = phash.phash(it.image_path)

    hashable = [it for it in items if it.hash is not None]
    unhashable = [it for it in items if it.hash is None]
    if unhashable:
        logger.warning("%d image(s) could not be hashed; kept without dedup",
                       len(unhashable))

    clusters = phash.cluster_near_duplicates([it.hash for it in hashable], threshold)
    dropped_idx: set[int] = set()
    dropped_log: list[tuple[Item, Item, int]] = []

    def rank(it: Item) -> tuple:
        return (it.num_boxes, 1 if it.classes & military else 0, it.stem)

    for cluster in clusters:
        rep = max((hashable[i] for i in cluster), key=rank)
        rep_i = next(i for i in cluster if hashable[i] is rep)
        for i in cluster:
            if i == rep_i:
                continue
            dropped_idx.add(i)
            dropped_log.append(
                (hashable[i], rep, phash.hamming(hashable[i].hash, rep.hash))
            )

    kept = [it for k, it in enumerate(hashable) if k not in dropped_idx] + unhashable
    for dup, rep, dist in dropped_log:
        logger.info("dup dropped: %s ~= %s (hamming=%d, datasets %s/%s)",
                    dup.stem, rep.stem, dist, dup.dataset, rep.dataset)
    logger.info("dedup: dropped %d near-duplicate image(s), kept %d",
                len(dropped_log), len(kept))
    return kept, dropped_log


# ---------------------------------------------------------------------------
# Stratified split (class x domain)
# ---------------------------------------------------------------------------
def _primary_class(item: Item, freq: Counter) -> int:
    """The rarest class present in the image (keeps rare classes balanced)."""
    return min(item.classes, key=lambda c: (freq[c], c))


def _allocate(n: int, train: float, val: float, test: float) -> tuple[int, int, int]:
    """Split `n` into (train, val, test) via largest-remainder rounding.

    Preserves the ratios far better than per-stratum floor(), so small strata
    (e.g. a scarce domain) still get val/test representation instead of all
    landing in train.
    """
    ratios = (train, val, test)
    raw = [n * r for r in ratios]
    counts = [int(x) for x in raw]
    remainder = n - sum(counts)
    order = sorted(range(3), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in range(remainder):
        counts[order[i]] += 1
    return counts[0], counts[1], counts[2]


def stratified_split(
    items: list[Item], val: float, test: float, seed: int
) -> None:
    """Assign each item a split, stratifying over (domain, rarest-class)."""
    train = 1.0 - val - test
    freq: Counter = Counter()
    for it in items:
        for c in it.classes:
            freq[c] += 1

    strata: dict[tuple[str, int], list[Item]] = defaultdict(list)
    for it in items:
        strata[(it.domain, _primary_class(it, freq))].append(it)

    rng = random.Random(seed)
    for key in sorted(strata):
        group = sorted(strata[key], key=lambda it: it.stem)
        rng.shuffle(group)
        n_train, n_val, _ = _allocate(len(group), train, val, test)
        for it in group[:n_train]:
            it.split = "train"
        for it in group[n_train:n_train + n_val]:
            it.split = "val"
        for it in group[n_train + n_val:]:
            it.split = "test"


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def _reset_processed(processed_root: Path) -> None:
    for split in SPLITS:
        for kind in ("images", "labels"):
            d = processed_root / kind / split
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)


def write_split(items: list[Item], processed_root: Path) -> None:
    _reset_processed(processed_root)
    for it in items:
        img_out = processed_root / "images" / it.split / f"{it.stem}{it.image_path.suffix}"
        lbl_out = processed_root / "labels" / it.split / f"{it.stem}.txt"
        shutil.copy2(it.image_path, img_out)
        shutil.copy2(it.label_path, lbl_out)

    manifest = processed_root / "manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "domain", "dataset", "stem", "num_boxes", "classes"])
        for it in sorted(items, key=lambda x: (x.split, x.stem)):
            w.writerow([it.split, it.domain, it.dataset, it.stem, it.num_boxes,
                        "|".join(str(c) for c in sorted(it.classes))])
    logger.info("wrote manifest %s", manifest)


def log_distribution(items: list[Item], schema: dict) -> None:
    names = schema_utils.id_to_name(schema)
    by_split: Counter = Counter(it.split for it in items)
    logger.info("=== split sizes ===")
    for s in SPLITS:
        logger.info("  %-5s %d", s, by_split[s])

    logger.info("=== per-class image counts by split ===")
    per: dict[int, Counter] = defaultdict(Counter)
    for it in items:
        for c in it.classes:
            per[c][it.split] += 1
    for c in sorted(names):
        row = per.get(c, Counter())
        logger.info("  %-16s train=%d val=%d test=%d",
                    names[c], row["train"], row["val"], row["test"])

    logger.info("=== per-domain image counts by split ===")
    dom: dict[str, Counter] = defaultdict(Counter)
    for it in items:
        dom[it.domain][it.split] += 1
    for d in sorted(dom):
        logger.info("  %-8s train=%d val=%d test=%d",
                    d, dom[d]["train"], dom[d]["val"], dom[d]["test"])


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def merge(
    interim_root: Path,
    processed_root: Path,
    schema_path: Path,
    val: float = 0.2,
    test: float = 0.1,
    seed: int = 42,
    dedup_threshold: int = 5,
    data_yaml: Path = schema_utils.DEFAULT_DATA_YAML,
) -> list[Item]:
    schema = schema_utils.load_schema(schema_path)
    military = schema_utils.military_class_ids(schema)

    items = collect_items(interim_root, schema, schema_utils.SURFACE)
    if not items:
        raise SystemExit(f"no labelled images found under {interim_root}")

    items, _ = deduplicate(items, dedup_threshold, military)
    stratified_split(items, val, test, seed)
    write_split(items, processed_root)

    try:
        processed_str = str(processed_root.resolve().relative_to(schema_utils.REPO_ROOT))
    except ValueError:
        processed_str = str(processed_root)
    schema_utils.write_data_yaml(schema, data_yaml, processed_root=processed_str)

    log_distribution(items, schema)
    return items


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(
        prog="python -m src.data.merge",
        description="Merge interim datasets into a stratified, deduped processed split.",
    )
    p.add_argument("--interim", type=Path, default=schema_utils.DEFAULT_INTERIM_ROOT)
    p.add_argument("--processed", type=Path, default=schema_utils.DEFAULT_PROCESSED_ROOT)
    p.add_argument("--schema", type=Path, default=schema_utils.DEFAULT_SCHEMA_PATH)
    p.add_argument("--val", type=float, default=0.2, help="val fraction (default 0.2)")
    p.add_argument("--test", type=float, default=0.1, help="test fraction (default 0.1)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-yaml", type=Path, default=schema_utils.DEFAULT_DATA_YAML,
                   help="where to regenerate data.yaml (default: configs/data.yaml)")
    p.add_argument("--dedup-threshold", type=int, default=5,
                   help="max hamming distance treated as a near-duplicate")
    p.add_argument("--data-yaml-only", action="store_true",
                   help="only regenerate configs/data.yaml from schema, then exit")
    args = p.parse_args(argv)

    if args.data_yaml_only:
        schema = schema_utils.load_schema(args.schema)
        schema_utils.write_data_yaml(schema)
        return 0

    if args.val + args.test >= 1.0:
        p.error("--val + --test must be < 1.0")

    merge(args.interim, args.processed, args.schema,
          val=args.val, test=args.test, seed=args.seed,
          dedup_threshold=args.dedup_threshold, data_yaml=args.data_yaml)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
