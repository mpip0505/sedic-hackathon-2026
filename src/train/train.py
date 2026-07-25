"""train.py — thin, config-driven wrapper over Ultralytics YOLO.

Every hyperparameter comes from the YAML config (configs/train_baseline.yaml);
nothing is hardcoded here. Individual keys may be overridden on the CLI with
repeatable ``--set key=value``.

    python -m src.train.train --config configs/train_baseline.yaml
    python -m src.train.train --config configs/train_baseline.yaml --set epochs=50
    python -m src.train.train --config configs/train_baseline.yaml --dry-run
    python -m src.train.train --config configs/train_baseline.yaml --resume

Before training it asserts configs/data.yaml still matches configs/schema.yaml
and that the TRAIN split has military instances (the recall gate is unmeetable
otherwise) — both abort loudly; --allow-empty-military overrides the latter.
After training it copies the best weights to models/<name>_best.pt and scores
the military-recall gate (src/eval/metrics.py) on the held-out TEST split.

Ultralytics/torch are imported lazily, so --dry-run works without them.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from collections import Counter
from pathlib import Path

import yaml

from src.data import schema_utils
from src.eval import metrics

logger = logging.getLogger(__name__)

# Config keys that are NOT Ultralytics train() arguments and must be stripped
# before the kwargs are forwarded.
_NON_TRAIN_KEYS = {"model", "conf", "conf_military"}
_SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config(path: Path | str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if not isinstance(config, dict):
        raise SystemExit(f"config {path} did not parse to a mapping")
    return config


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    """Apply repeatable `key=value` overrides; values are YAML-typed."""
    merged = dict(config)
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, _, raw = item.partition("=")
        key = key.strip()
        if key not in merged:
            logger.warning("override introduces new key %r not in config", key)
        merged[key] = yaml.safe_load(raw)
    return merged


def build_train_kwargs(config: dict) -> dict:
    """The subset of config forwarded to Ultralytics `model.train()`."""
    return {k: v for k, v in config.items() if k not in _NON_TRAIN_KEYS}


# ---------------------------------------------------------------------------
# Pre-flight validation + logging
# ---------------------------------------------------------------------------
def _repo_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else schema_utils.REPO_ROOT / p


def assert_consistent(schema_path: Path, data_yaml: Path) -> None:
    """Abort loudly if data.yaml has drifted from schema.yaml."""
    schema = schema_utils.load_schema(schema_path)
    with open(data_yaml, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    problems = schema_utils.data_schema_problems(schema, data)
    if problems:
        msg = "\n  - ".join(problems)
        raise SystemExit(
            f"ABORT: {data_yaml} has drifted from {schema_path}:\n  - {msg}\n"
            f"Regenerate with: python -m src.data.merge --data-yaml-only"
        )
    logger.info("schema/data consistency: OK (%d classes)",
                schema_utils.num_classes(schema))


def count_instances_in_split(data_yaml: Path, split: str) -> Counter:
    """Per-class instance counts for a single split, read from its label files."""
    with open(data_yaml, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    root = _repo_path(data.get("path", "data/processed"))
    img_rel = data.get(split)
    counter: Counter = Counter()
    if not img_rel:
        return counter
    labels_dir = root / str(img_rel).replace("images", "labels")
    if not labels_dir.is_dir():
        return counter
    for lf in sorted(labels_dir.glob("*.txt")):
        for line in lf.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                counter[int(float(line.split()[0]))] += 1
    return counter


def count_dataset(data_yaml: Path, schema: dict) -> tuple[dict[str, int], Counter]:
    """Return (per-split image counts, per-class instance counts) from labels."""
    with open(data_yaml, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    root = _repo_path(data.get("path", "data/processed"))

    images: dict[str, int] = {}
    per_class: Counter = Counter()
    for split in _SPLITS:
        img_rel = data.get(split)
        if not img_rel:
            continue
        labels_dir = root / str(img_rel).replace("images", "labels")
        if not labels_dir.is_dir():
            logger.warning("labels dir missing for %s split: %s", split, labels_dir)
            images[split] = 0
            continue
        label_files = sorted(labels_dir.glob("*.txt"))
        images[split] = len(label_files)
        for lf in label_files:
            for line in lf.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    per_class[int(float(line.split()[0]))] += 1
    return images, per_class


def log_startup(config: dict, data_yaml: Path, schema: dict) -> None:
    logger.info("=== resolved config ===")
    for key in sorted(config):
        logger.info("  %-14s %s", key, config[key])

    images, per_class = count_dataset(data_yaml, schema)
    logger.info("=== dataset counts (images per split) ===")
    for split in _SPLITS:
        logger.info("  %-5s %d", split, images.get(split, 0))

    names = schema_utils.id_to_name(schema)
    military = schema_utils.military_class_ids(schema)
    logger.info("=== per-class instance counts (all splits) ===")
    for cid in sorted(names):
        tag = "  <- military (recall gate)" if cid in military else ""
        logger.info("  %-16s %d%s", names[cid], per_class.get(cid, 0), tag)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run(config: dict, resume: bool = False, dry_run: bool = False,
        allow_empty_military: bool = False, gate_split: str = "test",
        schema_path: Path = schema_utils.DEFAULT_SCHEMA_PATH) -> int:
    data_yaml = _repo_path(config["data"])
    assert_consistent(schema_path, data_yaml)

    schema = schema_utils.load_schema(schema_path)
    log_startup(config, data_yaml, schema)

    # Military recall IS the pass/fail gate — training with zero military
    # instances can never satisfy it. Abort loudly (same style as schema drift).
    military_ids = schema_utils.military_class_ids(schema)
    train_counts = count_instances_in_split(data_yaml, "train")
    military_in_train = sum(train_counts.get(m, 0) for m in military_ids)
    if military_in_train == 0 and not allow_empty_military:
        raise SystemExit(
            "ABORT: zero military_vessel instances in the TRAIN split — the "
            ">90% military-recall gate can never be met. Add military data (or "
            "pass --allow-empty-military to override intentionally)."
        )
    logger.info("military instances in train split: %d", military_in_train)

    if dry_run:
        logger.info("--dry-run: config and data validated; not training.")
        return 0

    from ultralytics import YOLO  # lazy: keep torch out of import time

    train_kwargs = build_train_kwargs(config)
    project = _repo_path(train_kwargs.get("project", "outputs/runs"))
    name = train_kwargs.get("name", "train")
    train_kwargs["project"] = str(project)

    weights_source = config["model"]
    last_ckpt = project / name / "weights" / "last.pt"
    if resume:
        if last_ckpt.is_file():
            weights_source = str(last_ckpt)
            train_kwargs["resume"] = True
            logger.info("resuming from %s", last_ckpt)
        else:
            logger.warning("--resume set but %s not found; starting fresh", last_ckpt)

    logger.info("starting training: model=%s data=%s", weights_source, data_yaml)
    model = YOLO(weights_source)
    results = model.train(**train_kwargs)

    save_dir = Path(getattr(results, "save_dir", project / name))
    best = save_dir / "weights" / "best.pt"
    models_dir = schema_utils.REPO_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    dest = models_dir / f"{name}_best.pt"
    if best.is_file():
        shutil.copy2(best, dest)
        logger.info("copied best weights: %s -> %s", best, dest)
    else:
        logger.error("best weights not found at %s; skipping copy", best)
        return 1

    # Gate on the held-out TEST split, never val: YOLO uses val for checkpoint
    # selection / early stopping, so a val recall number is optimistically
    # biased. The reported gate number must come from data the model never saw
    # during training or selection.
    logger.info("=== running military-recall gate (src/eval/metrics.py) on %s split ===",
                gate_split)
    result = metrics.evaluate(
        weights=dest,
        data=data_yaml,
        schema_path=schema_path,
        conf=config.get("conf_military", 0.10),
        imgsz=config.get("imgsz", 640),
        split=gate_split,
    )
    logger.info("gate number reported from the %s split", result.split)
    return 0 if result.passed else 1


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(
        prog="python -m src.train.train",
        description="Config-driven Ultralytics YOLO training for Project Guardian.",
    )
    p.add_argument("--config", required=True, type=Path,
                   help="training config YAML (e.g. configs/train_baseline.yaml)")
    p.add_argument("--schema", type=Path, default=schema_utils.DEFAULT_SCHEMA_PATH)
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE", help="override a single config key (repeatable)")
    p.add_argument("--resume", action="store_true", help="resume from last checkpoint")
    p.add_argument("--dry-run", action="store_true",
                   help="validate config + data and exit without training")
    p.add_argument("--allow-empty-military", action="store_true",
                   help="override the zero-military abort (rarely correct)")
    p.add_argument("--gate-split", default="test", choices=["train", "val", "test"],
                   help="split the recall gate is scored on (default: test, held-out)")
    args = p.parse_args(argv)

    config = apply_overrides(load_config(args.config), args.overrides)
    return run(config, resume=args.resume, dry_run=args.dry_run,
               allow_empty_military=args.allow_empty_military,
               gate_split=args.gate_split, schema_path=args.schema)


if __name__ == "__main__":
    raise SystemExit(main())
