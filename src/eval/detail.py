"""detail.py — threshold sweep + per-domain recall for the baseline weights.

Companion to metrics.py. Where metrics.py runs Ultralytics' own validation pass
and reports the canonical gate number at a single conf, this module does ONE
CPU-friendly inference pass over a split (predictions collected at a low conf
floor) and then derives, in numpy, two things the gate number alone can't show:

  1. A `conf_military` THRESHOLD SWEEP for military_vessel — recall, precision,
     and pass/fail vs the >0.90 gate at each threshold — so the recall/precision
     tradeoff is visible and the lowest-precision-cost threshold that still
     clears the gate is easy to read off.

  2. A PER-DOMAIN breakdown (aerial / surface / surface_synth) using
     data/processed/domains.json — per-domain × per-class recall plus military
     recall per domain, to localise where the gate miss comes from.

Metric definition (both analyses): a detection is a true positive if it is the
highest-confidence prediction of its class with IoU >= `iou_thr` (default 0.50)
against an as-yet-unmatched ground-truth box of that class, greedy in confidence
order — the standard PASCAL-VOC @0.5 matching, computed at the ACTUAL operating
threshold (not Ultralytics' max-F1 point, so numbers may differ slightly from
metrics.py's box.r). recall = TP/(TP+FN), precision = TP/(TP+FP).

    python -m src.eval.detail --weights models/baseline_best.pt \\
        --data configs/data.yaml --split test --device cpu

Ultralytics/torch are imported lazily inside `collect_predictions`, so importing
this module never pulls in the heavy stack.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.data import schema_utils

logger = logging.getLogger(__name__)

GATE_DEFAULT = 0.90
DEFAULT_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
DEFAULT_IOU = 0.50
# Low conf floor for the single inference pass: capture every candidate down to
# well below the smallest swept threshold so precision denominators are complete.
PRED_CONF_FLOOR = 0.01


@dataclass
class Sample:
    """One image's ground truth + predictions, in absolute pixel xyxy."""

    stem: str
    domain: str
    gt_cls: np.ndarray          # (G,) int
    gt_xyxy: np.ndarray         # (G, 4) float
    pred_cls: np.ndarray        # (P,) int
    pred_conf: np.ndarray       # (P,) float
    pred_xyxy: np.ndarray       # (P, 4) float


@dataclass
class SweepRow:
    conf: float
    tp: int
    fp: int
    fn: int

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    def passes(self, gate: float) -> bool:
        return self.recall >= gate


@dataclass
class DomainClassResult:
    per_domain_recall: dict[str, dict[str, float]] = field(default_factory=dict)
    military_recall_by_domain: dict[str, float] = field(default_factory=dict)
    overall_per_class_recall: dict[str, float] = field(default_factory=dict)
    overall_military_recall: float | None = None
    gate: float = GATE_DEFAULT

    @property
    def passed(self) -> bool:
        m = self.overall_military_recall
        return m is not None and m >= self.gate


# ---------------------------------------------------------------------------
# Ground-truth + IoU helpers
# ---------------------------------------------------------------------------
def _load_gt(label_path: Path, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Read a YOLO label file → (cls (G,), xyxy (G,4)) in absolute pixels."""
    cls: list[int] = []
    boxes: list[list[float]] = []
    if label_path.is_file():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            c = int(float(parts[0]))
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            cls.append(c)
            boxes.append([x1, y1, x2, y2])
    if not boxes:
        return np.empty((0,), dtype=int), np.empty((0, 4), dtype=float)
    return np.asarray(cls, dtype=int), np.asarray(boxes, dtype=float)


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a (N,4) and b (M,4) → (N,M). xyxy pixels."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=float)
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def _tp_fp_fn(sample: Sample, class_id: int, conf: float,
              iou_thr: float) -> tuple[int, int, int]:
    """Greedy VOC@iou matching for one class in one image, at threshold `conf`."""
    pred_mask = (sample.pred_cls == class_id) & (sample.pred_conf >= conf)
    gt_mask = sample.gt_cls == class_id
    n_gt = int(gt_mask.sum())
    p_conf = sample.pred_conf[pred_mask]
    p_xyxy = sample.pred_xyxy[pred_mask]
    n_pred = p_conf.shape[0]
    if n_pred == 0:
        return 0, 0, n_gt
    if n_gt == 0:
        return 0, n_pred, 0

    order = np.argsort(-p_conf)  # highest confidence first
    ious = _iou_matrix(p_xyxy[order], sample.gt_xyxy[gt_mask])
    matched = np.zeros(n_gt, dtype=bool)
    tp = 0
    for i in range(n_pred):
        row = ious[i].copy()
        row[matched] = -1.0
        j = int(np.argmax(row))
        if row[j] >= iou_thr:
            matched[j] = True
            tp += 1
    fp = n_pred - tp
    fn = n_gt - tp
    return tp, fp, fn


def _aggregate(samples: list[Sample], class_ids: set[int], conf: float,
               iou_thr: float, domain: str | None = None) -> tuple[int, int, int]:
    tp = fp = fn = 0
    for s in samples:
        if domain is not None and s.domain != domain:
            continue
        for cid in class_ids:
            t, f, n = _tp_fp_fn(s, cid, conf, iou_thr)
            tp += t
            fp += f
            fn += n
    return tp, fp, fn


# ---------------------------------------------------------------------------
# Inference pass (single, CPU-friendly)
# ---------------------------------------------------------------------------
def _split_dirs(data_yaml: Path, split: str) -> tuple[Path, Path]:
    import yaml
    data = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    root = Path(data.get("path", "data/processed"))
    if not root.is_absolute():
        root = schema_utils.REPO_ROOT / root
    img_rel = data.get(split)
    if not img_rel:
        raise SystemExit(f"split {split!r} not defined in {data_yaml}")
    images = root / str(img_rel)
    labels = root / str(img_rel).replace("images", "labels")
    return images, labels


def _load_domains(processed_root: Path, split: str) -> dict[str, str]:
    dj_path = processed_root / "domains.json"
    if not dj_path.is_file():
        logger.warning("no domains.json at %s — per-domain rows will be empty", dj_path)
        return {}
    return json.loads(dj_path.read_text(encoding="utf-8")).get(split, {})


def collect_predictions(weights: Path | str, data_yaml: Path | str, split: str,
                        imgsz: int = 640, device: str = "cpu", batch: int = 4,
                        conf_floor: float = PRED_CONF_FLOOR,
                        limit: int | None = None,
                        start: int = 0, end: int | None = None) -> list[Sample]:
    """One inference pass → per-image GT + predictions (absolute pixel xyxy).

    `start`/`end` slice the (sorted) image list so a long split can be collected
    in bounded foreground chunks (see --dump / --from-dumps); `limit` caps the
    total for a quick smoke test.
    """
    import os
    # Single-image batches hit predictor.pre_transform's auto=True "minimum
    # rectangle" letterbox, so consecutive images rarely share a shape and each
    # one can provoke a fresh cuDNN workspace allocation. Over a long stream on
    # GPU these never get reused and the *reserved-but-unallocated* pool grows
    # until a normal-sized allocation fails on fragmentation (not on total
    # image size — confirmed by bisection: identical slices OOM at different
    # points depending only on how many images already streamed in-process).
    # expandable_segments lets the allocator extend/reuse existing segments
    # instead of demanding a new contiguous block; must be set before any CUDA
    # context (i.e. before importing ultralytics/torch) to take effect.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from ultralytics import YOLO  # lazy: keep torch out of import time
    is_cuda = str(device).lower() != "cpu"
    if is_cuda:
        import torch

    images_dir, labels_dir = _split_dirs(Path(data_yaml), split)
    processed_root = images_dir.parent.parent
    domains = _load_domains(processed_root, split)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in exts)
    if limit is not None:
        image_paths = image_paths[:limit]
    image_paths = image_paths[start:end]
    if not image_paths:
        raise SystemExit(f"no images found under {images_dir} for slice [{start}:{end}]")
    logger.info("collecting predictions on %d %s images [%d:%s] "
                "(device=%s, imgsz=%d, batch=%d)",
                len(image_paths), split, start, end, device, imgsz, batch)

    model = YOLO(str(weights))
    results = model.predict(
        source=[str(p) for p in image_paths], stream=True, conf=conf_floor,
        imgsz=imgsz, device=device, batch=batch, verbose=False,
    )

    samples: list[Sample] = []
    missing_domain = 0
    for n_done, res in enumerate(results, start=1):
        path = Path(res.path)
        stem = path.stem
        h, w = res.orig_shape
        gt_cls, gt_xyxy = _load_gt(labels_dir / f"{stem}.txt", w, h)
        boxes = res.boxes
        if boxes is None or boxes.shape[0] == 0:
            p_cls = np.empty((0,), dtype=int)
            p_conf = np.empty((0,), dtype=float)
            p_xyxy = np.empty((0, 4), dtype=float)
        else:
            p_cls = boxes.cls.cpu().numpy().astype(int)
            p_conf = boxes.conf.cpu().numpy().astype(float)
            p_xyxy = boxes.xyxy.cpu().numpy().astype(float)
        domain = domains.get(stem, "unknown")
        if domain == "unknown":
            missing_domain += 1
        samples.append(Sample(stem, domain, gt_cls, gt_xyxy, p_cls, p_conf, p_xyxy))
        # Ultralytics Results/Boxes hold the full-res orig_img and form reference
        # cycles; without prodding the cyclic collector they pile up (~37 MB/img)
        # and OOM-kill a long CPU pass. Drop refs and collect periodically so
        # peak RSS stays flat instead of scaling with the image count.
        del res, boxes
        if is_cuda:
            # Cheap relative to inference; the growth is front-loaded (new
            # cuDNN workspace per never-repeated letterboxed shape), so
            # clearing only every 50 images lets it OOM before the first
            # clear ever fires. Clear every image instead.
            torch.cuda.empty_cache()
        if n_done % 50 == 0:
            gc.collect()
            logger.info("  … %d/%d images", n_done, len(image_paths))
    if missing_domain:
        logger.warning("%d/%d images had no domains.json entry (tagged 'unknown')",
                       missing_domain, len(samples))
    return samples


# ---------------------------------------------------------------------------
# 1) Military threshold sweep
# ---------------------------------------------------------------------------
def sweep_military(samples: list[Sample], military_ids: set[int],
                   thresholds=DEFAULT_THRESHOLDS, iou_thr: float = DEFAULT_IOU,
                   ) -> list[SweepRow]:
    rows: list[SweepRow] = []
    for t in thresholds:
        tp, fp, fn = _aggregate(samples, military_ids, t, iou_thr)
        rows.append(SweepRow(conf=t, tp=tp, fp=fp, fn=fn))
    return rows


# ---------------------------------------------------------------------------
# 2) Per-domain × per-class recall
# ---------------------------------------------------------------------------
def per_domain_recall(samples: list[Sample], schema: dict, conf: float = 0.25,
                      conf_military: float = 0.10, iou_thr: float = DEFAULT_IOU,
                      gate: float = GATE_DEFAULT) -> DomainClassResult:
    """Per-domain × per-class recall at the operational thresholds.

    Each class is thresholded at its operating point — military at
    `conf_military`, everything else at `conf` — mirroring predict()/the gate,
    so the military rows reconcile with the >0.90 gate measured at conf_military.
    """
    names = schema_utils.id_to_name(schema)
    military_ids = schema_utils.military_class_ids(schema)

    def class_conf(cid: int) -> float:
        return conf_military if cid in military_ids else conf

    domains = sorted({s.domain for s in samples})
    result = DomainClassResult(gate=gate)

    for dom in domains:
        result.per_domain_recall[dom] = {}
        for cid in sorted(names):
            tp, _fp, fn = _aggregate(samples, {cid}, class_conf(cid), iou_thr, domain=dom)
            denom = tp + fn
            if denom == 0:
                continue  # class absent in this domain's test set
            result.per_domain_recall[dom][names[cid]] = tp / denom
        # military-only recall for this domain (aggregated over military ids)
        tp, _fp, fn = _aggregate(samples, military_ids, conf_military, iou_thr, domain=dom)
        if tp + fn > 0:
            result.military_recall_by_domain[dom] = tp / (tp + fn)

    # overall per-class + overall military (all domains together)
    for cid in sorted(names):
        tp, _fp, fn = _aggregate(samples, {cid}, class_conf(cid), iou_thr)
        denom = tp + fn
        if denom:
            result.overall_per_class_recall[names[cid]] = tp / denom
    tp, _fp, fn = _aggregate(samples, military_ids, conf_military, iou_thr)
    if tp + fn > 0:
        result.overall_military_recall = tp / (tp + fn)
    return result


# ---------------------------------------------------------------------------
# Chunked collection: dump a slice's predictions, reload + concatenate.
# Lets a long split be collected in bounded foreground chunks (the harness caps
# backgrounded processes harder), then aggregated in one cheap CPU-free step.
# ---------------------------------------------------------------------------
def dump_samples(samples: list[Sample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(samples, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("dumped %d samples → %s", len(samples), path)


def load_dumps(dump_dir: Path) -> list[Sample]:
    files = sorted(dump_dir.glob("*.pkl"))
    if not files:
        raise SystemExit(f"no *.pkl prediction dumps found in {dump_dir}")
    samples: list[Sample] = []
    seen: set[str] = set()
    for f in files:
        with open(f, "rb") as fh:
            chunk = pickle.load(fh)
        for s in chunk:
            if s.stem in seen:  # overlapping chunks → keep one copy
                continue
            seen.add(s.stem)
            samples.append(s)
    logger.info("loaded %d unique samples from %d dump(s) in %s",
                len(samples), len(files), dump_dir)
    return samples


# ---------------------------------------------------------------------------
# Rendering (plain-text logs + markdown for the README / brief)
# ---------------------------------------------------------------------------
def _fmt(x: float | None) -> str:
    return "—" if x is None else f"{x:.3f}"


def _checkpoint_facts(weights: Path) -> list[str]:
    """Training facts read out of the checkpoint itself (date, epochs, source).

    Ultralytics stores `date` and `train_args` inside the .pt. Those describe the
    run that produced the weights — unlike the file mtime, which only says when
    the file landed on this machine. Returns [] if torch is unavailable (the
    --from-dumps path must keep working without it) or the keys are absent.
    """
    try:
        import torch
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 — provenance is best-effort, never fatal
        logger.debug("could not read checkpoint metadata from %s: %s", weights, exc)
        return []

    facts: list[str] = []
    trained = ckpt.get("date")
    if trained:
        facts.append(f"trained **{str(trained)[:19].replace('T', ' ')}**")
    args = ckpt.get("train_args") or {}
    if args.get("epochs"):
        facts.append(f"{args['epochs']} epochs")
    if args.get("model"):
        facts.append(f"from `{args['model']}`")
    if args.get("project"):
        facts.append(f"run dir `{args['project']}`")
    del ckpt
    return [" · ".join(facts)] if facts else []


def _provenance(weights: Path | None, split: str, n_images: int, iou: float,
                device: str, conf: float, conf_military: float,
                from_dumps: Path | None) -> str:
    """Header block stating WHEN this ran, on WHICH weights, over WHICH split.

    This file is the number the team quotes in the GUI, the brief and the
    poster, so a report that doesn't date itself is worse than no report — a
    stale one is indistinguishable from a current one. Emitted unconditionally.
    """
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z")
    if weights is not None and Path(weights).is_file():
        wp = Path(weights)
        wdesc = f"`{wp.as_posix()}` ({wp.stat().st_size / 1e6:.1f} MB)"
        # Prefer the checkpoint's OWN embedded training metadata over the file
        # mtime: mtime records when the file was copied here, which is not when
        # (or for how long) it was trained. Reading the wrong model's numbers as
        # the current model's is the exact failure this header exists to stop.
        for line in _checkpoint_facts(wp):
            wdesc += f"<br>{line}"
    else:
        wdesc = f"`{weights}`" if weights else "_(unknown — rendered from dumps)_"

    src = (f"cached prediction dumps in `{Path(from_dumps).as_posix()}`"
           if from_dumps else f"a live inference pass (device `{device}`)")

    banner = ("> **This file is auto-generated and is the single source of truth "
              "for the numbers the team quotes.** Do not hand-edit. Regenerate "
              "with the command at the bottom after any retrain, then update the "
              "README.")
    operating = (f"| **Operating point** | military @ conf {conf_military:.2f} · "
                 f"all other classes @ conf {conf:.2f} |")
    footnote = ("_Numbers here come from `detail.py`'s explicit VOC matching at "
                "the actual operating threshold. `src/eval/metrics.py` "
                "(Ultralytics' own val pass, max-F1 matching) is the **canonical "
                "gate** and typically reads slightly lower. Quote `metrics.py` "
                "for the headline gate number and these tables for the "
                "per-domain/per-class breakdown._")

    return "\n".join([
        f"# Baseline evaluation — `{split}` split",
        "",
        banner,
        "",
        "| | |",
        "|---|---|",
        f"| **Generated** | {now} |",
        f"| **Model** | {wdesc} |",
        f"| **Split** | `{split}` — {n_images} images (held out; never trained on) |",
        f"| **Predictions from** | {src} |",
        f"| **Matching** | greedy PASCAL-VOC, IoU ≥ {iou:.2f} |",
        operating,
        "",
        footnote,
    ])


def sweep_markdown(rows: list[SweepRow], gate: float, split: str = "test",
                   iou: float = DEFAULT_IOU) -> str:
    out = [f"### `military_vessel` conf threshold sweep ({split} split, IoU {iou:.2f})",
           "",
           f"| conf | recall | precision | TP | FP | FN | gate >{gate:.2f} |",
           "|-----:|-------:|----------:|---:|---:|---:|:----------:|"]
    for r in rows:
        out.append(f"| {r.conf:.2f} | {r.recall:.3f} | {r.precision:.3f} | "
                   f"{r.tp} | {r.fp} | {r.fn} | {'✅ PASS' if r.passes(gate) else '❌ FAIL'} |")
    return "\n".join(out)


def per_domain_markdown(res: DomainClassResult, split: str = "test",
                        conf: float = 0.25, conf_military: float = 0.10,
                        iou: float = DEFAULT_IOU) -> str:
    domains = sorted(res.per_domain_recall)
    classes = sorted({c for d in res.per_domain_recall.values() for c in d}
                     | set(res.overall_per_class_recall))

    subtitle = (f"_military @ conf {conf_military:.2f}, all other classes @ conf "
                f"{conf:.2f}, IoU {iou:.2f}_")
    out = [f"### Per-domain × per-class recall ({split} split)", subtitle, ""]
    header = "| class | " + " | ".join(domains) + " | overall |"
    sep = "|-------|" + "|".join(["------:"] * (len(domains) + 1)) + "|"
    out += [header, sep]
    for c in classes:
        cells = [_fmt(res.per_domain_recall.get(d, {}).get(c)) for d in domains]
        row_label = f"**{c}**" if "military" in c else c
        out.append(f"| {row_label} | " + " | ".join(cells) + f" | "
                   f"{_fmt(res.overall_per_class_recall.get(c))} |")
    out += ["", "### Military recall per domain (the gate, localised)", "",
            f"| domain | military recall | gate >{res.gate:.2f} |",
            "|--------|----------------:|:----------:|"]
    for d in domains:
        mr = res.military_recall_by_domain.get(d)
        if mr is None:
            continue
        flag = "✅" if mr >= res.gate else "❌"
        out.append(f"| {d} | {mr:.3f} | {flag} |")
    overall_flag = "✅ PASS" if res.passed else "❌ FAIL"
    out.append(f"| **overall** | {_fmt(res.overall_military_recall)} | {overall_flag} |")
    return "\n".join(out)


def _log_sweep(rows: list[SweepRow], gate: float) -> None:
    logger.info("=== military_vessel conf sweep (test, IoU 0.50) ===")
    logger.info("  %-6s %-8s %-10s %-6s %-6s %-6s %s",
                "conf", "recall", "precision", "TP", "FP", "FN", "gate")
    for r in rows:
        logger.info("  %-6.2f %-8.3f %-10.3f %-6d %-6d %-6d %s",
                    r.conf, r.recall, r.precision, r.tp, r.fp, r.fn,
                    "PASS" if r.passes(gate) else "FAIL")


def _log_per_domain(res: DomainClassResult) -> None:
    logger.info("=== military recall per domain (test) ===")
    for d in sorted(res.military_recall_by_domain):
        mr = res.military_recall_by_domain[d]
        logger.info("  %-14s %.3f  [%s]", d, mr, "OK" if mr >= res.gate else "FAIL")
    logger.info("  %-14s %s  [%s]", "OVERALL", _fmt(res.overall_military_recall),
                "PASS" if res.passed else "FAIL")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # Windows consoles default stdout to cp1252, which can't encode the ✅/❌
    # markdown tables below; force utf-8 so this doesn't crash after the report
    # (and the md file) are already written.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(
        prog="python -m src.eval.detail",
        description="conf_military threshold sweep + per-domain recall (CPU-friendly).",
    )
    p.add_argument("--weights", type=Path, default=None,
                   help="model weights (required unless --from-dumps)")
    p.add_argument("--data", default=schema_utils.DEFAULT_DATA_YAML, type=Path)
    p.add_argument("--schema", default=schema_utils.DEFAULT_SCHEMA_PATH, type=Path)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--gate", type=float, default=GATE_DEFAULT)
    p.add_argument("--conf", type=float, default=0.25,
                   help="civilian operating conf for the per-domain table (default 0.25)")
    p.add_argument("--conf-military", type=float, default=0.10,
                   help="military operating conf for the per-domain table (default 0.10)")
    p.add_argument("--iou", type=float, default=DEFAULT_IOU)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="cpu", help="cpu (default) / 0 / mps")
    p.add_argument("--batch", type=int, default=4, help="inference batch (small for CPU)")
    p.add_argument("--limit", type=int, default=None,
                   help="only run the first N images (smoke test / quick check)")
    p.add_argument("--start", type=int, default=0, help="chunk start index (sorted list)")
    p.add_argument("--end", type=int, default=None, help="chunk end index (exclusive)")
    p.add_argument("--dump", type=Path, default=None,
                   help="collect predictions for [start:end] and pickle them here, then "
                        "exit (no tables). Use for bounded foreground chunks.")
    p.add_argument("--from-dumps", type=Path, default=None,
                   help="skip inference; load pickled chunks from this dir and report.")
    p.add_argument("--thresholds", default=",".join(str(t) for t in DEFAULT_THRESHOLDS),
                   help="comma-separated sweep thresholds")
    p.add_argument("--md-out", type=Path, default=None,
                   help="write the markdown tables here (default outputs/eval/<split>_eval.md)")
    args = p.parse_args(argv)

    if args.weights is None and args.from_dumps is None:
        p.error("--weights is required unless --from-dumps is given")

    schema = schema_utils.load_schema(args.schema)
    military_ids = schema_utils.military_class_ids(schema)
    thresholds = tuple(float(t) for t in args.thresholds.split(",") if t.strip())

    # --dump: collect a slice and persist it, nothing else.
    if args.dump is not None:
        samples = collect_predictions(args.weights, args.data, args.split,
                                      imgsz=args.imgsz, device=args.device,
                                      batch=args.batch, limit=args.limit,
                                      start=args.start, end=args.end)
        dump_samples(samples, args.dump)
        return 0

    # --from-dumps: reuse persisted chunks (no torch); else one live pass.
    if args.from_dumps is not None:
        samples = load_dumps(args.from_dumps)
    else:
        samples = collect_predictions(args.weights, args.data, args.split,
                                      imgsz=args.imgsz, device=args.device,
                                      batch=args.batch, limit=args.limit,
                                      start=args.start, end=args.end)

    rows = sweep_military(samples, military_ids, thresholds, args.iou)
    _log_sweep(rows, args.gate)

    res = per_domain_recall(samples, schema, conf=args.conf,
                            conf_military=args.conf_military, iou_thr=args.iou,
                            gate=args.gate)
    _log_per_domain(res)

    regen = (f"python -m src.eval.detail --weights "
             f"{Path(args.weights).as_posix() if args.weights else '<weights.pt>'} "
             f"--split {args.split} --device {args.device} --batch {args.batch}")
    md = (
        _provenance(args.weights, args.split, len(samples), args.iou, args.device,
                    args.conf, args.conf_military, args.from_dumps) + "\n\n"
        + sweep_markdown(rows, args.gate, args.split, args.iou) + "\n\n"
        + per_domain_markdown(res, args.split, args.conf, args.conf_military,
                              args.iou) + "\n\n"
        + "---\n\n### Regenerate\n\n```bash\n" + regen + "\n```\n\n"
        "On a long CPU/GPU pass, collect in fresh-process chunks instead "
        "(`--start`/`--end`/`--dump`), then render with `--from-dumps` — see the "
        "README `## Results` note.\n"
    )
    md_out = args.md_out or (schema_utils.REPO_ROOT / "outputs" / "eval"
                             / f"{args.split}_eval.md")
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(md, encoding="utf-8")
    logger.info("wrote markdown tables → %s", md_out)
    print("\n" + md)

    return 0 if res.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
