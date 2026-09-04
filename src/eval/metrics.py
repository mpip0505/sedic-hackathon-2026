"""metrics.py — per-class recall and the >90% military-recall gate.

Runs an Ultralytics validation pass and reports per-class recall, then checks
the one hard competition requirement: **recall > 90% on military classes**.
Called automatically at the end of training (src/train/train.py) and runnable
standalone. Exits nonzero when the gate fails.

    python -m src.eval.metrics --weights models/baseline2_best.pt \\
        --data configs/data.yaml --split val --conf 0.10

Ultralytics/torch are imported lazily inside `evaluate`, so importing this
module (e.g. for --dry-run wiring) never requires the heavy stack.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

from src.data import schema_utils

logger = logging.getLogger(__name__)

# The competition's hard requirement. Not a tunable hyperparameter.
GATE_DEFAULT = 0.90


@dataclass
class GateResult:
    gate: float
    split: str = "test"
    per_class_recall: dict[str, float] = field(default_factory=dict)
    military_recall: dict[str, float] = field(default_factory=dict)
    evaluated: bool = True

    @property
    def min_military(self) -> float | None:
        return min(self.military_recall.values()) if self.military_recall else None

    @property
    def passed(self) -> bool:
        m = self.min_military
        return m is not None and m >= self.gate


# Ultralytics' own val() NMS floor. Deliberately far below any real operating
# point (conf_military=0.10, conf=0.25) so the confidence curve stays fully
# resolved at both — see the note in `evaluate()` on why the floor and the
# reporting threshold must NOT be the same value.
_VAL_CONF_FLOOR = 0.001


def _recall_at_conf(box_metric, conf: float) -> dict[int, float]:
    """Per-class recall read off the confidence curve AT a fixed `conf`.

    `model.val()`'s own `box.r` is NOT recall at the `conf` passed to `val()` —
    that argument only sets the NMS candidate floor. The reported `box.r` is
    every class's recall read off ONE index shared across ALL classes: wherever
    `f1_curve.mean(0)` (F1 averaged ACROSS EVERY CLASS, not just this one) peaks
    (see ultralytics/utils/metrics.py, `ap_per_class()`). That index is rarely
    anywhere near `conf`, isn't military-specific, and drifts run-to-run with
    unrelated classes' precision/recall — confirmed against this repo's own
    checkpoints: it read 0.892 on a model whose recall at the ACTUAL deployed
    conf_military=0.10 was 0.938 (`src/eval/detail.py`, explicit VOC matching).

    `box_metric.r_curve` (nc, 1000) and `box_metric.px` (1000,) are the
    per-class recall-vs-confidence curve Ultralytics already computed — reading
    the value at the index nearest `conf` gives recall at the real operating
    point instead, with no extra inference pass.
    """
    import numpy as np

    idx = int(np.argmin(np.abs(box_metric.px - conf)))
    return {
        int(cls_id): float(box_metric.r_curve[row, idx])
        for row, cls_id in enumerate(box_metric.ap_class_index)
    }


def evaluate(
    weights: Path | str,
    data: Path | str,
    schema_path: Path | str = schema_utils.DEFAULT_SCHEMA_PATH,
    gate: float = GATE_DEFAULT,
    conf: float = 0.10,
    imgsz: int = 640,
    *,
    split: str,
) -> GateResult:
    """Run Ultralytics validation and compute the military-recall gate result.

    `split` is REQUIRED (keyword-only) so the gate can never silently default to
    the wrong split — it must be the held-out `test` split the model never saw
    for checkpoint selection. `conf` is the FIXED operating point recall is
    reported at (military's deployed conf_military=0.10 by default) — NOT the
    NMS floor passed to `model.val()`, which stays low (`_VAL_CONF_FLOOR`) so
    the confidence curve is fully resolved at `conf`. See `_recall_at_conf`.
    """
    from ultralytics import YOLO  # lazy: keep torch out of import time

    schema = schema_utils.load_schema(schema_path)
    military_ids = schema_utils.military_class_ids(schema)

    model = YOLO(str(weights))
    metrics = model.val(data=str(data), split=split, conf=_VAL_CONF_FLOOR,
                        imgsz=imgsz, verbose=False)

    names = metrics.names  # {class_id: name}
    recall_at_conf = _recall_at_conf(metrics.box, conf)
    per_class: dict[str, float] = {
        names[cls_id]: recall_at_conf[cls_id] for cls_id in recall_at_conf
    }

    military_names = {names[i] for i in military_ids if i in names}
    military = {n: per_class[n] for n in military_names if n in per_class}

    result = GateResult(gate=gate, split=split, per_class_recall=per_class,
                        military_recall=military)
    _log_result(result)
    return result


def _log_result(result: GateResult) -> None:
    logger.info("=== per-class recall (%s split) ===", result.split)
    for name in sorted(result.per_class_recall):
        logger.info("  %-16s %.3f", name, result.per_class_recall[name])
    logger.info("=== MILITARY RECALL GATE (> %.2f) on %s split ===",
                result.gate, result.split.upper())
    if not result.military_recall:
        logger.warning("  no military-class instances evaluated — gate INDETERMINATE")
        return
    for name, rec in sorted(result.military_recall.items()):
        flag = "OK" if rec >= result.gate else "FAIL"
        logger.info("  %-16s %.3f  [%s]", name, rec, flag)
    if result.passed:
        logger.info("  RESULT: PASS (min military recall %.3f >= %.2f)",
                    result.min_military, result.gate)
    else:
        logger.error("  RESULT: FAIL (min military recall %.3f < %.2f)",
                     result.min_military, result.gate)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(
        prog="python -m src.eval.metrics",
        description="Per-class recall + the >90% military-recall gate.",
    )
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--data", default=schema_utils.DEFAULT_DATA_YAML, type=Path)
    p.add_argument("--schema", default=schema_utils.DEFAULT_SCHEMA_PATH, type=Path)
    p.add_argument("--gate", type=float, default=GATE_DEFAULT)
    p.add_argument("--conf", type=float, default=0.10,
                   help="confidence threshold for eval (default: military 0.10)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--split", default="test", choices=["train", "val", "test"],
                   help="split to score the gate on (default: test, the held-out set)")
    args = p.parse_args(argv)

    result = evaluate(args.weights, args.data, args.schema, gate=args.gate,
                      conf=args.conf, imgsz=args.imgsz, split=args.split)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
