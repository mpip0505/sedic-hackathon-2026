"""predict.py — THE FROZEN INTERFACE.

Everything downstream (the Streamlit GUI, the detection log, the evaluator)
talks to this module and nothing else. The `Detection` dataclass and the
`predict()` signature are FROZEN — do not change their shape without a team
decision, because breaking them breaks every consumer at once.

`--stub` mode returns synthetic detections in the exact real format, with NO
weights and NO torch, so integration/GUI work can start before any model
exists. `--stub` must ALWAYS work.

Usage:
    python -m src.inference.predict --source path/to/img.jpg --stub
    python -m src.inference.predict --source none --stub --out dets.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Repo root = two levels up from src/inference/predict.py
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA_PATH = _REPO_ROOT / "configs" / "schema.yaml"


# ---------------------------------------------------------------------------
# The frozen Detection contract.
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    """A single detected object.

    Attributes:
        class_name: schema class name (str). Serialized under the JSON key
            "class" (NOT "class_name") — `class` is a Python keyword so we can't
            use it as an attribute, but the wire format is "class".
        confidence: detection confidence in [0, 1].
        bbox: [x1, y1, x2, y2] in ABSOLUTE pixels.
        frame: source video frame index, or None for still images.
        timestamp: source video timestamp in seconds, or None for stills.
    """

    class_name: str
    confidence: float
    bbox: list[float] = field(default_factory=list)
    frame: int | None = None
    timestamp: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize with the frozen wire key "class" (not "class_name")."""
        d = asdict(self)
        d["class"] = d.pop("class_name")
        # Keep a stable key order for readable, diff-friendly JSON.
        return {
            "class": d["class"],
            "confidence": d["confidence"],
            "bbox": d["bbox"],
            "frame": d["frame"],
            "timestamp": d["timestamp"],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Detection:
        """Inverse of `to_dict` — reads the "class" wire key."""
        return cls(
            class_name=d["class"],
            confidence=d["confidence"],
            bbox=list(d.get("bbox", [])),
            frame=d.get("frame"),
            timestamp=d.get("timestamp"),
        )


# ---------------------------------------------------------------------------
# Schema access — class names come from configs/schema.yaml, never hardcoded.
# ---------------------------------------------------------------------------
def load_classes(schema_path: Path | str = _SCHEMA_PATH) -> list[str]:
    """Return detector class names ordered by class ID, read from schema.yaml.

    This is the ONLY place class names enter this module. Do not hardcode them.
    """
    with open(schema_path, "r", encoding="utf-8") as fh:
        schema = yaml.safe_load(fh)
    classes: dict[int, str] = schema["classes"]
    return [classes[i] for i in sorted(classes)]


def _military_class_names(schema_path: Path | str = _SCHEMA_PATH) -> set[str]:
    """Return the set of class names in the `military` group (recall gate)."""
    with open(schema_path, "r", encoding="utf-8") as fh:
        schema = yaml.safe_load(fh)
    classes: dict[int, str] = schema["classes"]
    military_ids = schema.get("groups", {}).get("military", [])
    return {classes[i] for i in military_ids}


# ---------------------------------------------------------------------------
# Stub inference — synthetic detections in the exact real format.
# ---------------------------------------------------------------------------
def _stub_predict(source: str, conf: float, conf_military: float) -> list[Detection]:
    """Produce deterministic synthetic detections for integration/GUI work.

    Runs with no weights and no torch. Output is byte-for-byte the same shape as
    real inference, so consumers can be built and tested today. Includes at
    least one military_vessel so the GUI's military-warning path is exercised.
    """
    rng = random.Random(hash(source) & 0xFFFFFFFF)
    classes = load_classes()
    military = _military_class_names()

    # A fixed, representative scene: a couple of civilians plus one military.
    plan = [
        ("container_ship", 0.91, [40, 120, 360, 300]),
        ("fishing_boat", 0.62, [420, 260, 520, 330]),
        ("military_vessel", 0.14, [610, 150, 880, 280]),  # low conf on purpose
    ]

    dets: list[Detection] = []
    for name, base_conf, bbox in plan:
        if name not in classes:
            continue
        # jitter a little so repeated runs look plausible but stay deterministic
        c = round(min(0.99, max(0.01, base_conf + rng.uniform(-0.03, 0.03))), 3)
        threshold = conf_military if name in military else conf
        if c < threshold:
            continue
        dets.append(
            Detection(
                class_name=name,
                confidence=c,
                bbox=[float(v) for v in bbox],
                frame=None,
                timestamp=None,
            )
        )
    logger.info("stub produced %d detection(s) for source=%r", len(dets), source)
    return dets


# ---------------------------------------------------------------------------
# Public entry point — FROZEN signature.
# ---------------------------------------------------------------------------
def predict(
    source: str,
    weights: str | None = None,
    conf: float = 0.25,
    conf_military: float = 0.10,
    stub: bool = False,
) -> list[Detection]:
    """Run detection on `source` and return a list of `Detection`.

    Args:
        source: image path, video path, directory, or "none" (stub only).
        weights: path to model weights (.pt). Ignored in stub mode.
        conf: general confidence threshold (civilian classes).
        conf_military: LOWER threshold for military classes — the recall gate.
            Intentionally below `conf`; do not raise it to match.
        stub: if True, return synthetic detections with no model/torch.

    Returns:
        list[Detection] in absolute-pixel coordinates.
    """
    if stub:
        return _stub_predict(source, conf=conf, conf_military=conf_military)

    # ---- Real inference path (NOT YET IMPLEMENTED) --------------------------
    # TODO(P2/P3): implement Ultralytics-backed inference.
    #   1. from ultralytics import YOLO
    #   2. model = YOLO(weights)  # default to models/best.pt if None
    #   3. results = model.predict(source, conf=min(conf, conf_military),
    #                              imgsz=640, verbose=False)
    #      -> run at the LOWER of the two thresholds so nothing is dropped early,
    #         then filter per-class below.
    #   4. For each box: map model class id -> name via load_classes(); apply
    #      the per-class threshold: keep if
    #         (name in military and score >= conf_military) or
    #         (name not in military and score >= conf).
    #      This is what enforces the >90% military recall gate.
    #   5. Convert xyxy to absolute pixels, capture frame/timestamp for video,
    #      and wrap each in a Detection.
    raise NotImplementedError(
        "Real inference is not implemented yet. Use stub=True (or --stub) until "
        "a model exists. See the TODO in predict() for the Ultralytics approach."
    )


def detections_to_json(dets: list[Detection], indent: int = 2) -> str:
    """Serialize detections to a JSON array string (wire key 'class')."""
    return json.dumps([d.to_dict() for d in dets], indent=indent)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.inference.predict",
        description="Project Guardian detection interface (frozen).",
    )
    p.add_argument("--source", required=True,
                   help='image/video/dir path, or "none" with --stub')
    p.add_argument("--weights", default=None, help="model weights .pt path")
    p.add_argument("--conf", type=float, default=0.25,
                   help="general confidence threshold")
    p.add_argument("--conf-military", type=float, default=0.10,
                   help="military confidence threshold (recall gate; low on purpose)")
    p.add_argument("--stub", action="store_true",
                   help="synthetic detections, no weights/torch required")
    p.add_argument("--out", default=None, help="write JSON to this file")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_arg_parser().parse_args(argv)

    dets = predict(
        source=args.source,
        weights=args.weights,
        conf=args.conf,
        conf_military=args.conf_military,
        stub=args.stub,
    )
    payload = detections_to_json(dets)

    if args.out:
        Path(args.out).write_text(payload, encoding="utf-8")
        logger.info("wrote %d detection(s) to %s", len(dets), args.out)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
