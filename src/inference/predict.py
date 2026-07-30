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
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Repo root = two levels up from src/inference/predict.py
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA_PATH = _REPO_ROOT / "configs" / "schema.yaml"

# Used when `weights` is None on the real path. Gitignored — see CLAUDE.md.
DEFAULT_WEIGHTS = _REPO_ROOT / "models" / "baseline_best.pt"

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


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


# Group precedence when a class belongs to more than one group (e.g. `yacht` is
# both civilian and small_craft). Most operationally significant wins.
_GROUP_PRECEDENCE = ("military", "small_craft", "civilian")


def class_groups(schema_path: Path | str = _SCHEMA_PATH) -> dict[str, str]:
    """Map each class name to ONE group name, resolved by `_GROUP_PRECEDENCE`.

    Consumers (e.g. the GUI's colour coding) use this instead of hardcoding
    which class is military/civilian — the grouping lives in schema.yaml.
    """
    with open(schema_path, "r", encoding="utf-8") as fh:
        schema = yaml.safe_load(fh)
    classes: dict[int, str] = schema["classes"]
    groups: dict[str, list[int]] = schema.get("groups", {})

    # Later assignments must not override higher-precedence ones, so walk the
    # precedence order backwards and let the important groups write last.
    ordered = [g for g in groups if g not in _GROUP_PRECEDENCE]
    ordered += [g for g in reversed(_GROUP_PRECEDENCE) if g in groups]
    out: dict[str, str] = {name: "other" for name in classes.values()}
    for group in ordered:
        for cid in groups[group]:
            if cid in classes:
                out[classes[cid]] = group
    return out


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

    model = load_model(weights)
    military = _military_class_names()
    is_video = Path(source).suffix.lower() in VIDEO_SUFFIXES

    # Run at the LOWER of the two thresholds so nothing is dropped before the
    # per-class filter below can apply the military gate.
    results = model.predict(
        source=source,
        conf=min(conf, conf_military),
        imgsz=640,
        verbose=False,
        stream=is_video,  # videos stream frame-by-frame instead of filling RAM
    )

    dets: list[Detection] = []
    fps = _video_fps(source) if is_video else None
    for idx, result in enumerate(results):
        frame = idx if is_video else None
        timestamp = round(idx / fps, 3) if (is_video and fps) else None
        dets.extend(
            _detections_from_result(
                result, military, conf, conf_military, frame=frame, timestamp=timestamp
            )
        )
    logger.info("predicted %d detection(s) for source=%r", len(dets), source)
    return dets


# ---------------------------------------------------------------------------
# Video tracking — ADDITIVE. `predict()` and `Detection` stay frozen; this is a
# separate entry point for consumers that need stable per-vessel IDs.
# ---------------------------------------------------------------------------
@dataclass
class TrackedDetection:
    """A `Detection` plus the tracker's persistent ID (None if unassigned)."""

    detection: Detection
    track_id: int | None = None


@dataclass
class TrackedFrame:
    """One video frame: its index, timestamp, BGR pixels, and detections."""

    index: int
    timestamp: float | None
    image: Any  # np.ndarray (BGR) — typed loosely to keep numpy a soft import
    detections: list[TrackedDetection] = field(default_factory=list)


def track_video(
    source: str,
    weights: str | None = None,
    conf: float = 0.25,
    conf_military: float = 0.10,
    tracker: str = "botsort.yaml",
    vid_stride: int = 1,
) -> Iterator[TrackedFrame]:
    """Yield tracked frames from a video, one at a time.

    Uses the same per-class threshold rule as `predict()`, so the military
    recall gate holds here too. Streams frames so long videos never load whole
    into memory.

    Args:
        source: video path.
        weights: model weights (.pt); defaults to `DEFAULT_WEIGHTS`.
        conf: general confidence threshold.
        conf_military: LOWER threshold for military classes (the recall gate).
        tracker: Ultralytics tracker config — "botsort.yaml" or "bytetrack.yaml".
        vid_stride: process every Nth frame (>1 trades detail for speed).
    """
    model = load_model(weights)
    military = _military_class_names()
    fps = _video_fps(source)

    results = model.track(
        source=source,
        conf=min(conf, conf_military),
        imgsz=640,
        tracker=tracker,
        persist=True,
        stream=True,
        verbose=False,
        vid_stride=max(1, vid_stride),
    )

    for step, result in enumerate(results):
        index = step * max(1, vid_stride)
        timestamp = round(index / fps, 3) if fps else None
        ids = _track_ids(result)
        dets = _detections_from_result(
            result, military, conf, conf_military, frame=index, timestamp=timestamp
        )
        # _detections_from_result keeps source box order, so ids line up by index.
        kept = _kept_box_indices(result, military, conf, conf_military)
        tracked = [
            TrackedDetection(detection=d, track_id=ids[i] if i < len(ids) else None)
            for d, i in zip(dets, kept)
        ]
        yield TrackedFrame(
            index=index, timestamp=timestamp, image=result.orig_img, detections=tracked
        )


# ---------------------------------------------------------------------------
# Real-inference helpers (Ultralytics imported lazily — `--stub` needs no torch)
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict[str, Any] = {}


def load_model(weights: str | Path | None = None) -> Any:
    """Load (and memoise) an Ultralytics YOLO model.

    Imports ultralytics lazily so the stub path keeps working with no torch
    installed. Raises FileNotFoundError with an actionable message when the
    weights are missing — callers surface that instead of a stack trace.
    """
    path = Path(weights) if weights else DEFAULT_WEIGHTS
    key = str(path)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    if not path.exists():
        raise FileNotFoundError(
            f"Model weights not found at {path}. Train a model, or run with "
            f"stub=True / --stub."
        )
    from ultralytics import YOLO  # lazy: heavy import

    logger.info("loading weights from %s", path)
    model = YOLO(str(path))
    _MODEL_CACHE[key] = model
    return model


def _keep(name: str, score: float, conf: float, conf_military: float,
          military: set[str]) -> bool:
    """Per-class threshold rule — the military gate lives here."""
    return score >= (conf_military if name in military else conf)


def _kept_box_indices(result: Any, military: set[str], conf: float,
                      conf_military: float) -> list[int]:
    """Indices of `result.boxes` that survive the per-class threshold."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    names = result.names
    return [
        i
        for i, box in enumerate(boxes)
        if _keep(names[int(box.cls[0])], float(box.conf[0]), conf, conf_military,
                 military)
    ]


def _detections_from_result(
    result: Any,
    military: set[str],
    conf: float,
    conf_military: float,
    frame: int | None = None,
    timestamp: float | None = None,
) -> list[Detection]:
    """Convert one Ultralytics result into `Detection`s in absolute pixels."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    names = result.names
    dets: list[Detection] = []
    for box in boxes:
        name = names[int(box.cls[0])]
        score = float(box.conf[0])
        if not _keep(name, score, conf, conf_military, military):
            continue
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
        dets.append(
            Detection(
                class_name=name,
                confidence=round(score, 4),
                bbox=[x1, y1, x2, y2],
                frame=frame,
                timestamp=timestamp,
            )
        )
    return dets


def _track_ids(result: Any) -> list[int | None]:
    """Tracker IDs for `result.boxes`, or Nones when the tracker assigned none."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    if getattr(boxes, "id", None) is None:
        return [None] * len(boxes)
    return [int(v) for v in boxes.id.tolist()]


def _video_fps(source: str) -> float | None:
    """Frame rate of `source`, or None if it can't be read."""
    try:
        import cv2
    except ImportError:  # pragma: no cover - only when cv2 isn't installed
        return None
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0.0
    cap.release()
    return fps if fps and fps > 0 else None


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
