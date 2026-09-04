"""
Trajectory / behavior anomaly flagging.

Uses the same detection CSV log as src.reports.incident_report (schema:
frame,timestamp_s,track_id,class,group,confidence,x1,y1,x2,y2) and looks at
each tracked vessel's path over time to flag behavior worth an operator's
attention:

- Loitering: barely moves for an extended period.
- Sudden course change: direction of travel changes sharply between
  consecutive points.
- Restricted zone entry: the track's path crosses into a user-defined box.

Pure math over positions already produced by track_video()'s BoT-SORT/
ByteTrack output — no model, no GUI, no new dependencies beyond pandas.

Usage (CLI):
    python -m src.analysis.trajectory_anomaly --csv outputs/detections/demo.csv \
        --out outputs/anomalies.md

Usage (Python):
    from src.analysis.trajectory_anomaly import detect_anomalies
    flags = detect_anomalies(df)
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "frame",
    "timestamp_s",
    "track_id",
    "class",
    "group",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
]

# Tunable defaults — deliberately conservative so the demo doesn't flood with flags.
DEFAULT_LOITER_RADIUS_PX = 40.0       # a track that never leaves this radius...
DEFAULT_LOITER_MIN_DURATION_S = 4.0   # ...for at least this long is "loitering"
DEFAULT_COURSE_CHANGE_DEG = 60.0      # heading change between consecutive points
DEFAULT_MIN_MOVE_PX = 5.0             # ignore jitter smaller than this when computing heading


class AnomalyDetectionError(ValueError):
    """Raised when the input CSV doesn't match the expected detection log schema."""


@dataclass
class AnomalyFlag:
    track_id: str
    class_name: str
    reason: str  # "loitering" | "sudden_course_change" | "restricted_zone_entry"
    detail: str
    at_timestamp_s: float


def load_detection_log(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise AnomalyDetectionError(f"Detection log not found: {path}")
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise AnomalyDetectionError(f"Detection log is empty (no header): {path}") from exc

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise AnomalyDetectionError(
            f"Detection log {path} is missing expected column(s): {missing}. "
            f"Expected schema: {REQUIRED_COLUMNS}"
        )
    return df


def _centroids(df: pd.DataFrame) -> pd.DataFrame:
    """Add cx, cy centroid columns from the box corners."""
    df = df.copy()
    for col in ("x1", "y1", "x2", "y2", "timestamp_s"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["cx"] = (df["x1"] + df["x2"]) / 2.0
    df["cy"] = (df["y1"] + df["y2"]) / 2.0
    return df


def _track_groups(df: pd.DataFrame):
    """Yield (track_id, sorted_group_df) for tracked (non-blank track_id) rows only."""
    tracked_mask = df["track_id"].notna() & (df["track_id"].astype(str).str.strip() != "")
    tracked = df[tracked_mask]
    for track_id, group in tracked.groupby("track_id"):
        group = group.dropna(subset=["cx", "cy", "timestamp_s"]).sort_values("timestamp_s")
        if len(group) < 2:
            continue  # can't assess movement from a single point
        yield str(track_id), group


def _check_loitering(
    track_id: str,
    class_name: str,
    group: pd.DataFrame,
    radius_px: float,
    min_duration_s: float,
) -> AnomalyFlag | None:
    duration = float(group["timestamp_s"].max() - group["timestamp_s"].min())
    if duration < min_duration_s:
        return None

    cx0, cy0 = group["cx"].iloc[0], group["cy"].iloc[0]
    max_dist = ((group["cx"] - cx0) ** 2 + (group["cy"] - cy0) ** 2).pow(0.5).max()
    if max_dist <= radius_px:
        return AnomalyFlag(
            track_id=track_id,
            class_name=class_name,
            reason="loitering",
            detail=(
                f"stayed within {max_dist:.1f}px of its first position for "
                f"{duration:.1f}s (threshold: {radius_px:.0f}px / {min_duration_s:.0f}s)"
            ),
            at_timestamp_s=round(float(group["timestamp_s"].iloc[0]), 2),
        )
    return None


def _heading_deg(dx: float, dy: float) -> float:
    return math.degrees(math.atan2(dy, dx))


def _angle_diff_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two headings in degrees, in [0, 180]."""
    diff = abs(a - b) % 360.0
    return diff if diff <= 180.0 else 360.0 - diff


def _check_sudden_course_change(
    track_id: str,
    class_name: str,
    group: pd.DataFrame,
    angle_threshold_deg: float,
    min_move_px: float,
) -> AnomalyFlag | None:
    points = list(zip(group["timestamp_s"], group["cx"], group["cy"]))
    prev_heading = None
    for i in range(1, len(points)):
        _t0, x0, y0 = points[i - 1]
        t1, x1, y1 = points[i]
        dx, dy = x1 - x0, y1 - y0
        move = math.hypot(dx, dy)
        if move < min_move_px:
            continue  # ignore jitter / near-stationary steps for heading purposes
        heading = _heading_deg(dx, dy)
        if prev_heading is not None:
            change = _angle_diff_deg(heading, prev_heading)
            if change >= angle_threshold_deg:
                return AnomalyFlag(
                    track_id=track_id,
                    class_name=class_name,
                    reason="sudden_course_change",
                    detail=f"heading changed by {change:.0f}° at t={t1:.2f}s (threshold: {angle_threshold_deg:.0f}°)",
                    at_timestamp_s=round(float(t1), 2),
                )
        prev_heading = heading
    return None


def _segment_intersects_box(x0, y0, x1, y1, zone: tuple[float, float, float, float]) -> bool:
    """True if the segment (x0,y0)-(x1,y1) enters the axis-aligned box zone=(zx1,zy1,zx2,zy2).

    Checks both endpoint containment and a simple parametric sweep, which is
    sufficient for the short, mostly-linear per-frame segments in a tracking log.
    """
    zx1, zy1, zx2, zy2 = zone

    def _in_zone(px, py):
        return zx1 <= px <= zx2 and zy1 <= py <= zy2

    if _in_zone(x0, y0) or _in_zone(x1, y1):
        return True

    steps = 10
    for i in range(1, steps):
        t = i / steps
        px = x0 + (x1 - x0) * t
        py = y0 + (y1 - y0) * t
        if _in_zone(px, py):
            return True
    return False


def _check_restricted_zone(
    track_id: str,
    class_name: str,
    group: pd.DataFrame,
    zone: tuple[float, float, float, float],
) -> AnomalyFlag | None:
    points = list(zip(group["timestamp_s"], group["cx"], group["cy"]))
    for i in range(1, len(points)):
        _t0, x0, y0 = points[i - 1]
        t1, x1, y1 = points[i]
        if _segment_intersects_box(x0, y0, x1, y1, zone):
            return AnomalyFlag(
                track_id=track_id,
                class_name=class_name,
                reason="restricted_zone_entry",
                detail=f"path entered restricted zone {zone} at t={t1:.2f}s",
                at_timestamp_s=round(float(t1), 2),
            )
    return None


def detect_anomalies(
    df: pd.DataFrame,
    restricted_zone: tuple[float, float, float, float] | None = None,
    loiter_radius_px: float = DEFAULT_LOITER_RADIUS_PX,
    loiter_min_duration_s: float = DEFAULT_LOITER_MIN_DURATION_S,
    course_change_deg: float = DEFAULT_COURSE_CHANGE_DEG,
    min_move_px: float = DEFAULT_MIN_MOVE_PX,
) -> list[AnomalyFlag]:
    """Run all anomaly checks over every track in the log. Returns flags time-sorted."""
    if df.empty:
        return []

    df = _centroids(df)
    flags: list[AnomalyFlag] = []

    for track_id, group in _track_groups(df):
        class_name = group["class"].mode().iloc[0]

        loiter = _check_loitering(track_id, class_name, group, loiter_radius_px, loiter_min_duration_s)
        if loiter:
            flags.append(loiter)

        course = _check_sudden_course_change(track_id, class_name, group, course_change_deg, min_move_px)
        if course:
            flags.append(course)

        if restricted_zone is not None:
            zone_flag = _check_restricted_zone(track_id, class_name, group, restricted_zone)
            if zone_flag:
                flags.append(zone_flag)

    flags.sort(key=lambda f: f.at_timestamp_s)
    return flags


def render_markdown(flags: list[AnomalyFlag], source_name: str = "video") -> str:
    if not flags:
        return f"# Anomaly Report — {source_name}\n\nNo anomalies flagged for this clip.\n"

    lines = [f"# Anomaly Report — {source_name}", "", f"**{len(flags)} anomaly event(s) flagged.**", ""]
    reason_labels = {
        "loitering": "🕐 Loitering",
        "sudden_course_change": "↩️ Sudden course change",
        "restricted_zone_entry": "🚫 Restricted zone entry",
    }
    for f in flags:
        label = reason_labels.get(f.reason, f.reason)
        lines.append(f"- **Track {f.track_id}** ({f.class_name}) — {label} at {f.at_timestamp_s:.2f}s: {f.detail}")
    lines.append("")
    return "\n".join(lines)


def generate_anomaly_report(
    csv_path: str | Path,
    output_path: str | Path,
    restricted_zone: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Path:
    df = load_detection_log(csv_path)
    flags = detect_anomalies(df, restricted_zone=restricted_zone, **kwargs)
    text = render_markdown(flags, source_name=Path(csv_path).stem)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return out_path


def _parse_zone(value: str) -> tuple[float, float, float, float]:
    parts = [float(p) for p in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--zone must be 'x1,y1,x2,y2'")
    return tuple(parts)  # type: ignore[return-value]


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Flag suspicious vessel trajectories from a detection CSV log.")
    parser.add_argument("--csv", required=True, help="Path to the detection log CSV (from track_video()).")
    parser.add_argument("--out", required=True, help="Path to write the Markdown anomaly report to.")
    parser.add_argument("--zone", type=_parse_zone, default=None, help="Restricted zone as 'x1,y1,x2,y2' in pixel coords.")
    args = parser.parse_args()

    try:
        out_path = generate_anomaly_report(args.csv, args.out, restricted_zone=args.zone)
    except AnomalyDetectionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
