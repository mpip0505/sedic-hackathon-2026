"""
Incident report generator.

Turns a video detection log (the CSV produced by app/app.py's video export,
schema: frame,timestamp_s,track_id,class,group,confidence,x1,y1,x2,y2) into a
short, human-readable summary instead of a raw spreadsheet.

Does not import Ultralytics/torch and does not touch the model — it is a pure
post-processing step over `track_video()`'s CSV output, so it works with the
`--stub` pipeline too (any correctly-shaped CSV is enough to run this).

Usage (CLI):
    python -m src.reports.incident_report --csv outputs/detections/demo.csv \
        --out deliverables/incident_report.md

Usage (Python):
    from src.reports.incident_report import generate_report
    generate_report("outputs/detections/demo.csv", "outputs/incident_report.md")
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
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

MILITARY_CLASS = "military_vessel"
MILITARY_GROUP = "military"


class IncidentReportError(ValueError):
    """Raised when the input CSV doesn't match the expected detection log schema."""


@dataclass
class TrackSummary:
    track_id: str
    class_name: str
    group: str
    first_seen_s: float
    last_seen_s: float
    n_detections: int
    max_confidence: float
    mean_confidence: float

    @property
    def duration_s(self) -> float:
        return round(self.last_seen_s - self.first_seen_s, 2)


@dataclass
class ReportSummary:
    total_rows: int
    n_tracked_vessels: int
    n_untracked_detections: int
    n_military_vessels: int
    class_counts: dict = field(default_factory=dict)
    tracks: list = field(default_factory=list)  # list[TrackSummary], time-ordered
    military_tracks: list = field(default_factory=list)  # subset of tracks
    overall_mean_confidence: float = 0.0
    clip_duration_s: float = 0.0


def load_detection_log(csv_path: str | Path) -> pd.DataFrame:
    """Load and validate a detection CSV log.

    Raises IncidentReportError on a missing file, an empty file, or a file
    that's missing one of the frozen CSV columns — fail loudly rather than
    silently producing a misleading report.
    """
    path = Path(csv_path)
    if not path.exists():
        raise IncidentReportError(f"Detection log not found: {path}")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise IncidentReportError(f"Detection log is empty (no header): {path}") from exc

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise IncidentReportError(
            f"Detection log {path} is missing expected column(s): {missing}. "
            f"Expected schema: {REQUIRED_COLUMNS}"
        )

    return df


def summarize_detections(df: pd.DataFrame) -> ReportSummary:
    """Compute the stats the report is built from.

    track_id is blank for unconfirmed/untracked boxes (per the README: "leave
    track_id blank rather than writing nan") — those rows are counted but
    excluded from per-track summaries since they have no track to summarize.
    """
    if df.empty:
        return ReportSummary(
            total_rows=0,
            n_tracked_vessels=0,
            n_untracked_detections=0,
            n_military_vessels=0,
        )

    df = df.copy()
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["timestamp_s"] = pd.to_numeric(df["timestamp_s"], errors="coerce")

    tracked_mask = df["track_id"].notna() & (df["track_id"].astype(str).str.strip() != "")
    tracked_df = df[tracked_mask]
    untracked_count = int((~tracked_mask).sum())

    class_counts = df["class"].value_counts().to_dict()

    tracks: list[TrackSummary] = []
    for track_id, group_df in tracked_df.groupby("track_id"):
        group_df = group_df.sort_values("timestamp_s")
        # A track can (rarely) span more than one class label if tracking
        # re-identifies; take the most frequent label as the representative one.
        class_name = group_df["class"].mode().iloc[0]
        group_name = group_df["group"].mode().iloc[0]
        tracks.append(
            TrackSummary(
                track_id=str(track_id),
                class_name=class_name,
                group=group_name,
                first_seen_s=round(float(group_df["timestamp_s"].min()), 2),
                last_seen_s=round(float(group_df["timestamp_s"].max()), 2),
                n_detections=len(group_df),
                max_confidence=round(float(group_df["confidence"].max()), 3),
                mean_confidence=round(float(group_df["confidence"].mean()), 3),
            )
        )

    tracks.sort(key=lambda t: t.first_seen_s)
    military_tracks = [
        t for t in tracks if t.class_name == MILITARY_CLASS or t.group == MILITARY_GROUP
    ]

    overall_mean_conf = float(df["confidence"].mean()) if not df["confidence"].isna().all() else 0.0
    clip_duration = float(df["timestamp_s"].max() - df["timestamp_s"].min()) if len(df) > 1 else 0.0

    return ReportSummary(
        total_rows=len(df),
        n_tracked_vessels=len(tracks),
        n_untracked_detections=untracked_count,
        n_military_vessels=len(military_tracks),
        class_counts=class_counts,
        tracks=tracks,
        military_tracks=military_tracks,
        overall_mean_confidence=round(overall_mean_conf, 3),
        clip_duration_s=round(clip_duration, 2),
    )


def _format_track_line(t: TrackSummary) -> str:
    flag = " ⚠️ MILITARY" if t.class_name == MILITARY_CLASS or t.group == MILITARY_GROUP else ""
    return (
        f"- Track {t.track_id} — **{t.class_name}**{flag}: first seen at "
        f"{t.first_seen_s:.2f}s, tracked for {t.duration_s:.2f}s "
        f"({t.n_detections} detections, max confidence {t.max_confidence:.2f}, "
        f"mean confidence {t.mean_confidence:.2f})"
    )


def render_markdown(summary: ReportSummary, source_name: str = "video") -> str:
    """Build the report as a Markdown string using a plain template — no LLM required."""
    if summary.total_rows == 0:
        return (
            f"# Incident Report — {source_name}\n\n"
            "No detections were logged for this clip.\n"
        )

    lines = [
        f"# Incident Report — {source_name}",
        "",
        "## Summary",
        f"- Clip duration analysed: {summary.clip_duration_s:.2f}s",
        f"- Total detections logged: {summary.total_rows}",
        f"- Distinct tracked vessels: {summary.n_tracked_vessels}",
        f"- **Military vessels flagged: {summary.n_military_vessels}**",
        f"- Untracked detections (not yet confirmed by tracker): {summary.n_untracked_detections}",
        f"- Overall mean confidence: {summary.overall_mean_confidence:.2f}",
        "",
        "## Detections by class",
    ]
    for class_name, count in sorted(summary.class_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"- {class_name}: {count}")

    lines += ["", "## Timeline (by first appearance)"]
    if summary.tracks:
        for t in summary.tracks:
            lines.append(_format_track_line(t))
    else:
        lines.append("- No confirmed tracks in this clip.")

    if summary.military_tracks:
        lines += ["", "## ⚠️ Military vessels — detail"]
        for t in summary.military_tracks:
            lines.append(_format_track_line(t))

    lines.append("")
    return "\n".join(lines)


def render_llm_narrative(summary: ReportSummary, source_name: str = "video", client=None) -> str:
    """Optional: ask an LLM to turn the same stats into a short prose paragraph.

    `client` must expose `.messages.create(...)` (an Anthropic-SDK-shaped client)
    or be None, in which case this falls back to `render_markdown`. Kept
    separate from `render_markdown` so the deterministic report always works
    even with no API key / no network — this is a pure enhancement on top.
    """
    if client is None or summary.total_rows == 0:
        return render_markdown(summary, source_name)

    military_lines = "\n".join(_format_track_line(t) for t in summary.military_tracks) or "None"
    prompt = (
        f"Write a short (4-6 sentence), factual incident summary for a maritime "
        f"detection system, for source '{source_name}'. Use only these facts, do "
        f"not invent numbers:\n"
        f"- Clip duration: {summary.clip_duration_s:.2f}s\n"
        f"- Total detections: {summary.total_rows}\n"
        f"- Distinct tracked vessels: {summary.n_tracked_vessels}\n"
        f"- Military vessels flagged: {summary.n_military_vessels}\n"
        f"- Military vessel detail:\n{military_lines}\n"
        f"- Class breakdown: {summary.class_counts}\n"
    )
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
    return text.strip() or render_markdown(summary, source_name)


def generate_report(
    csv_path: str | Path,
    output_path: str | Path,
    use_llm: bool = False,
    llm_client=None,
) -> Path:
    """End-to-end: read CSV -> summarize -> write report file. Returns the output path."""
    df = load_detection_log(csv_path)
    summary = summarize_detections(df)
    source_name = Path(csv_path).stem

    if use_llm:
        text = render_llm_narrative(summary, source_name=source_name, client=llm_client)
    else:
        text = render_markdown(summary, source_name=source_name)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return out_path


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Generate an incident report from a detection CSV log.")
    parser.add_argument("--csv", required=True, help="Path to the detection log CSV (from track_video()).")
    parser.add_argument("--out", required=True, help="Path to write the Markdown report to.")
    parser.add_argument("--llm", action="store_true", help="Use an LLM to write the narrative (needs ANTHROPIC_API_KEY).")
    args = parser.parse_args()

    client = None
    if args.llm:
        try:
            import anthropic  # local import: optional dependency
            client = anthropic.Anthropic()
        except ImportError:
            print("anthropic package not installed; falling back to the template report.", file=sys.stderr)

    try:
        out_path = generate_report(args.csv, args.out, use_llm=args.llm, llm_client=client)
    except IncidentReportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
