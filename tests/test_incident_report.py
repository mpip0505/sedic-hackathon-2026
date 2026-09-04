"""
Tests for src.reports.incident_report.

Run with: pytest -q tests/test_incident_report.py
No torch/GPU/Ultralytics needed — this module never imports them.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.reports.incident_report import (
    IncidentReportError,
    generate_report,
    load_detection_log,
    render_markdown,
    summarize_detections,
)

CSV_COLUMNS = [
    "frame", "timestamp_s", "track_id", "class", "group",
    "confidence", "x1", "y1", "x2", "y2",
]


def _write_csv(tmp_path, rows, filename="detections.csv"):
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    path = tmp_path / filename
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# load_detection_log — input validation
# ---------------------------------------------------------------------------

def test_load_missing_file_raises(tmp_path):
    with pytest.raises(IncidentReportError, match="not found"):
        load_detection_log(tmp_path / "does_not_exist.csv")


def test_load_empty_file_raises(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(IncidentReportError, match="empty"):
        load_detection_log(path)


def test_load_missing_columns_raises(tmp_path):
    # Realistic mistake: someone exports without the 'group' column.
    df = pd.DataFrame(
        [[1, 0.1, "1", "cargo", 0.9, 0, 0, 10, 10]],
        columns=["frame", "timestamp_s", "track_id", "class", "confidence", "x1", "y1", "x2", "y2"],
    )
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    with pytest.raises(IncidentReportError, match="missing expected column"):
        load_detection_log(path)


def test_load_valid_file_ok(tmp_path):
    path = _write_csv(tmp_path, [
        [1, 0.0, "1", "cargo", "civilian", 0.9, 0, 0, 10, 10],
    ])
    df = load_detection_log(path)
    assert len(df) == 1


# ---------------------------------------------------------------------------
# summarize_detections — the core logic
# ---------------------------------------------------------------------------

def test_summarize_empty_dataframe():
    df = pd.DataFrame(columns=CSV_COLUMNS)
    summary = summarize_detections(df)
    assert summary.total_rows == 0
    assert summary.n_tracked_vessels == 0
    assert summary.n_military_vessels == 0


def test_summarize_counts_distinct_tracks(tmp_path):
    path = _write_csv(tmp_path, [
        [1, 0.0, "1", "cargo", "civilian", 0.90, 0, 0, 10, 10],
        [2, 0.5, "1", "cargo", "civilian", 0.92, 1, 1, 11, 11],
        [3, 1.0, "2", "yacht", "small_craft", 0.80, 5, 5, 20, 20],
    ])
    df = load_detection_log(path)
    summary = summarize_detections(df)
    assert summary.n_tracked_vessels == 2
    assert summary.total_rows == 3


def test_summarize_flags_military_by_class(tmp_path):
    path = _write_csv(tmp_path, [
        [1, 0.0, "1", "military_vessel", "military", 0.95, 0, 0, 10, 10],
        [2, 1.0, "2", "cargo", "civilian", 0.88, 0, 0, 10, 10],
    ])
    df = load_detection_log(path)
    summary = summarize_detections(df)
    assert summary.n_military_vessels == 1
    assert summary.military_tracks[0].track_id == "1"


def test_summarize_untracked_rows_counted_but_excluded_from_tracks(tmp_path):
    # Per the README: untracked boxes leave track_id blank, not "nan".
    path = _write_csv(tmp_path, [
        [1, 0.0, "", "cargo", "civilian", 0.5, 0, 0, 10, 10],
        [2, 1.0, "1", "cargo", "civilian", 0.9, 0, 0, 10, 10],
    ])
    df = load_detection_log(path)
    summary = summarize_detections(df)
    assert summary.n_untracked_detections == 1
    assert summary.n_tracked_vessels == 1


def test_summarize_track_duration_and_confidence_stats(tmp_path):
    path = _write_csv(tmp_path, [
        [1, 0.0, "7", "military_vessel", "military", 0.80, 0, 0, 10, 10],
        [2, 2.0, "7", "military_vessel", "military", 0.90, 0, 0, 10, 10],
        [3, 5.0, "7", "military_vessel", "military", 0.70, 0, 0, 10, 10],
    ])
    df = load_detection_log(path)
    summary = summarize_detections(df)
    track = summary.tracks[0]
    assert track.first_seen_s == 0.0
    assert track.last_seen_s == 5.0
    assert track.duration_s == 5.0
    assert track.n_detections == 3
    assert track.max_confidence == 0.9
    assert track.mean_confidence == pytest.approx(0.8, abs=1e-6)


def test_summarize_handles_non_numeric_confidence_gracefully(tmp_path):
    # Guards against a malformed export (e.g. a stray header row re-appended).
    path = _write_csv(tmp_path, [
        [1, 0.0, "1", "cargo", "civilian", "not_a_number", 0, 0, 10, 10],
        [2, 1.0, "1", "cargo", "civilian", 0.9, 0, 0, 10, 10],
    ])
    df = load_detection_log(path)
    summary = summarize_detections(df)  # should not raise
    assert summary.n_tracked_vessels == 1


def test_summarize_track_spanning_reidentified_class_uses_mode(tmp_path):
    # Edge case: a track_id briefly mislabeled mid-track (tracker re-ID noise).
    path = _write_csv(tmp_path, [
        [1, 0.0, "3", "cargo", "civilian", 0.9, 0, 0, 10, 10],
        [2, 1.0, "3", "cargo", "civilian", 0.9, 0, 0, 10, 10],
        [3, 2.0, "3", "tanker", "civilian", 0.6, 0, 0, 10, 10],
    ])
    df = load_detection_log(path)
    summary = summarize_detections(df)
    assert summary.tracks[0].class_name == "cargo"  # majority label wins


# ---------------------------------------------------------------------------
# render_markdown / generate_report — output shape
# ---------------------------------------------------------------------------

def test_render_markdown_empty_summary_has_no_crash():
    df = pd.DataFrame(columns=CSV_COLUMNS)
    summary = summarize_detections(df)
    text = render_markdown(summary, source_name="empty_clip")
    assert "No detections" in text


def test_render_markdown_flags_military_in_output(tmp_path):
    path = _write_csv(tmp_path, [
        [1, 0.0, "1", "military_vessel", "military", 0.95, 0, 0, 10, 10],
    ])
    df = load_detection_log(path)
    summary = summarize_detections(df)
    text = render_markdown(summary, source_name="clip1")
    assert "MILITARY" in text
    assert "Military vessels flagged: 1" in text


def test_generate_report_writes_file(tmp_path):
    csv_path = _write_csv(tmp_path, [
        [1, 0.0, "1", "cargo", "civilian", 0.9, 0, 0, 10, 10],
        [2, 1.0, "2", "military_vessel", "military", 0.93, 0, 0, 10, 10],
    ])
    out_path = tmp_path / "reports" / "incident.md"  # nested dir that doesn't exist yet
    result_path = generate_report(csv_path, out_path)
    assert result_path == out_path
    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")
    assert "Incident Report" in content
    assert "military_vessel" in content


def test_generate_report_raises_on_missing_csv(tmp_path):
    with pytest.raises(IncidentReportError):
        generate_report(tmp_path / "nope.csv", tmp_path / "out.md")


def test_generate_report_use_llm_false_never_needs_network_or_key(tmp_path, monkeypatch):
    # Make sure the default path can't accidentally reach for an API client.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    csv_path = _write_csv(tmp_path, [
        [1, 0.0, "1", "cargo", "civilian", 0.9, 0, 0, 10, 10],
    ])
    out_path = generate_report(csv_path, tmp_path / "out.md", use_llm=False)
    assert out_path.exists()
