"""
Tests for src.analysis.trajectory_anomaly.

Run with: pytest -q tests/test_trajectory_anomaly.py
No torch/GPU/Ultralytics needed — this module never imports them.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.analysis.trajectory_anomaly import (
    AnomalyDetectionError,
    detect_anomalies,
    generate_anomaly_report,
    load_detection_log,
    render_markdown,
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


def _row(frame, t, track_id, cls, group, conf, cx, cy, size=10):
    """Helper: build a row from a centroid (cx, cy) instead of raw corners."""
    half = size / 2
    return [frame, t, track_id, cls, group, conf, cx - half, cy - half, cx + half, cy + half]


# ---------------------------------------------------------------------------
# load_detection_log — same validation contract as incident_report
# ---------------------------------------------------------------------------

def test_load_missing_file_raises(tmp_path):
    with pytest.raises(AnomalyDetectionError, match="not found"):
        load_detection_log(tmp_path / "missing.csv")


def test_load_missing_columns_raises(tmp_path):
    df = pd.DataFrame([[1, 0.0, "1", "cargo", 0.9]], columns=["frame", "timestamp_s", "track_id", "class", "confidence"])
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    with pytest.raises(AnomalyDetectionError, match="missing expected column"):
        load_detection_log(path)


# ---------------------------------------------------------------------------
# detect_anomalies — empty / trivial inputs
# ---------------------------------------------------------------------------

def test_detect_anomalies_empty_dataframe_returns_no_flags():
    df = pd.DataFrame(columns=CSV_COLUMNS)
    assert detect_anomalies(df) == []


def test_single_point_track_is_skipped_not_crashed(tmp_path):
    # A track with only one detection has no "movement" to assess.
    path = _write_csv(tmp_path, [
        _row(1, 0.0, "1", "cargo", "civilian", 0.9, 100, 100),
    ])
    df = load_detection_log(path)
    flags = detect_anomalies(df)
    assert flags == []


def test_untracked_rows_are_ignored(tmp_path):
    path = _write_csv(tmp_path, [
        _row(1, 0.0, "", "cargo", "civilian", 0.5, 100, 100),
        _row(2, 1.0, "", "cargo", "civilian", 0.5, 101, 100),
    ])
    df = load_detection_log(path)
    assert detect_anomalies(df) == []


# ---------------------------------------------------------------------------
# Loitering
# ---------------------------------------------------------------------------

def test_loitering_flagged_when_stationary_long_enough(tmp_path):
    # Stays within a couple px of (100,100) for 10 seconds.
    rows = [_row(i, float(i), "1", "cargo", "civilian", 0.9, 100 + i * 0.5, 100) for i in range(11)]
    path = _write_csv(tmp_path, rows)
    df = load_detection_log(path)
    flags = detect_anomalies(df, loiter_radius_px=40, loiter_min_duration_s=4)
    reasons = [f.reason for f in flags]
    assert "loitering" in reasons


def test_loitering_not_flagged_when_duration_too_short(tmp_path):
    rows = [_row(i, float(i), "1", "cargo", "civilian", 0.9, 100, 100) for i in range(3)]  # only 2s span
    path = _write_csv(tmp_path, rows)
    df = load_detection_log(path)
    flags = detect_anomalies(df, loiter_radius_px=40, loiter_min_duration_s=4)
    assert not any(f.reason == "loitering" for f in flags)


def test_loitering_not_flagged_when_actually_moving_away(tmp_path):
    # Moves steadily 20px/frame for 10 frames -> travels 200px, well past radius.
    rows = [_row(i, float(i), "1", "cargo", "civilian", 0.9, 100 + i * 20, 100) for i in range(10)]
    path = _write_csv(tmp_path, rows)
    df = load_detection_log(path)
    flags = detect_anomalies(df, loiter_radius_px=40, loiter_min_duration_s=4)
    assert not any(f.reason == "loitering" for f in flags)


# ---------------------------------------------------------------------------
# Sudden course change
# ---------------------------------------------------------------------------

def test_sudden_course_change_flagged_on_sharp_turn(tmp_path):
    # Moves east for 3 points, then abruptly moves north (90 deg turn).
    rows = [
        _row(0, 0.0, "1", "cargo", "civilian", 0.9, 100, 100),
        _row(1, 1.0, "1", "cargo", "civilian", 0.9, 130, 100),
        _row(2, 2.0, "1", "cargo", "civilian", 0.9, 160, 100),
        _row(3, 3.0, "1", "cargo", "civilian", 0.9, 160, 60),   # turns north
    ]
    path = _write_csv(tmp_path, rows)
    df = load_detection_log(path)
    flags = detect_anomalies(df, course_change_deg=60, min_move_px=5)
    assert any(f.reason == "sudden_course_change" for f in flags)


def test_no_course_change_flagged_on_straight_line(tmp_path):
    rows = [_row(i, float(i), "1", "cargo", "civilian", 0.9, 100 + i * 30, 100) for i in range(6)]
    path = _write_csv(tmp_path, rows)
    df = load_detection_log(path)
    flags = detect_anomalies(df, course_change_deg=60, min_move_px=5)
    assert not any(f.reason == "sudden_course_change" for f in flags)


def test_jitter_below_min_move_does_not_trigger_false_course_change(tmp_path):
    # Tiny sub-pixel-scale jitter shouldn't be treated as a heading at all.
    rows = [
        _row(0, 0.0, "1", "cargo", "civilian", 0.9, 100.0, 100.0),
        _row(1, 1.0, "1", "cargo", "civilian", 0.9, 101.0, 100.0),  # move=1px, below min_move_px
        _row(2, 2.0, "1", "cargo", "civilian", 0.9, 100.0, 101.0),  # move=1px, below min_move_px
        _row(3, 3.0, "1", "cargo", "civilian", 0.9, 130.0, 100.0),  # first real move
    ]
    path = _write_csv(tmp_path, rows)
    df = load_detection_log(path)
    flags = detect_anomalies(df, course_change_deg=60, min_move_px=5)
    assert not any(f.reason == "sudden_course_change" for f in flags)


# ---------------------------------------------------------------------------
# Restricted zone entry
# ---------------------------------------------------------------------------

def test_restricted_zone_entry_flagged_when_path_crosses_zone(tmp_path):
    # Zone is a box from (150,150) to (250,250); path passes straight through.
    rows = [
        _row(0, 0.0, "1", "military_vessel", "military", 0.9, 100, 200),
        _row(1, 1.0, "1", "military_vessel", "military", 0.9, 200, 200),  # inside zone
        _row(2, 2.0, "1", "military_vessel", "military", 0.9, 300, 200),
    ]
    path = _write_csv(tmp_path, rows)
    df = load_detection_log(path)
    flags = detect_anomalies(df, restricted_zone=(150, 150, 250, 250))
    assert any(f.reason == "restricted_zone_entry" for f in flags)


def test_restricted_zone_not_flagged_when_path_never_enters(tmp_path):
    rows = [
        _row(0, 0.0, "1", "cargo", "civilian", 0.9, 0, 0),
        _row(1, 1.0, "1", "cargo", "civilian", 0.9, 10, 10),
        _row(2, 2.0, "1", "cargo", "civilian", 0.9, 20, 20),
    ]
    path = _write_csv(tmp_path, rows)
    df = load_detection_log(path)
    flags = detect_anomalies(df, restricted_zone=(150, 150, 250, 250))
    assert not any(f.reason == "restricted_zone_entry" for f in flags)


def test_restricted_zone_none_by_default_skips_check(tmp_path):
    # No zone provided -> zone check simply never runs, no crash.
    rows = [_row(i, float(i), "1", "cargo", "civilian", 0.9, 200, 200) for i in range(2)]
    path = _write_csv(tmp_path, rows)
    df = load_detection_log(path)
    flags = detect_anomalies(df, restricted_zone=None)
    assert not any(f.reason == "restricted_zone_entry" for f in flags)


# ---------------------------------------------------------------------------
# Multi-track isolation (bug class: one track's data leaking into another's)
# ---------------------------------------------------------------------------

def test_flags_are_correctly_attributed_per_track(tmp_path):
    rows = [
        # Track 1: loiters
        *[_row(i, float(i), "1", "cargo", "civilian", 0.9, 100, 100) for i in range(6)],
        # Track 2: moves in a straight line, should not be flagged for anything
        *[_row(i, float(i), "2", "yacht", "small_craft", 0.9, 300 + i * 30, 300) for i in range(6)],
    ]
    path = _write_csv(tmp_path, rows)
    df = load_detection_log(path)
    flags = detect_anomalies(df, loiter_radius_px=40, loiter_min_duration_s=4)
    track1_flags = [f for f in flags if f.track_id == "1"]
    track2_flags = [f for f in flags if f.track_id == "2"]
    assert any(f.reason == "loitering" for f in track1_flags)
    assert track2_flags == []


# ---------------------------------------------------------------------------
# Output rendering / end-to-end
# ---------------------------------------------------------------------------

def test_render_markdown_no_flags():
    text = render_markdown([], source_name="clip1")
    assert "No anomalies" in text


def test_generate_anomaly_report_writes_file(tmp_path):
    rows = [_row(i, float(i), "1", "cargo", "civilian", 0.9, 100, 100) for i in range(11)]
    csv_path = _write_csv(tmp_path, rows)
    out_path = tmp_path / "nested" / "anomalies.md"
    result = generate_anomaly_report(csv_path, out_path, loiter_radius_px=40, loiter_min_duration_s=4)
    assert result == out_path
    assert out_path.exists()
    assert "Loitering" in out_path.read_text(encoding="utf-8")


def test_generate_anomaly_report_raises_on_missing_csv(tmp_path):
    with pytest.raises(AnomalyDetectionError):
        generate_anomaly_report(tmp_path / "nope.csv", tmp_path / "out.md")
