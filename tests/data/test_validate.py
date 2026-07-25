"""Tests for validate.py."""

from __future__ import annotations

import shutil
from pathlib import Path

from src.data import merge, validate


def _prepare(interim: Path, tmp_path: Path, schema_path: Path) -> Path:
    processed = tmp_path / "processed"
    merge.merge(interim, processed, schema_path, seed=42, data_yaml=tmp_path / "data.yaml")
    return processed


def test_validate_passes_on_clean_merge(build_interim, tmp_path: Path, schema_path: Path):
    interim, _ = build_interim
    processed = _prepare(interim, tmp_path, schema_path)
    report = validate.validate(processed, schema_path)
    assert report.ok, report.errors[:5]
    assert report.checked_images > 0 and report.checked_boxes > 0


def test_validate_flags_out_of_range_coords(build_interim, tmp_path: Path, schema_path: Path):
    interim, _ = build_interim
    processed = _prepare(interim, tmp_path, schema_path)
    bad = next((processed / "labels" / "train").glob("*.txt"))
    bad.write_text("7 0.5 0.5 1.4 0.9\n", encoding="utf-8")  # w=1.4 out of range
    report = validate.validate(processed, schema_path)
    assert not report.ok
    assert any("outside [0,1]" in e for e in report.errors)


def test_validate_flags_class_out_of_range(build_interim, tmp_path: Path, schema_path: Path):
    interim, _ = build_interim
    processed = _prepare(interim, tmp_path, schema_path)
    bad = next((processed / "labels" / "train").glob("*.txt"))
    bad.write_text("99 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    report = validate.validate(processed, schema_path)
    assert not report.ok
    assert any("out of range" in e for e in report.errors)


def test_validate_flags_empty_label(build_interim, tmp_path: Path, schema_path: Path):
    interim, _ = build_interim
    processed = _prepare(interim, tmp_path, schema_path)
    empty = next((processed / "labels" / "train").glob("*.txt"))
    empty.write_text("", encoding="utf-8")
    report = validate.validate(processed, schema_path)
    assert not report.ok
    assert any("empty label" in e for e in report.errors)


def test_validate_flags_missing_label(build_interim, tmp_path: Path, schema_path: Path):
    interim, _ = build_interim
    processed = _prepare(interim, tmp_path, schema_path)
    orphan = next((processed / "labels" / "train").glob("*.txt"))
    orphan.unlink()
    report = validate.validate(processed, schema_path)
    assert not report.ok
    assert any("image without label" in e for e in report.errors)


def test_validate_flags_leakage(build_interim, tmp_path: Path, schema_path: Path):
    interim, _ = build_interim
    processed = _prepare(interim, tmp_path, schema_path)
    # Copy a train image+label into val -> exact-stem leakage.
    img = next((processed / "images" / "train").glob("*.png"))
    lbl = processed / "labels" / "train" / f"{img.stem}.txt"
    shutil.copy2(img, processed / "images" / "val" / img.name)
    shutil.copy2(lbl, processed / "labels" / "val" / lbl.name)
    report = validate.validate(processed, schema_path)
    assert not report.ok
    assert any("LEAKAGE" in e for e in report.errors)
