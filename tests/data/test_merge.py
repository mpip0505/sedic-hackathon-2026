"""Tests for merge.py."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.data import merge


def _run(interim: Path, tmp_path: Path, schema_path: Path):
    processed = tmp_path / "processed"
    data_yaml = tmp_path / "data.yaml"
    items = merge.merge(interim, processed, schema_path,
                        val=0.2, test=0.1, seed=42, data_yaml=data_yaml,
                        dedup_audit_dir=tmp_path / "dedup_audit")
    return items, processed, data_yaml


def test_merge_dedup_and_split(build_interim, tmp_path: Path, schema_path: Path):
    interim, expected_kept = build_interim
    items, processed, _ = _run(interim, tmp_path, schema_path)

    # Cross-dataset duplicate was dropped before splitting.
    assert len(items) == expected_kept

    # All three splits exist and together hold exactly the kept items.
    written = {s: list((processed / "images" / s).glob("*")) for s in ("train", "val", "test")}
    assert sum(len(v) for v in written.values()) == expected_kept
    assert len(written["train"]) > 0 and len(written["val"]) > 0

    # No stem appears in more than one split (no leakage).
    stems = {s: {p.stem for p in written[s]} for s in written}
    assert not (stems["train"] & stems["val"])
    assert not (stems["train"] & stems["test"])
    assert not (stems["val"] & stems["test"])

    # Every image has a matching label.
    for s in ("train", "val", "test"):
        for img in written[s]:
            assert (processed / "labels" / s / f"{img.stem}.txt").is_file()


def test_merge_military_in_train_and_val(build_interim, tmp_path: Path, schema_path: Path):
    interim, _ = build_interim
    items, _, _ = _run(interim, tmp_path, schema_path)
    mil = {it.split for it in items if 7 in it.classes}
    # Military (the rarest-ish, largest aerial stratum) should reach train+val.
    assert "train" in mil and "val" in mil


def test_merge_regenerates_data_yaml(build_interim, tmp_path: Path, schema_path: Path):
    interim, _ = build_interim
    _, _, data_yaml = _run(interim, tmp_path, schema_path)
    data = yaml.safe_load(data_yaml.read_text())
    schema = yaml.safe_load(schema_path.read_text())
    assert data["nc"] == len(schema["classes"])
    assert data["names"] == schema["classes"]
    assert data["train"] == "images/train"


def test_merge_deterministic(build_interim, tmp_path: Path, schema_path: Path):
    interim, _ = build_interim
    items1, _, _ = _run(interim, tmp_path / "a", schema_path)
    items2, _, _ = _run(interim, tmp_path / "b", schema_path)
    assign1 = {it.stem: it.split for it in items1}
    assign2 = {it.stem: it.split for it in items2}
    assert assign1 == assign2
