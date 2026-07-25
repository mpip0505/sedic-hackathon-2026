"""Tests for balance.py (copy-paste augmentation)."""

from __future__ import annotations

import json
from pathlib import Path

from src.data import balance, merge, validate


def _prepare(interim: Path, tmp_path: Path, schema_path: Path) -> Path:
    processed = tmp_path / "processed"
    merge.merge(interim, processed, schema_path, seed=42, data_yaml=tmp_path / "data.yaml",
                dedup_audit_dir=tmp_path / "dedup_audit")
    return processed


def test_copy_paste_train_only(build_interim, tmp_path: Path, schema_path: Path):
    interim, _ = build_interim
    processed = _prepare(interim, tmp_path, schema_path)

    val_before = sorted((processed / "images" / "val").glob("*"))
    test_before = sorted((processed / "images" / "test").glob("*"))

    n = balance.copy_paste(processed, schema_path, copies=10, max_paste=2, seed=1)
    assert n > 0

    # Synthetic images landed only in train.
    aug = list((processed / "images" / "train").glob("augcp_*"))
    assert len(aug) == n
    for a in aug:
        assert (processed / "labels" / "train" / f"{a.stem}.txt").is_file()

    # val/test are byte-for-byte untouched.
    assert sorted((processed / "images" / "val").glob("*")) == val_before
    assert sorted((processed / "images" / "test").glob("*")) == test_before
    assert not list((processed / "images" / "val").glob("augcp_*"))


def test_copy_paste_labels_valid_and_military(build_interim, tmp_path: Path, schema_path: Path):
    interim, _ = build_interim
    processed = _prepare(interim, tmp_path, schema_path)
    balance.copy_paste(processed, schema_path, copies=8, max_paste=2, seed=7)

    aug_labels = list((processed / "labels" / "train").glob("augcp_*.txt"))
    assert aug_labels
    for lbl in aug_labels:
        lines = [ln for ln in lbl.read_text().splitlines() if ln.strip()]
        assert lines
        assert any(ln.split()[0] == "7" for ln in lines)  # military pasted in
        for ln in lines:
            coords = [float(v) for v in ln.split()[1:]]
            assert all(0.0 <= c <= 1.0 for c in coords)

    # Augmented dataset still passes validation (labels sane, no leakage).
    report = validate.validate(processed, schema_path)
    assert report.ok, report.errors[:5]


def test_copy_paste_clean(build_interim, tmp_path: Path, schema_path: Path):
    interim, _ = build_interim
    processed = _prepare(interim, tmp_path, schema_path)
    balance.copy_paste(processed, schema_path, copies=6, seed=1)
    first = len(list((processed / "images" / "train").glob("augcp_*")))
    assert first > 0
    # Re-running with --clean replaces rather than accumulates.
    balance.copy_paste(processed, schema_path, copies=6, seed=1, clean=True)
    second = len(list((processed / "images" / "train").glob("augcp_*")))
    assert second == first


# --- cross-domain (aerial military -> surface backgrounds) -------------------
def test_cross_domain_surface_synth_train_only(build_interim, tmp_path: Path,
                                               schema_path: Path):
    interim, _ = build_interim
    processed = _prepare(interim, tmp_path, schema_path)

    val_before = sorted((processed / "images" / "val").glob("*"))
    test_before = sorted((processed / "images" / "test").glob("*"))

    gen, pasted = balance.cross_domain_copy_paste(
        processed, schema_path, copies=8, max_paste=2, seed=3)
    assert gen > 0 and pasted >= gen

    aug = list((processed / "images" / "train").glob("augxd_*"))
    assert len(aug) == gen
    # train only — val/test byte-for-byte unchanged, no synth there.
    assert sorted((processed / "images" / "val").glob("*")) == val_before
    assert sorted((processed / "images" / "test").glob("*")) == test_before
    assert not list((processed / "labels" / "val").glob("augxd_*"))
    assert not list((processed / "labels" / "test").glob("augxd_*"))


def test_cross_domain_labels_and_domains_json(build_interim, tmp_path: Path,
                                              schema_path: Path):
    interim, _ = build_interim
    processed = _prepare(interim, tmp_path, schema_path)
    balance.cross_domain_copy_paste(processed, schema_path, copies=8, seed=5)

    labels = list((processed / "labels" / "train").glob("augxd_*.txt"))
    assert labels
    for lbl in labels:
        lines = [ln for ln in lbl.read_text().splitlines() if ln.strip()]
        assert any(ln.split()[0] == "7" for ln in lines)  # military pasted in
        for ln in lines:
            coords = [float(v) for v in ln.split()[1:]]
            assert all(0.0 <= c <= 1.0 for c in coords)

    # domains.json gained surface_synth entries for exactly the synthetic images.
    dj = json.loads((processed / "domains.json").read_text())
    synth = {k: v for k, v in dj["train"].items() if v == "surface_synth"}
    assert synth
    assert all(k.startswith("augxd_") for k in synth)

    # Augmented set still validates (labels sane, no leakage).
    report = validate.validate(processed, schema_path)
    assert report.ok, report.errors[:5]


def test_cross_domain_new_surface_military_counts(build_interim, tmp_path: Path,
                                                  schema_path: Path):
    interim, _ = build_interim
    processed = _prepare(interim, tmp_path, schema_path)

    before = balance.military_counts_by_domain(processed, schema_path)
    assert before.get("surface", 0) == 0        # no real surface military
    assert before.get("aerial", 0) > 0

    balance.cross_domain_copy_paste(processed, schema_path, copies=8, seed=7)
    after = balance.military_counts_by_domain(processed, schema_path)
    assert after.get("surface_synth", 0) > 0    # synthetic surface military exists
    assert after.get("aerial", 0) == before.get("aerial", 0)  # real aerial unchanged
