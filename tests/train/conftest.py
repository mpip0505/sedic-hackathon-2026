"""Fixtures for train.py tests: a matching schema + data.yaml + label tree."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_CLASSES = {
    0: "container_ship", 1: "tanker", 2: "cargo", 3: "passenger_ferry",
    4: "yacht", 5: "speedboat", 6: "fishing_boat", 7: "military_vessel",
}
_SCHEMA = {"classes": _CLASSES, "groups": {"military": [7]}}


@pytest.fixture
def schema_path(tmp_path: Path) -> Path:
    path = tmp_path / "schema.yaml"
    path.write_text(yaml.safe_dump(_SCHEMA), encoding="utf-8")
    return path


@pytest.fixture
def processed(tmp_path: Path) -> Path:
    root = tmp_path / "processed"
    labels = {
        "train": [("t0", [2, 6]), ("t1", [7]), ("t2", [7, 2])],
        "val": [("v0", [7])],
        "test": [("e0", [2])],
    }
    for split, entries in labels.items():
        d = root / "labels" / split
        d.mkdir(parents=True)
        for stem, classes in entries:
            lines = [f"{c} 0.5 0.5 0.4 0.4" for c in classes]
            (d / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


@pytest.fixture
def data_yaml(tmp_path: Path, processed: Path) -> Path:
    path = tmp_path / "data.yaml"
    path.write_text(yaml.safe_dump({
        "path": str(processed),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(_CLASSES),
        "names": _CLASSES,
    }), encoding="utf-8")
    return path


@pytest.fixture
def config_path(tmp_path: Path, data_yaml: Path) -> Path:
    path = tmp_path / "train_test.yaml"
    path.write_text(yaml.safe_dump({
        "data": str(data_yaml),
        "model": "yolo11m.pt",
        "epochs": 100,
        "imgsz": 640,
        "batch": 16,
        "seed": 42,
        "project": "outputs/runs",
        "name": "unit",
        "copy_paste": 0.3,
        "conf": 0.25,
        "conf_military": 0.10,
    }), encoding="utf-8")
    return path
