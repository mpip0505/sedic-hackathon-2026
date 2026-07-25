"""Tests for train.py (everything up to the Ultralytics call)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.data import schema_utils
from src.train import train


def test_apply_overrides_types(config_path: Path):
    config = train.load_config(config_path)
    merged = train.apply_overrides(config, ["epochs=50", "copy_paste=0.5", "name=exp"])
    assert merged["epochs"] == 50 and isinstance(merged["epochs"], int)
    assert merged["copy_paste"] == 0.5 and isinstance(merged["copy_paste"], float)
    assert merged["name"] == "exp"
    assert config["epochs"] == 100  # original untouched


def test_apply_overrides_bad_format(config_path: Path):
    config = train.load_config(config_path)
    with pytest.raises(SystemExit):
        train.apply_overrides(config, ["epochsIS50"])


def test_build_train_kwargs_strips_non_train_keys(config_path: Path):
    config = train.load_config(config_path)
    kwargs = train.build_train_kwargs(config)
    for k in ("model", "conf", "conf_military"):
        assert k not in kwargs
    assert kwargs["epochs"] == 100
    assert kwargs["data"].endswith("data.yaml")


def test_assert_consistent_ok(schema_path: Path, data_yaml: Path):
    train.assert_consistent(schema_path, data_yaml)  # must not raise


def test_assert_consistent_detects_drift(schema_path: Path, data_yaml: Path, tmp_path: Path):
    data = yaml.safe_load(data_yaml.read_text())
    data["names"][7] = "submarine"  # drift the taxonomy
    drifted = tmp_path / "drifted.yaml"
    drifted.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        train.assert_consistent(schema_path, drifted)
    assert "drifted" in str(exc.value).lower()


def test_count_dataset(data_yaml: Path, schema_path: Path):
    schema = schema_utils.load_schema(schema_path)
    images, per_class = train.count_dataset(data_yaml, schema)
    assert images == {"train": 3, "val": 1, "test": 1}
    assert per_class[7] == 3      # military instances: t1, t2, v0
    assert per_class[2] == 3      # cargo: t0, t2, e0
    assert per_class[6] == 1      # fishing: t0


def test_dry_run_no_training(config_path: Path, schema_path: Path):
    # --dry-run must succeed without Ultralytics/torch installed.
    rc = train.main(["--config", str(config_path), "--schema", str(schema_path),
                     "--dry-run"])
    assert rc == 0


def test_dry_run_aborts_on_drift(config_path: Path, schema_path: Path, tmp_path: Path):
    config = yaml.safe_load(config_path.read_text())
    data = yaml.safe_load(Path(config["data"]).read_text())
    data["nc"] = 7  # drift
    Path(config["data"]).write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(SystemExit):
        train.main(["--config", str(config_path), "--schema", str(schema_path),
                    "--dry-run"])


def _strip_military_from_train(processed: Path) -> None:
    """Remove military (class 7) instances from every train label file."""
    for lf in (processed / "labels" / "train").glob("*.txt"):
        kept = [ln for ln in lf.read_text().splitlines()
                if ln.strip() and ln.split()[0] != "7"]
        # leave a non-military box so files aren't empty
        lf.write_text(("\n".join(kept) or "2 0.5 0.5 0.4 0.4") + "\n", encoding="utf-8")


def test_aborts_when_no_military_in_train(config_path: Path, schema_path: Path,
                                          processed: Path):
    _strip_military_from_train(processed)
    with pytest.raises(SystemExit) as exc:
        train.main(["--config", str(config_path), "--schema", str(schema_path),
                    "--dry-run"])
    assert "military" in str(exc.value).lower()


def test_allow_empty_military_overrides_abort(config_path: Path, schema_path: Path,
                                              processed: Path):
    _strip_military_from_train(processed)
    rc = train.main(["--config", str(config_path), "--schema", str(schema_path),
                     "--dry-run", "--allow-empty-military"])
    assert rc == 0


def test_count_instances_in_split(data_yaml: Path):
    train_counts = train.count_instances_in_split(data_yaml, "train")
    assert train_counts[7] == 2      # t1, t2
    assert train.count_instances_in_split(data_yaml, "val")[7] == 1
    assert train.count_instances_in_split(data_yaml, "test")[7] == 0
