"""Tests for metrics.py that don't require Ultralytics/torch."""

from __future__ import annotations

import pytest

from src.eval import metrics


def test_gate_pass():
    r = metrics.GateResult(gate=0.90, split="test",
                           military_recall={"military_vessel": 0.94})
    assert r.passed is True
    assert r.min_military == 0.94
    assert r.split == "test"


def test_gate_fail_below_threshold():
    r = metrics.GateResult(gate=0.90, split="test",
                           military_recall={"military_vessel": 0.83})
    assert r.passed is False
    assert r.min_military == 0.83


def test_gate_indeterminate_without_military():
    r = metrics.GateResult(gate=0.90, split="test", military_recall={})
    assert r.passed is False
    assert r.min_military is None


def test_evaluate_requires_explicit_split():
    # split is keyword-only with no default: calling without it is a TypeError,
    # raised before any Ultralytics import, so the gate can't silently use val.
    with pytest.raises(TypeError):
        metrics.evaluate("weights.pt", "data.yaml")  # type: ignore[call-arg]
