"""Tests for phash dedup — greedy (no chaining) vs legacy cluster (chains)."""

from __future__ import annotations

from src.data import phash


def test_greedy_does_not_chain():
    # A~B (dist 3) and B~C (dist 3), but A~C (dist 6) > threshold 3.
    a = 0
    b = 0b000_00111            # 3 bits set -> hamming(a,b)=3
    c = 0b000_111111           # 6 bits set -> hamming(b,c)=3, hamming(a,c)=6
    assert phash.hamming(a, b) == 3
    assert phash.hamming(b, c) == 3
    assert phash.hamming(a, c) == 6

    assign, dist = phash.greedy_representative_dedup([a, b, c], threshold=3)
    # A kept (rep); B dropped in favour of A; C kept (too far from the only kept
    # representative A — B never becomes a bridge).
    assert assign[0] == -1
    assert assign[1] == 0 and dist[1] == 3
    assert assign[2] == -1


def test_legacy_cluster_chains_them_together():
    a, b, c = 0, 0b111, 0b111111
    # Single-linkage merges A-B-C into ONE cluster via the B bridge.
    clusters = phash.cluster_near_duplicates([a, b, c], threshold=3)
    assert len(clusters) == 1
    assert sorted(clusters[0]) == [0, 1, 2]


def test_greedy_keeps_distinct_images():
    # Two far-apart images are both kept.
    assign, _ = phash.greedy_representative_dedup([0, (1 << 40) - 1], threshold=3)
    assert assign == [-1, -1]
