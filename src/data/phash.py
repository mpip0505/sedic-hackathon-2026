"""Perceptual hashing + near-duplicate detection.

A DCT-based perceptual hash (pHash) tolerant to resizing/recompression, plus a
banded-LSH grouping that finds near-duplicate clusters without an O(n^2) sweep.
Used by merge.py to drop cross-dataset overlaps (FGSCR-42 ⊂ DOTA+HRSC;
ShipRSImageNet ⊂ HRSC+FGSD) and by validate.py to assert no train/val leakage.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_HASH_SIZE = 8              # -> 64-bit hash
_HIGHFREQ_FACTOR = 4       # DCT computed on a 32x32 image
_BANDS = 8                 # 8 bands x 8 bits = 64; catches hamming distance <= 7


def _dct_matrix(n: int) -> np.ndarray:
    """Orthonormal DCT-II basis matrix (so D @ x @ D.T is a 2-D DCT-II)."""
    k = np.arange(n).reshape(-1, 1)
    x = np.arange(n).reshape(1, -1)
    d = np.cos(np.pi * (2 * x + 1) * k / (2 * n)) * np.sqrt(2.0 / n)
    d[0] *= 1 / np.sqrt(2)
    return d


_IMG_SIZE = _HASH_SIZE * _HIGHFREQ_FACTOR
_DCT = _dct_matrix(_IMG_SIZE)


def phash_pixels(im: Image.Image) -> int:
    """Perceptual hash of an in-memory PIL image (no file I/O)."""
    gray = im.convert("L").resize((_IMG_SIZE, _IMG_SIZE), Image.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float64)
    dct = _DCT @ pixels @ _DCT.T
    low = dct[:_HASH_SIZE, :_HASH_SIZE]
    med = np.median(low)
    bits = (low > med).flatten()

    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def phash(path: Path | str) -> int | None:
    """Return a 64-bit perceptual hash for an image file, or None if unreadable."""
    try:
        with Image.open(path) as im:
            return phash_pixels(im)
    except Exception as exc:  # noqa: BLE001 - malformed images must not crash
        logger.warning("cannot hash %s: %s", path, exc)
        return None


def hamming(a: int, b: int) -> int:
    """Hamming distance between two 64-bit hashes."""
    return (a ^ b).bit_count()


def _band(value: int, index: int, band_bits: int = 8) -> int:
    return (value >> (index * band_bits)) & ((1 << band_bits) - 1)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def cluster_near_duplicates(
    hashes: list[int], threshold: int = 5, bands: int = _BANDS
) -> list[list[int]]:
    """Group hash indices into near-duplicate clusters (hamming <= threshold).

    Uses banded LSH: two hashes within `threshold` bits must share a band as
    long as threshold <= bands - 1 (pigeonhole), so only same-band candidates
    are compared. Returns clusters of size >= 2 (singletons omitted).
    """
    n = len(hashes)
    uf = _UnionFind(n)
    band_bits = 64 // bands

    buckets: dict[tuple[int, int], list[int]] = {}
    for i, h in enumerate(hashes):
        for b in range(bands):
            buckets.setdefault((b, _band(h, b, band_bits)), []).append(i)

    checked: set[tuple[int, int]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for a_idx in range(len(members)):
            for b_idx in range(a_idx + 1, len(members)):
                i, j = members[a_idx], members[b_idx]
                key = (i, j) if i < j else (j, i)
                if key in checked:
                    continue
                checked.add(key)
                if hamming(hashes[i], hashes[j]) <= threshold:
                    uf.union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(uf.find(i), []).append(i)
    return [sorted(c) for c in clusters.values() if len(c) >= 2]


def greedy_representative_dedup(
    hashes: list[int], threshold: int, bands: int = _BANDS
) -> tuple[list[int], list[int]]:
    """Greedy leader dedup — drop only images near an already-KEPT representative.

    Processes `hashes` in the given order. An image is KEPT (becomes a
    representative) unless it is within `threshold` of some already-kept
    representative, in which case it is DROPPED and attributed to the nearest
    such representative. Because candidates are only ever compared against KEPT
    representatives (never pairwise-chained), no cluster can grow by transitively
    linking sequential-but-distinct frames.

    Returns (assign, dist): assign[i] is -1 if image i was kept, else the index
    of the representative it was dropped in favour of; dist[i] is the hamming
    distance to that representative (0 for kept images).
    """
    band_bits = 64 // bands
    buckets: dict[tuple[int, int], list[int]] = {}   # kept representatives only
    assign = [-1] * len(hashes)
    dist = [0] * len(hashes)

    for i, h in enumerate(hashes):
        candidates: set[int] = set()
        for b in range(bands):
            candidates.update(buckets.get((b, _band(h, b, band_bits)), ()))
        best_j, best_d = -1, threshold + 1
        for j in candidates:
            d = hamming(h, hashes[j])
            if d <= threshold and d < best_d:
                best_j, best_d = j, d
        if best_j == -1:
            for b in range(bands):
                buckets.setdefault((b, _band(h, b, band_bits)), []).append(i)
        else:
            assign[i] = best_j
            dist[i] = best_d
    return assign, dist


def cross_group_near_duplicates(
    hashes_a: list[int], hashes_b: list[int], threshold: int = 5, bands: int = _BANDS
) -> list[tuple[int, int, int]]:
    """Find near-duplicate pairs BETWEEN two groups.

    Returns (index_in_a, index_in_b, hamming) tuples. Used for the train/val
    leakage check, where a match across groups is a failure.
    """
    band_bits = 64 // bands
    buckets_b: dict[tuple[int, int], list[int]] = {}
    for j, h in enumerate(hashes_b):
        for b in range(bands):
            buckets_b.setdefault((b, _band(h, b, band_bits)), []).append(j)

    matches: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int]] = set()
    for i, h in enumerate(hashes_a):
        candidates: set[int] = set()
        for b in range(bands):
            candidates.update(buckets_b.get((b, _band(h, b, band_bits)), []))
        for j in candidates:
            if (i, j) in seen:
                continue
            seen.add((i, j))
            dist = hamming(h, hashes_b[j])
            if dist <= threshold:
                matches.append((i, j, dist))
    return matches
