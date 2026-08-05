# Datasets — provenance & licence log

Every dataset used in Project Guardian **must** have a row here **before** its
data is used. This log reflects the datasets **actually integrated** as of the
current `data/processed/` build. Counts are real (raw folders / merge output).

> 🔎 **Licence:** all four in-use sets are **Roboflow YOLO exports declaring
> CC BY 4.0** — attribute the Roboflow workspace. NOTE: `shiprsimagenet`'s
> underlying imagery derives from the original **ShipRSImageNet** (academic /
> research use only); we use the CC BY 4.0 Roboflow re-export, but cite both for
> attribution. Never commit dataset images or weights (`data/`, `models/` are
> git-ignored).

## In use (integrated into `data/processed/`)

| Dataset (folder) | Domain | Roboflow source | Ver | Licence | Raw imgs | After remap (imgs / boxes) | Classes contributed |
|---|---|---|---|---|---:|---:|---|
| `military_ships` | aerial | [`hanif-noer-r/military-ships`](https://universe.roboflow.com/hanif-noer-r/military-ships) | 1 | CC BY 4.0 | 2,746 | 2,641 / 11,208 | military + civilian (50→8 collapse) |
| `seaships` | surface | [`ship-detection-cedpa/seaships-spcag`](https://universe.roboflow.com/ship-detection-cedpa/seaships-spcag) | 1 | CC BY 4.0 | 6,979 | 6,979 / 9,198 | civilian + small craft (6→schema) |
| `shiprsimagenet` | aerial | [`convertvoctoyolo/shiprsimagenet`](https://universe.roboflow.com/convertvoctoyolo/shiprsimagenet) | 39 | CC BY 4.0 (derives from academic ShipRSImageNet) | 4,579 | 4,535 / 34,299 | military + civilian (50→8 collapse) |
| `military_surface` | **surface** | [`hannah-agkvq/military-ship-detection-qxv5m`](https://universe.roboflow.com/hannah-agkvq/military-ship-detection-qxv5m) | 2 | CC BY 4.0 | 3,011 | 3,000 / 3,713 | **military_vessel only** (single-class `ship`→military; first REAL surface-military source) |

_"After remap" = output of `yolo2yolo` into `data/interim/` (images with zero
kept boxes are dropped; polygon labels enveloped to horizontal boxes). Added by
P1, 2026-07-25; `military_surface` added 2026-07-27._

### Merged split (`data/processed/`, seed 42, greedy dedup @ threshold 3)
_Rebuilt from scratch 2026-07-28 with `military_surface` included and
cross-domain `surface_synth` copy-paste **disabled** (real surface-military
data now exists — see below)._
- Collected 17,155 → **3,655 near-duplicates dropped (21.3%)** → **13,500 kept**.
- Split: train **9,452** · val **2,699** · test **1,349**.
- `military_vessel` (images): train 5,768 · val 1,653 · test 835 — now a mix of
  aerial + real surface, not aerial-only.
- `military_vessel` by domain (boxes / images):
  | split | aerial boxes | aerial imgs | surface boxes | surface imgs |
  |---|---:|---:|---:|---:|
  | train | 19,058 | 3,718 | 2,541 | 2,050 |
  | val   | 5,422  | 1,068 | 722   | 585   |
  | test  | 2,633  | 542   | **371** | 293   |
- Synthetic **surface-military** cross-domain copy-paste (`balance.py`): **0** —
  ran with `--no-cross-domain` this build since `military_surface` supplies
  real frontal-view military data (371 real surface-military boxes now sit in
  TEST, unaugmented). Re-enable in `configs/train_baseline.yaml`
  (`balance.cross_domain.enabled`) if surface recall needs a further boost.

> ✅ **Dedup note (de-chained):** the old single-linkage clustering dropped 46%
> (6,545), but ~78% of that was **transitive chaining** (largest cluster ≈ 2,961
> distinct SeaShips frames merged via A~B~C…). Switched to **greedy leader dedup**
> (drop only if within threshold of a KEPT representative) and lowered the
> threshold 5 → **3** (in `configs/schema.yaml` `dedup:`). New drop rate 25%,
> largest cluster 38, every drop ≤ threshold — no chaining. See
> `outputs/dedup_audit/audit.md` for evidence. Genuine `military_ships` ⊂
> `shiprsimagenet` exact overlaps (hamming 0) are still caught.

> 🔗 **Shared taxonomy:** `military_ships` and `shiprsimagenet` use the SAME
> ShipRSImageNet 50-class taxonomy, so both go through one collapse in
> `configs/schema.yaml` (anchor `&shiprs50`). `military_ships` is NOT
> all-military — it contains civilians too.

## Acquired — downloaded, NOT yet integrated

Fetched into `data/raw/` but **not** in `data/processed/`, **not** in
`configs/schema.yaml` (`mappings:` / `domains:`), and **not** trained into
`models/baseline_best.pt`. Every result and recall number in this repo is from a
build that **excludes** this set.

| Dataset (folder) | Domain | Roboflow source | Ver | Licence | Raw imgs | Fetch | Purpose |
|---|---|---|---|---|---:|---|---|
| `civilian_gapfill` | surface | [`boats-ri7td/speedboat`](https://universe.roboflow.com/boats-ri7td/speedboat) | 2 (generated 2025-02-20) | CC BY 4.0 (declared by the Roboflow project) | 6,213 | `python scripts/download_civilian_gapfill.py` | Close-view **civilian** surface imagery — intended to reduce civilian-vessels-detected-as-`military_vessel` false positives |

**Real class list.** The Roboflow project is *named* "speedboat", but it is a merge
of several upstream sets and the exported taxonomy is **18 class names, most of them
junk**. Read from the Roboflow project API on 2026-08-05 (class name → annotation
count; 10,816 boxes total):

| Kind | Classes (annotation count) |
|---|---|
| **Vessel classes (usable)** | `Fishing-boats` 2,820 · `speedboat` 211 · `Yacht` 170 · `tugboat` 89 — **3,290 boxes (30%)** |
| Non-vessel | `Human Fall` 245 |
| **Junk** — Roboflow README/export boilerplate ingested as class names | `undefined` 1,619 · `ship_detection - v1 2025-01-02 1:34pm` 1,338 · `==============================` 776 · `fishing boat - v2 2022-01-27 11:32pm` 690 · `all - v2 2023-05-24 7:41pm` 544 · `Roboflow is an end-to-end computer vision platform that helps you` 466 · `* annotate, and create datasets` 431 · `* understand and search unstructured image data` 361 · `* export, train, and deploy computer vision models` 357 · `* collaborate with your team on computer vision projects` 321 · `* use active learning to improve your dataset over time` 276 · `This dataset was exported via roboflow.com on January 11, 2024 at 6:42 AM GMT` 70 · `* collect & organize images` 32 — **7,281 boxes (67%)** |

**Ingest notes (LEAD, before this can be merged):**
- The junk entries are **real boxes carrying unusable label text**, not empty
  classes. Each must be mapped explicitly in `configs/schema.yaml` — most of them to
  `null` (dropped) — or they poison the taxonomy. Do **not** use a `"*":` wildcard
  here (see the 2026-07-25 decision on `military_ships`).
- Likely mapping: `Fishing-boats` → `fishing_boat`, `speedboat` → `speedboat`,
  `Yacht` → `yacht`, `Human Fall` → `null`. `tugboat` has no schema class — decide
  `null` or fold into a civilian class; do not add a class to satisfy one source.
- **No `tanker` class exists in this set**, so it does *not* close the `tanker`
  surface gap. It does supply surface `speedboat` and `yacht` — two of the three
  zero-surface classes.
- Roboflow's own split (train 4,370 / valid 1,121 / test 722) is discarded as usual;
  `merge.py` does its own stratified split.
- `seaships` is also surface + civilian — expect dedup overlap; run the normal greedy
  dedup and re-check `validate`'s leakage report.
- After any retrain that includes this set, the **military recall gate must be
  re-measured** (`src/eval/detail.py` → `outputs/eval/test_eval.md`) before any number
  is quoted anywhere.

## Candidate / not yet integrated

Listed in `configs/schema.yaml` `mappings`/`domains` as stubs but **not used** in
the current build. Mind the dedup overlaps if added (they cause train/test leak):

| Dataset | Domain | Status | Notes |
|---|---|---|---|
| Singapore Maritime (SMD) | surface | candidate | viewpoint variety + small craft; stub mapping only |
| HRSC2016 | aerial | candidate | warships; **overlaps** ShipRSImageNet (dedup risk) |
| DOTA | aerial | excluded | generic `ship`; mapped to `null` to avoid leakage |

## Planned / pending

| Dataset | Domain | Purpose | Status |
|---|---|---|---|
| Custom RMN set | surface + aerial | val + bonus fine-grained (`malaysian_rmn` vs `foreign`), and to help close the **surface-military gap** | not started (P4) |

### Notes
- Map each source's native class names in `configs/schema.yaml` under
  `mappings:` — do not translate labels anywhere else.
- The `surface_synth` cross-domain augmentation is a stopgap for zero real
  surface-military data; real frontal military imagery (Custom RMN + Roboflow
  frontal warship sets) is still the priority.
