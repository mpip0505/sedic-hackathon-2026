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
- Collected 14,155 → **3,583 near-duplicates dropped (25.3%)** → **10,572 kept**.
- Split: train **7,403** · val **2,114** · test **1,055**.
- `military_vessel`: train 3,721 · val 1,079 · test 528 (all real, aerial).
- Synthetic **surface-military** (cross-domain copy-paste, `balance.py`): ~380
  images / ~572 military instances tagged `surface_synth` in `domains.json`,
  **train split only** (with a val/test near-duplicate leak guard).

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
