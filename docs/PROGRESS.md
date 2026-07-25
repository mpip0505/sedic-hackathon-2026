# Project Guardian — Progress Report

_Last updated: 2026-07-25 · SEDIC 2026 Visual Track · Maritime Domain Awareness_

This report reflects the **actual state of the repo** (code, configs, tests) as of
the date above. No training has run yet, so all accuracy/recall numbers are marked
**PENDING** — none are invented.

---

## 1. Current status

The **data pipeline is built and has been run end-to-end on real data**; a clean,
deduplicated, stratified `data/processed/` split exists and passes validation. The
**training and evaluation wrappers are written but have not been executed** (no
GPU / Ultralytics run yet). Inference beyond the stub, the GUI beyond its skeleton,
the bonus tracks, and all deliverables are not yet started.

Against the two core competition requirements:

| Requirement | Status |
|---|---|
| **Multi-angle detection (frontal + aerial)** | Both domains present in the dataset and every split. ⚠️ but `military_vessel` is currently **aerial-only** (0 surface instances) — see §4. |
| **Recall > 90% on military classes** | Infrastructure ready (low `conf_military` threshold, gate measured on held-out **test** split, abort-if-zero-military). Actual recall **PENDING** (training not run). |

### Pipeline checklist

| Stage | State | Notes |
|---|---|---|
| Phase 0 scaffold | ✅ Done | dirs, configs, CI, `.gitignore`, requirements |
| Contracts (`schema.yaml`, `predict()`/`Detection`, `--stub`) | ✅ Done | stub runs with no torch/weights |
| Data acquisition (3 datasets) | ✅ Done | Roboflow YOLO exports in `data/raw/` |
| Converters (voc / dota / cls / **yolo2yolo**) | ✅ Done | all unit-tested |
| Schema class mapping (50 → 8 collapse) | ✅ Done | anchored, shared by 2 datasets |
| Merge → dedup → stratified split | ✅ Done | ran; `data/processed/` produced |
| `validate` (label sanity + leakage) | ✅ Done | **PASS** on current processed set |
| Balance / copy-paste augmentation | 🟡 Built, not yet run on real data | train-split-only, military-first |
| Training wrapper (`src/train/train.py`) | 🟡 Built, **not run** | needs GPU + Ultralytics + weights |
| Eval / military-recall gate (`src/eval/metrics.py`) | 🟡 Built, **not run** | runs after training |
| Real inference path (`predict()` non-stub) | 🔴 Not started | `NotImplementedError` placeholder |
| Detection log on Qualifier Clip | 🔴 Not started | clip not yet provided |
| GUI (`app/app.py`) | 🟡 Skeleton done (stub-wired) | no box drawing / video overlay yet |
| Bonus: oriented boxes (OBB) | 🔴 Not started | HBB envelope in place as precursor |
| Bonus: RMN-vs-foreign 2nd stage | 🔴 Not started | `src/fine_grained/` is an empty package |
| Deliverables (brief / video / poster) | 🔴 Not started | |
| CI (ruff + schema + stub + pytest) | ✅ Done | 34 tests, all green |

---

## 2. Data pipeline

### 2.1 Datasets in use

All three are **Roboflow YOLO exports** placed in `data/raw/<name>/` (git-ignored).

| Dataset (folder) | Domain | Source (Roboflow) | Ver | Licence (as declared) | Raw imgs |
|---|---|---|---|---|---:|
| `seaships` | surface | `ship-detection-cedpa/seaships-spcag` | 1 | CC BY 4.0 | 6,979 |
| `military_ships` | aerial | `hanif-noer-r/military-ships` | 1 | CC BY 4.0 | 2,746 |
| `shiprsimagenet` | aerial | `convertvoctoyolo/shiprsimagenet` | 39 | CC BY 4.0 | 4,579 |

> ⚠️ **Licence caveat:** the Roboflow exports each declare **CC BY 4.0**, but the
> *original* SeaShips and ShipRSImageNet datasets are academic-use-only. Do not
> claim commercial rights; attribute all three. **`data/DATASETS.md` is stale** —
> it still lists academic-only stubs with `~TBD` counts and must be reconciled to
> the table above (P1).

### 2.2 The 50 → 8 class collapse (`configs/schema.yaml`)

`military_ships` and `shiprsimagenet` share the **identical ShipRSImageNet 50-class
taxonomy**, so one collapse (a YAML anchor `&shiprs50`) serves both. `seaships` has
its own 6-class taxonomy. The detector taxonomy is deliberately coarse — 8 classes.

Collapse of the 50 fine types:

| → schema class | # source types | Examples |
|---|---:|---|
| `military_vessel` | 35 | Arleigh Burke DD, Nimitz, Submarine, Perry FF, Ticonderoga, LHA/LSD/Osumi/Yu* landing, Other Warship/Destroyer/Frigate/Auxiliary |
| `cargo` | 3 | Cargo, Other Merchant, RoRo |
| `yacht` | 2 | Yacht, Sailboat |
| `container_ship` | 1 | Container Ship |
| `tanker` | 1 | Oil Tanker |
| `passenger_ferry` | 1 | Ferry |
| `fishing_boat` | 1 | Fishing Vessel |
| `speedboat` | 1 | Motorboat |
| `null` (dropped) | 5 | Barge, Dock, Other Ship, Test Ship, Tugboat |

**Judgment calls** (as actually implemented in `schema.yaml`):

| Source type | Mapped to | Rationale |
|---|---|---|
| Hovercraft | `military_vessel` | LCAC landing craft |
| Patrol | `military_vessel` | patrol combatant |
| Training Ship | `military_vessel` | naval training / auxiliary |
| **Medical Ship** | **`military_vessel`** | hospital ship = military auxiliary/support |
| Other Merchant, RoRo | `cargo` | generic merchant → coarse cargo bucket |
| Barge, Tugboat | `null` | working craft with no clean schema class |
| Test Ship, Dock, Other Ship | `null` | unmappable / shore infrastructure |

> 🔶 **Flag for confirmation:** the task brief listed **Medical Ship → null**, but
> the repo currently maps **Medical Ship → `military_vessel`**. If `null` was
> intended, flip it in `schema.yaml` (`shiprsimagenet` block) and re-remap. All
> other judgment calls match the brief.

### 2.3 Counts, domains, and split

Boxes written per source (converter output → `data/interim/`):
`seaships` 9,198 · `military_ships` 11,235 · `shiprsimagenet` 34,390.

After merge (dedup + stratified 70/20/10, seed 42): **14,155 collected → 6,545
near-duplicates dropped → 7,610 kept.**

**Split sizes:** train **5,330** · val **1,520** · test **760**.

**Per-class images by split:**

| Class | train | val | test |
|---|---:|---:|---:|
| container_ship | 679 | 189 | 94 |
| tanker | 351 | 98 | 46 |
| cargo | 2,587 | 730 | 369 |
| passenger_ferry | 300 | 86 | 42 |
| yacht | 272 | 77 | 39 |
| speedboat | 501 | 133 | 69 |
| fishing_boat | 289 | 85 | 44 |
| **military_vessel** | **3,729** | **1,057** | **528** |

**Per-domain × per-class (images, summed over splits):**

| Class | aerial | surface | total |
|---|---:|---:|---:|
| container_ship | 794 | 168 | 962 |
| tanker | 495 | 0 | 495 |
| cargo | 2,633 | 1,053 | 3,686 |
| passenger_ferry | 351 | 77 | 428 |
| yacht | 388 | 0 | 388 |
| speedboat | 703 | 0 | 703 |
| fishing_boat | 244 | 174 | 418 |
| **military_vessel** | **5,314** | **0** | **5,314** |
| **TOTAL** | **10,922** | **1,472** | **12,394** |

_(An image with multiple classes counts once per class, so totals exceed the 7,610
kept images. Per-image domain map is written to `data/processed/domains.json`.)_

### 2.4 Key findings

- **Shared taxonomy:** `military_ships` and `shiprsimagenet` are the same 50-class
  ShipRSImageNet taxonomy → one collapse, shared via YAML anchor to prevent drift.
- **Wildcard-stub bug (caught & fixed):** the scaffold stub mapped
  `roboflow_military_ships: {"*": military_vessel}`, i.e. *everything* → military.
  But `military_ships` actually contains civilians (6,061 military vs ~5,000
  civilian boxes). That stub would have poisoned the military class. Replaced with
  the real 50→8 collapse; the wildcard was retired.
- **46% dedup drop:** 6,545 of 14,155 images removed as perceptual near-duplicates
  (hamming ≤ 5). Expected — ShipRSImageNet ↔ military_ships overlap, plus Roboflow
  augmentation/tiling producing near-identical frames. Dedup runs **before** the
  split, so a duplicate can never straddle train/val/test. Rate to revisit (§4).
- **Polygon → HBB envelope:** the ShipRSImageNet/military_ships Roboflow exports are
  **polygon/oriented labels**, not horizontal boxes. `yolo2yolo` collapses each
  polygon to its min/max horizontal envelope (horizontal-boxes-first plan); true
  HBB inputs (SeaShips) pass through unchanged. This was caught when `validate`
  rejected 11-token label lines.

---

## 3. Architecture & code

### 3.1 Contracts (the things nobody breaks unilaterally)

- **`configs/schema.yaml`** — the single source of truth for class IDs, groups
  (`military` = the recall gate), per-dataset `domains`, `fine_grained` labels, and
  all source→schema `mappings`. Nothing hardcodes class names anywhere.
- **`configs/data.yaml`** — Ultralytics dataset descriptor, **auto-generated from
  `schema.yaml`** by `merge.py` so the two can't drift (CI asserts they match).
- **`src/inference/predict.py`** — the frozen interface: a `Detection` dataclass
  (`class`, `confidence`, `bbox` [x1,y1,x2,y2] abs px, `frame`, `timestamp`) and
  `predict(source, weights, conf, conf_military, stub)`. `--stub` returns synthetic
  detections in the exact real format **with no torch/weights**, so GUI and
  integration work runs today. Real path is a documented `NotImplementedError`.

### 3.2 `src/` modules

| Module | Purpose |
|---|---|
| `data/converters/_common.py` | Shared `ClassMapper` (schema-driven), stats, HBB clamp, YOLO writer |
| `data/converters/voc2yolo.py` | Pascal VOC XML → YOLO (SeaShips/ShipRSImageNet native form) |
| `data/converters/dota2yolo.py` | DOTA oriented txt → HBB envelope |
| `data/converters/cls2det.py` | Classification folders → full-image YOLO box |
| `data/converters/yolo2yolo.py` | **YOLO passthrough remap** + polygon→HBB envelope (used for all 3 current datasets) |
| `data/phash.py` | DCT perceptual hash + banded-LSH near-duplicate clustering |
| `data/schema_utils.py` | Schema/group/domain lookups, `data.yaml` regen, drift check |
| `data/merge.py` | Combine interim → dedup → stratified split (class × domain) → `processed/` + `manifest.csv` + `domains.json` |
| `data/balance.py` | Copy-paste augment military onto same-domain backgrounds, **train only** |
| `data/validate.py` | Label sanity (coords ∈ [0,1], class range, non-empty) + train/val/test leakage; exits nonzero on failure |
| `train/train.py` | Config-driven Ultralytics wrapper (`--config`, `--set`, `--resume`, `--dry-run`) |
| `eval/metrics.py` | Per-class recall + the >90% military gate (keyword-only `split`) |
| `inference/predict.py` | Frozen detection interface + working `--stub` |
| `app/app.py` | Streamlit skeleton wired to `predict()` (stub toggle, sliders, military banner) |
| `fine_grained/` | Empty package — bonus RMN-vs-foreign 2nd stage (not started) |

### 3.3 Safeguards built in

- **Schema-drift gate:** `train` aborts loudly if `data.yaml` ≠ `schema.yaml`.
- **Abort-on-zero-military:** `train` refuses to run if the train split has zero
  `military_vessel` instances (the gate would be unmeetable); `--allow-empty-military`
  overrides deliberately.
- **Gate measured on held-out `test`:** the recall number comes from `test`, never
  `val` (val drives checkpoint selection and would be optimistically biased). The
  `split` argument is required/keyword-only so it can't silently default.
- **`conf_military = 0.10 < conf = 0.25`:** intentional lower military threshold to
  favour recall; commented as such so nobody "fixes" it.
- **Dedup before split:** perceptual dedup guarantees no near-duplicate leakage.
- **CI:** `ruff check .`, schema/data-match assertion, stub-contract run, and the
  full 34-test pytest suite on every push (torch/ultralytics not required).

---

## 4. Known gaps & risks

- **🔴 Military is aerial-only (0 surface instances).** The competition wants
  multi-angle military detection, but every `military_vessel` box comes from the two
  aerial sets; SeaShips (surface) has no military. A frontal-view military ship at
  test time is out-of-distribution. **Highest-priority data gap.** Mitigations:
  source surface/frontal military imagery (the planned Custom RMN set, Roboflow
  frontal warship sets), and/or copy-paste military onto surface backgrounds
  (`balance.py` currently pastes within the same domain only — would need a
  cross-domain mode).
- **🟡 Thin / zero classes on the surface side:** `tanker`, `yacht`, `speedboat` have
  **0 surface** instances; `yacht` (388) and `passenger_ferry` (428) are the
  thinnest overall. Civilian aerial coverage leans on ShipRSImageNet.
- **🟡 46% dedup rate to revisit.** Dropping ~6.5k images is aggressive; if it's
  removing legitimate distinct frames, raise `--dedup-threshold` (currently 5) or
  audit a sample of dropped pairs before committing to this split for final runs.
- **🟡 `data/DATASETS.md` is stale** — wrong licences (says academic-only vs the
  actual CC BY 4.0 exports), placeholder counts, and lists datasets not yet used
  (Singapore Maritime, HRSC2016, Custom RMN). Reconcile to §2.1.
- **🔶 Medical Ship mapping** differs from the brief (`military_vessel` vs `null`) —
  confirm (§2.2).
- **⚪ Nothing trained or evaluated yet.** No mAP, no recall, no confusion matrix —
  all **PENDING** a real Ultralytics run on GPU. The train→copy-weights→gate path is
  built and unit-tested up to the `YOLO(...)` call, but **not exercised end-to-end**.
- **⚪ No Qualifier Clip yet**, so the detection-log deliverable and real inference
  path are unstarted; the GUI only runs in stub mode.

---

## 5. Next steps

**P1 — Data (critical path).**
1. Close the **surface-military gap**: source frontal military imagery and/or add a
   cross-domain copy-paste mode to `balance.py`.
2. Reconcile `data/DATASETS.md` to actual datasets/licences/counts.
3. Audit a sample of dedup drops; tune `--dedup-threshold` if needed.
4. Confirm the Medical Ship mapping decision.

**P2 — Training.**
1. Run the real baseline: `python -m src.train.train --config configs/train_baseline.yaml`
   (yolo11m, 100 epochs, seed 42) on Colab/Kaggle GPU.
2. Run `balance.py` first if military recall is short; report per-class recall from
   the **test** gate.
3. Populate the README/PROGRESS results tables with the first real numbers.

**P3 — Integration + GUI.**
1. Implement the real `predict()` path (Ultralytics + per-class threshold filtering)
   behind the frozen interface.
2. Add box drawing + video overlay to `app/app.py`; wire the detection log
   (`frame/timestamp/class/confidence/bbox`) for the Qualifier Clip.

**P4 — Deliverables + bonus.**
1. Start the Custom RMN image collection (feeds both the surface-military gap and the
   bonus).
2. Draft the technical brief (dataset provenance, the 50→8 logic, military-recall
   approach) — most of §2–§3 here is reusable.
3. Bonus (only once the mandatory gate passes): OBB, then the RMN-vs-foreign 2nd
   stage in `src/fine_grained/`.

---

## 6. Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-07-24 | 8 coarse detector classes; `military_vessel` stays **one** class | Fragmenting military into fine ship types spreads positives thin and collapses recall — the entire pass/fail gate is military recall |
| 2026-07-24 | `schema.yaml` is the single contract; no hardcoded class names | 4 people pushing; one taxonomy edit must propagate everywhere without drift |
| 2026-07-24 | Frozen `Detection`/`predict()` + working `--stub` | Lets GUI/integration/eval build before any model exists |
| 2026-07-24 | `conf_military = 0.10` < `conf = 0.25` (intentional) | Recall gate rewards catching over precision on military |
| 2026-07-24 | Auto-generate `data.yaml` from `schema.yaml` | Prevents the class list drifting between the two files |
| 2026-07-25 | Perceptual dedup **before** splitting | Guarantees no near-duplicate leaks across train/val/test |
| 2026-07-25 | Stratify split across **class × domain** | Keeps rare classes (military) and both visual domains represented in every split |
| 2026-07-25 | Recall gate measured on **test**, not val | Val drives checkpoint selection → biased; test is truly held out |
| 2026-07-25 | Abort training if zero military in train split | The gate is unmeetable otherwise; fail loud, not silent |
| 2026-07-25 | Retire `"*": military_vessel` wildcard for `military_ships` | That set carries civilians too; wildcard would poison the military class |
| 2026-07-25 | Share one 50→8 collapse across both aerial sets (YAML anchor) | Identical taxonomy; anchor prevents the two mappings drifting apart |
| 2026-07-25 | **Horizontal boxes first**; `yolo2yolo` envelopes polygons to HBB | Simpler, enough to pass; OBB is a later bonus. Roboflow exports were polygon-format |
| 2026-07-25 | `military_ships` tagged **aerial** (was surface stub) | It uses the aerial ShipRSImageNet taxonomy/imagery |

---

_Numbers here are from the current `data/processed/` build (seed 42). Regenerate this
section after any re-merge or the first training run. Accuracy/recall remain PENDING
until training is executed._
