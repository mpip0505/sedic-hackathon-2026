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
| **Multi-angle detection (frontal + aerial)** | Both domains present in every split. `military_vessel` now has **real surface** coverage via `military_surface` (2,928 surface imgs; test holds 371 real surface-military instances). The earlier `surface_synth` copy-paste stopgap is set to zero for the next run — see §2.3 / §4. |
| **Recall > 90% on military classes** | Infrastructure ready (low `conf_military` threshold, gate measured on held-out **test** split, abort-if-zero-military). Actual recall **PENDING** (training not run). |

### Pipeline checklist

| Stage | State | Notes |
|---|---|---|
| Phase 0 scaffold | ✅ Done | dirs, configs, CI, `.gitignore`, requirements |
| Contracts (`schema.yaml`, `predict()`/`Detection`, `--stub`) | ✅ Done | stub runs with no torch/weights |
| Data acquisition (3 datasets) | ✅ Done | Roboflow YOLO exports in `data/raw/` |
| Converters (voc / dota / cls / **yolo2yolo**) | ✅ Done | all unit-tested |
| Schema class mapping (50 → 8 collapse) | ✅ Done | anchored, shared by 2 datasets |
| Merge → dedup → stratified split | ✅ Done | greedy de-chained dedup; `data/processed/` produced |
| `validate` (label sanity + leakage) | ✅ Done | **PASS** (leak threshold = dedup threshold) |
| Balance / copy-paste augmentation | ✅ Available | same-domain + cross-domain `surface_synth` (train-only); `surface_synth` **set to zero for next run** now that real surface-military exists — current build is real-only |
| Training wrapper (`src/train/train.py`) | 🟡 Built, **not run** | needs GPU + Ultralytics + weights |
| Eval / military-recall gate (`src/eval/metrics.py`) | 🟡 Built, **not run** | runs after training |
| Real inference path (`predict()` non-stub) | 🔴 Not started | `NotImplementedError` placeholder |
| Detection log on Qualifier Clip | 🔴 Not started | clip not yet provided |
| GUI (`app/app.py`) | 🟡 Skeleton done (stub-wired) | no box drawing / video overlay yet |
| Bonus: oriented boxes (OBB) | 🔴 Not started | HBB envelope in place as precursor |
| Bonus: RMN-vs-foreign 2nd stage | 🔴 Not started | `src/fine_grained/` is an empty package |
| Deliverables (brief / video / poster) | 🔴 Not started | |
| CI (ruff + schema + stub + pytest) | ✅ Done | 40 tests, all green |

---

## 2. Data pipeline

### 2.1 Datasets in use

All four are **Roboflow YOLO exports** placed in `data/raw/<name>/` (git-ignored).

| Dataset (folder) | Domain | Source (Roboflow) | Ver | Licence (as declared) | Raw imgs |
|---|---|---|---|---|---:|
| `seaships` | surface | `ship-detection-cedpa/seaships-spcag` | 1 | CC BY 4.0 | 6,979 |
| `military_ships` | aerial | `hanif-noer-r/military-ships` | 1 | CC BY 4.0 | 2,746 |
| `shiprsimagenet` | aerial | `convertvoctoyolo/shiprsimagenet` | 39 | CC BY 4.0 | 4,579 |
| `military_surface` | surface | `hannah-agkvq/military-ship-detection-qxv5m` | 2 | CC BY 4.0 | 3,011 |

> ⚠️ **Licence caveat:** the Roboflow exports each declare **CC BY 4.0**, but the
> *original* SeaShips and ShipRSImageNet datasets are academic-use-only. Do not
> claim commercial rights; attribute all three. `data/DATASETS.md` has been
> reconciled to these real datasets/counts/licences.

### 2.2 The 50 → 8 class collapse (`configs/schema.yaml`)

`military_ships` and `shiprsimagenet` share the **identical ShipRSImageNet 50-class
taxonomy**, so one collapse (a YAML anchor `&shiprs50`) serves both. `seaships` has
its own 6-class taxonomy. The detector taxonomy is deliberately coarse — 8 classes.

Collapse of the 50 fine types:

| → schema class | # source types | Examples |
|---|---:|---|
| `military_vessel` | 34 | Arleigh Burke DD, Nimitz, Submarine, Perry FF, Ticonderoga, LHA/LSD/Osumi/Yu* landing, Other Warship/Destroyer/Frigate/Auxiliary |
| `cargo` | 3 | Cargo, Other Merchant, RoRo |
| `yacht` | 2 | Yacht, Sailboat |
| `container_ship` | 1 | Container Ship |
| `tanker` | 1 | Oil Tanker |
| `passenger_ferry` | 1 | Ferry |
| `fishing_boat` | 1 | Fishing Vessel |
| `speedboat` | 1 | Motorboat |
| `null` (dropped) | 6 | Barge, Dock, Other Ship, Test Ship, Tugboat, Medical Ship |

**Judgment calls** (as actually implemented in `schema.yaml`):

| Source type | Mapped to | Rationale |
|---|---|---|
| Hovercraft | `military_vessel` | LCAC landing craft |
| Patrol | `military_vessel` | patrol combatant |
| Training Ship | `military_vessel` | naval training / auxiliary |
| **Medical Ship** | **`null`** | hospital ship — not a threat/combatant; dropped |
| Other Merchant, RoRo | `cargo` | generic merchant → coarse cargo bucket |
| Barge, Tugboat | `null` | working craft with no clean schema class |
| Test Ship, Dock, Other Ship | `null` | unmappable / shore infrastructure |

### 2.3 Counts, domains, and split

Boxes written per source (converter output → `data/interim/`, after Medical Ship
→ null): `seaships` 9,198 (6,979 imgs) · `military_ships` 11,208 (2,641 imgs) ·
`shiprsimagenet` 34,299 (4,535 imgs) · `military_surface` 3,713 (3,000 imgs).

After merge (**greedy** dedup at threshold 3 + stratified 70/20/10, seed 42):
**17,155 collected → 3,655 near-duplicates dropped → 13,500 kept** (21.3% drop —
see §2.4 for the de-chaining). Of the 3,655 drops, only **72** were from the new
`military_surface` set, and **all 72 matched other `military_surface` frames**
(internal Roboflow near-dupes) — none collapsed against the aerial sets.

**Split sizes:** train **9,452** · val **2,699** · test **1,349**.

**Per-class images by split:**

| Class | train | val | test |
|---|---:|---:|---:|
| container_ship | 982 | 278 | 140 |
| tanker | 348 | 101 | 47 |
| cargo | 3,964 | 1,158 | 562 |
| passenger_ferry | 414 | 122 | 59 |
| yacht | 273 | 78 | 39 |
| speedboat | 502 | 138 | 65 |
| fishing_boat | 818 | 244 | 112 |
| **military_vessel** | **5,768** | **1,653** | **835** |

**Per-domain × per-class (images, summed over splits):**

| Class | aerial | surface | total |
|---|---:|---:|---:|
| container_ship | 794 | 606 | 1,400 |
| tanker | 496 | 0 | 496 |
| cargo | 2,637 | 3,047 | 5,684 |
| passenger_ferry | 352 | 243 | 595 |
| yacht | 390 | 0 | 390 |
| speedboat | 705 | 0 | 705 |
| fishing_boat | 246 | 928 | 1,174 |
| **military_vessel** | **5,328** | **2,928** | **8,256** |
| **TOTAL** | **10,948** | **7,752** | **18,700** |

_(An image with multiple classes counts once per class, so totals exceed the
13,500 kept images. Per-image domain map is written to
`data/processed/domains.json`.)_

**Real surface-military (NEW, `military_surface`):** `military_vessel` now has real
frontal warship coverage in **all three splits** — surface `military_vessel`
**instances** by split: train **2,541** / val **722** / test **371** (images
2,050 / 585 / 293). This is the first time the >0.90 gate can be measured on the
**surface** domain: the held-out test split now holds **371 real surface-military
instances** (293 images), vs **0** before.

**Synthetic `surface_synth` — set to zero for the next run.** Real frontal warships
supersede the cross-domain copy-paste stopgap, so `balance.py`'s cross-domain
generation is disabled for the next training build; the current `data/processed/`
is **real-only** (no `surface_synth`). The synth code is retained and toggleable for
ablation/comparison — see the decision log.

### 2.4 Key findings

- **Shared taxonomy:** `military_ships` and `shiprsimagenet` are the same 50-class
  ShipRSImageNet taxonomy → one collapse, shared via YAML anchor to prevent drift.
- **Wildcard-stub bug (caught & fixed):** the scaffold stub mapped
  `roboflow_military_ships: {"*": military_vessel}`, i.e. *everything* → military.
  But `military_ships` actually contains civilians (6,034 military vs ~5,000
  civilian boxes). That stub would have poisoned the military class. Replaced with
  the real 50→8 collapse; the wildcard was retired.
- **Dedup was over-pruning by chaining — fixed.** The original single-linkage
  clustering (threshold 5) dropped 6,545 images (46%), but ~78% of drops came from
  transitive **chaining** (A~B~C…, each hop ≤5) — one cluster swallowed **2,961**
  distinct SeaShips frames. Switched to **greedy leader dedup**: an image is
  dropped only if within threshold of an already-**kept** representative (never
  pairwise-chained), and the threshold was lowered **5 → 3** (conservative). New
  result: **3,583 dropped (25%)**, largest cluster **38**, every drop ≤ threshold
  (histogram `{0: 1114, 2: 2469}`). ~2,962 distinct frames recovered; surface
  civilian coverage roughly tripled (e.g. cargo surface 1,053 → 3,047). Dedup runs
  **before** the split; greedy guarantees no two kept images are within threshold,
  so nothing leaks. Configurable in `schema.yaml` (`dedup.method`/`.threshold`).
- **Synthetic surface-military + leak guard (now superseded by real data).** When
  `military_vessel` had zero real surface examples, `balance.py` pasted aerial
  military crops onto surface (SeaShips) backgrounds → `surface_synth`, TRAIN only,
  with a leak guard (each candidate hashed and **dropped if it near-duplicates a
  val/test image**, same threshold). With `military_surface` now supplying real
  surface warships, this generation is **set to zero for the next run**; the code +
  leak guard stay toggleable for ablation.
- **One duplicate threshold everywhere.** merge dedup, `validate`'s leakage check,
  and the `balance` synth guard all read `schema.dedup.threshold`, so "duplicate"
  means the same thing in all three (otherwise validate false-alarms on pairs
  dedup deliberately kept).
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
| `data/phash.py` | DCT perceptual hash + **greedy leader dedup** (no chaining) + legacy banded-LSH clustering |
| `data/schema_utils.py` | Schema/group/domain/**dedup** lookups, `data.yaml` regen, drift check |
| `data/merge.py` | Combine interim → dedup (greedy) → stratified split (class × domain) → `processed/` + `manifest.csv` + `domains.json` + `outputs/dedup_audit/` |
| `data/balance.py` | Copy-paste augment military — **same-domain + cross-domain (`surface_synth`)** with val/test leak guard, **train only** |
| `data/validate.py` | Label sanity (coords ∈ [0,1], class range, non-empty) + train/val/test leakage (threshold = `schema.dedup.threshold`); exits nonzero on failure |
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
- **Dedup before split, no chaining:** greedy leader dedup drops only images within
  threshold of a **kept** representative, so kept images are pairwise > threshold —
  no near-duplicate can straddle splits, and distinct frames aren't chained away.
- **One duplicate threshold** shared by merge / validate / balance guard, so the
  leakage check never contradicts the dedup decision.
- **CI:** `ruff check .`, schema/data-match assertion, stub-contract run, and the
  full 40-test pytest suite on every push (torch/ultralytics not required).

---

## 4. Known gaps & risks

- **🟢 Surface-military data — RESOLVED (was the highest-priority gap).** The
  `military_surface` set adds **real** frontal warship imagery: `military_vessel` now
  has surface coverage in all three splits (test: 371 real instances / 293 images),
  so the gate can finally be measured on the surface domain. The synthetic
  `surface_synth` copy-paste stopgap is set to zero for the next run (real frontal
  warships supersede it); the code stays toggleable for ablation. Remaining follow-up:
  a baseline retrain on this data, then re-measure the surface gate (currently
  PENDING — `models/baseline_best.pt` predates this set).
- **🟡 Thin / zero classes on the surface side:** `tanker`, `yacht`, `speedboat` have
  **0 surface** instances; `yacht` (390) is the thinnest overall. Surface civilian
  coverage improved a lot after de-chaining (cargo 3,047, container 606, fishing 928
  on surface).
- **🟡 Dedup threshold vs video frames.** Threshold lowered to 3 (greedy) fixed the
  chaining over-prune, but SeaShips is video-derived: adjacent frames at hamming 4–5
  are now treated as distinct and can land in different splits. Greedy guarantees no
  *kept* pair ≤ threshold (so leak-free by our definition), but if near-adjacent
  frames should count as leakage regardless, the fuller fix is **scene/group-aware
  splitting** (keep a scene's frames in one split) — a larger change, not yet done.
- **⚪ Nothing trained or evaluated yet.** No mAP, no recall, no confusion matrix —
  all **PENDING** a real Ultralytics run on GPU. The train→copy-weights→gate path is
  built and unit-tested up to the `YOLO(...)` call, but **not exercised end-to-end**.
- **⚪ No Qualifier Clip yet**, so the detection-log deliverable and real inference
  path are unstarted; the GUI only runs in stub mode.

---

## 5. Next steps

**P1 — Data (critical path).** _(Done: DATASETS.md reconciled; dedup de-chained +
audited; Medical Ship → null; `surface_synth` copy-paste built; **real
surface-military `military_surface` added → `surface_synth` set to zero for next
run**.)_
1. ✅ Real frontal/surface military imagery sourced (`military_surface`; test now has
   371 real surface-military instances). Optional: more frontal warship sets / Custom
   RMN for the bonus fine-grained track.
2. Decide whether SeaShips near-adjacent frames need **scene/group-aware splitting**
   (vs the current distance-threshold dedup) before final training.

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
| 2026-07-25 | **Medical Ship → `null`** (was military_vessel) | Hospital ship isn't a threat/combatant; keep the military class to actual warfighting vessels |
| 2026-07-25 | **De-chain dedup**: greedy leader (compare to kept reps only), threshold 5 → 3 | Single-linkage chaining collapsed 2,961 distinct SeaShips frames into one cluster (46% drop, 78% chaining). Greedy + lower threshold drops only true near-dups (25%), recovers ~2,962 distinct frames, and can't chain |
| 2026-07-25 | **`surface_synth` cross-domain copy-paste** (aerial military → surface backgrounds), train-only, distinctly tagged | 0 real surface-military data vs the multi-angle requirement; a reversible stopgap that P2 can ablate. Real frontal imagery still needed |
| 2026-07-25 | **One duplicate threshold** across merge / validate / balance guard (`schema.dedup.threshold`) | A conservative dedup threshold below validate's old fixed 5 caused false leakage alarms; unifying the definition keeps them consistent |
| 2026-07-26 | **Scene-aware split deferred until pre-submission** | SeaShips is video-derived; at dedup threshold 3, near-adjacent frames (hamming 4–5) count as distinct and can split across train/test, slightly inflating civilian-surface metrics (military/aerial largely unaffected). Accepted for the BASELINE run; scene-aware (group-by-source-video) splitting is REQUIRED before any numbers enter the technical brief. Owner: P1; trigger: before the final training run / any brief numbers |
| 2026-07-27 | **Gate currently validated on AERIAL military only** (test military recall **0.929**, PASS) | The test split has **0 real surface-military** instances (`surface_synth` is train-only), so the >0.90 gate is measured entirely on aerial. `conf_military` sweep clears 0.90 across 0.05–0.30 (0.25–0.30 = lowest-precision-cost point above gate). Real held-out **surface/frontal military imagery is REQUIRED** before the gate can be claimed for the frontal domain. Owner: P1; trigger: before the technical brief. Eval: `src/eval/detail.py` → `outputs/eval/test_eval.md` |
| 2026-07-27 | **Real surface-military data added (`military_surface`); `surface_synth` reduced to zero for the next run** | Real frontal warship imagery (`hannah-agkvq/military-ship-detection-qxv5m` v2, CC BY 4.0, surface) now supplies `military_vessel` on the surface domain in all three splits (test: 371 instances / 293 images). Real frontal warships supersede the cross-domain copy-paste stopgap, so `balance.py`'s `surface_synth` generation is disabled for the next training build (current `data/processed/` is real-only). Synth code retained/toggleable for comparison. Only 72/3,000 dropped in dedup, all internal — no aerial collapse. Owner: P1 |
| 2026-07-28 | **Baseline retrained (100 epochs, RTX 3060 local) on the real-surface build — gate PASSES at 0.904** | First run to include real surface-military data end to end. Two mid-run crashes, both resumed cleanly from checkpoint (not data/config issues): (1) `results.csv` locked by Excel opened to check progress — closed Excel, resumed; (2) an unrelated harness-side background-task timeout killed the process at epoch 7 — relaunched as a detached OS process, survived to completion. Gate (via `src/eval/metrics.py`, TEST split, `conf_military=0.10`): overall military recall **0.904** (PASS, >0.90), mAP50 0.851, mAP50-95 0.651. Per-class recall mostly 0.77–0.93; `speedboat` is a low outlier at 0.384 (not gate-relevant, flagged for follow-up). Owner: P2 |
| 2026-07-28 | **`src/eval/detail.py` (conf sweep + per-domain aerial/surface recall) INCOMPLETE — deferred** | Repeated non-deterministic crashes on this run: GPU path hit a `cudnn`/`model.predict()` OOM (requested ~33GB on a 12GB card, same size every time regardless of batch/content); CPU path hit intermittent native access-violation crashes. Bisected a "failing" image range down to individual images that then passed clean on isolated retry and even on a straight retest of the original failing range — points to something environmental (a remote-session/display-context change on the machine mid-run is the leading theory) rather than a corrupt image or code bug. Two incidentally-oversized source images in `military_surface` (native res up to 6000×4000, unlike the rest of the dataset which is Roboflow-preprocessed to ≤2048px) were downsized in `data/processed` as a precaution but did not fix the crash, confirming they weren't the cause. The canonical gate (line above, via `metrics.py`) is unaffected — separate code path, completed cleanly. **TODO**: rerun `detail.py` (chunked via `--start`/`--end`/`--dump` + `--from-dumps` if still flaky) to get surface-only and aerial-only military recall specifically — the whole point of adding `military_surface`. Consider a defensive resize step in the data pipeline for any future oversized source images, since the frozen `predict()` interface could hit the same class of issue on a huge user-submitted photo. Owner: P1/P3 |

---

_Numbers here are from the current `data/processed/` build (seed 42). Regenerate this
section after any re-merge or the first training run. Accuracy/recall remain PENDING
until training is executed._
