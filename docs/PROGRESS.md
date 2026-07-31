# Project Guardian — Progress Report

_Last updated: 2026-07-31 · SEDIC 2026 Visual Track · Maritime Domain Awareness_

This report reflects the **actual state of the repo** (code, configs, tests) as of
the date above. The baseline has been trained and the military-recall gate has been
measured on real, held-out test data — see §1 and the decision log for numbers.

---

## 1. Current status

The **data pipeline is built and has been run end-to-end on real data**; a clean,
deduplicated, stratified `data/processed/` split exists and passes validation. The
**baseline has been trained (100 epochs, RTX 3060 local) and the military-recall
gate has been measured** on the held-out test split, on the first build to include
real surface-military data. The **real inference path and the presentation GUI are
now built** — `predict()` runs the trained model, and `app/app.py` does live
image/video detection with tracked IDs and a CSV detection log. The bonus tracks
and all deliverables are not yet started.

Against the two core competition requirements:

| Requirement | Status |
|---|---|
| **Multi-angle detection (frontal + aerial)** | Both domains present in every split. `military_vessel` now has **real surface** coverage via `military_surface` (2,928 surface imgs; test holds 371 real surface-military instances), and the gate has been measured on both domains independently — see below. |
| **Recall > 90% on military classes** | ✅ **PASS.** Canonical gate (`metrics.py`, TEST, `conf_military=0.10`): **0.904** overall. Per-domain breakdown (`detail.py`): aerial **0.940**, surface **0.954**, overall **0.942** — both domains individually clear >0.90. |

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
| Balance / copy-paste augmentation | ✅ Available | same-domain + cross-domain `surface_synth` (train-only); `surface_synth` **set to zero** for the current build now that real surface-military exists |
| Training wrapper (`src/train/train.py`) | ✅ **Done — run** | `yolo11m`, 100 epochs, RTX 3060 local, `models/baseline_best.pt` |
| Eval / military-recall gate (`src/eval/metrics.py` + `src/eval/detail.py`) | ✅ **Done — run** | overall **0.904** PASS; per-domain aerial 0.940 / surface 0.954 PASS |
| Real inference path (`predict()` non-stub) | ✅ **Done** | Ultralytics-backed, lazy import, per-class thresholds; `--stub` unaffected |
| Detection log on Qualifier Clip | 🟡 Mechanism ready | GUI exports `frame,timestamp_s,track_id,class,group,confidence,bbox` CSV; clip not yet provided |
| GUI (`app/app.py`) | ✅ **Done** | box drawing, live thresholds, military alert, before/after, BoT-SORT video playback + CSV log |
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
  detections in the exact real format **with no torch/weights**. The real path is
  now implemented (Ultralytics, lazy-imported); the frozen shapes are unchanged and
  everything new — `load_model()`, `class_groups()`, `track_video()` — is additive.

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
| `eval/detail.py` | `conf_military` sweep + per-domain × per-class recall; chunked collection (`--start`/`--end`/`--dump`/`--from-dumps`) |
| `inference/predict.py` | Frozen detection interface + working `--stub` + real Ultralytics path + `track_video()` (BoT-SORT) |
| `app/app.py` | Streamlit demo GUI: colour-coded boxes by schema group, live threshold sliders, military alert banner, metrics strip, before/after toggle, video tracking + CSV detection log |
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

_Numbers in this section are sourced from the freshly-regenerated
`outputs/eval/test_eval.md` — read figures from there, not from memory, if
they need to go in a slide or the brief._

- **🟢 Surface-military data — RESOLVED for the gate class (was the
  highest-priority gap).** The `military_surface` set adds **real** frontal
  warship imagery: `military_vessel` now has surface coverage in all three
  splits (test: 371 real instances / 293 images). The baseline has been
  retrained on this data and the gate measured on both domains independently:
  aerial 0.940, surface 0.954 (surface actually scores *higher*). The
  synthetic `surface_synth` copy-paste stopgap is set to zero for this build;
  the code stays toggleable for ablation. **This resolves the mandatory
  requirement (military recall, multi-angle) — it does not mean every class
  in the taxonomy has both-domain coverage; see the next bullet.**
- **🟡 Multi-angle coverage is honest for 5 of 8 classes, not all 8.** Per
  `outputs/eval/test_eval.md`'s per-domain table, `cargo`, `container_ship`,
  `fishing_boat`, `military_vessel`, and `passenger_ferry` all have both
  aerial and surface instances (and recall measured on both). `speedboat`,
  `tanker`, and `yacht` have **zero surface instances** — they exist in the
  detector's test set only via aerial imagery, so their "multi-angle" claim
  does not hold. This does not touch the mandatory gate (military is one of
  the 5 covered classes), but a submission that implied every class was
  multi-angle would be overstating the data. **This is DATA-FOR's gap-fill
  job** (`docs/TEAM_TASKS.md`) — it's cheap to close (find an existing
  surface/frontal export for these three classes, not hand-annotation), and
  it now runs **in parallel** with the foreign-navy collection rather than
  after it, because it defends a mandatory requirement (multi-angle) while
  the RMN/foreign bonus classifier is optional.
- **🟡 `speedboat` recall is low (0.485 aerial, 0 surface instances)** — not part of
  the military gate, but flagged for follow-up. **Root cause diagnosed without
  retraining; see the dedicated subsection below.**
- **🟡 Dedup threshold vs video frames.** Threshold lowered to 3 (greedy) fixed the
  chaining over-prune, but SeaShips is video-derived: adjacent frames at hamming 4–5
  are now treated as distinct and can land in different splits. Greedy guarantees no
  *kept* pair ≤ threshold (so leak-free by our definition), but if near-adjacent
  frames should count as leakage regardless, the fuller fix is **scene/group-aware
  splitting** (keep a scene's frames in one split) — a larger change, not yet done.
  Worth revisiting before final numbers go in the technical brief.
- **⚪ No Qualifier Clip yet.** The detection-log mechanism is built and tested (GUI
  CSV export with tracked IDs), but it has only been exercised on a locally-made
  test clip — the real clip hasn't been provided.
- **🟡 GUI environment is fragile in two known ways, both worked around.** Streamlit's
  file watcher segfaults walking `torch.classes` (disabled in `.streamlit/config.toml`
  — edits need a manual restart), and `st.dataframe` segfaults inside pyarrow on some
  numpy/pyarrow combinations (results tables are hand-built HTML instead). Both kill
  the server process with no traceback, so anyone changing `app/app.py` should know
  the symptom: "Cannot load Streamlit frontend code" in the browser.
- **🟡 GUI runs at `conf_military = 0.25`, not the gate's 0.10.** The demo default
  favours a clean picture (recall 0.921 / precision 0.888) over max recall (0.942 at
  0.10). It's a slider, but the number quoted in the brief must be the 0.10 one.

### `speedboat` recall — root-cause diagnosis (2026-07-31, no retrain)

Diagnosed on the existing `models/baseline_best.pt`, TEST split, **no retraining
run** — this is analysis for the jury pitch and to decide whether fixing it is
worth the time before submission. Not gate-relevant (the >90% requirement
applies only to `military_vessel`, which passes at 0.942 overall).

**Confirmed recall / precision / instance count, by domain** (project
methodology — `src/eval/detail.py`, conf 0.25, IoU 0.50, same numbers as
`outputs/eval/test_eval.md`):

| | aerial | surface |
|---|---:|---:|
| instances (test) | 268 | 0 |
| recall | 0.485 | — |
| precision | 0.489 | — |

_(Ultralytics' own `model.val()` reports recall 0.384 for this class — lower
than 0.485 because it's measured at Ultralytics' internally-chosen max-F1
confidence point, not at the project's actual `conf=0.25` operating
threshold. 0.485/0.489 is the pair that reflects what the deployed system
actually does; both figures are cited around the repo, so don't mix them.)_

**Confusion matrix — where the 138/268 non-recalled instances go.** Every
speedboat GT box not matched at the operating threshold was checked against
*all* predictions on its image (any class, any confidence down to the 0.01
floor) to tell apart "never seen," "seen but under-confident," and "seen and
called something else":

| Outcome | Count | Share |
|---|---:|---:|
| Correctly detected (TP) | 130 | 48.5% |
| Missed entirely — no prediction of any class overlaps it, at any confidence | 62 | 23.1% |
| Right class, but confidence fell below the 0.25 operating threshold | 37 | 13.8% |
| Mislabeled as another class at ≥0.25 confidence | 39 | 14.6% |

Of the 39 mislabeled instances, **27 (69%) were called `yacht`**; the rest
split thinly across `fishing_boat` (5), `military_vessel` (4),
`passenger_ferry` (1), `cargo` (1), `container_ship` (1). Ultralytics' own
confusion matrix (independently computed via `model.val(plots=True)`, IoU
0.45) corroborates the same asymmetric pattern: 33 true-speedboat detections
called `yacht` vs. only 6 true-yacht detections called `speedboat` — the
model's bias runs one way, toward the larger, more common class.

**Assessment of likely causes:**

1. **Object size, not confusion, is the dominant driver.** Median GT box
   size (short side, px) on the TEST split:

   | Class | median short side (px) |
   |---|---:|
   | speedboat | **13.9** |
   | yacht | 28.6 |
   | fishing_boat | 58.5 |
   | passenger_ferry | 96.0 |
   | container_ship | 108.2 |
   | military_vessel | 133.5 |
   | cargo | 141.0 |
   | tanker | 164.0 |

   Speedboat is by far the smallest object class in the taxonomy — half the
   median size of the next-smallest class (`yacht`) and 4–12× smaller than
   everything else; a quarter of speedboat boxes are ≤9.5px on the short
   side. That little pixel signal plausibly explains both the 23%
   "missed entirely" and the 14% "right class, not confident enough" —
   symptoms of weak features, not misclassification.
2. **Small fast craft are hard to detect from directly overhead**, consistent
   with (1): speedboat is aerial-only, so it is only ever evaluated in the
   domain where its apparent size is smallest.
3. **Genuine confusion, when it happens, is concentrated on `yacht`** — the
   next-smallest, visually closest class (both are small planing-hull small
   craft; `schema.yaml` groups them together under `small_craft`). At
   ~14–29px the two are hard to distinguish even for a person.
4. **Zero surface instances (see the gap above) means cause (2) can't be
   cross-checked against a surface baseline** — there is no same-class,
   different-domain comparison point today.

**Plain-English explanation for the jury pitch (DELIV, use verbatim or adapt):**

> Speedboat has the lowest recall in the system — about 49% at our operating
> threshold — because speedboats are, on average, the smallest object in our
> entire taxonomy in the aerial imagery: a typical speedboat covers roughly a
> 14×14-pixel patch, about half the size of the next-smallest class and a
> tenth the size of a cargo ship. At that scale the model either doesn't see
> it at all (23% of the misses) or sees it but isn't confident enough to call
> it (14%), and when it does confuse it with something else, it's almost
> always with `yacht` — the visually closest small-craft class. It is not a
> gate-relevant class — the mandatory >90% recall requirement applies only to
> military vessels, which pass at 94% overall — but it's the clearest
> remaining weak spot in the system, and it's a resolution problem, not a
> labeling or data-quality problem.

**Recommendation:** a surface-domain speedboat dataset (the DATA-FOR gap-fill
job above) would close the multi-angle-coverage gap and is worth doing for
that reason alone, but it would **not** be expected to meaningfully fix the
low aerial recall — the root cause is object scale, not domain coverage, so
a real fix would be resolution-side (larger `imgsz` / tiled inference for
small aerial craft), not a data-collection one; not worth pursuing before
submission given speedboat isn't a gate class.

---

## 5. Next steps

Team expanded to **5 people on 2026-07-31**; handles below are the new ones. Full
task packets, data handoff formats and acceptance criteria live in
**`docs/TEAM_TASKS.md`** — this is the summary view.

**LEAD — ML core.** _(Done: data pipeline built and run; dedup de-chained + audited;
real surface-military added; baseline trained; gate PASSING on both domains — overall
0.904, aerial 0.940, surface 0.954; real `predict()` + presentation GUI shipped.)_
1. **Decide the scene/group-aware split question** (decision log, 2026-07-26) — flagged
   as required before any numbers enter the technical brief, so it **blocks DELIV**.
   Either implement + retrain, or record the decision to accept the current split.
2. Investigate the low `speedboat` recall (0.384) — not gate-relevant, but it is an
   obvious jury question.
3. Ingest the DATA-RMN / DATA-FOR handoffs: `schema.yaml` mapping + domain entries,
   convert → merge → dedup → validate, `data/DATASETS.md` rows.
4. Build `src/fine_grained/` (crop `military_vessel` detections → classify
   `malaysian_rmn` vs `foreign`); flip `fine_grained.enabled` when it works.
5. Run the detection log against the **Qualifier Clip** once provided — the export
   format is already in place, only the clip is missing.
6. Rehearse the Phase 2 live stress test, including the stub fallback path.

**GUI — landing page.** _(New lane; unblocked today.)_
1. Add `app/pages/` navigation with a mission/results **landing page** as the entry
   point; move the existing detection view into a page unchanged.
2. Constraints are non-negotiable and already cost this project two segfaults: no
   `st.dataframe`, no hot reload (watcher disabled), **no hardcoded metrics or class
   names** — read them from `outputs/eval/test_eval.md` and `schema.yaml`.

**DATA-RMN / DATA-FOR — the bonus set.** _(The only real critical path left.)_
1. DATA-RMN: ≥6 RMN (TLDM) classes, ~150–300 images each, both domains, delivered as
   labelled crops + `manifest.csv` with per-image source URL and licence.
2. DATA-FOR: a `foreign` bucket at rough parity, ≥5 navies, deliberately including the
   visually-similar regional ones (RSN, TNI-AL) as hard negatives. **Not** sourced from
   ShipRSImageNet/`military_ships` — those are already in the detector's train split.
3. DATA-FOR, running **in parallel** with item 2 (not after it): surface imagery for
   `speedboat` / `tanker` / `yacht`, all of which have **0 surface instances** today —
   see §4. Cheap to close (an existing export, not hand-annotation) and defends the
   mandatory multi-angle requirement, so it's no longer deprioritized behind the
   optional bonus classifier.

**DELIV — submission package.**
1. Draft the technical brief now for everything that isn't a number (dataset
   provenance, the 50→8 logic, the dual-threshold military-classification logic) —
   most of §2–§3 here is directly reusable. **Results table waits on LEAD item 1.**
2. Record the ≤5 min video off the real GUI, showing both an aerial and a frontal
   image (multi-angle is the headline requirement).
3. Phase 2 only: poster, jury pitch (prepare a **scalability** answer), final package.

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
| 2026-07-30 | **`src/eval/detail.py` root cause found + fixed; per-domain gate CLEARS on both domains (aerial 0.940, surface 0.954, overall 0.942)** | Root cause of the 2026-07-28 GPU OOM: `model.predict(stream=True, ...)` lets every single-image batch pick its own letterboxed shape (`auto=True` in Ultralytics' `pre_transform` whenever a batch's images share one shape, which single-image batches always do), and a long stream of differently-shaped images fragments/accumulates GPU memory until a normal-sized allocation fails — not a corrupt image (bisection showed identical slices OOM at *different* points depending only on how many images had already streamed in that process; the earlier "remote-session" theory doesn't hold, since it reproduced deterministically today back at the machine). Fixed by chunking collection into fresh-process slices via `--start`/`--end`/`--dump` (each fresh process resets the accumulation) then combining with `--from-dumps` — now the standard way to run this script on GPU; documented in the README. Also fixed a separate, unrelated bug: the final `print()` crashed with `UnicodeEncodeError` on Windows' cp1252 console default when printing the ✅/❌ markdown tables (after the report file was already written) — `sys.stdout.reconfigure(encoding="utf-8")` added. Results (test split, IoU 0.50): `conf_military` sweep clears the gate 0.05–0.30; per-domain military recall aerial 0.940, surface 0.954 (surface — the whole reason `military_surface` was added — actually scores *higher*), overall 0.942, all ✅ PASS. Full tables in `outputs/eval/test_eval.md` and the README `## Results` section. Owner: P1/P3 |
| 2026-07-30 | **Real `predict()` path implemented + presentation GUI built** | `predict()`'s `NotImplementedError` branch replaced with the Ultralytics path: run at the **lower** of the two thresholds, then filter per class (military at `conf_military`, rest at `conf`) — that filter is what enforces the gate. Ultralytics/torch stay **lazy imports** so `--stub` still runs with no torch. The frozen `Detection`/`predict()` shapes are untouched; tracking is **additive** (`track_video()` → `TrackedFrame`/`TrackedDetection` with BoT-SORT IDs) rather than bent into the frozen contract, and `class_groups()` reads the GUI's colour grouping from `schema.yaml` so no class name is hardcoded in `app/`. GUI: colour-coded boxes (military red / small craft amber / civilian teal), military alert banner, per-group counts, metrics strip, before/after toggle, live threshold sliders (images re-run on slider move, cached per threshold), video playback with tracked IDs and a CSV detection log. Verified end to end in-browser on real weights. Owner: P3 |
| 2026-07-31 | **Team expanded to 5; remaining work re-cut into 4 parallel lanes** (`docs/TEAM_TASKS.md`) | The serial core (data → model → gate) is finished, so the old 3-person, critical-path-first split no longer describes reality. New handles: **LEAD** (ML core + `src/fine_grained/`), **GUI** (landing page), **DATA-RMN** + **DATA-FOR** (the two halves of the bonus fine-grained set, one person each), **DELIV** (brief/video/poster). Two people on the bonus data rather than one because a binary MY-vs-Foreign classifier needs *both* halves at comparable scale, and the foreign half is the larger, messier collection job — plus the regionally-similar navies (RSN, TNI-AL) are the hard negatives that decide whether the classifier learns "Malaysian vs not" or just "grey ship vs US carrier". Data handoff is contract-first (fixed folder layout + `manifest.csv` with per-image source URL and licence) for the same reason `schema.yaml` and `predict()` are: four people collecting into one pipeline drift without a fixed shape. Sequencing: LEAD's scene-split decision goes first because it gates the numbers DELIV puts in the brief; the bonus data is the only remaining critical path, and also still the first thing to cut. Owner: LEAD |
| 2026-07-30 | **GUI avoids `st.dataframe`; Streamlit file watcher disabled** | Two separate **SIGSEGV**s killed the whole server during GUI work — no Python traceback, browser just shows "Cannot load Streamlit frontend code". (1) Streamlit's source watcher walks `torch.classes` and crashes → `fileWatcherType = "none"` in `.streamlit/config.toml` (cost: manual restart after edits, which also prevents an accidental mid-demo rerun). (2) `st.dataframe` serializes via pyarrow, which segfaulted in `pandas_compat.convert_column` on pyarrow 25 + numpy 1.26 → results tables are hand-built HTML in `render_table()`. Diagnosed with `PYTHONFAULTHANDLER=1`, which is how you get a traceback out of a native crash. Both worked around rather than version-pinned, because a Phase 2 live stress test on someone else's machine must not depend on an exact pyarrow build. `lapx` (BoT-SORT's assignment solver) added to `requirements.txt`. Owner: P3 |
| 2026-07-31 | **`speedboat` low recall root-caused without retraining — object size, not confusion or domain** | Diagnosed on the existing `models/baseline_best.pt` (no retrain) for the jury pitch: at the project's op. point (conf 0.25, IoU 0.50) recall is 0.485 / precision 0.489, all 268 test instances aerial (0 surface). Per-GT-box outcome breakdown: 48.5% correctly detected, 23.1% missed entirely (no prediction of any class at any confidence), 13.8% right class but under the 0.25 threshold, 14.6% mislabeled (69% of those as `yacht`). Median GT box short side is 13.9px — smaller than every other class by 2–12×, including the next-smallest (`yacht`, 28.6px) — which plausibly explains the missed/under-confident buckets as a weak-signal problem, not a labeling problem; the confusion that does occur concentrates on `yacht` (the visually closest `small_craft`-group class), corroborated independently by Ultralytics' own confusion matrix (33 speedboat→yacht vs. 6 yacht→speedboat). **Recommendation: a surface speedboat dataset closes the multi-angle gap but would not be expected to fix the aerial recall** — the fix for that would be resolution-side (larger `imgsz`/tiling), not more data, and isn't worth pursuing pre-submission since speedboat isn't a gate class. Full breakdown in §4. Owner: LEAD |

---

_Data numbers here are from the current `data/processed/` build (seed 42); accuracy/
recall numbers are from the baseline trained on that build (`models/baseline_best.pt`,
2026-07-28). Regenerate this section after any re-merge or retrain._
