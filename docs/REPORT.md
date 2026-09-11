# Project Guardian — Technical Brief

**SEDIC 2026 · Visual Track · Phase 1 Preliminary Qualifier**
_Advanced Maritime Domain Awareness — multi-angle vessel detection_

_Compiled 2026-09-06. Every number below is quoted from a repo artefact, named
inline. Nothing here is estimated or recalled._

---

## How to read this document

This brief is organised **one section per Phase 1 scoring criterion**, so a judge
can tick each off in order. Each section opens with a claim box naming the
criterion, its point value, and the repo file that evidences it.

| § | Criterion | Points | Verdict |
|---|---|---:|---|
| [1](#1-mandatory-classification--30-pts) | Mandatory Classification (Civilian / Small Craft / Military, multi-angle, open-source data) | 30 | ✅ All three categories + both domains, all data open-source |
| [2](#2-performance-benchmark--25-pts) | Performance Benchmark (military/threat accuracy) | 25 | ✅ **0.936 military recall** on held-out TEST — clears both the 80% rubric bar and the brief's own stricter >90% |
| [3](#3-competitive-advantage--local-vs-foreign--10-pts) | Competitive Advantage (Malaysian vs foreign military) | 10 | ✅ Built and running — presented with a stated domain-gap caveat |
| [4](#4-technical-brief--20-pts) | Technical Brief (dataset · architecture · military-classification logic) | 20 | ✅ §4.1 / §4.2 / §4.3 below |
| [5](#5-video-demonstration--15-pts--what-this-brief-supplies) | Video Demonstration | 15 | Brief supplies the script beats — §5 |

**Global operating point for every number in this brief** (unless a table says
otherwise):

| | |
|---|---|
| **Model** | `models/baseline2_best.pt` — YOLO11m, 100 epochs, trained 2026-08-07 |
| **Split** | **TEST** — 1,588 images, held out, never trained on |
| **Military threshold** | `conf_military = 0.10` |
| **All other classes** | `conf = 0.25` |
| **IoU** | 0.50 |

---

## 1. Mandatory Classification — 30 pts

> **Criterion.** Detect and classify **Civilian** (container ships, tankers,
> cargo, passenger ferries), **Small Craft** (yachts, speedboats, fishing
> boats), and **High-Priority Military** vessels; handle **multi-angle**
> (frontal *and* aerial/satellite) views; train on **open-source** datasets.
>
> **Evidence:** `configs/schema.yaml` · `data/DATASETS.md` · `docs/PROGRESS.md` §1

### 1.1 Multi-angle proof — lead with this

The mandatory requirement is not merely "we have both kinds of image in the
training set." It is that the model **performs on both**. We therefore measured
the military recall gate **separately per domain** on the held-out TEST split
(`src/eval/detail.py`, explicit greedy PASCAL-VOC matching at the real
`conf_military = 0.10` operating point):

| Domain | Military recall | Gate > 0.90 |
|---|---:|:---:|
| **aerial / satellite** | **0.932** | ✅ |
| **surface / frontal** | **0.977** | ✅ |
| **overall** | **0.938** | ✅ **PASS** |

_Source: `README.md` § Results; `docs/PROGRESS.md` §1 and the 2026-09-04
decision-log entry. TEST split, `conf_military = 0.10`, IoU 0.50._

**Each domain independently clears the bar.** This is the multi-angle evidence:
the gate is not carried by one strong domain averaging out a weak one. Surface
(frontal) actually scores *higher* than aerial.

This was earned, not assumed. As of 2026-07-27 the TEST split held **zero real
surface-military instances**, and the decision log recorded plainly that the
gate was then "validated on AERIAL military only" and that real held-out
frontal imagery was **required** before any frontal claim could be made
(`docs/PROGRESS.md` decision log, 2026-07-27). The `military_surface` dataset
was then sourced and ingested. In the current `data/processed/` build TEST holds
**335 real surface-military images** (`data/DATASETS.md`, rebuilt 2026-08-06), and
the gate was re-measured per domain.

**Domain coverage of the data itself** (`data/DATASETS.md`, `military_vessel`
images by split and domain, current `data/processed/` build):

| Split | aerial | surface |
|---|---:|---:|
| train | 3,722 | 2,344 |
| val | 1,067 | 670 |
| test | 539 | 335 |

### 1.2 The three mandatory categories map 1:1 onto our schema

`configs/schema.yaml` defines 8 detector classes and semantic groups over them.
The groups were written to match the competition brief's own categories:

| Brief category | `schema.yaml` group | Classes (IDs) |
|---|---|---|
| **Civilian** | `civilian` | `container_ship` (0), `tanker` (1), `cargo` (2), `passenger_ferry` (3), `yacht` (4) |
| **Small Craft** | `small_craft` | `yacht` (4), `speedboat` (5), `fishing_boat` (6) |
| **High Priority (MILITARY)** | `military` | `military_vessel` (7) — **the recall gate applies here** |

Every class named in the brief's mandatory list is a first-class detector class.
`yacht` intentionally sits in both `civilian` and `small_craft`; the groups are
semantic views over classes, not a partition.

**Per-class recall, TEST split** (`README.md` § Results, canonical
`src/eval/metrics.py` pass):

| Class | Group | Recall |
|---|---|---:|
| cargo | civilian | 0.944 |
| container_ship | civilian | 0.904 |
| tanker | civilian | 0.918 |
| passenger_ferry | civilian | 0.875 |
| yacht | civilian / small craft | 0.802 |
| fishing_boat | small craft | 0.872 |
| speedboat | small craft | 0.716 |
| **military_vessel** | **military** | **0.936** |

Overall across all classes on TEST: precision 0.838, recall 0.795,
**mAP50 0.856**, **mAP50-95 0.627**.

### 1.3 Training data is entirely open-source

The brief requires participants source their own training data from open-source
datasets. **All five integrated datasets are public Roboflow Universe YOLO
exports declaring CC BY 4.0.** Full provenance log: `data/DATASETS.md`.

| Dataset | Domain | Source (Roboflow Universe) | Ver | Licence (declared) | Raw imgs | After remap (imgs / boxes) |
|---|---|---|---|---|---:|---:|
| `seaships` | surface | `ship-detection-cedpa/seaships-spcag` | 1 | CC BY 4.0 | 6,979 | 6,979 / 9,198 |
| `military_ships` | aerial | `hanif-noer-r/military-ships` | 1 | CC BY 4.0 | 2,746 | 2,641 / 11,208 |
| `shiprsimagenet` | aerial | `convertvoctoyolo/shiprsimagenet` | 39 | CC BY 4.0 | 4,579 | 4,535 / 34,299 |
| `military_surface` | **surface** | `hannah-agkvq/military-ship-detection-qxv5m` | 2 | CC BY 4.0 | 3,011 | 3,000 / 3,713 |
| `civilian_gapfill` | **surface** | `boats-ri7td/speedboat` | 2 | CC BY 4.0 | 6,213 | 2,521 / 4,229 |

**Licence honesty note, stated deliberately** (`data/DATASETS.md`; `docs/PROGRESS.md`
§2.1): the Roboflow *exports* each declare CC BY 4.0, but the *original* SeaShips
and ShipRSImageNet corpora are academic/research-use-only. We use the CC BY 4.0
re-exports and attribute both the Roboflow workspace and the original authors. We
make no commercial-rights claim. `DOTA` is deliberately **excluded** (mapped to
`null`) because it overlaps other sources and would cause train/test leakage.

Bonus-track data (§3) is logged separately in the same file: `foreign` from
`navy-ip6vd/navy-ship-a6prh` (Roboflow, CC BY 4.0, 2,125 imgs), and `malaysia_rmn`,
346 images collected by the team, licensed per-image via the collection's own
manifest.

### 1.4 Honest scope limit on multi-angle

Recorded rather than hidden (`docs/PROGRESS.md` §4): **5 of the 8 classes have
both-domain test coverage** — `cargo`, `container_ship`, `fishing_boat`,
`military_vessel`, `passenger_ferry`. `speedboat`, `tanker` and `yacht` have zero
*surface* instances in TEST, so their individual multi-angle claim does not hold.

This does **not** touch the mandatory gate: `military_vessel` is one of the five
fully covered classes, and it is the class the >90% requirement applies to.

---

## 2. Performance Benchmark — 25 pts

> **Criterion.** Prove accuracy on military / threat classes. The scoring rubric
> sets **>80%**; the official brief (`docs/SEDIC2026-track2.pdf`, §4) sets the
> stricter bar: **"Must achieve a Recall > 90% on military and threat-based
> classes."** We report against the stricter one.
>
> **Evidence:** `src/eval/metrics.py` · `src/eval/detail.py` · `README.md` § Results

### 2.1 The headline number

> ### Military recall = **0.936** (93.6%)
> **Split:** held-out **TEST** (1,588 images, never trained on)
> **Threshold:** `conf_military = 0.10`
> **Model:** `models/baseline2_best.pt`
> **Measured by:** `python -m src.eval.metrics` (canonical gate)
>
> ✅ Clears the rubric's 80% bar by **13.6 points**.
> ✅ Clears the official brief's stricter >90% requirement.

### 2.2 It is not threshold-shopped — the whole sweep passes

A single favourable threshold would be a weak claim, so we publish the full
`conf_military` sweep. **The gate passes at every operating point from 0.05 to
0.30** (`src/eval/detail.py`, TEST split, IoU 0.50; `README.md` § Results):

| `conf_military` | Recall | Precision | Gate > 0.90 |
|---:|---:|---:|:---:|
| 0.05 | 0.949 | 0.705 | ✅ |
| **0.10 (shipped)** | **0.938** | 0.796 | ✅ |
| 0.15 | 0.929 | 0.842 | ✅ |
| 0.20 | 0.922 | 0.871 | ✅ |
| 0.25 (GUI demo default) | 0.915 | 0.891 | ✅ |
| 0.30 | 0.909 | 0.906 | ✅ |

Recall degrades gracefully — there is no cliff just past our operating point.

### 2.3 Two independent measurements agree

| Method | Matching | Military recall (TEST, conf 0.10) |
|---|---|---:|
| `src/eval/metrics.py` (canonical gate) | Ultralytics val pass, per-class recall read off `r_curve` at the real `conf` | **0.936** |
| `src/eval/detail.py` (diagnostic) | Explicit greedy PASCAL-VOC, IoU ≥ 0.50, at the actual operating point | **0.938** |

Two different matching implementations, 0.002 apart. We quote **0.936** as
authoritative (the more conservative of the two).

### 2.4 Why the benchmark is credible — three integrity measures

1. **Measured on TEST, never val.** The `split` argument in `src/eval/metrics.py`
   is keyword-only and required so it cannot silently default; val drives
   checkpoint selection and would be optimistically biased (`docs/PROGRESS.md` §3.3).
2. **Leakage-controlled.** Cross-dataset near-duplicates are removed **before**
   splitting (19,676 collected → 3,786 dropped, 19.2% → 15,890 kept;
   `data/DATASETS.md`). `python -m src.data.validate` reports **PASS** —
   15,890 images / 55,614 boxes, zero schema errors, **zero cross-split
   leakage**. Merge dedup, the leakage check and the augmentation guard all read
   one shared threshold from `configs/schema.yaml` so "duplicate" means the same
   thing everywhere.
3. **The dedup method was corrected to avoid *inflating* the numbers.** The
   original single-linkage clustering dropped 46% of images, but ~78% of those
   drops were transitive chaining — one cluster swallowed **2,961 distinct**
   SeaShips frames. Switched to **greedy leader dedup** (drop only against an
   already-*kept* representative, never chained) at threshold 3: drop rate 25%,
   largest cluster 38, every drop within threshold. Evidence with image pairs:
   `outputs/dedup_audit/audit.md`. This *added* real evaluation data rather than
   removing it, and guarantees no two kept images are within threshold.

### 2.5 Two negative results we report rather than bury

- **Civilian-as-military false positives.** Running military at 0.10
  deliberately trades precision for recall, so some civilian craft are called
  military. The recorded fix is **more close-view civilian surface data**, never
  raising the threshold — raising it would attack the one mandatory requirement
  (`docs/PROGRESS.md` §4, decision log 2026-08-05). Ingesting `civilian_gapfill`
  cut military false positives on TEST at `conf_military=0.10` by **~27%**
  (965 → 708 boxes; `data/DATASETS.md`).
- **A measurement bug found and fixed against the source.** On 2026-08-07 this
  model was logged as **failing** at 0.892 and was not shipped. On 2026-09-04
  the cause was traced to `src/eval/metrics.py` reading Ultralytics'
  `metrics.box.r`, which is recall at **one confidence index shared across all
  classes** (wherever mean F1 averaged over every class peaks) — not at the
  `conf` actually passed, and not per-class. Confirmed by reading the installed
  `ultralytics` source (`utils/metrics.py::ap_per_class`). Fixed to read
  per-class recall off `r_curve` at the index nearest the real operating `conf`;
  the gate then read **0.936 PASS**, reproduced twice, deterministic, and
  agreeing with `detail.py`'s always-correct 0.938. Full write-up:
  `docs/PROGRESS.md` decision log, 2026-09-04.

### 2.6 Reproduction

```bash
python -m src.eval.metrics --weights models/baseline2_best.pt --split test   # canonical gate
python -m src.eval.detail  --weights models/baseline2_best.pt --split test   # sweep + per-domain
```

---

## 3. Competitive Advantage — Local vs Foreign — 10 pts

> **Criterion.** "Models that can distinguish between Local (Malaysian) and
> Foreign military assets will be awarded significantly higher technical scores."
>
> **Evidence:** `src/fine_grained/` · `configs/fine_grained.yaml` ·
> `docs/PROGRESS.md` §4 · `data/DATASETS.md`

### 3.1 What is built

A **2nd-stage nationality classifier** that runs *after* the detector, on crops
the detector has already labelled `military_vessel`:

```
image ──▶ YOLO11m detector ──▶ military_vessel crop ──▶ ResNet18 ──▶ malaysian_rmn
                                                                  │  foreign
                                                                  └─ unknown (abstain)
```

| Property | Value | Source |
|---|---|---|
| Backbone | ResNet18, ImageNet-pretrained, fine-tuned | `configs/fine_grained.yaml` |
| Input size | 224 px | `configs/fine_grained.yaml` |
| Training | 16 epochs (early-stopped from a 25 max), RTX 3060 | `docs/PROGRESS.md` §4 |
| Class imbalance handling | majority capped at 3× minority in TRAIN **plus** inverse-frequency weighted loss | `configs/fine_grained.yaml` |
| **Abstention** | softmax < **0.55** → `unknown` | `configs/fine_grained.yaml` |
| Weights | `models/fine_grained_rmn_classifier.pt` (gitignored) | `docs/PROGRESS.md` §4 |

The **abstention floor is a deliberate design choice**: for a nationality call
in a maritime-security context, an honest "unknown" is worth more than a
confident wrong attribution.

Crop counts (`docs/PROGRESS.md` §4): TRAIN `malaysian_rmn` 294 / `foreign` 882
(capped from 4,290 raw boxes); VAL `malaysian_rmn` 52 / `foreign` 760 (left
uncapped, natural distribution).

### 3.2 The result — and its caveat, in the same breath

> **0.98 validation accuracy** (confusion matrix: `malaysian_rmn` 51/52 correct,
> `foreign` 760/760 correct) — **but this number is inflated by a domain gap
> between the two data sources, not a clean measure of nationality recognition.**

The two class pools are entirely disjoint in origin and differ systematically in
photographic style: `malaysian_rmn` is curated RMN press photography (**median
crop area 691K px**, sharp); `foreign` is small, low-resolution Roboflow
detection crops (**median 58K px**, hazy). That is a **>10× resolution gap that
has nothing to do with the vessel itself**, so the model partly separates on
image-source style rather than purely on vessel identity.

**We tested the caveat rather than merely asserting it.** A
resolution/blur-degradation ablation — RMN validation images downsampled to
foreign-like resolution before inference — **did not flip predictions**, ruling
out that specific shortcut. But TRAIN and VAL for each class still share the same
source pool, so other domain cues (colour grading, framing and angle
distribution, compression artefacts) cannot be ruled out without independently
sourced RMN imagery at realistic field-camera quality.

**The claim we actually make:** this classifier reliably distinguishes *"RMN
press-photo domain"* from *"this one foreign Roboflow dataset"* — narrower than
"distinguishes RMN vessels from foreign vessels in general." On real detector
output (small, low-resolution field crops) it may be systematically biased
toward predicting `foreign` regardless of true nationality, since real crops
resemble the foreign-domain distribution more closely.

### 3.3 Decision record and the path to deployment

Two options were weighed on 2026-09-04 (`docs/PROGRESS.md` decision log):
**(A)** present the classifier honestly with the caveat, or **(B)** source
domain-matched/degraded RMN data and retrain for a fairer measurement.
**Path A was chosen** — the RMN set is thin (346 images, 4 hull classes:
`kasturi_corvette`, `kedah_ngpv`, `keris_lms`, `lekiu_ff`) and a second
data-collection pass was not justified for an optional bonus track on the
submission timeline.

**Deployment-readiness note:** the real fix is domain-matched RMN training data
at field-camera resolution (not press photography), and/or **fusing this
classifier's call with AIS/IFF signals** rather than relying on vision alone for
a nationality determination.

This is not gate-relevant: the mandatory >90% recall requirement applies to the
detector's `military_vessel` class, not to this optional second stage. §2's
numbers are unaffected by anything in this section.

---

## 4. Technical Brief — 20 pts

> **Criterion.** The official brief (`docs/SEDIC2026-track2.pdf`, §4) requires a
> PDF "detailing **the dataset used**, **the model architecture**, and **the
> logic used for military classification**." All three are below, in that order.

### 4.1 The dataset used

**Five open-source Roboflow YOLO exports** across both domains — full table,
licences and provenance in §1.3 above and `data/DATASETS.md`.

**Unified taxonomy — the 50→8 collapse.** `military_ships` and `shiprsimagenet`
share the identical ShipRSImageNet 50-class taxonomy, so a single collapse (a
YAML anchor `&shiprs50` in `configs/schema.yaml`) serves both and cannot drift
apart. `seaships` has its own 6-class taxonomy. All map into the 8 detector
classes:

| → schema class | # source types | Examples |
|---|---:|---|
| `military_vessel` | 34 | Arleigh Burke DD, Nimitz, Submarine, Perry FF, Ticonderoga, LHA/LSD/Osumi landing craft, Other Warship/Destroyer/Frigate/Auxiliary |
| `cargo` | 3 | Cargo, Other Merchant, RoRo |
| `yacht` | 2 | Yacht, Sailboat |
| `container_ship` | 1 | Container Ship |
| `tanker` | 1 | Oil Tanker |
| `passenger_ferry` | 1 | Ferry |
| `fishing_boat` | 1 | Fishing Vessel |
| `speedboat` | 1 | Motorboat |
| `null` (dropped) | 6 | Barge, Dock, Other Ship, Test Ship, Tugboat, **Medical Ship** |

Documented judgment calls: `Hovercraft` → `military_vessel` (LCAC landing craft);
`Patrol` → `military_vessel` (patrol combatant); `Training Ship` →
`military_vessel` (naval training/auxiliary); **`Medical Ship` → `null`** (a
hospital ship is not a threat/combatant, so including it would pollute the
military class the gate is measured on).

**Pipeline: convert → dedup → stratified split → validate.**

| Stage | What happens |
|---|---|
| **Convert** | Per-source converters (`src/data/converters/`) remap native labels into the schema. ShipRSImageNet/`military_ships` exports are **polygon/oriented** labels, not horizontal boxes; `yolo2yolo` collapses each polygon to its min/max horizontal envelope (horizontal-boxes-first plan; OBB is a later upgrade). |
| **Dedup** | Perceptual-hash **greedy leader** dedup at threshold 3, run *before* splitting so no near-duplicate can straddle splits: 19,676 collected → **3,786 dropped (19.2%)** → **15,890 kept**. |
| **Split** | Stratified by **class × domain**, seed 42 → train **11,125** / val **3,177** / test **1,588**. Val and test stay **un-augmented** so per-domain performance is measurable on real, unseen data. |
| **Validate** | `python -m src.data.validate` → **PASS**: 15,890 images / 55,614 boxes, zero schema errors, **zero cross-split leakage**. |

**A data forensics note that matters for label quality.** The `civilian_gapfill`
export's `data.yaml` is not usable at face value: the upstream Roboflow project
is a broken merge, and 13 of its 18 class names are Roboflow README boilerplate
ingested as labels (~67% of boxes carry an unusable label), plus one entirely
non-maritime class (`Human Fall`). Rather than wildcarding, every one of the 18
names was resolved explicitly — by tracing each class's underlying source
filenames back to its original dataset and visually spot-checking samples. That
recovered **422 images / 431 boxes of real frontal-view military vessels** from a
junk-named slot (confirmed: a Royal Navy L14 hull, a naval auxiliary, others),
which would otherwise have been discarded. Everything unresolvable maps to `null`
explicitly, no `"*"` wildcard. Full mapping: `configs/schema.yaml`
`mappings.civilian_gapfill`; write-up: `data/DATASETS.md`.

**Synthetic augmentation, and why it is now switched off.** When
`military_vessel` had zero real surface examples, `src/data/balance.py` pasted
aerial military crops onto surface backgrounds (`surface_synth`, **TRAIN split
only**, with a leak guard hashing every synthetic image against val/test). With
real `military_surface` data now covering all three splits, **`surface_synth`
generation is set to zero** — the shipped build is **real-only**. The code stays
toggleable for ablation. Every number in this brief is measured on real imagery.

### 4.2 Model architecture

| | | Source |
|---|---|---|
| **Detector** | **Ultralytics YOLO11m** (medium) | `configs/train_baseline.yaml` |
| **Box format** | Horizontal bounding boxes (HBB); polygon labels enveloped to HBB | `docs/PROGRESS.md` §2.4 |
| **Input size** | 640 × 640 | `configs/train_baseline.yaml` |
| **Epochs** | 100 | `configs/train_baseline.yaml` |
| **Batch** | 16 | `configs/train_baseline.yaml` |
| **Seed** | 42 (fixed across the team) | `configs/train_baseline.yaml` |
| **Hardware** | RTX 3060, local | `docs/PROGRESS.md` §1 |
| **Shipped weights** | `models/baseline2_best.pt` (trained 2026-08-07) | `src/inference/predict.py` `DEFAULT_WEIGHTS` |
| **2nd stage (bonus)** | ResNet18 @ 224 px — see §3 | `configs/fine_grained.yaml` |

**Augmentation, tuned for the aerial domain** (`configs/train_baseline.yaml`) —
aerial/satellite ships appear at arbitrary orientations, so rotation and
copy-paste matter more here than in a purely frontal dataset:
`copy_paste: 0.3` · `mosaic: 1.0` · `mixup: 0.15` · `degrees: 10` ·
`fliplr: 0.5` · `hsv_h/s/v: 0.015 / 0.7 / 0.4`.

**Everything is config-driven.** `src/train/train.py` reads
`configs/train_baseline.yaml`; nothing about training is hardcoded in Python.
`configs/data.yaml` is **auto-generated from `configs/schema.yaml`** by `merge.py`
so the two cannot drift, and CI asserts they match.

**Architectural safeguards** (`docs/PROGRESS.md` §3.3):
- **Schema-drift gate** — training aborts loudly if `data.yaml` ≠ `schema.yaml`.
- **Abort-on-zero-military** — training refuses to run if the TRAIN split holds
  zero `military_vessel` instances, since the gate would then be unmeetable.
- **Frozen inference interface** — `src/inference/predict.py` exposes a
  `Detection` dataclass (`class`, `confidence`, `bbox` [x1,y1,x2,y2] abs px,
  `frame`, `timestamp`) and `predict(source, weights, conf, conf_military, stub)`.
  `--stub` returns synthetic detections **in the exact real format with no torch
  and no weights**, so the demo path can never hard-fail on stage.
- **CI on every push** — `ruff`, schema/data-match assertion, stub-contract run,
  and a 40-test pytest suite, none of which require torch or ultralytics.

**Deployment path**: `predict()` → Streamlit GUI (`app/app.py`) with
colour-coded boxes by schema group, live threshold sliders, a military alert
banner, before/after toggle, and **BoT-SORT video tracking with a CSV detection
log** exporting `frame, timestamp_s, track_id, class, group, confidence, bbox`
— the format required for the Qualifier Video Clip deliverable.

### 4.3 The logic used for military classification

Two design decisions carry the entire mandatory requirement.

**(a) `military_vessel` is deliberately ONE coarse class.**

The source taxonomy offers 34 distinct military ship types (Arleigh Burke,
Nimitz, Submarine, Perry, Ticonderoga, …). We collapse all 34 into a single
detector class. **Rationale** (`configs/schema.yaml` header comment;
`docs/PROGRESS.md` decision log 2026-07-24): fragmenting the military domain
into fine ship types spreads positive examples thin across many low-count
classes and **collapses recall on exactly the class the pass/fail gate is
measured on**. Recall on one well-populated class beats recall on 34 sparse
ones. Fine-grained identity work is pushed to the separate 2nd-stage classifier
in §3, where a mistake costs a label, not a detection.

`military_vessel` accordingly holds the most training data of any class in the
schema — 6,066 TRAIN images (`data/DATASETS.md`).

*The nuance this protects against:* `military_ships` is **not** an all-military
dataset. The original scaffold stub mapped `{"*": military_vessel}` for it,
which would have poisoned the gate class with ~5,000 civilian boxes. Caught and
replaced with the real 50→8 collapse; the wildcard was retired
(`docs/PROGRESS.md` §2.4).

**(b) The dual-threshold, threat-aware rule.**

```yaml
conf: 0.25            # all civilian and small-craft classes
conf_military: 0.10   # military — INTENTIONALLY LOWER
```

Implemented in `src/inference/predict.py` as a per-class threshold applied at
filter time — `score >= (conf_military if name in military_group else conf)` —
where the military group is read from `configs/schema.yaml`, never hardcoded.

**Why the asymmetry is correct, not a bug.** The two error types are not
symmetric in a maritime-security context:

| | Cost |
|---|---|
| **Missing a military vessel** (false negative) | A threat goes unreported — the failure mode the mission exists to prevent |
| **Flagging a civilian as military** (false positive) | An operator dismisses one box |

So we buy recall on the threat class with precision we can afford to spend, and
**only on that class** — civilian classes keep the standard 0.25 and are
unaffected. The sweep in §2.2 quantifies exactly what the trade costs: moving
0.25 → 0.10 buys **+2.3 points of military recall (0.915 → 0.938)** for 9.5
points of precision.

This threshold is commented as deliberate in `configs/train_baseline.yaml`,
listed as a non-negotiable in `CLAUDE.md`, and guarded in the decision log, so no
future contributor "fixes" it upward and silently breaks the mandatory
requirement.

**The gate applies to the group, not a hardcoded name.** `configs/schema.yaml`
declares `groups.military: [7]` and every consumer — trainer, evaluator, the
inference threshold rule, the GUI's colour coding and alert banner — reads the
group from that one file. Adding a future threat class to the group extends the
gate automatically, with no code change.

---

## 5. Video Demonstration — 15 pts — what this brief supplies

Not part of this document, but the demo is already built and these sections give
it its script beats. Suggested running order, each with the artefact behind it:

| Beat | Show | Backing artefact |
|---|---|---|
| 1 | Multi-angle: run the same model on a frontal harbour image, then an aerial/satellite scene | §1.1 per-domain table |
| 2 | The three mandatory categories rendered live, colour-coded by schema group, with the military alert banner firing | `app/app.py`, §1.2 |
| 3 | Drag the `conf_military` slider 0.25 → 0.10 and watch borderline military detections appear — the recall/precision trade, live | §2.2 sweep, `app/app.py` |
| 4 | Video tracking with persistent IDs + the exported CSV detection log | `app/app.py`, BoT-SORT; format in §4.2 |
| 5 | Bonus: a `military_vessel` crop routed into the RMN-vs-foreign classifier, including an `unknown` abstention | §3, `src/fine_grained/infer.py` |
| 6 | Close on the honesty beats — dedup de-chaining, the `metrics.py` bug, the domain-gap caveat | §2.4, §2.5, §3.2 |

The GUI ships at `conf_military = 0.25` for a cleaner picture on screen; **the
number quoted in this brief is the 0.10 one.** State which is which on camera.

---

## Appendix A — repo artefacts referenced

| File | What it supplied |
|---|---|
| `docs/SEDIC2026-track2.pdf` | The official requirements (authority for all criteria wording) |
| `data/DATASETS.md` | All dataset rows, licences, counts, split sizes, per-domain military counts, FP reduction, dedup note |
| `configs/schema.yaml` | 8 classes, the three groups, the 50→8 collapse, dedup config |
| `configs/train_baseline.yaml` | YOLO11m, 100 epochs, 640 px, batch 16, seed 42, augmentation, `conf` / `conf_military` |
| `configs/fine_grained.yaml` | ResNet18, 224 px, 3× cap, 0.55 abstention floor |
| `README.md` § Results | Shipped-model gate 0.936, per-class recall, conf sweep, per-domain table, mAP |
| `docs/PROGRESS.md` §1–§4 + decision log | Pipeline state, collapse rationale, safeguards, known gaps, the `metrics.py` bug, the RMN caveat and Path A decision |
| `src/inference/predict.py` | `DEFAULT_WEIGHTS`, the per-class threshold rule, frozen `Detection` interface |
| `outputs/dedup_audit/audit.md` | Dedup evidence (paired images) |

## Appendix B — data-freshness flag for the brief writer

**`outputs/eval/test_eval.md` on disk is STALE — do not quote it.** It still
reports the *previous* model `baseline_best.pt` (generated 2026-08-04, test =
1,349 images: aerial 0.940 / surface 0.954 / overall 0.942). `docs/PROGRESS.md`'s
2026-09-04 decision log states it was regenerated for `baseline2_best.pt` and the
old one archived as `outputs/eval/test_eval_baseline_2026-07-28.md`; **neither the
regeneration nor the archive is present in this working copy** (`outputs/eval/`
holds only the stale `test_eval.md` and `chunks_v2/`, and `models/` holds only
`baseline_best.pt` — `baseline2_best.pt` is gitignored and lives on the training
machine).

Every number in this brief is therefore taken from the shipped-model figures in
`README.md` § Results, `docs/PROGRESS.md` §1, and `data/DATASETS.md`, which agree
with each other. **Action before submission:** regenerate `outputs/eval/test_eval.md`
against `models/baseline2_best.pt` on the training machine and confirm it
reproduces 0.936 / aerial 0.932 / surface 0.977 / overall 0.938.

A related count inconsistency, flagged rather than smoothed over: `README.md` and
`docs/PROGRESS.md` §1 both state TEST holds "371 real surface-military
**instances**", a figure measured on the **pre-`civilian_gapfill` build** (293
images, `docs/PROGRESS.md` §2.3). `data/DATASETS.md` gives the current build's
TEST surface-military count as **335 images** (+42). This brief quotes the 335
image count and omits the stale instance count; regenerate the instance figure
alongside `test_eval.md` if the brief writer wants to cite it.

Two further open items recorded in `docs/PROGRESS.md` that a judge may probe:
- The `src/eval/metrics.py` recall fix is **verified correct but still
  uncommitted** pending team review.
- `docs/PROGRESS.md` §2's count tables still describe the pre-`civilian_gapfill`
  build (13,500 images); §1, `data/DATASETS.md` and this brief use the current
  15,890-image build. §2 needs a refresh pass.
