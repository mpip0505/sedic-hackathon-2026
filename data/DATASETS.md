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
| `civilian_gapfill` | **surface** | [`boats-ri7td/speedboat`](https://universe.roboflow.com/boats-ri7td/speedboat) | 2 | CC BY 4.0 (declared) | 6,213 | 2,521 / 4,229 | civilian gap-fill (`fishing_boat`, `speedboat`, `yacht`, `passenger_ferry`) **+ 421 real surface `military_vessel`** recovered from a corrupted class list (see below) |

_"After remap" = output of `yolo2yolo` into `data/interim/` (images with zero
kept boxes are dropped; polygon labels enveloped to horizontal boxes). Added by
P1, 2026-07-25; `military_surface` added 2026-07-27; `civilian_gapfill` added
2026-08-06._

### `civilian_gapfill` — corrupted class list, forensically recovered before mapping

The exported `data.yaml` is **not usable at face value**: the Roboflow project
("speedboat") is a broken merge of several upstream sets. Most of its 18
`names:` entries are literal Roboflow README/export boilerplate text ingested
as class names (e.g. `"- annotate- and create datasets"`), several carrying
real annotated boxes. One entry (`Human Fall`, 245 polygon-format instances)
is a human-fall-detection dataset that got merged in by mistake — completely
non-maritime. Two entries (`all - v2 2023-05-24...` / `fishing boat - v2
2022-01-27...`) are duplicate annotations of the same images under two
different corrupted slots. One entry (`ship_detection - v1 2025-01-02...`) is
polygon/segmentation format, not HBB.

**Correction to the earlier "likely mapping" note below:** that prior pass
(2026-08-05) treated every boilerplate-named class as pure junk to drop. It
missed that one of those junk-named slots (`"- annotate- and create
datasets"`, 431 boxes / 422 images) is actually a clean, homogeneous bucket of
**real frontal-view military vessels** — recovered by tracing each class's
underlying source filenames (e.g. `640_640_military0707.jpg`) back to their
original per-category dataset, then visually spot-checking sampled images
(confirmed: a Royal Navy L14 hull, a naval auxiliary/RFA ship, and others).
Full recovered mapping lives in `configs/schema.yaml` under
`mappings.civilian_gapfill`, with a comment on every non-obvious line.

Final disposition (18 native names → schema, all explicit, no `"*"` wildcard):
`Fishing-boats`→`fishing_boat`, `Yacht`→`yacht`, `speedboat`→`speedboat`,
`"- annotate- and create datasets"`→`military_vessel`, `"- use active
learning..."`→`passenger_ferry`, `"- collaborate with your team..."`→
`speedboat`; everything else (`-`, `tugboat` [no schema class for it],
`Human Fall`, the two duplicate stock-photo slots, the polygon-format slot,
`undefined`, and three low-confidence/ambiguous slots incl. a pilot-boat
class) → `null`.

### Merged split (`data/processed/`, seed 42, greedy dedup @ threshold 3)
_Rebuilt 2026-08-06 with `civilian_gapfill` included._
- Collected 19,676 → **3,786 near-duplicates dropped (19.2%)** → **15,890 kept**.
- Split: train **11,125** · val **3,177** · test **1,588**.
- `military_vessel` (images): train 6,066 · val 1,737 · test 874 — up from
  5,768 / 1,653 / 835 (the previous build), an increase of ~421 images,
  matching `civilian_gapfill`'s real surface-military contribution almost
  exactly. **No military images were lost or diluted.**
- `military_vessel` by domain (images):
  | split | aerial | surface |
  |---|---:|---:|
  | train | 3,722 | 2,344 |
  | val   | 1,067 |   670 |
  | test  |   539 |   335 |
  Surface military rose in **every** split (train +294, val +85, test +42 vs.
  the previous build) — the gap-fill data landed in train AND test, not just
  train.
- Civilian surface classes also rose: `fishing_boat` +1,186, `speedboat` +351,
  `passenger_ferry` +273, `yacht` +163 (all from `civilian_gapfill`). `tanker`
  remains 0 on surface — unchanged, `civilian_gapfill` has no tanker content.
- `python -m src.data.validate`: **PASS** — 15,890 images / 55,614 boxes
  checked, zero schema errors, zero cross-split leakage.
- Synthetic **surface-military** cross-domain copy-paste (`balance.py`): **0** —
  still disabled; real surface-military data now covers train/val/test.

> ❌ **Retrain attempt on this build (2026-08-07): FAILED the military gate, NOT
> shipped.** 100 epochs, `yolo11m`, same `configs/train_baseline.yaml`, weights
> preserved at `outputs/runs/baseline2/weights/best.pt` (not copied to `models/`).
> Canonical gate (`src/eval/metrics.py`, TEST): **military recall 0.892 < 0.90 —
> FAIL.** `models/baseline_best.pt` was restored from a pre-retrain backup
> (MD5-verified) within the same session, so the shipped model is unaffected —
> `outputs/eval/test_eval.md` and the README `## Results` table still describe
> the original 2026-07-28 model (aerial 0.942 / surface 0.984 / overall 0.948
> per `detail.py`; 0.904 per the canonical `metrics.py` gate), unchanged.
>
> Nuance worth recording: `src/eval/detail.py`'s per-domain breakdown (explicit
> VOC matching at the actual `conf_military=0.10` operating point, not
> Ultralytics' own max-F1-point matching) reads the *failed* checkpoint at
> aerial 0.932 / surface 0.977 / **overall 0.938** — all three individually
> clear >0.90. The two methodologies disagree on pass/fail here, not just
> magnitude. Per this repo's own convention, `metrics.py` is authoritative, so
> the FAIL verdict and the decision not to ship stand — but it means this
> retrain was close, not a clear miss.
>
> The retrain's actual goal — fewer civilian-vessels-detected-as-`military_vessel`
> false positives — **was achieved**: military_vessel FP count on TEST @
> `conf_military=0.10` dropped from **965 (then-shipped model) to 708 (this
> attempt)**, a ~27% reduction. So the false-positive fix direction is sound;
> this attempt just gave some recall back to earn it. A future attempt might
> recover the lost margin with fewer epochs / earlier stopping, or by
> rebalancing so the added civilian volume doesn't shift the model's confidence
> calibration on military as much.

> ✅ **CORRECTION (2026-09-04): the 0.892 FAIL above was a bug in `metrics.py`,
> not a real gate failure — `baseline2_best.pt` actually PASSES, and is now
> the shipped model.** `src/eval/metrics.py` read Ultralytics' `metrics.box.r`
> directly, which — confirmed against the installed `ultralytics` source
> (`utils/metrics.py`, `ap_per_class()`) — is computed at **one confidence
> index shared across every class**, chosen by maximising `f1_curve.mean(0)`
> (mean F1 averaged over ALL classes), not at the `conf` argument actually
> passed in and not per-class. That's an unrelated, run-drifting number, not
> recall at the real `conf_military=0.10` operating point — which is exactly
> why it disagreed with `detail.py`'s explicit VOC matching (0.892 vs. 0.938)
> in the "nuance" paragraph above. The fix reads recall directly off
> `box_metric.r_curve` at the index nearest the actual `conf`, per class.
> Re-verified today: canonical gate on `baseline2_best.pt` now reads
> **military recall 0.936 — PASS** (reproduced twice, deterministic), and a
> freshly regenerated `detail.py` per-domain report reproduces the *original*
> 2026-08-07 numbers exactly (aerial 0.932 / surface 0.977 / overall 0.938),
> confirming `detail.py` itself was never the buggy one. `baseline2_best.pt`
> is now `src/inference/predict.py`'s `DEFAULT_WEIGHTS` and the app's shipped
> baseline; `outputs/eval/test_eval.md` was regenerated for it (old
> `baseline_best.pt` report archived as
> `outputs/eval/test_eval_baseline_2026-07-28.md`). Full write-up + the fix's
> commit status: `docs/PROGRESS.md` decision log, 2026-09-04.

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

## Bonus: RMN-vs-Foreign fine-grained classifier data (`src/fine_grained/`)

This data feeds **only** the optional 2nd-stage nationality classifier that
runs on `military_vessel` crops the main detector has already produced. It is
entirely separate from `data/processed/` above — it is never merged, deduped,
or split alongside the detector data, and does not touch `configs/schema.yaml`.

| Dataset (folder) | Domain | Source | Ver | Licence | Raw imgs | Classes contributed |
|---|---|---|---|---|---:|---|
| `foreign` | surface | [`navy-ip6vd/navy-ship-a6prh`](https://universe.roboflow.com/navy-ip6vd/navy-ship-a6prh) (Roboflow, "Navy Ship") | 3 | CC BY 4.0 | 2,125 | foreign-navy warship crops. 9 of 11 native classes kept (`AC`, `Coastguard`, `Destroyer`, `Frigate`, `Fuqing`, `INS Vikrant`, `INS Visakhapatnam`, `INSVikramaditya`, `Ship`); `Non-millitary` and `Pennant no` excluded (not warship-crop content) |
| `malaysia_rmn` | surface | own collection (RMN — Royal Malaysian Navy / TLDM), by DATA-RMN | — | per-image — see the collection's own manifest, not a single blanket licence | 346 (4 hull classes: `kasturi_corvette`, `kedah_ngpv`, `keris_lms`, `lekiu_ff`) | `malaysian_rmn` class |

> ⚠️ **Known limitation, recorded honestly (2026-09-04):** `malaysia_rmn` and
> `foreign` differ systematically in photographic style/resolution (median crop
> area 691K px vs 58K px — curated press photography vs low-resolution
> detection crops), a domain gap unrelated to actual vessel identity. The
> trained classifier (`models/fine_grained_rmn_classifier.pt`) reports 0.98 val
> accuracy, but that number is partly inflated by this domain gap, not a clean
> measure of nationality recognition. Presented honestly rather than retrained
> (Path A). Full write-up + decision log: `docs/PROGRESS.md` §4.

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
| Custom RMN set | surface + aerial | val + bonus fine-grained (`malaysian_rmn` vs `foreign`), and to help close the **surface-military gap** | ✅ RMN/foreign halves sourced and in use for the bonus classifier — see "Bonus: RMN-vs-Foreign fine-grained classifier data" above. Surface-military-gap contribution to the *main detector* not pursued (`malaysia_rmn`/`foreign` are not mapped into `configs/schema.yaml` or `data/processed/`) |

### Notes
- Map each source's native class names in `configs/schema.yaml` under
  `mappings:` — do not translate labels anywhere else.
- The `surface_synth` cross-domain augmentation is a stopgap for zero real
  surface-military data; real frontal military imagery (Custom RMN + Roboflow
  frontal warship sets) is still the priority.
