# 🛥️ Project Guardian

A maritime object-detection system for the **SEDIC 2026 Visual Track**. It
detects vessels across two visual domains — **frontal/surface** camera views and
**aerial/satellite** views — and is built on **Ultralytics YOLO** with a
**Streamlit** GUI.

## 🎯 The gate
> **Recall > 90% on military / threat classes.**
> Overall mAP is secondary. When a tradeoff appears, favour military recall.

## Status
- [x] Phase 0 — repo scaffold, contracts, stub predictor, GUI skeleton
- [x] Datasets: `military_ships` + `seaships` + `shiprsimagenet` + `military_surface` (Roboflow YOLO)
- [x] Data pipeline: remap → unified schema, **greedy dedup**, class×domain split, `validate` PASS
      (`civilian_gapfill`-merged build; train 11,125 / val 3,177 / test 1,588 — see `docs/PROGRESS.md`)
- [x] **Surface-military gap CLOSED** with real frontal warships (`military_surface`):
      `military_vessel` now in all 3 splits on surface (test: 371 real instances / 293 imgs).
      `surface_synth` copy-paste set to zero for the next run (code retained/toggleable)
- [x] Baseline training (`yolo11m`, HBB, 100 epochs) — **DONE** (RTX 3060, local). Shipped:
      `baseline2_best.pt` (2026-08-07 retrain, fewer civilian-as-military false positives)
- [x] Evaluation harness + military recall gate report — **PASS** (0.936, real surface+aerial)
- [x] Real inference path in `predict()` (Ultralytics, per-class thresholds)
- [x] GUI: box drawing, live thresholds, video tracking + CSV detection log
- [ ] Bonus: oriented boxes (OBB)
- [x] Bonus: fine-grained RMN-vs-foreign 2nd stage (`src/fine_grained/`, ResNet18) — 0.98 val accuracy, but inflated by an RMN/foreign domain gap, not purely vessel identity; see `docs/PROGRESS.md` §4
- [ ] Deliverables: technical brief, video, poster

---

# 🚀 Getting started (read this first)

**You do not need a trained model, a GPU, or the datasets to start working.**
The stub predictor returns fake detections in the real format, so GUI and
integration work runs from day one.

## Prerequisites
- **Python 3.10–3.12** (`python3 --version`). On macOS: `brew install python@3.11`
- **Git**
- A **Roboflow account** (free) — only if you need to download datasets yourself

## 1. Clone and enter
```bash
git clone https://github.com/mpip0505/sedic-hackathon-2026.git
cd sedic-hackathon-2026
```

## 2. Create your virtual environment
Everyone makes their own — `.venv/` is gitignored and never committed.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

Your prompt should now show `(.venv)`. **You must re-run the `activate` line
every time you open a new terminal.**

## 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Verify your setup ✅
```bash
python -m src.inference.predict --source none --stub
```
You should get a JSON array of fake detections. **If this works, you're set up
correctly** — it exercises the schema, the config loader, and the frozen contract.

```bash
streamlit run app/app.py
```
Opens the GUI at `localhost:8501`. With no weights present it starts in **stub
mode** — no model needed. Full usage: see **Running the GUI** below.

---

## 5. Datasets (P1/P2 only — skip if you're on GUI or deliverables)

Datasets are **not in git** (too large, and several are licence-restricted).
Download them into `data/raw/` yourself.

**Get a Roboflow API key:** roboflow.com → your workspace → Settings → API Keys →
copy the *Private API Key*.

**Store it in `.env` at the repo root** (gitignored — never commit this):
```
ROBOFLOW_API_KEY=your_key_here
```

**Load it and download:**
```bash
export $(grep -v '^#' .env | xargs)      # Windows PS: $env:ROBOFLOW_API_KEY="..."
python scripts/download_military_ships.py
python scripts/download_seaships.py
python scripts/download_shiprsimagenet.py
python scripts/download_military_surface.py
python scripts/download_civilian_gapfill.py   # acquired, not yet ingested — see below
```

**Verify:**
```bash
find data/raw/military_ships -name "*.jpg" | wc -l     # ~2.7k
find data/raw/seaships -name "*.jpg" | wc -l           # ~7k
find data/raw/shiprsimagenet -name "*.jpg" | wc -l     # ~4.6k
find data/raw/military_surface -name "*.jpg" | wc -l   # ~3.0k
find data/raw/civilian_gapfill -name "*.jpg" | wc -l   # ~6.2k
cat data/raw/seaships/data.yaml                        # check class name strings
```

### Datasets currently in use
| Name | Domain | Images | Format | Source | Licence |
|------|--------|-------:|--------|--------|--------|
| `military_ships` | aerial | ~2.7k | YOLO (Roboflow) | `hanif-noer-r/military-ships` v1 | CC BY 4.0 |
| `seaships` | surface | ~7k | YOLO (Roboflow) | `ship-detection-cedpa/seaships-spcag` v1 | CC BY 4.0 |
| `shiprsimagenet` | aerial | ~4.6k | YOLO (Roboflow) | `convertvoctoyolo/shiprsimagenet` v39 | CC BY 4.0¹ |
| `military_surface` | **surface** | ~3.0k | YOLO (Roboflow) | `hannah-agkvq/military-ship-detection-qxv5m` v2 | CC BY 4.0 |
| `civilian_gapfill` | surface | 6,213 raw / 2,521 kept | YOLO (Roboflow) | `boats-ri7td/speedboat` v2 | CC BY 4.0 |

All arrive **already in YOLO format** with their own `train/valid/test` split —
that split is **discarded**; `merge.py` does its own stratified split. The Roboflow
labels are polygon/oriented; `yolo2yolo` envelopes them to horizontal boxes. Full
provenance and licences: `data/DATASETS.md`.

_¹ the Roboflow re-export declares CC BY 4.0; the underlying ShipRSImageNet is
academic-use-only — attribute both, don't claim commercial rights._

Fetched with `python scripts/download_civilian_gapfill.py` into
`data/raw/civilian_gapfill/`. Close-view **civilian** surface imagery, added to
reduce civilian vessels being detected as `military_vessel`, and merged into the
current `data/processed/` build (train 11,125 / val 3,177 / test 1,588) — the
build `models/baseline2_best.pt` (the shipped model, see `## Results`) is trained
on. Despite the project name, its real taxonomy is 18 classes of which only four
are vessels (`Fishing-boats`, `speedboat`, `Yacht`, `tugboat`) and ~67% of boxes
carry junk boilerplate labels; the full class list and the ingest caveats are in
`data/DATASETS.md`.

> **Surface-military gap — closed.** `military_surface` is the first **real**
> frontal-view warship set, giving `military_vessel` surface coverage in all three
> splits (test: 371 real instances / 293 imgs), so the >90% gate can finally be
> measured on frontal views. The old `surface_synth` copy-paste stopgap is set to
> zero for the next run. Still thin on the surface side: `tanker`/`yacht`/`speedboat`
> have 0 surface instances.

## 6. Build the unified dataset (P1)

With the four sets in `data/raw/`, remap → merge → validate:

```bash
# 1. remap each Roboflow set → data/interim/<name>/ (schema class IDs, HBB)
python -m src.data.converters.yolo2yolo --dataset seaships
python -m src.data.converters.yolo2yolo --dataset military_ships
python -m src.data.converters.yolo2yolo --dataset shiprsimagenet
python -m src.data.converters.yolo2yolo --dataset military_surface

# 2. merge → greedy dedup → stratified class×domain split → data/processed/
python -m src.data.merge                 # also regenerates configs/data.yaml

# 3. sanity-check labels + train/val/test leakage (exits nonzero on failure)
python -m src.data.validate
```

Current build (seed 42, greedy dedup @ hamming 3): **13,500 images kept** from
17,155 (21% near-duplicate drop), split train 9,452 / val 2,699 / test 1,349.
`military_vessel` now has real **surface** coverage in all three splits (test: 371
instances). Dedup method/threshold live in `configs/schema.yaml` (`dedup:`); a drop
audit is written to `outputs/dedup_audit/`. Per-class tables: `docs/PROGRESS.md`.

> The build above is **real-only** — the `surface_synth` cross-domain copy-paste
> (`python -m src.data.balance --clean`) is a stopgap kept for ablation but **set to
> zero** now that real surface-military data exists. Enable it only for comparison.

---

# 🖥️ Running the GUI

`app/app.py` is the demo interface: upload an image or video, get colour-coded
boxes, a live summary, and (for video) tracked IDs plus a downloadable detection
log. It talks **only** to `predict()` / `track_video()` — it never imports
Ultralytics itself.

```bash
streamlit run app/app.py                 # venv must have ultralytics + streamlit
```

Opens `localhost:8501`. It loads `models/baseline2_best.pt` by default and warms
the weights at startup, so the first upload is already fast.

## The controls

| Sidebar control | What it does |
|---|---|
| **Stub mode** | Synthetic detections, no model. Auto-on if weights are missing — the fallback if anything breaks on demo day |
| **Weights (.pt)** | Path to the model. Defaults to `models/baseline2_best.pt` |
| **Confidence — civilian** | Threshold for the 7 non-military classes |
| **Confidence — military** | Threshold for `military_vessel` — **the recall knob** |
| **Tracker** | `botsort.yaml` (stable IDs) or `bytetrack.yaml` (faster). Video only |
| **Frame stride** | Process every Nth frame. Raise it if playback drags |
| **Max frames** | Safety stop so a long clip can't stall a live demo |

Images run **automatically** on upload and re-run on every slider move — no
button. Video needs an explicit *▶ Run detection on video*.

## Using the thresholds

The two sliders are independent, which is the whole point: **military gets its
own, lower threshold** so we catch warships at the cost of some false alarms.
That asymmetry is the competition gate, not an oversight.

| Setting | Effect | Use it for |
|---|---|---|
| Military **0.10**, civilian 0.25 | Max recall (0.938 on test) — canonical gate 0.936 | The **gate posture** — matches `predict()`'s default and the eval numbers |
| Military **0.25**, civilian 0.25 | Recall 0.915 at precision 0.891 | The **GUI default** — cleanest picture for a live demo |
| Military **0.30+** | Recall drifts toward the 0.90 floor | Only to show the tradeoff; don't ship it |
| Military **above** civilian | Inverts the priority — the sidebar warns you | Illustrating *why* the asymmetry exists |

**How to feel it:** load an image with a borderline vessel and drag the military
slider from 0.50 down to 0.05. Boxes, counts and the red banner update live;
results are cached per threshold, so sliding back and forth is instant.

**The GUI can't tell you whether you're still above the gate** — it's
qualitative. For the actual recall/precision numbers, sweep with the evaluator:

```bash
# dump predictions once, then re-score at any thresholds for free
python -m src.eval.detail --weights models/baseline2_best.pt --split val \
    --thresholds 0.05,0.10,0.15,0.20,0.25,0.30 --dump outputs/sweep_val.json
python -m src.eval.detail --from-dumps outputs/sweep_val.json \
    --thresholds 0.22,0.24,0.26,0.28 --md-out outputs/sweep.md

# single pass/fail check (exits nonzero below 0.90)
python -m src.eval.metrics --weights models/baseline2_best.pt --split val --conf 0.10
```

Sweep on **val**; confirm the final pick once on **test**. Repeatedly tuning
against test is how you end up reporting a number that won't survive Phase 2.

## Video and the detection log

Tracking runs through BoT-SORT, so each vessel keeps one ID across frames. The
CSV export is the Qualifier-Clip deliverable format:

```
frame,timestamp_s,track_id,class,group,confidence,x1,y1,x2,y2
```

Untracked boxes (tracker not yet confirmed) leave `track_id` blank rather than
writing `nan`.

## Two things that will bite you

- **Edits need a restart.** `.streamlit/config.toml` sets
  `fileWatcherType = "none"` because the watcher walks `torch.classes` and
  **segfaults the server** on macOS. `Ctrl+C` and re-run after changing `app.py`.
- **Don't swap the results table back to `st.dataframe`.** It serializes through
  pyarrow, which segfaults the whole process on some numpy/pyarrow combinations
  (seen with pyarrow 25 + numpy 1.26). `render_table()` builds HTML on purpose —
  a demo that can hard-crash on the results table isn't worth sortable columns.

Both failures kill the server with no traceback; the browser just says it can't
load the frontend. If you see that, check the terminal.

---

# 🤝 Working in this repo

## Ownership (team of 5, from 2026-07-31)
The mandatory gate is passed and the core is built, so the remaining work is the
**bonus**, the **presentation layer**, and the **submission package**.

| Handle | Owns | Current task |
|--------|------|-----------|
| **LEAD** | ML core — `src/data/`, `src/train/`, `src/eval/`, `src/inference/`, `src/fine_grained/`, `configs/` | Scene-split decision → then the RMN-vs-Foreign 2nd stage |
| **GUI** | `app/pages/`, `app/assets/` | Landing page + navigation over the existing detection view |
| **DATA-RMN** | `data/raw/fine_grained/malaysian_rmn/` | Collect + annotate the RMN (TLDM) half of the bonus set |
| **DATA-FOR** | `data/raw/fine_grained/foreign/`, gap-fill | Foreign-navy half, then surface `speedboat`/`tanker`/`yacht` |
| **DELIV** | `deliverables/` | Technical brief PDF, ≤5 min video, Phase 2 poster |

👉 **Full task packets, handoff formats and acceptance criteria:
[`docs/TEAM_TASKS.md`](docs/TEAM_TASKS.md).** Read your section before starting.

## Branches & PRs
```bash
git checkout -b feat/<area>-<short-desc>     # e.g. feat/data-merge, feat/app-boxes
```
Before opening a PR (matches CI):
```bash
ruff check .                                  # lint (whole repo)
pytest -q                                     # 40 tests (no torch/GPU needed)
python -m src.inference.predict --source none --stub   # contract must not break
```
Merge to `main` via PR. Touching someone else's area? Tag them.

## Non-negotiables
1. **`configs/schema.yaml` is the contract.** Class names/IDs are read from it —
   never hardcode them. Changing it invalidates everyone's converted data, so
   announce it first.
2. **The `Detection` interface is frozen.** `--stub` must always work.
3. **`military_vessel` stays one coarse class** in the detector. Fine-grained
   RMN-vs-foreign lives in `src/fine_grained/`.
4. **Never commit** data, weights, `.venv/`, or `.env`.
5. **Every new dataset gets a row in `data/DATASETS.md`** with its licence.

See `CLAUDE.md` for the full brief (auto-read by Claude Code) and `docs/` for the
competition PDF, data sourcing map, project playbook, and `TEAM_TASKS.md`.

---

## The Detection contract (frozen)
`src/inference/predict.py` is the single interface every consumer talks to.

```python
@dataclass
class Detection:
    class_name: str          # serialized under JSON key "class"
    confidence: float
    bbox: list[float]        # [x1, y1, x2, y2] absolute pixels
    frame: int | None        # video frame index, else None
    timestamp: float | None  # video seconds, else None

predict(source, weights=None, conf=0.25, conf_military=0.10, stub=False)
-> list[Detection]
```

> `conf_military` is **deliberately lower** than `conf`. The competition gates on
> military *recall*, so we accept more false positives to avoid misses. This is
> intentional — don't "fix" it.

The real (non-stub) path is implemented on top of Ultralytics: it runs at the
**lower** of the two thresholds so nothing is dropped early, then applies the
per-class rule (military at `conf_military`, everything else at `conf`) — that
filter is what enforces the recall gate. Ultralytics/torch are imported **lazily**,
so `--stub` still works with no torch installed.

Additive helpers alongside the frozen pair (the frozen shapes are untouched):

| Symbol | Purpose |
|---|---|
| `load_model(weights)` | Memoised YOLO loader; clear `FileNotFoundError` if weights are missing |
| `class_groups()` | class name → `military`/`small_craft`/`civilian`, read from `schema.yaml` (the GUI's colours) |
| `track_video(...)` | Streams `TrackedFrame`s with BoT-SORT/ByteTrack IDs — same per-class threshold rule |

## Structure
```
configs/      schema.yaml (contract + dedup/domains), data.yaml (generated), train_baseline.yaml
data/         raw/ → interim/ → processed/ (all gitignored) + DATASETS.md
              processed/ holds images|labels/{train,val,test} + manifest.csv + domains.json
src/          data/ (converters · merge · balance · validate · phash) · train · inference · eval · fine_grained
app/          app.py (Streamlit) + assets
models/       trained weights (gitignored)
outputs/      detections/ · runs/ · dedup_audit/ (gitignored)
deliverables/ technical_brief · video · poster
docs/         competition brief, data sourcing map, project playbook, PROGRESS.md
scripts/      setup.sh · dataset download scripts
```

## Label schema
| ID | Class | Group |
|----|-------|-------|
| 0 | container_ship | civilian |
| 1 | tanker | civilian |
| 2 | cargo | civilian |
| 3 | passenger_ferry | civilian |
| 4 | yacht | small_craft |
| 5 | speedboat | small_craft |
| 6 | fishing_boat | small_craft |
| 7 | **military_vessel** | **military (>90% recall gate)** |

Deliberately coarse — fragmenting military into many ship types collapses recall.

## Results

Shipped model `models/baseline2_best.pt` — `yolo11m`, 100 epochs, trained 2026-08-07
(RTX 3060, local) on the `civilian_gapfill`-merged build (train 11,125 / val 3,177 /
test 1,588; `military_surface` supplies real surface-military coverage, 371 real
instances in TEST). Replaced `baseline_best.pt` (the original 2026-07-28 run) on
2026-09-04 — ~27% fewer civilian-vessels-detected-as-`military_vessel` false positives,
see `docs/DATASETS.md`. Gate scored on the held-out **TEST** split via
`python -m src.eval.metrics` (Ultralytics val pass, `conf_military = 0.10`).

**The gate — military recall, real surface+aerial combined:**

| | Military recall (conf 0.10) | Gate >0.90 |
|---|----------------------------:|:----------:|
| **overall (TEST)** | **0.936** | ✅ **PASS** |

Per-class recall (TEST, same pass):

| Class | Recall |
|---|---:|
| cargo | 0.944 |
| container_ship | 0.904 |
| tanker | 0.918 |
| **military_vessel** | **0.936** |
| fishing_boat | 0.872 |
| yacht | 0.802 |
| passenger_ferry | 0.875 |
| speedboat | 0.716 |

Overall (all classes, TEST, conf 0.10): precision 0.838, recall 0.795, **mAP50 0.856**,
**mAP50-95 0.627**.

**`conf_military` threshold sweep** (test split, IoU 0.50, via `src/eval/detail.py`):

| conf | recall | precision | gate >0.90 |
|-----:|-------:|----------:|:----------:|
| 0.05 | 0.949 | 0.705 | ✅ PASS |
| **0.10** | **0.938** | 0.796 | ✅ PASS |
| 0.15 | 0.929 | 0.842 | ✅ PASS |
| 0.20 | 0.922 | 0.871 | ✅ PASS |
| 0.25 | 0.915 | 0.891 | ✅ PASS |
| 0.30 | 0.909 | 0.906 | ✅ PASS |

**Per-domain military recall — the whole reason `military_surface` was added:**

| domain | military recall | gate >0.90 |
|--------|-----------------:|:----------:|
| aerial | 0.932 | ✅ |
| surface | 0.977 | ✅ |
| **overall** | **0.938** | ✅ **PASS** |

Both domains individually clear the gate — surface (real `military_surface` data)
actually scores *higher* recall than aerial.

<details>
<summary>Note on `detail.py`'s numbers vs. the canonical gate (0.938 vs 0.936)</summary>

`metrics.py`/`model.val()` (the canonical gate) and `detail.py` (this sweep) use
different matching: Ultralytics' internal per-class recall-at-conf reading vs. this
script's explicit greedy PASCAL-VOC @0.5 matching *at the actual `conf_military=0.10`
operating point*. Both pass the gate and now agree closely; treat `metrics.py`'s 0.936
as the authoritative number and this table as the diagnostic breakdown.

`metrics.py` itself had a real bug until 2026-09-04: it read Ultralytics'
`metrics.box.r` directly, which is recall at **one confidence index shared across every
class** (wherever mean F1 *averaged over all classes* peaks) — not at the actual
`conf` argument, and not per-class. That's why this table and the canonical gate used
to disagree by a much larger margin (0.942 vs 0.904 on the previous model). Fixed by
reading recall directly off Ultralytics' per-class `r_curve` at the index nearest the
real operating `conf` — see `src/eval/metrics.py` and `docs/PROGRESS.md`'s decision log
(2026-09-04) for the full story, including why the fix is still uncommitted pending
team review.

Getting `detail.py` to run also required a real fix: `model.predict(stream=True, ...)`
lets every image pick its own letterboxed shape (`auto=True` for single-image
batches), and on a long GPU stream that fragments/accumulates until a normal-sized
allocation OOMs partway through — not a bad image, confirmed by bisection (identical
slices OOM at different points depending only on how many images had already
streamed in-process). Fixed by chunking collection into fresh-process slices via
`--start`/`--end`/`--dump`, then combining with `--from-dumps` — chunked collection
is now the standard way to run this script on GPU. On Windows specifically,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (this script's other mitigation)
silently no-ops (confirmed via the runtime `UserWarning`), so chunks need to stay
small — 200 images/chunk worked reliably where 400 still OOM'd.
</details>

---

## Troubleshooting

**`source: no such file or directory: .venv/bin/activate`**
The venv doesn't exist yet — run `python3 -m venv .venv` first (step 2).

**`ModuleNotFoundError: No module named 'src'`**
Run from the **repo root** and use module syntax: `python -m src.inference.predict`,
not `python src/inference/predict.py`.

**`ROBOFLOW_API_KEY not set`**
Create `.env` (step 5) and load it: `export $(grep -v '^#' .env | xargs)`.

**Streamlit shows a blank page / import error**
Confirm your venv is active (`(.venv)` in the prompt) and deps installed.

**GUI dies mid-use: "Cannot load Streamlit frontend code"**
The server process died — check the terminal. Known causes are the torch file
watcher and pyarrow; see [Two things that will bite you](#two-things-that-will-bite-you).

**`ModuleNotFoundError: No module named 'lap'` when running video**
The tracker needs the assignment solver: `pip install lapx` (pinned in
`requirements.txt`).

**`.env` or `.venv/` shows up in `git status`**
Stop — don't commit. Check `.gitignore` contains both, then
`git rm -r --cached .venv .env` if already staged.

## Licence note
Code in this repo is the team's. The three in-use datasets are **Roboflow YOLO
exports declaring CC BY 4.0** — attribute the Roboflow workspaces. Note the
*underlying* SeaShips and ShipRSImageNet are academic-use-only, so don't claim
commercial rights (see `data/DATASETS.md`). No dataset images or model weights
are committed.