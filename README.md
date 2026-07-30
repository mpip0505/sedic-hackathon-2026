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
      (13,500 imgs kept; train 9,452 / val 2,699 / test 1,349 — see `docs/PROGRESS.md`)
- [x] **Surface-military gap CLOSED** with real frontal warships (`military_surface`):
      `military_vessel` now in all 3 splits on surface (test: 371 real instances / 293 imgs).
      `surface_synth` copy-paste set to zero for the next run (code retained/toggleable)
- [x] Baseline training (`yolo11m`, HBB, 100 epochs) — **DONE** (RTX 3060, local)
- [x] Evaluation harness + military recall gate report — **PASS** (0.904, real surface+aerial)
- [ ] Real inference path in `predict()`
- [ ] GUI: box drawing + video playback
- [ ] Bonus: oriented boxes (OBB)
- [ ] Bonus: fine-grained RMN-vs-foreign 2nd stage
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
Opens the GUI at `localhost:8501` with stub mode on. No model needed.

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
```

**Verify:**
```bash
find data/raw/military_ships -name "*.jpg" | wc -l     # ~2.7k
find data/raw/seaships -name "*.jpg" | wc -l           # ~7k
find data/raw/shiprsimagenet -name "*.jpg" | wc -l     # ~4.6k
find data/raw/military_surface -name "*.jpg" | wc -l   # ~3.0k
cat data/raw/seaships/data.yaml                        # check class name strings
```

### Datasets currently in use
| Name | Domain | Images | Format | Source | Licence |
|------|--------|-------:|--------|--------|--------|
| `military_ships` | aerial | ~2.7k | YOLO (Roboflow) | `hanif-noer-r/military-ships` v1 | CC BY 4.0 |
| `seaships` | surface | ~7k | YOLO (Roboflow) | `ship-detection-cedpa/seaships-spcag` v1 | CC BY 4.0 |
| `shiprsimagenet` | aerial | ~4.6k | YOLO (Roboflow) | `convertvoctoyolo/shiprsimagenet` v39 | CC BY 4.0¹ |
| `military_surface` | **surface** | ~3.0k | YOLO (Roboflow) | `hannah-agkvq/military-ship-detection-qxv5m` v2 | CC BY 4.0 |

All arrive **already in YOLO format** with their own `train/valid/test` split —
that split is **discarded**; `merge.py` does its own stratified split. The Roboflow
labels are polygon/oriented; `yolo2yolo` envelopes them to horizontal boxes. Full
provenance and licences: `data/DATASETS.md`.

_¹ the Roboflow re-export declares CC BY 4.0; the underlying ShipRSImageNet is
academic-use-only — attribute both, don't claim commercial rights._

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

# 🤝 Working in this repo

## Ownership (team of 4)
| Person | Owns | First task |
|--------|------|-----------|
| **P1** | Data pipeline — `src/data/`, `configs/schema.yaml`, `data/DATASETS.md` | ✅ 4 sets remapped+merged, real surface-military added → next: support P2's retrain |
| **P2** | Model training — `src/train/`, augmentation, baseline | Training wrapper over Ultralytics |
| **P3** | Integration + GUI — `src/inference/`, `app/`, `src/eval/` | Build GUI against `--stub` |
| **P4** | Deliverables + bonus — brief/video/poster, `src/fine_grained/`, OBB | Start RMN image collection |

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
competition PDF, data sourcing map, and project playbook.

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

Baseline `yolo11m`, 100 epochs, trained 2026-07-28 (RTX 3060, local) on the current
build — **the first run to include real surface-military data** (`military_surface`,
371 real instances in TEST). `models/baseline_best.pt`, gate scored on the held-out
**TEST** split via `python -m src.eval.metrics` (Ultralytics val pass, `conf_military
= 0.10`).

**The gate — military recall, real surface+aerial combined:**

| | Military recall (conf 0.10) | Gate >0.90 |
|---|----------------------------:|:----------:|
| **overall (TEST)** | **0.904** | ✅ **PASS** |

Per-class recall (TEST, same pass):

| Class | Recall |
|---|---:|
| cargo | 0.934 |
| container_ship | 0.892 |
| tanker | 0.872 |
| **military_vessel** | **0.904** |
| fishing_boat | 0.815 |
| yacht | 0.769 |
| passenger_ferry | 0.768 |
| speedboat | 0.384 ⚠️ |

Overall (all classes, TEST, conf 0.10): precision 0.841, recall 0.792, **mAP50 0.851**,
**mAP50-95 0.651**.

> ⚠️ **`speedboat` recall (0.384) is low and worth investigating** — not part of the
> military gate, but flagged here rather than left silent.

**`conf_military` threshold sweep** (test split, IoU 0.50, via `src/eval/detail.py`):

| conf | recall | precision | gate >0.90 |
|-----:|-------:|----------:|:----------:|
| 0.05 | 0.952 | 0.726 | ✅ PASS |
| **0.10** | **0.942** | 0.805 | ✅ PASS |
| 0.15 | 0.933 | 0.846 | ✅ PASS |
| 0.20 | 0.927 | 0.869 | ✅ PASS |
| 0.25 | 0.921 | 0.888 | ✅ PASS |
| 0.30 | 0.915 | 0.901 | ✅ PASS |

**Per-domain military recall — the whole reason `military_surface` was added:**

| domain | military recall | gate >0.90 |
|--------|-----------------:|:----------:|
| aerial | 0.940 | ✅ |
| surface | 0.954 | ✅ |
| **overall** | **0.942** | ✅ **PASS** |

Both domains individually clear the gate — surface (real `military_surface` data)
actually scores *higher* recall than aerial. `speedboat` (see warning above) is aerial-only
in TEST (0.485) with no surface instances to average against, which is why it doesn't
show up here as a gate risk despite the low number.

<details>
<summary>Note on `detail.py`'s numbers vs. the canonical gate (0.942 vs 0.904)</summary>

`metrics.py`/`model.val()` (the canonical gate) and `detail.py` (this sweep) use
different matching: Ultralytics' internal max-F1-point matching vs. this script's
explicit greedy PASCAL-VOC @0.5 matching *at the actual `conf_military=0.10`
operating point*. Both pass the gate; treat `metrics.py`'s 0.904 as the
authoritative number and this table as the diagnostic breakdown.

Getting this to run also required a real fix: `model.predict(stream=True, ...)`
lets every image pick its own letterboxed shape (`auto=True` for single-image
batches), and on a long GPU stream that fragments/accumulates until a normal-sized
allocation OOMs partway through — not a bad image, confirmed by bisection (identical
slices OOM at different points depending only on how many images had already
streamed in-process). Fixed by chunking collection into fresh-process slices via
`--start`/`--end`/`--dump`, then combining with `--from-dumps` — chunked collection
is now the standard way to run this script on GPU.
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

**`.env` or `.venv/` shows up in `git status`**
Stop — don't commit. Check `.gitignore` contains both, then
`git rm -r --cached .venv .env` if already staged.

## Licence note
Code in this repo is the team's. The three in-use datasets are **Roboflow YOLO
exports declaring CC BY 4.0** — attribute the Roboflow workspaces. Note the
*underlying* SeaShips and ShipRSImageNet are academic-use-only, so don't claim
commercial rights (see `data/DATASETS.md`). No dataset images or model weights
are committed.