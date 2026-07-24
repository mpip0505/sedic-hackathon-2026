# CLAUDE.md — Project Guardian

Repo brief for future Claude Code sessions. Read this before touching anything.

## What this is
**Project Guardian** is a maritime object-detection system for the **SEDIC 2026
Visual Track** competition. It detects vessels across two visual domains:

- **frontal / surface** camera views (e.g. shore/ship cameras), and
- **aerial / satellite** views.

Built on **Ultralytics YOLO**. Python only — no npm/Node. The GUI is Streamlit.

## The one hard requirement
> **Recall > 90% on military / threat classes.**

Overall mAP is secondary. **When a tradeoff appears, favour military recall.**
The lower military confidence threshold (`conf_military: 0.10`) exists solely to
serve this gate.

## Non-negotiables
1. **`configs/schema.yaml` is THE CONTRACT.** Never hardcode class names or IDs
   anywhere. Converters, trainer, and eval all read classes from it via
   `load_classes()`.
2. **The `predict()` / `Detection` interface is FROZEN.** `--stub` must ALWAYS
   work with no weights and no torch. Everything downstream depends on it.
3. **`military_vessel` stays ONE coarse class in the detector.** Do not split it
   into ship types — that fragments positives and collapses recall.
   Fine-grained work (RMN vs foreign) lives in `src/fine_grained/` as a separate
   2nd-stage classifier.
4. **Never commit data or weights.** No images, no `.pt`/`.onnx`. See
   `.gitignore`.
5. **Every dataset gets a row in `data/DATASETS.md`** with its licence before
   its data is used.

## Path → purpose
| Path | Purpose |
|------|---------|
| `configs/schema.yaml` | The label-taxonomy contract + source→schema mappings |
| `configs/data.yaml` | YOLO dataset descriptor (mirrors schema classes) |
| `configs/train_baseline.yaml` | Config-driven training params |
| `data/{raw,interim,processed}/` | Datasets by pipeline stage (gitignored) |
| `data/DATASETS.md` | Provenance + licence log |
| `src/data/converters/` | Per-source label converters → unified schema |
| `src/data/` | Dataset assembly, splitting, dedup |
| `src/train/` | Training entry points (read train_baseline.yaml) |
| `src/inference/predict.py` | **Frozen** detection interface + `--stub` |
| `src/eval/` | Metrics, especially the military recall gate |
| `src/fine_grained/` | Bonus 2nd-stage RMN-vs-foreign classifier |
| `app/app.py` | Streamlit GUI, wired to `predict()` |
| `models/` | Trained weights (gitignored) |
| `outputs/{detections,runs}/` | Detection JSON + training runs (gitignored) |
| `deliverables/` | Technical brief, video, poster |
| `scripts/` | `setup.sh`, `download_datasets.sh` |

## Conventions
- **Python 3.10+.** Use modern typing (`X | None`, built-in generics).
- **Run modules, not scripts:** `python -m src.inference.predict ...`.
- Use **`pathlib`**, not `os.path`.
- **Type hints** on public functions.
- **`logging`, not `print`** (except deliberate CLI stdout output like JSON).
- **Config-driven:** parameters live in `configs/*.yaml`, not in code.
- Lint with **ruff** before every PR.

## Domain gotchas
- **Cross-dataset duplicates cause train/test leakage.** Known overlaps:
  - FGSCR-42 derives from **DOTA + HRSC**.
  - **ShipRSImageNet** overlaps **HRSC + FGSD**.
  Dedup across sources before splitting. `dota` maps to `null` (dropped) by
  default for this reason.
- **The val set stays UN-augmented**, and is **split per domain** (frontal vs
  aerial) so per-domain performance is measurable.
- **Horizontal boxes before oriented boxes.** Get HBB working first; OBB is a
  later upgrade for arbitrarily-rotated aerial ships (`degrees: 10` augments
  toward this).
- **The low military threshold is DELIBERATE.** `conf_military: 0.10 < conf:
  0.25`. Do not "fix" it — it is the recall gate.
