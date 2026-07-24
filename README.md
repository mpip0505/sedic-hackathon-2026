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
- [ ] Data pipeline: converters → unified schema, dedup, per-domain split
- [ ] Baseline training (`yolo11m`, HBB)
- [ ] Evaluation harness + military recall gate report
- [ ] Real inference path in `predict()`
- [ ] GUI: box drawing + video playback
- [ ] Bonus: oriented boxes (OBB)
- [ ] Bonus: fine-grained RMN-vs-foreign 2nd stage
- [ ] Deliverables: technical brief, video, poster

## Quickstart
```bash
# 1. environment + deps + stub smoke test
bash scripts/setup.sh

# 2. run the frozen stub predictor (no weights, no torch needed)
python -m src.inference.predict --source none --stub

# 3. launch the GUI (stub mode on by default)
streamlit run app/app.py
```

## Ownership (team of 4)
| Person | Owns |
|--------|------|
| **P1** | Data pipeline — converters, dedup, splits, `data/DATASETS.md` |
| **P2** | Model training — `src/train/`, augmentation, baseline |
| **P3** | Integration + GUI — `src/inference/`, `app/`, eval wiring |
| **P4** | Deliverables + bonus — brief/video/poster, `src/fine_grained/`, OBB |

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
`--stub` returns synthetic detections in the exact real format and must always
work. See `CLAUDE.md` for the non-negotiables.

## Structure
```
configs/     schema.yaml (contract), data.yaml, train_baseline.yaml
data/        raw/ interim/ processed/ (gitignored) + DATASETS.md
src/         data/ (+converters) · train · inference · eval · fine_grained
app/         app.py (Streamlit) + assets
models/      trained weights (gitignored)
outputs/     detections/ · runs/ (gitignored)
deliverables/ technical_brief · video · poster
scripts/     setup.sh · download_datasets.sh
```

## Label schema
| ID | Class | Group |
|----|-------|-------|
| 0 | container_ship | civilian |
| 1 | tanker | civilian |
| 2 | cargo | civilian |
| 3 | passenger_ferry | civilian |
| 4 | yacht | civilian / small_craft |
| 5 | speedboat | small_craft |
| 6 | fishing_boat | small_craft |
| 7 | **military_vessel** | **military (>90% recall gate)** |

## Results
| Model | Domain | mAP50 | mAP50-95 | Military recall |
|-------|--------|-------|----------|-----------------|
| _tbd_ | frontal | — | — | — |
| _tbd_ | aerial  | — | — | — |

## Licence note
Code in this repo is the team's. **Several datasets are academic-use-only**
(see `data/DATASETS.md`); respect each dataset's licence. No dataset images or
model weights are committed.
