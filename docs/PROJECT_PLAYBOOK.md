# SEDIC 2026 — Project Playbook

**Project Guardian: Maritime Domain Awareness**
5-person team · 3–4 coders · Phase 1 online POC → Phase 2 grand finale

This is the team's single source of truth: repo layout, who owns what, and how to approach each step. Pair it with the **Data Sourcing Map** for the dataset side.

---

## 1. The pipeline at a glance

```
                        ┌─────────────────────────────────────────────┐
                        │  configs/schema.yaml  (label taxonomy = law) │
                        └─────────────────────────────────────────────┘
                                          │
   ┌──────────────┐   ┌──────────────┐   ▼   ┌──────────────┐   ┌──────────────┐
   │ 1. ACQUIRE   │──▶│ 2. CONVERT   │──▶│ 3. MERGE +   │──▶│ 4. TRAIN     │
   │ raw datasets │   │ →YOLO format │   │ dedupe+split │   │ YOLO model   │
   └──────────────┘   └──────────────┘   └──────────────┘   └──────┬───────┘
                                                                    │
                        ┌───────────────────────────────────────────┤
                        ▼                                           ▼
              ┌──────────────────┐                        ┌──────────────────┐
              │ 5. TUNE for      │                        │ 6. BONUS: MY vs  │
              │ >90% mil. recall │                        │ Foreign (2-stage)│
              └────────┬─────────┘                        └────────┬─────────┘
                       │                                           │
                       └─────────────────┬─────────────────────────┘
                                         ▼
                        ┌──────────────────────────────┐
                        │ 7. INFERENCE + detection log │
                        │    on Qualifier Video Clip   │
                        └───────────────┬──────────────┘
                                        ▼
        ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
        │ 8. EVALUATE  │   │ 9. GUI /     │   │ 10. Tech     │   │ 11. YouTube  │
        │ recall gate  │   │ live demo    │   │ Brief PDF    │   │ demo (5 min) │
        └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
              PHASE 1 SUBMISSION ═══════════════════════════════════════════════╝
                                        │
                                        ▼  (top 10 only)
              PHASE 2: poster + live stress-test station + jury pitch
```

---

## 2. Repository structure

```
project-guardian/
├── README.md                  # setup, how to reproduce, results table
├── requirements.txt           # pinned deps (ultralytics, opencv, etc.)
├── .gitignore                 # ignore data/raw, models/, outputs/runs
│
├── configs/
│   ├── schema.yaml            # ⭐ unified label taxonomy — THE contract
│   ├── data.yaml              # YOLO dataset paths + class names
│   └── train_baseline.yaml    # hyperparameters per experiment
│
├── data/
│   ├── raw/                   # downloaded datasets, untouched  (gitignored)
│   ├── interim/               # each source converted to YOLO, pre-merge
│   ├── processed/             # final unified train/ val/ test
│   └── DATASETS.md            # provenance + licence log (feeds the brief)
│
├── src/
│   ├── data/
│   │   ├── converters/        # voc2yolo.py · dota2yolo.py · cls2det.py
│   │   ├── merge.py           # applies schema.yaml, dedupes, splits
│   │   ├── balance.py         # oversample + augment rare military classes
│   │   └── validate.py        # label sanity checks (boxes in range, etc.)
│   ├── train/
│   │   └── train.py           # thin wrapper over Ultralytics
│   ├── inference/
│   │   ├── detect.py          # run on image / folder / video
│   │   ├── detection_log.py   # emits the required detection log
│   │   └── tracker.py         # optional multi-frame smoothing for video
│   ├── eval/
│   │   ├── metrics.py         # per-class recall — the >90% gate check
│   │   └── report.py          # confusion matrix, PR curves
│   └── fine_grained/          # MY-vs-Foreign 2nd-stage classifier
│
├── models/                    # trained weights  (gitignored → GH Releases)
├── notebooks/                 # EDA + experiment scratch
├── app/                       # Streamlit/Gradio GUI (Phase 2 live demo)
│   ├── app.py
│   └── assets/
├── outputs/
│   ├── detections/            # logs + annotated frames
│   └── runs/                  # Ultralytics training runs
├── deliverables/
│   ├── technical_brief/       # the required PDF
│   ├── video/                 # 5-min YouTube demo assets
│   └── poster/                # Phase 2
└── scripts/
    ├── download_datasets.sh
    └── setup.sh
```

**Why this shape:** `configs/schema.yaml` is the one file nobody edits without telling the team — it's the contract every converter and the trainer read from. Judges reward reproducibility, so a clean `data/DATASETS.md` provenance log + a `README` results table directly lift your technical score.

---

## 3. Division of labour (3-person team)

Everyone codes with AI help (Claude Code). Roles are drawn to match strengths and to keep the person on the critical path (you) from being overloaded — everything *downstream* of the model is pushed onto the other two.

| # | Role | Owns (repo folders) | Core responsibility | AI leverage |
|---|------|---------------------|---------------------|-------------|
| **P1** | **ML Core** *(you)* | `data/`, `src/data/`, `src/train/`, `configs/`, `models/` | The whole ML core: data pipeline (source, convert, merge, balance) **and** model training + the fight to >90% military recall. You hand off a trained model + clean dataset — nothing downstream. | Claude Code writes conversion scripts, merge/balance logic, training configs. |
| **P2** | **Integration + GUI** | `src/inference/`, `src/eval/`, `app/` | Everything the model plugs into: the inference wrapper, the **detection log** (required deliverable), the eval/metrics scripts, and the Streamlit/Gradio GUI + live-demo station. Owns "the thing that runs the model." | Claude Code scaffolds the FastAPI/wrapper, the GUI, and eval scripts. |
| **P3** | **Deliverables + Bonus data** | `deliverables/`, `data/raw/` (RMN set), `src/fine_grained/` (data side) | Their strength — technical brief PDF, 5-min YouTube video, poster + jury pitch (Phase 2). Early on, owns **Malaysian/foreign image collection** (scrape RMN images, annotate in Roboflow) to stay productive before deliverables cluster at the end. | Claude Code for scrapers + drafting the brief; presentation polish is all them. |

**Why this shape:** you hold the critical path (data → model → recall gate), so P2 and P3 are deliberately loaded with everything else. P2 gets a *full* role (not just a few-day GUI) by owning the entire serving + eval layer. P3 stays busy from Day 1 via the bonus data work — which also feeds their pitch narrative — instead of idling until deliverables are due.

**The two rules that make this parallelize (not bottleneck on you):**
- **Agree the API contract on Day 1.** Your model exposes `predict(image_or_video) → [{class, confidence, bbox}]` (a function or a tiny FastAPI/Flask endpoint returning JSON). P2's GUI just calls that. Fix the contract once and you both work independently.
- **P2 builds against a stub first.** On Day 1 you hand P2 a **stub predictor** returning fake detections in that exact format, so they build the whole GUI + serving layer *before* your real model exists. Then swapping in the trained model is a one-line change.

**Other rules of engagement**
- **Schema is frozen after Day 1.** Any change goes through P1 and gets announced — it invalidates converted data.
- **Ship a rough model fast.** Everything downstream waits on your model, so get a *weak* baseline out in the first few days. Existence before quality — P2 needs something real to integrate against.
- **One git repo, feature branches, PRs.** No "three notebooks that never merge."
- **The bonus is the first thing to cut.** P3's fine-grained MY/Foreign work only proceeds if the mandatory pipeline is locked. Drop it without guilt if time is tight — it's bonus points, not a requirement.
- **Deliverables are a whole-team push.** In the final ~3 days, all three pause new features to split the brief, video, and (if top 10) poster + pitch.

---

## 4. Step-by-step playbook

### Step 1 — Setup & schema lock  *(whole team, Day 1)*
- **Goal:** repo scaffolded, environment reproducible, label taxonomy agreed.
- **Do:** stand up the repo (Claude Code can scaffold the whole tree). Everyone installs from a pinned `requirements.txt`. Then the team agrees `schema.yaml`:
  ```
  Civilian:    [container_ship, tanker, cargo, passenger_ferry]
  Small_Craft: [yacht, speedboat, fishing_boat]
  Military:    [military_vessel]      # + optional MY / Foreign split
  ```
- **Tip:** keep the mandatory taxonomy *coarse* (matches the brief). Keep fine ship types as an optional sub-label for the bonus — don't fragment your Military class into 50 pieces or recall will collapse.
- **Also on Day 1:** lock the **API contract** — `predict(image_or_video) → [{class, confidence, bbox}]` — and P1 hands P2 a **stub predictor** returning fake detections in that format, so P2 starts the GUI + serving layer immediately.
- **Output:** `configs/schema.yaml`, a working repo everyone can clone and run, agreed API contract + stub.

### Step 2 — Acquire raw data  *(P1)*
- **Goal:** all source datasets downloaded into `data/raw/`.
- **Do:** run the starter stack from the Sourcing Map (SeaShips 7000, SMD, Roboflow Military Ships, ShipRSImageNet, HRSC2016). Log each in `DATASETS.md` with source + licence.
- **Tip:** grab the Roboflow sets in **YOLO export** to skip conversion; grab the academic sets in native format and convert.

### Step 3 — Convert & merge  *(P1)*
- **Goal:** one unified YOLO dataset in `data/processed/`.
- **Do:** converters map each source's format → YOLO txt (`voc2yolo`, `dota2yolo`, `cls2det` for classification chips). `merge.py` remaps every native class into `schema.yaml`, dedupes overlaps (FGSCR-42 ⊂ DOTA+HRSC; ShipRSImageNet ⊂ HRSC+FGSD), and writes a clean train/val/test split.
- **Tip:** **horizontal boxes first** (standard YOLO detect) — simpler and enough to pass. Consider oriented boxes (YOLO-OBB, supported in YOLO26/YOLO11) only if aerial recall lags. Hold out a genuine, *un-augmented* val set per domain so your recall numbers are honest.
- **Output:** `data/processed/`, `configs/data.yaml`.

### Step 4 — Baseline training  *(P1)*
- **Goal:** a first model that detects all mandatory classes.
- **Do:** train **YOLO26** (latest, NMS-free, edge-friendly for the live demo) — or **YOLO11** if you want the most battle-tested stability. Start with a **medium** variant (`yolo26m`/`yolo11m`) for accuracy; keep a **nano** variant for the fast Phase-2 live demo. Compute: Colab / Kaggle free GPU (Kaggle ≈ 30 hrs/week per account — 3 accounts is still ~90 hrs/week, plenty for this scale).
- **Tip:** turn on strong augmentation — mosaic, mixup, HSV, and especially **copy-paste**, which is gold for boosting rare-class (military) recall. Log runs to `outputs/runs/`.

### Step 5 — Tune for >90% military recall  *(P1)*
- **Goal:** clear the hard gate: **Recall > 90% on military/threat classes.**
- **Do:** three levers, in order of leverage:
  1. **Lower the confidence threshold** for military classes. The gate is *recall*, so you deliberately shift the operating point to catch more (accepting more false positives). This is the single biggest, cheapest win.
  2. **Rebalance data:** oversample + copy-paste-augment military images so the class isn't starved.
  3. **Class-weighted loss** to penalise missed military detections harder.
- **Tip:** track **per-class recall**, not just overall mAP — a great mAP can still hide a failing military recall. `metrics.py` should print the gate check as pass/fail.

### Step 6 — Bonus: Malaysian vs Foreign  *(P3 data collection + P1 classifier, only after mandatory is locked)*
- **Goal:** the differentiator for "significantly higher technical scores."
- **Do:** **two-stage** approach — the main model detects `military_vessel`; crop each detection; feed to a small fine-grained classifier (`src/fine_grained/`) fine-tuned on FGSCR-42/FGSC-23 + your custom RMN image set (Wikimedia Commons + Google Earth over Lumut/Sepanggar). Keeps the main detector clean and the bonus decoupled.
- **Tip:** be realistic — aim for a few well-represented RMN classes vs a "Foreign" bucket. Don't gate the whole submission on hitting 90% here; any working MY/Foreign distinction earns bonus credit. **P3 can start the image collection/annotation from Day 1** (it's independent of the model), so the data is ready the moment the mandatory pipeline is locked.

### Step 7 — Inference + detection log  *(P2)*
- **Goal:** run the model on the **Qualifier Video Clip** and produce the required log.
- **Do:** `detect.py` runs on the clip frame-by-frame; add a **tracker** (ByteTrack/BoT-SORT, built into Ultralytics) to smooth detections across frames and avoid flicker. `detection_log.py` emits a clean log: `frame/timestamp · class · confidence · bbox`.
- **Tip:** confirm the clip's nature early (harbour/surface vs aerial) — it tells you which domain to weight. Save an annotated output video; it doubles as YouTube demo footage.

### Step 8 — Evaluate & benchmark  *(P2 + P1)*
- **Goal:** defensible numbers for the brief and the jury.
- **Do:** per-class recall/precision table, confusion matrix, PR curves, inference speed (FPS). Put a results table in the `README`.
- **Tip:** report the honest val-set recall *and* the qualifier-clip results separately.

### Step 9 — GUI / live-demo station  *(P2)*
- **Goal:** the "significant competitive advantage" GUI + the Phase-2 station.
- **Do:** Streamlit or Gradio — upload image/video → run model → show annotated output + a live detection table. Must handle **fresh, unseen** data fast (Phase 2's "hidden verification" stress test).
- **Tip:** use the **nano** model here for speed; pre-warm the model so on-the-spot inference is instant in front of judges.

### Step 10 — Technical Brief PDF  *(P3 + whole team)*
- **Goal:** the required PDF: dataset used, model architecture, **and the logic for military classification** (judges specifically want this).
- **Do:** pull provenance from `DATASETS.md`, architecture from the training config, metrics from `eval/`. Explain *why* your military-recall approach works.
- **Tip:** a clean pipeline diagram (reuse §1) + a results table reads as maturity.

### Step 11 — YouTube demo (≤5 min)  *(P3 + whole team)*
- **Goal:** the video deliverable.
- **Do:** problem → pipeline → live detection on the qualifier clip → GUI walkthrough → results. Screen-record the GUI; use the annotated video from Step 7.
- **Tip:** lead with a strong 20-second hook showing detections firing — judges skim.

---

## 5. Suggested sequencing (3 people, ≈3-week sprint)

You (P1) hold the critical path (data → model → recall gate). The stub predictor + fixed API contract let P2 build the entire GUI/serving layer in parallel before your model exists, and P3 runs the bonus data collection independently — so all three are productive from Day 1 despite the serial core.

```
Week 1 │ Day 1   Setup + schema lock + API contract + stub handoff (all three)
       │ D2–4    P1: acquire → convert → merge     ║ P2: build GUI + serving vs STUB
       │         │                                 ║ P3: start RMN image scrape + annotate
       │ D5–7    P1: baseline training (rough!)    ║ P2: detection-log + eval scripts
       │         └─ ship weak model early ─────────╫─▶ P2 swaps stub → real model
─────────────────────────────────────────────────────────────────────────────────────
Week 2 │ D8–11   P1: recall-tuning loop to >90%    ║ P2: qualifier-clip inference + log
       │         │  ← the critical gate            ║ P3: finish RMN dataset
       │ D12–14  P1: freeze best model once gate passes ║ P2: GUI v1 polished
─────────────────────────────────────────────────────────────────────────────────────
Week 3 │ D15–17  P1+P3: MY-vs-Foreign bonus  ← ONLY if gate passed; else skip
       │         P2+P1: evaluate + benchmark, results table
       │ D18–21  ALL: technical brief + record video + package submission
```

You're on semester break, so calendar isn't the constraint — the real limits are compute, data volume, and three people. The two habits that keep it unblocked: **P1 ships a rough model by ~D7** so P2 isn't stuck on the stub forever, and **the whole team converges on deliverables for the last ~3 days** (P3 leads, since it's their strength, but brief + video are big enough to need all three).

**If you get a 4th person back:** best addition is a second modeller to share P1's load (split frontal vs aerial), since you're currently the single point of failure on the critical path.

---

## 6. High-leverage decisions cheat-sheet

| Decision | Recommendation | Why |
|---|---|---|
| Model | **YOLO26** (or YOLO11 for max stability) | NMS-free, fast edge inference for live demo; strong small-object handling |
| Model size | `m` for accuracy, `n` for the live demo | Balance the recall gate vs on-the-spot speed |
| Box type | Horizontal first, OBB only if aerial lags | OBB is more work; horizontal usually passes |
| Domains | **One unified model** on frontal+aerial union first | Simpler; split into 2 models + router only if metrics show one domain dragging |
| Hitting >90% recall | Lower military conf threshold + copy-paste aug + class weights | Recall gate rewards catching over precision |
| MY vs Foreign | Two-stage: detect military → crop → classify | Decouples the risky bonus from the mandatory detector |
| Compute | Colab / Kaggle free GPU (×3 accounts) | Free, ~90 GPU-hrs/week total — enough for this scale |
| Annotation | Roboflow (team + versioning + YOLO export) | Removes format pain when several people label |

---

## 7. Deliverables checklist (maps to the brief)

**Phase 1**
- [ ] Model source code (standard OSS libs) — the repo
- [ ] Detection log on the Qualifier Video Clip — Step 7
- [ ] Recall > 90% on military/threat classes — Step 5, proven in Step 8
- [ ] Technical Brief PDF (dataset · architecture · military-classification logic) — Step 10
- [ ] YouTube video demo (≤5 min) — Step 11
- [ ] *(bonus)* Malaysian vs Foreign distinction — Step 6

**Phase 2 (top 10)**
- [ ] Display poster (pipeline · methods · accuracy)
- [ ] Live demo station (real-time inference)
- [ ] Functional GUI — Step 9 (competitive advantage)
- [ ] Jury pitch (technical approach + scalability)
- [ ] Live stress-test readiness (process hidden verification data on the spot)
```
