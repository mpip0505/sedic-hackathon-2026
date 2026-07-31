# Team Tasks — remaining work, 5 people

_Created 2026-07-31 · supersedes the 3-person split in `PROJECT_PLAYBOOK.md` §3_

The mandatory gate is **already passed** (military recall 0.904 overall; aerial
0.940 / surface 0.954 — see `outputs/eval/test_eval.md`). The detector, the
inference path, the eval harness and the detection GUI are **built**. What is
left is: the **bonus** (Malaysian vs Foreign), the **presentation layer**, and
the **submission package**.

Find your handle below and read only that section. Everything above your
section is context; everything below it is someone else's job.

| Handle | Person | Owns | Ships |
|---|---|---|---|
| **LEAD** | you | `src/data/`, `src/train/`, `src/eval/`, `src/inference/`, `src/fine_grained/`, `configs/` | model, gate, 2nd-stage classifier, all merges |
| **GUI** | 1 person | `app/pages/`, `app/assets/` | landing page + navigation |
| **DATA-RMN** | 1 person | `data/raw/fine_grained/malaysian_rmn/` | the RMN half of the bonus set |
| **DATA-FOR** | 1 person | `data/raw/fine_grained/foreign/`, surface gap-fill | the Foreign half + thin-class top-up |
| **DELIV** | 1 person | `deliverables/` | technical brief PDF, ≤5 min video, poster |

**Rules that apply to everyone:**

1. **Never commit images, videos or weights.** `data/`, `models/`, `outputs/`
   are git-ignored and stay that way. Data moves by shared drive / Roboflow
   link, not by git.
2. **`configs/schema.yaml` is the contract.** Nobody edits it except LEAD, and
   only after announcing it — a change invalidates converted data.
3. **Branch + PR.** `git checkout -b feat/<area>-<desc>`. Before opening a PR:
   `ruff check . && pytest -q && python -m src.inference.predict --source none --stub`.
4. **Numbers come from `outputs/eval/test_eval.md`, never from memory.** If you
   need a figure for a slide, a page or the brief, read it from there or ask
   LEAD to regenerate it. Do not hardcode metrics anywhere.
5. **Every image source gets a licence.** No licence recorded → the image does
   not enter the repo. See rule 5 in `CLAUDE.md`.

---

## LEAD — ML core (Afif)

Unchanged ownership: data merge, training, eval, `predict()`, and now the
2nd-stage classifier. Remaining:

1. **Scene/group-aware split decision** (decision log, 2026-07-26). SeaShips is
   video-derived; at dedup threshold 3 near-adjacent frames can land in
   different splits. This is flagged as **REQUIRED before any numbers enter the
   technical brief** — so it blocks DELIV. Either do it and retrain, or record
   the decision to accept the current split and why. **Do this first; DELIV is
   waiting on the final number.**
2. **`speedboat` recall 0.384** — not gate-relevant, but it will be asked about
   in the jury pitch. Understand it, then either fix it or write the one-line
   explanation DELIV can use.
3. **Ingest DATA-RMN / DATA-FOR handoffs**: add `schema.yaml` mapping + domain
   entries, run the converter → merge → dedup → validate, add the
   `data/DATASETS.md` rows.
4. **Build `src/fine_grained/`** once both halves of the bonus set land: crop
   → classify `malaysian_rmn` vs `foreign`, running on `military_vessel`
   detections only. Flip `fine_grained.enabled` in `schema.yaml` when it works.
5. **Qualifier Clip detection log** the moment the clip is provided (mechanism
   is already built — GUI CSV export).
6. **Phase 2 live stress test rehearsal**: judge-supplied media through the GUI,
   including the stub fallback path if weights fail to load.

---

## GUI — landing page

**Goal:** the first screen a judge sees. Right now `app/app.py` drops straight
into an uploader; there is no framing of what the system *is*. Add a
landing page and the navigation between it and the existing detection view.

### What to build

`app/pages/` as a Streamlit multipage app, entered from `app/app.py`:

- **Home / landing** (new, yours) — mission framing:
  - Hero: Project Guardian · Advanced Maritime Domain Awareness · SEDIC 2026.
  - **The gate, front and center**: military recall, overall + per domain, with
    the threshold it was measured at (`conf_military = 0.10`) and the split
    (TEST). Judges are scoring against >90% — make it impossible to miss.
  - **Pipeline diagram**: raw datasets → convert → dedup → stratified split →
    YOLO11m → detect → track. Static SVG or HTML/CSS, no external CDN.
  - **The taxonomy**: the 8 classes as colour-coded chips, grouped
    civilian / small craft / military, read from `schema.yaml` via
    `class_groups()` — do **not** retype the class names.
  - **Data provenance**: the four datasets, domains, licences, image counts.
  - **CTA button** → the detection page.
- **Detect** (existing) — `app/app.py`'s current `main()`, moved into a page,
  behaviour unchanged.

### Hard constraints (these have already caused crashes — read `docs/PROGRESS.md` §4)

- **No `st.dataframe` / `st.table`.** pyarrow segfaults the whole server on some
  numpy/pyarrow builds. Build tables as HTML — copy the pattern in
  `render_table()`.
- **The Streamlit file watcher is disabled** (`.streamlit/config.toml`), because
  it segfaults walking `torch.classes`. Your edits will *not* hot-reload —
  restart the server manually. If you see "Cannot load Streamlit frontend code"
  in the browser, that is a native crash, not your Python; rerun with
  `PYTHONFAULTHANDLER=1` to get a traceback.
- **No hardcoded metrics.** Parse `outputs/eval/test_eval.md`, or read a small
  JSON that LEAD generates. The numbers *will* change if the model is retrained,
  and a stale number on the landing page in front of a judge is a real cost.
- **No hardcoded class names.** `get_class_groups()` already reads them from
  `schema.yaml`.
- **No external assets at runtime** — no CDN fonts, no remote images. The demo
  station may have no network.
- **Don't refactor the detection page.** Move it, don't rewrite it; LEAD is
  still changing `predict()`.

### Done when

- `streamlit run app/app.py` opens on the landing page, and the CTA reaches a
  detection page that still works on image, video and `--stub`.
- Retraining changes the numbers on the landing page with no code edit.
- `ruff check .` clean, `pytest -q` green.

---

## DATA-RMN — the Malaysian half of the bonus set

**Why this matters:** the brief awards "significantly higher technical scores"
for distinguishing Local (Malaysian) from Foreign military assets. This is the
single highest-value item left. **There is no ready-made dataset** — it is a
collect-and-annotate job. See `docs/DATA_SOURCING_MAP.md` §5 for sources.

### Target

Royal Malaysian Navy (TLDM) classes, **~150–300 images each**, ≥6 classes:

Kedah-class NGPV · Lekiu-class frigate · Kasturi-class corvette ·
Laksamana-class corvette · Scorpène-class submarine · Keris-class LMS ·
Mahamiru-class MCMV · Gagah Samudera training ship

Mix **surface/frontal and aerial** views — the system is judged on both.
A class you can only find 40 usable images of is better dropped than padded.

### Sources, in order of preference

| Source | Licence |
|---|---|
| Wikimedia Commons — "Ships of the Royal Malaysian Navy" | CC / PD — **attribute** |
| TLDM / MINDEF Malaysia official press + social | check terms per item |
| shipspotting.com, MarineTraffic | copyright varies — **verify each** |
| Google Earth over Lumut / Sepanggar / Kota Kinabalu | Google Earth ToU (aerial) |
| Naval defence media (Naval News, MalaysianDefence, Janes) | editorial — **reference only, do not redistribute** |

### Handoff format — this is the contract

Two products. **A is required, B is a bonus if you have time.**

**A. Fine-grained crops** (feeds `src/fine_grained/`, LEAD's 2nd stage):

```
data/raw/fine_grained/
  malaysian_rmn/
    kedah_ngpv/       img_0001.jpg …
    lekiu_ff/         …
    scorpene_ssk/     …
  manifest.csv
```

- **One vessel per crop**, cropped to the hull with ~10% margin.
- **≥96 px on the short side**; anything blurrier than "you can tell the class"
  is not usable.
- Folder name = ship class slug, lowercase, underscores. The *classifier* label
  is the parent folder (`malaysian_rmn`) — the ship-class subfolder is for
  provenance and for a possible finer split later.
- Label vocabulary is fixed by `schema.yaml` → `fine_grained.labels`
  (`malaysian_rmn`, `foreign`). **Do not invent labels.**

`manifest.csv`, one row per image, exact header:

```csv
filename,label,ship_class,navy,domain,source_url,licence,collected_by,date
```

`label` ∈ {`malaysian_rmn`}, `domain` ∈ {`surface`, `aerial`},
`licence` = the actual licence string (e.g. `CC BY-SA 4.0`, `Public Domain`).
A row with an empty `source_url` or `licence` will be rejected.

**B. Detection-format set** (optional; feeds the main detector):
a Roboflow project, single class `ship`, exported as **YOLOv8**, delivered as
the export zip. LEAD wires the `schema.yaml` mapping.

### Rules

- **Deduplicate against yourself** before handing over — no two crops of the
  same photo, no near-identical burst frames. LEAD dedups across datasets, but
  garbage in is still garbage.
- **Don't pre-split** into train/val/test. LEAD's stratified splitter does that.
- Deliver by shared drive / Roboflow link. **Nothing goes in git.**
- Interim handoffs are fine and preferred — send the first 2 classes as soon as
  they're done so LEAD can validate the format before you've done all 8.

### Done when

`manifest.csv` validates (no missing licences), ≥6 classes at ≥150 images, both
domains represented, and LEAD has confirmed one interim batch loads cleanly.

---

## DATA-FOR — the Foreign half + surface gap-fill

Two jobs. The first is the counterpart to DATA-RMN and has the same contract;
the second closes measured holes in the detector.

### Job 1 — the `foreign` bucket (primary)

A binary classifier needs both halves. Aim for **rough parity with DATA-RMN's
total** (~1,000–1,500 images), spread across navies that are regionally
plausible so the classifier learns "Malaysian vs not" and not "grey ship vs
American ship":

US Navy (Arleigh Burke, Ticonderoga, carriers) · China PLAN (Type 052D, Type
054A) · Singapore RSN (Formidable, Independence) · Australia (Hobart, ANZAC) ·
Indonesia TNI-AL · India

**Identical handoff format to DATA-RMN**, under `foreign/` instead:

```
data/raw/fine_grained/
  foreign/
    usn_arleigh_burke/  …
    plan_type054a/      …
    rsn_formidable/     …
  manifest.csv          # label column = "foreign", navy column = usn|plan|rsn|ran|tni_al|in
```

**Watch for the trap:** RSN, TNI-AL and RMN operate visually similar hulls in
the same waters. Those are the *hard* negatives and the most valuable images in
your set — collect them deliberately rather than filling up on US carriers,
which are trivially separable and teach the classifier nothing.

**Do not source foreign warships from ShipRSImageNet / `military_ships`.** Those
are already in the detector's train split — reusing them leaks into the
2nd stage. Fresh imagery only.

### Job 2 — surface civilian gap-fill (secondary)

`docs/PROGRESS.md` §4 records real holes in the current build:

| Gap | Current state |
|---|---|
| `speedboat` | recall **0.384** — worst class by far; **0 surface** instances |
| `tanker` | **0 surface** instances |
| `yacht` | thinnest class overall (390); **0 surface** instances |

Find surface/frontal imagery for these three. Roboflow Universe and the
Singapore Maritime Dataset (already mapped in `schema.yaml`) are the first
places to look — an existing YOLO export you can hand over whole beats
hand-annotation every time. Give LEAD the Roboflow URL, version, licence and
class list; LEAD does the remap.

This is genuinely secondary — the gate is already passed and these classes are
not gate classes. Do it only when Job 1 is done or blocked.

### Done when

`foreign/` matches `malaysian_rmn/` in scale with ≥5 navies and deliberate
RSN/TNI-AL coverage, manifest validates, and any gap-fill sets are handed over
as source links with licences.

---

## DELIV — technical brief, video, poster

Owns the whole submission package. **Everything here is graded directly.**

### 1. Technical Brief (PDF) — mandatory

The brief names exactly three required contents: **the dataset used, the model
architecture, and the logic used for military classification.** Cover all three
explicitly and in that order — a judge should be able to tick them off.

Most of the content already exists in `docs/PROGRESS.md`; your job is
selection, narrative and design, not original research.

| Section | Source |
|---|---|
| Problem + approach | `docs/PROJECT_PLAYBOOK.md` §1 |
| **Dataset** — 4 sources, licences, counts, both domains | PROGRESS §2.1, `data/DATASETS.md` |
| Data pipeline — convert → dedup → stratified split | PROGRESS §2.3–2.4 |
| **Architecture** — YOLO11m, 100 epochs, input size, HBB | `configs/train_baseline.yaml`, PROGRESS §3 |
| **Military-classification logic** — the 50→8 collapse and *why* military stays one coarse class; the dual-threshold rule (`conf_military` 0.10 < `conf` 0.25) as the recall mechanism | PROGRESS §2.2, CLAUDE.md non-negotiable #3 |
| Results — the gate, per-domain, per-class, conf sweep | `outputs/eval/test_eval.md` |
| Bonus — RMN vs Foreign 2nd stage | LEAD, once built |
| Reproducibility — repo layout, one-command rebuild | README |

**Three things that will win or lose points here:**

- **Quote the gate at `conf_military = 0.10`, on the TEST split.** The GUI demo
  runs at 0.25 for a cleaner picture; the *number in the brief* is the 0.10 one.
  Say which is which — do not let the two get mixed up.
- **Per-domain recall is the multi-angle evidence.** Aerial and surface each
  clearing 0.90 independently is what proves "multi-angle classification", not
  the overall number. Lead with the table.
- **The dedup story is a credibility asset.** "We found single-linkage chaining
  was collapsing 2,961 distinct frames and fixed it" tells a judge the numbers
  are honest. `docs/dedup_audit.md` has the evidence.

⚠️ **Blocked on LEAD:** the scene-aware-split decision (PROGRESS decision log,
2026-07-26) is marked *required before any numbers enter the brief*. Write
every section that isn't a number now; leave the results table for last and
fill it from `outputs/eval/test_eval.md` once LEAD confirms it is final.

### 2. Video demonstration — mandatory, **max 5 minutes**, YouTube

Over-length risks disqualification. Suggested cut:

| Time | Beat |
|---|---|
| 0:00–0:20 | **Hook** — detections firing live on a busy scene, military boxes going red. No preamble; judges skim. |
| 0:20–1:00 | Problem + what Project Guardian is |
| 1:00–2:00 | Data + pipeline (reuse the landing page's diagram from GUI) |
| 2:00–3:30 | **Live GUI walkthrough** — upload → detect → military alert → video tracking with IDs → CSV detection log export |
| 3:30–4:20 | Results: the gate, per-domain, both view angles side by side |
| 4:20–5:00 | Bonus (RMN vs Foreign) + scalability close |

Screen-record the real GUI, not slides. Show **both** an aerial and a frontal
image — multi-angle is the headline requirement. Upload unlisted first, share
the link with the team before making it public.

### 3. Poster — Phase 2 (top 10 only)

Required: **AI pipeline · data-processing methods · model accuracy.** Don't
build it until Phase 1 results are in, but reuse the brief's diagrams so it's a
half-day job when it's needed.

### 4. Jury pitch + submission packaging — Phase 2

- Pitch = technical approach + **scalability**; the brief explicitly asks for
  scalability, so prepare an answer for "what happens with 100 cameras".
- Assemble the final package: source code link, detection log from the Qualifier
  Clip, benchmark numbers, brief PDF, video URL.
- Prepare answers for the obvious hostile questions: *"Why is speedboat recall
  low?"*, *"How do you know there's no train/test leakage?"*, *"What happens on
  a vessel type you've never seen?"*

### Done when

Brief PDF is in `deliverables/technical_brief/`, video is uploaded and under
5:00, and the submission checklist in `PROJECT_PLAYBOOK.md` §7 is fully ticked.

---

## Dependency map

```
LEAD: scene-split decision ──────────────► final numbers ──► DELIV: brief results
LEAD: (already done) gate + GUI ─────────► GUI: landing page ──► DELIV: video footage
DATA-RMN: malaysian_rmn crops ──┐
                                 ├──► LEAD: src/fine_grained/ ──► DELIV: bonus section
DATA-FOR: foreign crops ────────┘
DATA-FOR: surface gap-fill ──────────────► LEAD: optional retrain (only if time)
```

**The critical path to a better score is the bonus set.** GUI and DELIV can both
finish without it; the fine-grained classifier cannot. If DATA-RMN or DATA-FOR
slips, that is the one to escalate.

**The bonus is also the first thing to cut.** If it can't land in time, drop it
without guilt — the mandatory gate is already passed, and a polished submission
with no bonus beats a broken one with a half-trained classifier.
