# SEDIC 2026 — Data Sourcing Map (Data Lead)

**Project Guardian: Maritime Domain Awareness — Phase 1 dataset plan**

Your job as data lead: assemble clean, unified, **YOLO-ready** data across two domains (surface/frontal-view + aerial/satellite), with the **Military** classes populated well enough to clear **Recall > 90%**. This doc lists every source worth touching, plus the workflow to merge them.

---

## 0. How to think about the data

The competition demands **multi-angle** detection. Frontal (camera-on-water) and aerial (satellite/drone) are basically **different visual domains** — a model trained only on one will fail on the other. So you're sourcing two parallel piles and unifying their labels.

| Competition class | Where the data comes from |
|---|---|
| Civilian (container, tanker, cargo, ferry) | SeaShips, SMD (surface) + ShipRSImageNet, DOTA (aerial) |
| Small Craft (yacht, speedboat, fishing) | SeaShips, SMD, ABOShips (surface) + ShipRSImageNet (aerial) |
| Military (mandatory, >90% recall) | Roboflow Military Ships, ShipRSImageNet, HRSC2016, FGSCR-42 |
| Malaysian vs Foreign military (bonus) | **Custom build — no dataset exists** (see §5) |

---

## 1. Surface / frontal-view datasets (civilian + small craft)

| Dataset | Size | Classes | Access | Notes |
|---|---|---|---|---|
| **SeaShips** | 31,455 imgs | 6 (ore carrier, bulk cargo, general cargo, container, fishing boat, passenger) | GitHub: `jiaming-wang/SeaShips`; common **SeaShips(7000)** subset also mirrored on Roboflow (`marine-cv6x4/seaships-zhqhn`) | Coastline surveillance cameras. Your best civilian + fishing-boat source. VOC XML → convert to YOLO. |
| **Singapore Maritime Dataset (SMD)** | ~20k RGB imgs / 157k objects (from ~51–80 videos) | 10 ship classes, incl. small craft | Search "Singapore Maritime Dataset" (NUS/ARL release) | On-shore + on-board viewpoints, day + night, has near-IR. Great viewpoint variety. Frame-by-frame labels. |
| **ABOShips** | inshore + offshore | vessels + buoys + small craft | arXiv 2102.05869 / linked repo | Ferry-mounted cameras (Finnish archipelago). Precise annotations, good small-craft coverage. |
| **SeaSAw (Sea Machines)** | 1.9M imgs / 14.6M objects | 12+ | Request from Sea Machines | Huge & diverse but access is gated/commercial. Aspirational, not day-one. |
| **Roboflow Universe (surface)** | varies | varies | `universe.roboflow.com` search: `boat`, `ship`, `maritime` | Many are already **YOLO-export ready** — zero conversion. Quality varies, check annotations. |

---

## 2. Aerial / satellite datasets (civilian + military)

| Dataset | Size | Classes | Access | Notes |
|---|---|---|---|---|
| **ShipRSImageNet** ⭐ | 3,435 imgs / 17,573 instances | **50 types** (incl. carriers, destroyers, frigates, submarines) | GitHub: `zzndream/ShipRSImageNet` | **The key aerial-military source.** 4-level hierarchical taxonomy (Class→Category→Type). H-box + oriented box. Dev kit converts VOC→COCO. *Academic use only.* |
| **HRSC2016** | ~1,061 imgs / 2,976 instances | ~25 (incl. warships, aircraft carriers) | Kaggle: `guofeng/hrsc2016` | 6 famous ports on Google Earth. Oriented boxes. Classic warship-in-aerial benchmark. |
| **DOTA (v1.0 / v2.0)** | 2,806 / 11,268 imgs | "Ship" is 1 of 15 classes | `captain-whu.github.io/DOTA` | Generic ship detection in large aerial scenes (not fine-grained military). Oriented boxes, huge images — needs tiling. |
| **FGSCR-42** | 9,320 chips | 42 fine-grained (from 10 broad: carrier, cruiser, destroyer, frigate, etc.) | Baidu pan (link in `jasonmanesis` repo) | **Classification** (cropped chips, not detection). Use as a 2nd-stage fine-grained warship classifier. Built from DOTA+HRSC+NWPU. |
| **FGSC-23** | 4,080 chips | 23 fine-grained | GitHub: `xiong577/ship-datasets` (Google Drive) | Classification chips from Google Earth + GF-2. Same 2nd-stage use as FGSCR-42. |
| **VHRShips** | 5,312 imgs / 11,179 instances | 35 | WACV 2024 refs / Google Earth | Detection, H-box, wide class range. |
| **FGSD2021** | 636 imgs / 5,274 instances | fine-grained | arXiv refs | High-res Google Earth, standardized 1 m GSD. |
| **UOW-Vessel** | large | high-res optical | WACV 2024 paper | Newer benchmark, worth checking for extra volume. |

---

## 3. Military-specific, detection-ready (fast starters)

These are the quickest way to get a working **Military** class, most already YOLO-export:

| Dataset | Size | Classes | Access |
|---|---|---|---|
| **Military Ships (Hanif Noer R)** ⭐ | ~2,746 imgs | **50** (Nimitz, Arleigh Burke, Ticonderoga, submarine, LHA, etc.) | Roboflow: `hanif-noer-r/military-ships` — exports straight to YOLO |
| Warship Classification (Nikhil) | ~2,141 imgs | warship + ships/boats/yacht | Roboflow: `nikhil-rdybh/warship-classification` |
| warship (Tokir) | ~1,068 imgs | war-ship | Roboflow: `tokir/warship-axend` |
| Navy Ship | ~2,125 imgs | ship | Roboflow: `navy-ip6vd/navy-ship-a6prh` |
| WAR SHIPS (vunc) | 144 imgs | aircraft_carrier, warship | Roboflow: `vunc/war-ships` |

**Tip:** the Military Ships (Hanif) set uses the ShipRSImageNet 50-class taxonomy, so it merges cleanly with ShipRSImageNet — collapse both down to your single "Military" competition class (and keep the fine type as a sub-label for the bonus work).

---

## 4. SAR / radar satellite (only if the qualifier clip is radar imagery)

Different modality — **skip unless needed.** Radar ≠ optical; a model trained on optical won't read these.

| Dataset | Notes |
|---|---|
| **OpenSARShip** | 34,528 instances, Sentinel-1, 16 classes — but 61% is Cargo (heavy imbalance) |
| **SSDD** (SAR Ship Detection Dataset) | GitHub: `TianwenZhang0825/Official-SSDD` |
| **HRSID / LS-SSDD / FUSARShip** | Additional SAR options, in the aggregator repo below |

---

## 5. Malaysian vs Foreign military (the bonus differentiator — custom build)

**There is no ready-made dataset.** This is a scrape-and-annotate job, but it's the thing that earns the "significantly higher technical scores." Realistic target: a handful of well-represented RMN classes vs a "Foreign" bucket — don't promise yourself >90% recall here.

**RMN (TLDM) classes to target:**
Kedah-class NGPV · Lekiu-class frigate · Kasturi-class corvette · Laksamana-class corvette · Scorpène-class submarine · Keris-class LMS · Mahamiru-class MCMV · Gagah Samudera training ship

**Foreign navies likely relevant regionally (for the "Foreign" bucket):**
US Navy (Arleigh Burke, Ticonderoga, carriers) · China PLAN (Type 052D, Type 054A) · Singapore RSN (Formidable, Independence) · Australia (Hobart, ANZAC) · Indonesia, India

**Where to source images (mind the licence):**
| Source | Use | Licence |
|---|---|---|
| **Wikimedia Commons** — category "Ships of the Royal Malaysian Navy" + per foreign navy | Best legal reuse | CC / Public Domain (attribute) |
| TLDM / MINDEF Malaysia press + official social media | Authentic RMN imagery | Check terms |
| shipspotting.com, MarineTraffic photos | Community photos, high volume | Copyright varies — verify |
| Naval defence media (MalaysianDefence.com, Naval News, Janes) | Class references | Editorial copyright — reference, don't redistribute |
| Google Earth over Lumut / Sepanggar / Kota Kinabalu naval bases | **Aerial** RMN views | Google Earth ToU |

**Method:** aim ~150–300 images per RMN class, annotate in Roboflow, treat it as a fine-grained layer on top of the "Military" detection (either a 2-stage classifier on military crops, or extra classes in the main model).

---

## 6. Aggregators & search (bookmark these)

- **`jasonmanesis/Satellite-Imagery-Datasets-Containing-Ships`** (GitHub) — curated master list of radar + optical ship datasets with links. Your single best index.
- **Roboflow Universe** — `universe.roboflow.com/search?q=class:warship` and `class:ship`. Many YOLO-ready, team annotation built in.
- **Papers With Code** — search "ship detection" / "maritime" for dataset + SOTA leaderboards.
- **Kaggle Datasets** — search "ship detection", "maritime", "warship".

---

## 7. Data-lead workflow (the order I'd run it)

1. **Lock the label schema on Day 1.** Map every source's classes into the competition taxonomy: Civilian {container, tanker, cargo, passenger ferry} · Small Craft {yacht, speedboat, fishing boat} · Military {+ optional MY/Foreign}. Changing this later is expensive for the whole team.
2. **Convert everything to YOLO format.** Sources arrive as VOC XML (SeaShips, ShipRSImageNet), DOTA txt (oriented), or classification folders (FGSC-23). Get Claude Code to write the converters. Decide horizontal-box (YOLO detect) vs oriented-box (YOLO-OBB — YOLO26/YOLO11 both support it; oriented is better for tilted aerial ships but more work).
3. **Deduplicate.** FGSCR-42 is built *from* DOTA + HRSC + NWPU; ShipRSImageNet pulls from HRSC + FGSD. Cross-check to avoid train/test leakage.
4. **Balance for the recall gate.** The >90% bar is on **Military** — those classes are scarcer in aerial data. Oversample + augment them so recall doesn't tank.
5. **Mind the domain gap.** Either train one model on the frontal+aerial union (needs enough of *both*) or two specialist models + a router. As data lead, make sure *both* domains are well-populated before the model team commits.
6. **Clean val/test split per domain.** Hold out a genuine, un-augmented validation set for each domain so the reported recall is real.
7. **Keep a licence log.** Track licence per source (many are academic-only, Google Earth ToU, or CC-BY needing attribution). You'll need this for the Technical Brief PDF anyway.

---

## 8. Recommended "download first" starter stack

To get the whole team a working baseline across all mandatory classes with minimal friction:

1. **SeaShips (7000 subset)** → civilian + fishing boat (surface)
2. **Singapore Maritime Dataset** → viewpoint variety + small craft (surface)
3. **Military Ships (Hanif Noer R, Roboflow)** → military, YOLO-ready instantly
4. **ShipRSImageNet** → aerial + fine-grained military
5. **HRSC2016** → aerial warships

That five-set combo covers **every mandatory class across both view angles**, and three of the five are already YOLO-export or trivial VOC→YOLO. Ship that to the model team first, then layer in DOTA/VHRShips/FGSCR-42 and the custom Malaysian set.

---

*Licence reminder: ShipRSImageNet, FGSC-23, FGSCR-42 and most Google-Earth-derived sets are **academic/research use only** — fine for a competition POC, but note it in your brief and don't claim commercial rights.*
