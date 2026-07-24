# Datasets — provenance & licence log

Every dataset used in Project Guardian **must** have a row here **before** its
data is used. Record source, licence, who added it, and when.

> ⚠️ **Dedup warning — cross-dataset leakage.** Several of these overlap and
> will leak between train and val if merged naively:
> - **FGSCR-42** derives from **DOTA + HRSC**.
> - **ShipRSImageNet** overlaps **HRSC + FGSD**.
> Deduplicate across sources before splitting. `dota` is mapped to `null`
> (dropped) in `configs/schema.yaml` for this reason.

> 🔒 **Academic-use-only** datasets are marked below. Respect each licence — do
> not redistribute their images, and never commit dataset images or weights.

| Dataset | Domain | Images | Used for | Source | Licence | Added by | Date |
|---------|--------|-------:|----------|--------|---------|----------|------|
| SeaShips | frontal/surface | ~31k | train + val | [SeaShips paper/site](https://github.com/jiaming-wang/SeaShips) | Academic-only 🔒 | P1 | _tbd_ |
| Singapore Maritime | frontal/surface | ~TBD | train + val | [SMD](https://sites.google.com/site/dilipprasad/home/singapore-maritime-dataset) | Academic-only 🔒 | P1 | _tbd_ |
| Roboflow Military Ships | frontal + aerial | ~TBD | train (military) | [Roboflow Universe](https://universe.roboflow.com/) | Check project licence (CC-BY / other) | P1 | _tbd_ |
| ShipRSImageNet | aerial/satellite | ~3.4k | train + val | [ShipRSImageNet](https://github.com/zzndream/ShipRSImageNet) | Academic-only 🔒 (overlaps HRSC+FGSD) | P1 | _tbd_ |
| HRSC2016 | aerial/satellite | ~1k | train + val | [HRSC2016](https://www.kaggle.com/datasets/guofeng/hrsc2016) | Academic-only 🔒 (source of FGSCR-42) | P1 | _tbd_ |
| Custom RMN set | frontal + aerial | ~TBD | val + fine-grained | Team-collected (Royal Malaysian Navy imagery) | Team-owned; verify image rights | P1 | _tbd_ |

### Notes
- Fill in `Images`, `Added by`, and `Date` as each set is actually ingested.
- The **Custom RMN set** feeds both detection val and the bonus fine-grained
  `malaysian_rmn` vs `foreign` classifier (`src/fine_grained/`).
- Map each source's native class names in `configs/schema.yaml` under
  `mappings:` — do not translate labels anywhere else.
