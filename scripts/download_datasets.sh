#!/usr/bin/env bash
# download_datasets.sh — placeholder for fetching raw datasets into data/raw/.
#
# Each dataset has its own access terms (several are academic-only — see
# data/DATASETS.md). Do NOT commit any downloaded images. Fill in the real
# fetch commands per dataset as access is obtained.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$REPO_ROOT/data/raw"
mkdir -p "$RAW"

echo ">> Target raw data directory: $RAW"

# TODO(P1): SeaShips (frontal/surface) — academic-only.
#   mkdir -p "$RAW/seaships" && <download + extract into it>

# TODO(P1): Singapore Maritime Dataset (frontal/surface) — academic-only.
#   mkdir -p "$RAW/singapore_maritime" && <download + extract>

# TODO(P1): Roboflow Military Ships — check project licence.
#   mkdir -p "$RAW/roboflow_military_ships" && <roboflow export>

# TODO(P1): ShipRSImageNet (aerial/satellite) — academic-only. DEDUP RISK.
#   mkdir -p "$RAW/shiprsimagenet" && <download + extract>

# TODO(P1): HRSC2016 (aerial/satellite) — academic-only. DEDUP RISK.
#   mkdir -p "$RAW/hrsc2016" && <download + extract>

# TODO(P1): Custom RMN set — team-collected. Verify image rights.
#   mkdir -p "$RAW/rmn_custom" && <sync from team storage>

echo ">> Placeholder only — no datasets fetched. See TODOs above and data/DATASETS.md."
