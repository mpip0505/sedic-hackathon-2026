import os
import shutil
import tempfile
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from roboflow import Roboflow

load_dotenv(find_dotenv())

api_key = os.environ.get("ROBOFLOW_API_KEY")
if not api_key:
    raise SystemExit("ROBOFLOW_API_KEY not set - check your .env")

target_dir = Path("data/raw/civilian_gapfill")

# Clean target directory if it already exists to avoid overwrite errors
if target_dir.exists():
    shutil.rmtree(target_dir)
target_dir.mkdir(parents=True, exist_ok=True)

rf = Roboflow(api_key=api_key)
project = rf.workspace("boats-ri7td").project("speedboat")
version = project.version(2)

# Stage the export in a throwaway dir, then move its contents into target_dir —
# roboflow errors if it downloads into a folder that already has content.
# tempfile picks the right place per OS (%TEMP% on Windows, /tmp on macOS/Linux);
# a hardcoded "C:/..." would create a literal "C:" folder in the repo elsewhere.
staging_root = Path(tempfile.mkdtemp(prefix="gapfill_"))
try:
    # download into a path that does NOT exist yet, inside the staging dir
    download_dir = staging_root / "export"
    version.download("yolov8", location=str(download_dir))

    for item in download_dir.iterdir():
        shutil.move(str(item), str(target_dir))
finally:
    shutil.rmtree(staging_root, ignore_errors=True)

print(f"Done -> {target_dir.resolve()}")