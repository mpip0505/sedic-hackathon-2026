import os
from pathlib import Path
from roboflow import Roboflow

api_key = os.environ.get("ROBOFLOW_API_KEY")
if not api_key:
    raise SystemExit("ROBOFLOW_API_KEY not set — check your .env")

target_dir = Path("data/raw/fine_grained/foreign")
target_dir.parent.mkdir(parents=True, exist_ok=True)

rf = Roboflow(api_key=api_key)
project = rf.workspace("navy-ip6vd").project("navy-ship-a6prh")
version = project.version(3)
dataset = version.download("yolov11", location=str(target_dir))

print(f"Done -> {dataset.location}")