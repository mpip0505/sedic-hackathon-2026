import os
from pathlib import Path

from roboflow import Roboflow

api_key = os.environ.get("ROBOFLOW_API_KEY")
if not api_key:
    raise SystemExit("ROBOFLOW_API_KEY not set — check your .env")

Path("data/raw").mkdir(parents=True, exist_ok=True)

rf = Roboflow(api_key=api_key)
project = rf.workspace("hannah-agkvq").project("military-ship-detection-qxv5m")
version = project.version(2)
dataset = version.download("yolov11", location="data/raw/military_surface")

print(f"Done → {dataset.location}")