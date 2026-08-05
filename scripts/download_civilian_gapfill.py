import os
import shutil
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
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

temp_dir = Path("C:/tmp_gapfill")
if temp_dir.exists():
    shutil.rmtree(temp_dir)

dataset = version.download("yolov8", location=str(temp_dir))

for item in temp_dir.iterdir():
    shutil.move(str(item), str(target_dir))

shutil.rmtree(temp_dir)

print(f"Done -> {target_dir.resolve()}")