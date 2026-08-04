import os
from roboflow import Roboflow

# Safely fetch API key from environment variables (.env)
api_key = os.getenv("ROBOFLOW_API_KEY")
if not api_key:
    raise ValueError("ROBOFLOW_API_KEY environment variable is not set! Check your .env file.")

# Connect to Roboflow workspace and project
rf = Roboflow(api_key=api_key)
project = rf.workspace("boats-ri7td").project("speedboat")
version = project.version(2)

# Download dataset in YOLOv8 format into data/raw/
dataset = version.download("yolov8", location="data/raw/civilian_gapfill")