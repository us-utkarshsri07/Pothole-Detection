from pathlib import Path
from ultralytics import YOLO

# Project root
ROOT = Path(__file__).resolve().parent.parent

# Correct paths
MODEL_PATH = (
    ROOT / "results" / "training" /
    "baseline_30_epochs" / "weights" / "best.pt"
)

DATA_PATH = ROOT / "dataset" / "processed" / "data.yaml"

# Verify paths
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset YAML not found: {DATA_PATH}")

print("Model found:", MODEL_PATH)
print("Dataset found:", DATA_PATH)

# Load trained model
model = YOLO(str(MODEL_PATH))

# Evaluate model
results = model.val(
    data=str(DATA_PATH),
    imgsz=640,
    plots=True
)

print(results)