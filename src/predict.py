
from pathlib import Path
from ultralytics import YOLO

# Project paths
ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    ROOT / "results" / "training" /
    "baseline_30_epochs" / "weights" / "best.pt"
)

TEST_IMAGES = ROOT / "dataset" / "processed" / "images" / "test"

OUTPUT_DIR = ROOT / "results" / "predictions"

# Configuration
IMAGE_SIZE = 512
CONFIDENCE = 0.25

# Validate paths
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not TEST_IMAGES.exists():
    raise FileNotFoundError(f"Test images not found: {TEST_IMAGES}")

# Load trained model
model = YOLO(str(MODEL_PATH))

# Run inference
results = model.predict(
    source=str(TEST_IMAGES),
    imgsz=IMAGE_SIZE,
    conf=CONFIDENCE,
    iou=0.7,
    device="cpu",
    save=True,
    save_conf=True,
    project=str(OUTPUT_DIR),
    name="baseline_predictions",
    exist_ok=True,
    verbose=True
)

# Summarise detections
total_detections = 0

for result in results:
    total_detections += len(result.boxes)

print("\n========== PREDICTION SUMMARY ==========")
print("Images processed:", len(results))
print("Total predicted potholes:", total_detections)
print("Confidence threshold:", CONFIDENCE)
print("Saved predictions:", OUTPUT_DIR / "baseline_predictions")
print("========================================")
