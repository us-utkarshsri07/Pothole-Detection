
from pathlib import Path
import csv
import yaml
import torch

from ultralytics import YOLO


# ==================================================
# PROJECT PATHS
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    ROOT / "results" / "training" /
    "baseline_30_epochs" / "weights" / "best.pt"
)

DATASET_DIR = ROOT / "dataset" / "processed"
ORIGINAL_YAML = DATASET_DIR / "data.yaml"

OUTPUT_DIR = ROOT / "results" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EVAL_YAML = OUTPUT_DIR / "test_data.yaml"


# ==================================================
# CONFIGURATION
# ==================================================

IMAGE_SIZE = 512
BATCH_SIZE = 8

DEVICE = 0 if torch.cuda.is_available() else "cpu"


# ==================================================
# PRE-EVALUATION VALIDATION
# ==================================================

print("\n========== TEST EVALUATION ==========")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not ORIGINAL_YAML.exists():
    raise FileNotFoundError(f"Dataset YAML not found: {ORIGINAL_YAML}")

with open(ORIGINAL_YAML, "r") as file:
    config = yaml.safe_load(file)

# Force the dataset path to the local project directory.
config["path"] = str(DATASET_DIR.resolve())

# Ensure the test directory exists.
TEST_IMAGES = DATASET_DIR / "images" / "test"
TEST_LABELS = DATASET_DIR / "labels" / "test"

if not TEST_IMAGES.exists() or not TEST_LABELS.exists():
    raise FileNotFoundError("Test images or labels directory missing.")

# Save a separate evaluation configuration.
with open(EVAL_YAML, "w") as file:
    yaml.safe_dump(config, file, sort_keys=False)

print("Model:", MODEL_PATH)
print("Dataset:", DATASET_DIR)
print("Test images:", len(list(TEST_IMAGES.iterdir())))
print("Device:", DEVICE)


# ==================================================
# LOAD TRAINED MODEL
# ==================================================

model = YOLO(str(MODEL_PATH))


# ==================================================
# EVALUATE ON UNTOUCHED TEST SET
# ==================================================

results = model.val(
    data=str(EVAL_YAML),
    split="test",
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE,
    workers=0,
    plots=True,
    save_json=True,
    project=str(OUTPUT_DIR),
    name="yolov8n_test",
    exist_ok=True
)


# ==================================================
# EXTRACT METRICS
# ==================================================

precision = float(results.box.mp)
recall = float(results.box.mr)
map50 = float(results.box.map50)
map50_95 = float(results.box.map)

f1_score = (
    2 * precision * recall / (precision + recall)
    if precision + recall > 0
    else 0.0
)


# ==================================================
# DISPLAY RESULTS
# ==================================================

print("\n========== FINAL TEST METRICS ==========")

metrics = {
    "Precision": precision,
    "Recall": recall,
    "F1-score": f1_score,
    "mAP50": map50,
    "mAP50-95": map50_95
}

for name, value in metrics.items():
    print(f"{name}: {value:.4f} ({value * 100:.2f}%)")


# ==================================================
# SAVE METRICS TO CSV
# ==================================================

METRICS_FILE = OUTPUT_DIR / "test_metrics.csv"

with open(METRICS_FILE, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Metric", "Value"])

    for name, value in metrics.items():
        writer.writerow([name, round(value, 6)])

print("\nMetrics saved:", METRICS_FILE)
print("Evaluation plots:", OUTPUT_DIR / "yolov8n_test")
print("\n========== EVALUATION COMPLETE ==========")
