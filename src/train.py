
from pathlib import Path
import torch
from ultralytics import YOLO



# CONFIGURATION


ROOT = Path(__file__).resolve().parent.parent

DATA_YAML = ROOT / "dataset" / "processed" / "data.yaml"

RESULTS_DIR = ROOT / "results" / "training"

MODEL_NAME = "yolov8n.pt"

# Start with 2 epochs to verify the complete pipeline.
# Change to 30 after the smoke test succeeds.
EPOCHS = 2

IMAGE_SIZE = 512
BATCH_SIZE = 2
WORKERS = 0

DEVICE = "cpu"

PROJECT_NAME = "pothole_detection"
RUN_NAME = "smoke_test"



# CHECK CONFIGURATION


def check_configuration():

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {DATA_YAML}"
        )

    print("\n========== TRAINING CONFIGURATION ==========")
    print("Dataset:", DATA_YAML)
    print("Model:", MODEL_NAME)
    print("Epochs:", EPOCHS)
    print("Image size:", IMAGE_SIZE)
    print("Batch size:", BATCH_SIZE)
    print("Device:", DEVICE)
    print("CUDA available:", torch.cuda.is_available())
    print("============================================\n")



# TRAIN MODEL


def main():

    check_configuration()

    # Load pretrained YOLOv8n weights
    model = YOLO(MODEL_NAME)

    print("Starting YOLOv8n training...\n")

    results = model.train(
        data=str(DATA_YAML),

        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,

        device=DEVICE,
        workers=WORKERS,

        project=str(RESULTS_DIR),
        name=RUN_NAME,
        exist_ok=True,

        pretrained=True,

        optimizer="auto",
        patience=10,

        cache=False,
        amp=False,

        seed=42,
        deterministic=True,

        plots=True,
        save=True,
        val=True,
        verbose=True
    )

    print("\n========== TRAINING FINISHED ==========")
    print("Results directory:", RESULTS_DIR / RUN_NAME)
    print("Best weights:", RESULTS_DIR / RUN_NAME / "weights" / "best.pt")
    print("Last weights:", RESULTS_DIR / RUN_NAME / "weights" / "last.pt")
    print("=======================================\n")


if __name__ == "__main__":
    main()
