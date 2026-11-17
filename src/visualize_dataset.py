
from pathlib import Path
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
import random
import math



# CONFIGURATION


ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT / "dataset" / "processed"
RESULTS_DIR = ROOT / "results"

SPLITS = ["train", "val", "test"]

RANDOM_SEED = 42
SAMPLE_COUNT = 12

THUMBNAIL_SIZE = (480, 320)
GRID_COLUMNS = 3
PADDING = 15



# LOAD FONT


try:
    FONT = ImageFont.truetype("arial.ttf", 18)
except OSError:
    FONT = ImageFont.load_default()



# VALIDATE YOLO LABELS


def read_labels(label_path):

    annotations = []

    if not label_path.exists():
        raise FileNotFoundError(
            f"Missing label file: {label_path}"
        )

    with open(label_path, "r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                continue

            values = line.split()

            if len(values) != 5:
                raise ValueError(
                    f"Invalid label format: {label_path}, "
                    f"line {line_number}"
                )

            class_id = int(values[0])
            xc, yc, width, height = map(float, values[1:])

            if class_id != 0:
                raise ValueError(
                    f"Unexpected class ID in {label_path}"
                )

            if not all(
                0.0 <= value <= 1.0
                for value in [xc, yc, width, height]
            ):
                raise ValueError(
                    f"Coordinates outside [0, 1]: "
                    f"{label_path}, line {line_number}"
                )

            if width <= 0 or height <= 0:
                raise ValueError(
                    f"Invalid bounding box size: {label_path}"
                )

            if (
                xc - width / 2 < -1e-6
                or xc + width / 2 > 1.000001
                or yc - height / 2 < -1e-6
                or yc + height / 2 > 1.000001
            ):
                raise ValueError(
                    f"Bounding box exceeds image boundaries: "
                    f"{label_path}"
                )

            annotations.append((class_id, xc, yc, width, height))

    return annotations


# ============================================================
# VALIDATE EACH DATASET SPLIT
# ============================================================

def validate_split(split):

    image_dir = DATASET_DIR / "images" / split
    label_dir = DATASET_DIR / "labels" / split

    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(
            f"Missing image or label directory for {split}"
        )

    images = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower()
        in {".jpg", ".jpeg", ".png"}
    ]

    labels = list(label_dir.glob("*.txt"))

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    missing_labels = image_stems - label_stems
    extra_labels = label_stems - image_stems

    if missing_labels or extra_labels:
        raise ValueError(
            f"Image-label mismatch in {split}.\n"
            f"Missing labels: {len(missing_labels)}\n"
            f"Extra labels: {len(extra_labels)}"
        )

    total_boxes = 0

    for label_path in labels:
        annotations = read_labels(label_path)
        total_boxes += len(annotations)

    return len(images), total_boxes



# DRAW BOUNDING BOXES


def create_annotated_image(image_path):

    label_path = (
        DATASET_DIR
        / "labels"
        / "train"
        / f"{image_path.stem}.txt"
    )

    image = Image.open(image_path).convert("RGB")

    original_width, original_height = image.size

    annotations = read_labels(label_path)

    draw = ImageDraw.Draw(image)

    for _, xc, yc, width, height in annotations:

        x1 = (xc - width / 2) * original_width
        y1 = (yc - height / 2) * original_height

        x2 = (xc + width / 2) * original_width
        y2 = (yc + height / 2) * original_height

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=max(2, original_width // 400)
        )

        draw.text(
            (x1, max(0, y1 - 22)),
            "POTHOLE",
            fill="yellow",
            font=FONT
        )

    image.thumbnail(THUMBNAIL_SIZE)

    return image



# CREATE CONTACT SHEET


def create_contact_sheet():

    image_dir = DATASET_DIR / "images" / "train"

    images = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower()
        in {".jpg", ".jpeg", ".png"}
    ]

    if not images:
        raise ValueError("No training images found.")

    random.seed(RANDOM_SEED)

    selected = random.sample(
        images,
        min(SAMPLE_COUNT, len(images))
    )

    annotated_images = [
        create_annotated_image(path)
        for path in selected
    ]

    rows = math.ceil(len(annotated_images) / GRID_COLUMNS)

    cell_width, cell_height = THUMBNAIL_SIZE

    sheet_width = (
        GRID_COLUMNS * cell_width
        + (GRID_COLUMNS + 1) * PADDING
    )

    sheet_height = (
        rows * (cell_height + 35)
        + (rows + 1) * PADDING
    )

    sheet = Image.new(
        "RGB",
        (sheet_width, sheet_height),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    for index, (image, source_path) in enumerate(
        zip(annotated_images, selected)
    ):

        row = index // GRID_COLUMNS
        column = index % GRID_COLUMNS

        x = PADDING + column * (cell_width + PADDING)
        y = PADDING + row * (cell_height + 35)

        sheet.paste(image, (x, y))

        draw.text(
            (x, y + cell_height + 5),
            source_path.name[:55],
            fill="black",
            font=FONT
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RESULTS_DIR / "annotation_preview.jpg"

    sheet.save(output_path, quality=95)

    print("\nVisualization saved:", output_path)



# MAIN


def main():

    print("\n PHASE 4: DATASET VALIDATION ==========\n")

    total_images = 0
    total_boxes = 0

    for split in SPLITS:

        image_count, box_count = validate_split(split)

        total_images += image_count
        total_boxes += box_count

        print(
            f"{split.upper()}: "
            f"{image_count} images, "
            f"{box_count} pothole boxes"
        )

    print("\n= TOTALS =")
    print("Total images:", total_images)
    print("Total pothole boxes:", total_boxes)

    create_contact_sheet()

    print("\nAll dataset labels passed validation.")
    print("====")


if __name__ == "__main__":
    main()
