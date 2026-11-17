
from pathlib import Path
from collections import Counter
import random
import shutil
import xml.etree.ElementTree as ET

from PIL import Image
import yaml


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

SOURCE_DIR = ROOT / "dataset" / "raw" / "RDD2022_India"
OUTPUT_DIR = ROOT / "dataset" / "processed"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# RDD2022 class to retain
TARGET_CLASS = "D40"

# YOLO class ID for pothole
TARGET_CLASS_ID = 0

# Reproducible random split
RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10


# ============================================================
# FIND IMAGES AND XML FILES
# ============================================================

def find_dataset_files():

    images = [
        p for p in SOURCE_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    xml_files = list(SOURCE_DIR.rglob("*.xml"))

    if not images:
        raise FileNotFoundError("No images found in source directory.")

    if not xml_files:
        raise FileNotFoundError("No XML annotations found.")

    image_map = {}
    xml_map = {}

    for image in images:
        key = image.stem.lower()

        if key in image_map:
            raise ValueError(
                f"Duplicate image filename stem detected: {key}"
            )

        image_map[key] = image

    for xml in xml_files:
        key = xml.stem.lower()

        if key in xml_map:
            raise ValueError(
                f"Duplicate XML filename stem detected: {key}"
            )

        xml_map[key] = xml

    missing_xml = set(image_map) - set(xml_map)
    missing_images = set(xml_map) - set(image_map)

    if missing_xml or missing_images:
        raise ValueError(
            f"Image/XML mismatch.\n"
            f"Images without XML: {len(missing_xml)}\n"
            f"XML files without images: {len(missing_images)}\n"
            f"Example missing XML: {list(missing_xml)[:5]}\n"
            f"Example missing images: {list(missing_images)[:5]}"
        )

    return image_map, xml_map


# ============================================================
# CONVERT PASCAL VOC BOUNDING BOX TO YOLO FORMAT
# ============================================================

def convert_bbox_to_yolo(bbox, image_width, image_height):

    xmin = float(bbox.findtext("xmin"))
    ymin = float(bbox.findtext("ymin"))
    xmax = float(bbox.findtext("xmax"))
    ymax = float(bbox.findtext("ymax"))

    # Clip coordinates to image boundaries
    xmin = max(0.0, min(xmin, image_width))
    xmax = max(0.0, min(xmax, image_width))
    ymin = max(0.0, min(ymin, image_height))
    ymax = max(0.0, min(ymax, image_height))

    if xmax <= xmin or ymax <= ymin:
        return None

    # YOLO normalized center coordinates and dimensions
    x_center = ((xmin + xmax) / 2.0) / image_width
    y_center = ((ymin + ymax) / 2.0) / image_height

    width = (xmax - xmin) / image_width
    height = (ymax - ymin) / image_height

    return (
        TARGET_CLASS_ID,
        x_center,
        y_center,
        width,
        height
    )


# ============================================================
# READ XML AND EXTRACT ONLY D40 OBJECTS
# ============================================================

def extract_pothole_annotations(xml_path, image_width, image_height):

    root = ET.parse(xml_path).getroot()

    annotations = []

    for obj in root.findall("object"):

        class_name = obj.findtext("name", default="").strip()

        # Ignore all classes except D40
        if class_name != TARGET_CLASS:
            continue

        bbox = obj.find("bndbox")

        if bbox is None:
            continue

        yolo_bbox = convert_bbox_to_yolo(
            bbox,
            image_width,
            image_height
        )

        if yolo_bbox is not None:
            annotations.append(yolo_bbox)

    return annotations


# ============================================================
# SPLIT DATASET
# ============================================================

def split_dataset(records):

    random.seed(RANDOM_SEED)

    random.shuffle(records)

    total = len(records)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_records = records[:train_end]
    val_records = records[train_end:val_end]
    test_records = records[val_end:]

    return {
        "train": train_records,
        "val": val_records,
        "test": test_records
    }


# ============================================================
# SAVE PROCESSED DATASET
# ============================================================

def save_dataset(splits):

    for split_name, records in splits.items():

        image_dir = OUTPUT_DIR / "images" / split_name
        label_dir = OUTPUT_DIR / "labels" / split_name

        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for record in records:

            image_path = record["image"]
            annotations = record["annotations"]

            # Copy image
            destination_image = image_dir / image_path.name
            shutil.copy2(image_path, destination_image)

            # Create corresponding YOLO label file
            label_path = label_dir / f"{image_path.stem}.txt"

            with open(label_path, "w", encoding="utf-8") as file:

                for annotation in annotations:

                    class_id, xc, yc, width, height = annotation

                    file.write(
                        f"{class_id} "
                        f"{xc:.6f} "
                        f"{yc:.6f} "
                        f"{width:.6f} "
                        f"{height:.6f}\n"
                    )


# ============================================================
# CREATE YOLO DATA CONFIGURATION
# ============================================================

def create_data_yaml():

    config = {
        "path": OUTPUT_DIR.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "pothole"
        }
    }

    yaml_path = OUTPUT_DIR / "data.yaml"

    with open(yaml_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    print("\nYOLO configuration created:", yaml_path)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {SOURCE_DIR}"
        )

    print("\n========== PHASE 3: PREPROCESSING ==========\n")

    image_map, xml_map = find_dataset_files()

    print("Images found:", len(image_map))
    print("XML annotations found:", len(xml_map))

    records = []

    total_potholes = 0
    images_with_potholes = 0
    images_without_potholes = 0

    ignored_classes = Counter()

    for key, image_path in image_map.items():

        xml_path = xml_map[key]

        with Image.open(image_path) as image:
            image_width, image_height = image.size
            image.verify()

        # Read XML annotations
        xml_root = ET.parse(xml_path).getroot()

        for obj in xml_root.findall("object"):
            class_name = obj.findtext("name", default="").strip()

            if class_name != TARGET_CLASS:
                ignored_classes[class_name] += 1

        annotations = extract_pothole_annotations(
            xml_path,
            image_width,
            image_height
        )

        if annotations:
            images_with_potholes += 1
        else:
            images_without_potholes += 1

        total_potholes += len(annotations)

        records.append({
            "image": image_path,
            "annotations": annotations
        })

    print("\n========== ANNOTATION SUMMARY ==========")
    print("Total images:", len(records))
    print("Images with potholes:", images_with_potholes)
    print("Images without potholes:", images_without_potholes)
    print("Valid pothole bounding boxes:", total_potholes)
    print("Ignored non-pothole objects:", sum(ignored_classes.values()))

    # Split records
    splits = split_dataset(records)

    print("\n========== DATASET SPLIT ==========")

    for name, split_records in splits.items():
        potholes = sum(
            len(record["annotations"])
            for record in split_records
        )

        print(
            f"{name.upper()}: "
            f"{len(split_records)} images, "
            f"{potholes} pothole objects"
        )

    # Save output
    save_dataset(splits)

    # Generate YOLO YAML
    create_data_yaml()

    print("\n=== PREPROCESSING COMPLETE ==")
    print("Processed dataset:", OUTPUT_DIR)
    print("====")


if __name__ == "__main__":
    main()
