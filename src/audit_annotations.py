
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset" / "processed"

# Report boxes exceeding either threshold.
MAX_AREA_RATIO = 0.25
MAX_DIMENSION_RATIO = 0.80

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

flagged = []
total_boxes = 0

for split in ["train", "val", "test"]:

    image_dir = DATASET / "images" / split
    label_dir = DATASET / "labels" / split

    for label_path in label_dir.glob("*.txt"):

        image_path = next(
            (
                image_dir / (label_path.stem + ext)
                for ext in IMAGE_EXTENSIONS
                if (image_dir / (label_path.stem + ext)).exists()
            ),
            None
        )

        if image_path is None:
            continue

        with Image.open(image_path) as image:
            image_width, image_height = image.size

        with open(label_path, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):

                values = line.split()
                if len(values) != 5:
                    continue

                _, xc, yc, width, height = map(float, values)

                total_boxes += 1

                area_ratio = width * height
                large_dimension = max(width, height)

                if (
                    area_ratio > MAX_AREA_RATIO
                    or large_dimension > MAX_DIMENSION_RATIO
                ):
                    flagged.append({
                        "split": split,
                        "image": image_path.name,
                        "line": line_number,
                        "area_percent": area_ratio * 100,
                        "width_percent": width * 100,
                        "height_percent": height * 100,
                        "image_size": (
                            image_width,
                            image_height
                        )
                    })

print("\n========== ANNOTATION AUDIT ==========")
print("Total boxes checked:", total_boxes)
print("Potentially oversized boxes:", len(flagged))

for item in flagged:
    print("\nImage:", item["image"])
    print("Split:", item["split"])
    print("Label line:", item["line"])
    print(f"Box area: {item['area_percent']:.2f}%")
    print(f"Box width: {item['width_percent']:.2f}%")
    print(f"Box height: {item['height_percent']:.2f}%")
    print("Image dimensions:", item["image_size"])

print("\n======================================")
