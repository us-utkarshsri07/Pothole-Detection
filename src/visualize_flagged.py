
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset" / "processed"
OUTPUT = ROOT / "results"

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

MAX_AREA_RATIO = 0.25
MAX_DIMENSION_RATIO = 0.80

COLUMNS = 3
CELL_WIDTH = 480
CELL_HEIGHT = 340
PADDING = 15


def find_image(image_dir, stem):
    for ext in IMAGE_EXTENSIONS:
        path = image_dir / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def main():

    flagged_images = []

    for split in ["train", "val", "test"]:

        image_dir = DATASET / "images" / split
        label_dir = DATASET / "labels" / split

        for label_path in label_dir.glob("*.txt"):

            image_path = find_image(image_dir, label_path.stem)

            if image_path is None:
                continue

            with Image.open(image_path) as img:
                width, height = img.size

            flagged_boxes = []

            with open(label_path, "r", encoding="utf-8") as file:

                for line_no, line in enumerate(file, start=1):

                    values = line.split()

                    if len(values) != 5:
                        continue

                    _, xc, yc, bw, bh = map(float, values)

                    area = bw * bh

                    if (
                        area > MAX_AREA_RATIO
                        or max(bw, bh) > MAX_DIMENSION_RATIO
                    ):
                        flagged_boxes.append(
                            (xc, yc, bw, bh, area, line_no)
                        )

            if flagged_boxes:
                flagged_images.append({
                    "path": image_path,
                    "split": split,
                    "width": width,
                    "height": height,
                    "boxes": flagged_boxes
                })

    if not flagged_images:
        print("No oversized boxes found.")
        return

    rows = math.ceil(len(flagged_images) / COLUMNS)

    sheet = Image.new(
        "RGB",
        (
            COLUMNS * (CELL_WIDTH + PADDING) + PADDING,
            rows * (CELL_HEIGHT + 45 + PADDING) + PADDING
        ),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype("arial.ttf", 17)
    except OSError:
        font = ImageFont.load_default()

    for index, item in enumerate(flagged_images):

        with Image.open(item["path"]) as original:
            image = original.convert("RGB")

        original_w, original_h = image.size

        scale = min(
            CELL_WIDTH / original_w,
            CELL_HEIGHT / original_h
        )

        new_w = int(original_w * scale)
        new_h = int(original_h * scale)

        image = image.resize((new_w, new_h))

        image_draw = ImageDraw.Draw(image)

        for xc, yc, bw, bh, area, line_no in item["boxes"]:

            x1 = (xc - bw / 2) * new_w
            y1 = (yc - bh / 2) * new_h
            x2 = (xc + bw / 2) * new_w
            y2 = (yc + bh / 2) * new_h

            image_draw.rectangle(
                [x1, y1, x2, y2],
                outline="red",
                width=3
            )

        row = index // COLUMNS
        col = index % COLUMNS

        x = PADDING + col * (CELL_WIDTH + PADDING)
        y = PADDING + row * (CELL_HEIGHT + 45 + PADDING)

        sheet.paste(image, (x, y))

        details = ", ".join(
            f"{box[4] * 100:.1f}%"
            for box in item["boxes"]
        )

        caption = (
            f"{item['path'].name} | {item['split']} | "
            f"Area: {details}"
        )

        draw.text(
            (x, y + CELL_HEIGHT + 5),
            caption,
            fill="black",
            font=font
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT / "flagged_annotations_review.jpg"
    sheet.save(output_path, quality=95)

    print("Flagged images:", len(flagged_images))
    print("Review saved to:", output_path)


if __name__ == "__main__":
    main()
