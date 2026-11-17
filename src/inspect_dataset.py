
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset" / "raw" / "RDD2022_India"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

images = [
    p for p in DATASET.rglob("*")
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
]

xml_files = list(DATASET.rglob("*.xml"))

print("== DATASET INSPECTION ==")
print("Total images:", len(images))
print("Total XML annotations:", len(xml_files))

print("\nImage extensions:")
print(Counter(p.suffix.lower() for p in images))

print("\nDirectories:")

for path in DATASET.iterdir():
    print(path.name, "[DIR]" if path.is_dir() else "[FILE]")

print("====")
