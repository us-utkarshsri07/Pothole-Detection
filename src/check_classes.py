
from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset" / "raw" / "RDD2022_India"

counts = Counter()
annotated_images = 0

for xml_file in DATASET.rglob("*.xml"):

    try:
        root = ET.parse(xml_file).getroot()

        objects = root.findall("object")

        if objects:
            annotated_images += 1

        for obj in objects:
            label = obj.findtext("name", default="UNKNOWN")
            counts[label] += 1

    except ET.ParseError:
        print("Invalid XML:", xml_file)

print("\n== CLASS DISTRIBUTION ==")

for label, count in counts.most_common():
    print(f"{label}: {count}")

print("\nImages containing annotations:", annotated_images)
print("Total objects:", sum(counts.values()))
print("====")
