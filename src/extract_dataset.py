
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent.parent

ZIP_PATH = ROOT / "dataset" / "raw" / "RDD2022_India.zip"
OUTPUT_DIR = ROOT / "dataset" / "raw" / "RDD2022_India"

def extract():

    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"ZIP not found: {ZIP_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as archive:

        bad_file = archive.testzip()

        if bad_file:
            raise RuntimeError(f"Corrupt archive member: {bad_file}")

        destination = OUTPUT_DIR.resolve()

        for member in archive.infolist():
            target = (OUTPUT_DIR / member.filename).resolve()

            if target != destination and destination not in target.parents:
                raise ValueError(f"Unsafe archive path: {member.filename}")

        archive.extractall(OUTPUT_DIR)

    print("Extraction successful.")
    print("Location:", OUTPUT_DIR)


if __name__ == "__main__":
    extract()
