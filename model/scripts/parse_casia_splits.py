import csv
import os
from pathlib import Path
from collections import defaultdict
from PIL import Image

from model.scripts.generate_casia_splits import read_gnt_file  # adjust import


# === CONFIG ===
GNT_FOLDER = Path("model/data/raw")
OUTPUT_FOLDER = Path("model/data/processed")
SPLITS_CSV = Path("model/data/casia_splits.csv")
IMAGE_SIZE = (64, 64)


# === LOAD SPLITS CSV ===
def load_splits(csv_path):
    splits = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            splits[row["sample_id"]] = row["split"]
    return splits


# === APPLY SPLITS ===
def apply_splits():
    splits = load_splits(SPLITS_CSV)

    # Track per-(split,char) counters for filenames
    counters = defaultdict(int)

    total = 0
    saved = 0
    skipped = 0

    print(f"Loaded {len(splits)} split assignments")
    print(f"Processing GNT files from {GNT_FOLDER}")

    # Group sample_ids by file for efficient parsing
    samples_by_file = defaultdict(dict)
    for sample_id, split in splits.items():
        filename, char, idx = sample_id.rsplit(":", 2)
        samples_by_file[filename][int(idx)] = (char, split)

    for filename, wanted_samples in samples_by_file.items():
        gnt_path = GNT_FOLDER / filename
        if not gnt_path.exists():
            print(f"WARNING: Missing GNT file {filename}, skipping")
            continue

        print(f"Parsing {filename}")

        for idx, (char, img) in enumerate(read_gnt_file(gnt_path)):
            total += 1

            if idx not in wanted_samples:
                continue

            expected_char, split = wanted_samples[idx]
            if char != expected_char:
                print(
                    f"WARNING: Character mismatch in {filename} "
                    f"(expected {expected_char}, got {char})"
                )
                skipped += 1
                continue

            # Prepare output path
            out_dir = OUTPUT_FOLDER / split / char
            out_dir.mkdir(parents=True, exist_ok=True)

            counters[(split, char)] += 1
            img_resized = img.resize(IMAGE_SIZE)

            out_path = out_dir / f"{counters[(split, char)]:04d}.png"
            img_resized.save(out_path)

            saved += 1

    print("\n=== DONE ===")
    print(f"Total samples examined : {total}")
    print(f"Images saved           : {saved}")
    print(f"Samples skipped        : {skipped}")


# === MAIN ===
if __name__ == "__main__":
    apply_splits()
