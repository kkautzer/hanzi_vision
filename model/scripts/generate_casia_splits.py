import csv
import os
import struct
import numpy as np
from PIL import Image
from pathlib import Path
import random


# === CONFIGURATION ===
# Path to the folder containing raw .gnt files extracted from CASIA dataset
GNT_FOLDER = "model/data/raw"

# === FUNCTION: Read a .gnt file and yield (character, image) pairs ===
def read_gnt_file(filepath):
    with open(filepath, 'rb') as f:
        while True:  
            header = f.read(10)  # Each sample starts with a 10-byte header
            if not header:
                break

            # Parse metadata from header
            sample_size = struct.unpack('<I', header[0:4])[0]
            tagcode = header[4:6]  # Encoded character label
            width = struct.unpack('<H', header[6:8])[0]
            height = struct.unpack('<H', header[8:10])[0]

            # Read image pixel data (grayscale bitmap)
            bitmap = f.read(width * height)
            if not bitmap:
                break

            try:
                # Convert the character code to a Unicode character using GB2312 encoding
                char = tagcode.decode('gb2312')
                if not char or not char.isprintable() or '\x00' in char or len(char) != 1:
                    continue
            except UnicodeDecodeError:
                continue  # Skip if decoding fails

            try:
                # Reshape pixel buffer into a NumPy array and convert to PIL image
                img = Image.fromarray(np.frombuffer(bitmap, dtype=np.uint8).reshape((height, width)))
            except Exception as e:
                print(f"Error processing image for character in '{filepath}' ('{char}'): {e}")
                continue  # Skip if image processing fails - likely corrupted data and should be ignored
            yield char, img


# === FUNCTION: Process all .gnt files and save images into class folders ===
def process_all_gnt(gnt_folder):
    """
    Iterates over CASIA GNT files, safely skipping failed parses,
    and ensuring deterministic splits via splits.csv.

    Yields: (char, img, split)
    """


    print(f"Processing GNT files from '{gnt_folder}'")


    splits_csv_path="model/data/casia_splits.csv"

    splits_csv_path = Path(splits_csv_path)

    # Load existing splits if present
    splits = {}
    if splits_csv_path.exists():
        with open(splits_csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                splits[row["sample_id"]] = row["split"]
    else:
        print("No existing splits CSV found; starting fresh.")

    new_rows = []
    total = 0
    skipped = 0

    print(f"Reading GNT files from '{gnt_folder}'...")

    for filename in os.listdir(gnt_folder):
        filepath = Path(gnt_folder) / filename

        print(f"Processing file: {filename}")

        # CASIA-defined split (authoritative)
        if filename.endswith("-t.gnt"):
            default_split = "train"
        elif filename.endswith("-f.gnt"):
            default_split = "test"
        else:
            raise ValueError(f"Unrecognized CASIA split in filename: {filename}")

        for idx, (char, img) in enumerate(read_gnt_file(filepath)):
            sample_id = f"{filename}:{char}:{idx}"
            total += 1

            # Skip failed parses (img was never created)
            if img is None:
                skipped += 1
                continue

            # Determine split
            if sample_id not in splits:
                splits[sample_id] = default_split
                new_rows.append((sample_id, default_split))

            yield char, img, splits[sample_id]

    print(f"Persisting splits to '{splits_csv_path}'...")

    # Persist new split assignments (append-only)
    if new_rows:
        write_header = not splits_csv_path.exists()
        with open(splits_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["sample_id", "split"])
            writer.writerows(new_rows)

    print(f"Finished parsing CASIA:")
    print(f"  Total samples seen : {total}")
    print(f"  Samples skipped    : {skipped}")
    print(f"  Valid samples      : {total - skipped}")



# === MAIN RUN ===
if __name__ == "__main__":
    for _ in process_all_gnt(GNT_FOLDER):
        pass