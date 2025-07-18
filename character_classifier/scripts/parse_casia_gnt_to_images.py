import os
import struct
import numpy as np
from PIL import Image
from pathlib import Path
import random


# === CONFIGURATION ===
# Path to the folder containing raw .gnt files extracted from CASIA dataset
GNT_FOLDER = "character_classifier/data/raw"

# Output base directory where processed images will be saved
OUTPUT_FOLDER = "character_classifier/data/processed"

# Ratios for splitting data into train, val, and test sets
SPLIT_RATIOS = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

# Create base output folders for each split if they don’t already exist
for split in SPLIT_RATIOS.keys():
    os.makedirs(os.path.join(OUTPUT_FOLDER, split), exist_ok=True)


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

            # Reshape pixel buffer into a NumPy array and convert to PIL image
            img = Image.fromarray(np.frombuffer(bitmap, dtype=np.uint8).reshape((height, width)))
            yield char, img


# === FUNCTION: Process all .gnt files and save images into class folders ===
def process_all_gnt(gnt_folder, output_base):
    image_counter = {}  # Track how many images we've saved per (split, character)
    all_data = []       # Collect all image/label pairs to later split into sets

    # Loop over all .gnt files in the provided folder
    for file in Path(gnt_folder).glob("*.gnt"):
        for char, img in read_gnt_file(file):
            all_data.append((char, img.copy()))  # Copy to decouple from file handle

    print(f"Total samples parsed: {len(all_data)}")

    # Shuffle data to randomize class distribution
    random.shuffle(all_data)

    # Compute split indices
    n = len(all_data)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])

    # Slice into train, val, test
    split_data = {
        "train": all_data[:n_train],
        "val": all_data[n_train:n_train + n_val],
        "test": all_data[n_train + n_val:]
    }

    # Process each split
    for split, data in split_data.items():
        print(f"Saving {split} set: {len(data)} images")
        for char, img in data:
            # Build output folder path for this character
            char_dir = os.path.join(output_base, split, char)
            os.makedirs(char_dir, exist_ok=True)

            # Count and increment how many images we've saved for this character in this split
            count = image_counter.get((split, char), 0) + 1
            image_counter[(split, char)] = count

            # Resize to consistent dimensions (e.g., 64x64)
            img_resized = img.resize((64, 64))

            # Save image with a zero-padded filename
            img_path = os.path.join(char_dir, f"{count:04d}.png")
            img_resized.save(img_path)


# === MAIN RUN ===
if __name__ == "__main__":
    process_all_gnt(GNT_FOLDER, OUTPUT_FOLDER)
