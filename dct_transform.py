import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

# ── Input folders (from preprocess.py output) ───────────────────
PROCESSED_REAL = "processed/real"
PROCESSED_FAKE = "processed/fake"

# ── Output base folder ───────────────────────────────────────────
OUTPUT_BASE = "dct_data"

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# ── Create output directories ────────────────────────────────────
for split in ["train", "validation", "test"]:
    for cls in ["real", "fake"]:
        os.makedirs(os.path.join(OUTPUT_BASE, split, cls), exist_ok=True)


def apply_dct_to_image(img_path):
    """Read image, apply DCT, return normalized uint8 array."""
    img = cv2.imread(img_path, 0)   # grayscale
    if img is None:
        return None
    img = np.float32(img)
    dct = cv2.dct(img)
    
    # Logarithmic scaling to preserve high-frequency artifacts
    dct_log = np.log(np.abs(dct) + 1)
    
    # Normalize log-scaled values to 0-255
    dct_normalized = cv2.normalize(dct_log, None, 0, 255, cv2.NORM_MINMAX)
    
    return np.uint8(dct_normalized)


def process_class(input_folder, class_name):
    """Split files into train/val/test and save DCT images."""
    files = [f for f in os.listdir(input_folder)
             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    if len(files) == 0:
        print(f"  WARNING: No images found in {input_folder}")
        return

    # Split into train / val+test, then val+test → val / test
    train_files, temp_files = train_test_split(
        files, train_size=TRAIN_RATIO, random_state=42
    )
    val_ratio_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_files, test_files = train_test_split(
        temp_files, train_size=val_ratio_adjusted, random_state=42
    )

    splits = {
        "train":      train_files,
        "validation": val_files,
        "test":       test_files
    }

    for split_name, split_files in splits.items():
        out_dir = os.path.join(OUTPUT_BASE, split_name, class_name)
        count = 0
        for file in split_files:
            src_path = os.path.join(input_folder, file)
            dct_img  = apply_dct_to_image(src_path)
            if dct_img is None:
                continue
            dst_path = os.path.join(out_dir, file)
            cv2.imwrite(dst_path, dct_img)
            count += 1
        print(f"  [{split_name}/{class_name}] Saved {count} DCT images -> {out_dir}")


# ── Run ──────────────────────────────────────────────────────────
print("\n[DCT Transform] Processing real images...")
process_class(PROCESSED_REAL, "real")

print("\n[DCT Transform] Processing fake images...")
process_class(PROCESSED_FAKE, "fake")

print("\nDone! dct_data/ is ready for train_model.py")