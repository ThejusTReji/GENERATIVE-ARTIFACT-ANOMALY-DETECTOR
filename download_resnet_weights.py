"""
download_resnet_weights.py
Run this ONCE before train_generalized.py.
Downloads ResNet50 ImageNet weights robustly with retries.
"""
import os
import ssl
import requests

# ── Target cache path ────────────────────────────────────────────
CACHE_DIR  = os.path.join(os.path.expanduser("~"), ".keras", "models")
FILE_NAME  = "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
DEST       = os.path.join(CACHE_DIR, FILE_NAME)
URL        = "https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
EXPECTED_BYTES = 94_765_736  # ~90 MB

os.makedirs(CACHE_DIR, exist_ok=True)

# ── Skip if already fully downloaded ────────────────────────────
if os.path.exists(DEST):
    actual = os.path.getsize(DEST)
    if actual >= EXPECTED_BYTES:
        print(f"[✓] Weights already cached: {DEST}")
        print(f"    Size: {actual / 1e6:.1f} MB — Skipping download.")
        exit(0)
    else:
        print(f"[!] Found incomplete file ({actual / 1e6:.1f} MB). Re-downloading...")
        os.remove(DEST)

# ── Download with retry ──────────────────────────────────────────
MAX_RETRIES = 5
CHUNK_SIZE  = 1024 * 1024   # 1 MB chunks

# Disable SSL verification to avoid DECRYPTION_FAILED_OR_BAD_RECORD_MAC
session = requests.Session()
session.verify = False
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

for attempt in range(1, MAX_RETRIES + 1):
    try:
        print(f"\n[Attempt {attempt}/{MAX_RETRIES}] Downloading ResNet50 weights (~90 MB)...")
        response = session.get(URL, stream=True, timeout=120)
        response.raise_for_status()

        total = int(response.headers.get("content-length", EXPECTED_BYTES))
        downloaded = 0

        with open(DEST, "wb") as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    pct = downloaded / total * 100
                    print(f"\r    Progress: {downloaded / 1e6:.1f} MB / {total / 1e6:.1f} MB  ({pct:.1f}%)", end="", flush=True)

        print(f"\n[✓] Download complete! Saved to: {DEST}")
        break

    except Exception as e:
        print(f"\n[!] Attempt {attempt} failed: {e}")
        if os.path.exists(DEST):
            os.remove(DEST)   # Remove partial file before retry
        if attempt == MAX_RETRIES:
            print("\n[✗] All retries exhausted. Check your internet connection and try again.")
            exit(1)
        print("    Retrying...")

print("\n[✓] ResNet50 weights ready. You can now run: train_generalized.py")
