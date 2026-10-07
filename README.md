# Generative Artifact Anomaly Detector (ForensicAI) 🛡️

A multi-modal deep learning & forensic analysis platform for detecting AI-generated images and visual artifacts.

## Overview

ForensicAI combines multiple detection strategies to identify synthetic and manipulated images:
- **Semantic / Vision Features**: Deep feature representation using neural architectures (ResNet-based).
- **Frequency Domain Analysis**: Discrete Cosine Transform (DCT) residual analysis to catch high-frequency generative artifacts.
- **EXIF & Metadata Analysis**: Classifier trained on structural image metadata and compression traces.
- **Streamlit Web Dashboard**: Interactive forensics dashboard with heatmap visualizations, confidence scores, and forensic report breakdown.

## Project Structure

```text
├── app.py                      # Streamlit forensics dashboard
├── generalized_model.py        # Model loading & inference logic
├── dct_transform.py            # DCT frequency analysis pipeline
├── preprocess.py               # Image preprocessing routines
├── extract.py                  # Feature extraction utilities
├── download_resnet_weights.py  # Script to download model weights
├── train_model.py              # Vision model training script
├── train_generalized.py        # Generalized architecture training
├── train_exif.py               # EXIF classifier training
├── requirements.txt            # Python dependencies
└── models/                     # Trained weights & metadata (exif configs, plots)
```

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux / macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download / prepare model weights:**
   ```bash
   python download_resnet_weights.py
   ```

5. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```
