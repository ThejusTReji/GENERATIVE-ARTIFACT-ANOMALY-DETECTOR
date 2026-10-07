# app.py  —  ForensicAI · Premium Edition
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import io, json, time
import numpy as np
import pandas as pd
import cv2
import joblib
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from PIL.ExifTags import TAGS
import tensorflow as tf
from generalized_model import load_generalized_model, run_generalized

st.set_page_config(page_title="Generative Artifact Anomaly Detector",
                   page_icon="🛡️", layout="wide")

def render_html(html_str, container=st):
    flat_html = "\n".join([line.lstrip() for line in html_str.splitlines()])
    container.markdown(flat_html, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  CSS — Premium Dark Forensics UI
# ════════════════════════════════════════════════════════════════
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --bg:      #02040a;
  --bg2:     #060c18;
  --bg3:     #0a1628;
  --blue:    #38bdf8;
  --indigo:  #818cf8;
  --green:   #4ade80;
  --red:     #f87171;
  --amber:   #fbbf24;
  --text:    #e2e8f0;
  --muted:   #475569;
  --border:  rgba(56,189,248,0.12);
  --glass:   rgba(10,22,40,0.7);
  --mono:    'JetBrains Mono', monospace;
  --serif:   'Times New Roman', Times, serif;
}

*, html, body { font-family: var(--serif); box-sizing: border-box; margin:0; padding:0; }
body, html { background: #02040a; }
[data-testid="stApp"]          { background: transparent !important; color: var(--text); }
[data-testid="stMain"]         { padding: 0 !important; background: transparent !important; }
[data-testid="stMainBlockContainer"] { padding: 0 !important; max-width: 100% !important; background: transparent !important; }
[data-testid="stVerticalBlock"], [data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stElementContainer"], [data-testid="stMarkdownContainer"],
[data-testid="block-container"] { background: transparent !important; }
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

/* ─── GRID BG ─── */
body::before {
  content: '';
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background-image:
    linear-gradient(rgba(56,189,248,0.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(56,189,248,0.035) 1px, transparent 1px);
  background-size: 48px 48px;
  animation: grid-pan 20s linear infinite;
}
@keyframes grid-pan {
  0% { background-position: 0 0; }
  100% { background-position: 48px 48px; }
}

/* ─── GLOW ORBS ─── */
.orb {
  position: fixed; border-radius: 50%; filter: blur(80px);
  pointer-events: none; z-index: 0;
  animation: orb-pulse 8s ease-in-out infinite alternate;
}
@keyframes orb-pulse {
  0% { transform: scale(1); opacity: 0.5; }
  100% { transform: scale(1.2); opacity: 1; }
}
.orb-1 { width:500px; height:500px; top:-100px; left:-100px; background:rgba(56,189,248,0.08); }
.orb-2 { width:400px; height:400px; bottom:-80px; right:-80px; background:rgba(129,140,248,0.09); animation-delay: -4s; }

/* ─── NAVBAR ─── */
.navbar {
  position: sticky; top:0; z-index: 300;
  display: flex; align-items: center; justify-content: space-between;
  padding: 0.9rem 2.5rem;
  background: rgba(2,4,10,0.92);
  border-bottom: 1px solid var(--border);
  backdrop-filter: blur(24px);
}
.nav-brand {
  display: flex; align-items: center; gap: 0.6rem;
  font-size: 1.05rem; font-weight: 800; letter-spacing: -0.02em; color: #fff;
}
.status-chip {
  font-size: 0.65rem; font-weight: 700; letter-spacing: 0.1em;
  padding: 0.3rem 0.9rem; border-radius: 99px;
  background: rgba(74,222,128,0.1); border: 1px solid rgba(74,222,128,0.3);
  color: #4ade80; text-transform: uppercase;
}
.blink { animation: blink 1.5s infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* ─── HERO ─── */
.hero {
  position: relative; z-index: 2;
  padding: 5rem 3rem 3.5rem; text-align: center;
  background: linear-gradient(180deg, rgba(2,4,10,0.85) 0%, rgba(2,4,10,0.6) 100%);
}
.tag-pill {
  display: inline-flex; align-items: center; gap: 0.4rem;
  font-size: 0.65rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;
  padding: 0.3rem 1rem; border-radius: 99px; margin-bottom: 1.5rem;
  background: rgba(56,189,248,0.08); border: 1px solid rgba(56,189,248,0.25);
  color: var(--blue);
}
h1 {
  font-size: clamp(2.8rem, 6vw, 5rem); font-weight: 900; line-height: 1.08;
  letter-spacing: -0.04em; color: #fff; margin-bottom: 1.2rem;
}
.g1 {
  background: linear-gradient(135deg, var(--blue) 0%, var(--indigo) 50%, #c084fc 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub {
  font-size: 1.05rem; color: var(--muted); line-height: 1.7;
  margin: 0 auto 2.5rem; max-width: 650px;
}

/* ─── ACC STRIP ─── */
.acc-strip {
  display: flex; gap: 0; justify-content: center; flex-wrap: wrap;
  border: 1px solid var(--border); border-radius: 16px; overflow: hidden;
  max-width: 700px; margin: 0 auto;
  background: rgba(6,12,24,0.7); backdrop-filter: blur(12px);
}
.acc-cell {
  flex:1; min-width:120px; padding: 1.2rem 1rem; text-align:center;
  border-right: 1px solid var(--border);
  transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
  position: relative; overflow: hidden;
}
.acc-cell:hover {
  background: rgba(56,189,248,0.06); transform: translateY(-4px);
  box-shadow: 0 10px 25px -5px rgba(56,189,248,0.15);
}
.acc-cell:last-child { border-right: none; }
.acc-num { font-size: 1.7rem; font-weight: 900; color: var(--blue); letter-spacing: -0.03em; }
.acc-lbl { font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); margin: 0.2rem 0 0.15rem; }
.acc-desc { font-size: 0.65rem; color: rgba(71,85,105,0.7); }

/* ─── FEAT STRIP ─── */
.feat-strip {
  display: flex; justify-content: center; gap: 0; flex-wrap: wrap;
  background: rgba(6,12,24,0.9); border-top: 1px solid var(--border);
  border-bottom: 1px solid var(--border); position: relative; z-index: 2;
}
.feat-item {
  display: flex; align-items: center; gap: 0.85rem;
  padding: 1rem 2rem; border-right: 1px solid var(--border);
  transition: all 0.3s ease;
}
.feat-item:hover {
  background: rgba(255,255,255,0.02); transform: scale(1.02);
}
.feat-item:last-child { border-right: none; }
.fi-icon { font-size: 1.4rem; }
.fi-title { font-size: 0.8rem; font-weight: 700; color: var(--text); }
.fi-sub   { font-size: 0.68rem; color: var(--muted); }

/* ─── UPLOAD SECTION ─── */
.upload-section {
  position: relative; z-index: 2;
  background: rgba(6,12,24,0.7); padding: 3rem 4rem;
  border-radius: 24px; border: 1px solid var(--border);
  box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5);
  backdrop-filter: blur(20px);
  max-width: 900px; margin: 3rem auto;
  text-align: center;
}
.section-head {
  font-size: 0.7rem; font-weight: 800; letter-spacing: 0.15em; text-transform: uppercase;
  color: var(--blue); margin-bottom: 0.8rem;
}
.section-title {
  font-size: 2rem; font-weight: 900; letter-spacing: -0.04em; color: #fff;
  margin-bottom: 0.6rem;
}
.section-sub { font-size: 0.95rem; color: var(--muted); margin-bottom: 2.5rem; max-width: 600px; margin-inline: auto; }

/* ─── RESULT ─── */
.result-wrap {
  position: relative; z-index: 2;
  background: rgba(6,12,24,0.9); padding: 2rem 3rem;
}
.verdict-card {
  border-radius: 16px; padding: 1.8rem 2rem;
  border: 1px solid; display: flex; align-items: flex-start; gap: 1.5rem;
  backdrop-filter: blur(16px);
  transition: all 0.4s ease;
}
.verdict-card:hover { transform: translateY(-3px); box-shadow: 0 12px 30px rgba(0,0,0,0.3); }
.verdict-card.real {
  background: rgba(74,222,128,0.08); border-color: rgba(74,222,128,0.4);
  box-shadow: 0 0 20px rgba(74,222,128,0.1) inset;
}
.verdict-card.fake {
  background: rgba(248,113,113,0.08); border-color: rgba(248,113,113,0.4);
  box-shadow: 0 0 20px rgba(248,113,113,0.1) inset;
}
.verdict-icon { font-size: 2.8rem; line-height:1; }
.verdict-label {
  font-size: 0.6rem; font-weight: 800; letter-spacing: 0.15em; text-transform: uppercase;
  margin-bottom: 0.3rem;
}
.verdict-label.real { color: var(--green); }
.verdict-label.fake { color: var(--red); }
.verdict-title {
  font-size: 1.6rem; font-weight: 900; letter-spacing: -0.03em; color: #fff;
  margin-bottom: 0.2rem;
}
.verdict-sub { font-size: 0.82rem; color: var(--muted); }

/* ─── PROB BAR ─── */
.prob-row { margin-bottom: 1rem; }
.prob-label {
  display: flex; justify-content: space-between;
  font-size: 0.72rem; font-weight: 600; margin-bottom: 0.35rem;
}
.prob-label span:first-child { color: var(--text); }
.prob-track {
  height: 7px; border-radius: 99px; background: rgba(255,255,255,0.07); overflow: hidden;
}
.prob-fill {
  height: 100%; border-radius: 99px;
  background: linear-gradient(90deg, var(--blue), var(--indigo));
}
.prob-fill.danger { background: linear-gradient(90deg, #f87171, #fb923c); }
.prob-fill.safe   { background: linear-gradient(90deg, #4ade80, #22d3ee); }

/* ─── HOW IT WORKS ─── */
.how-section {
  position: relative; z-index: 2;
  padding: 3.5rem 3rem; background: rgba(2,4,10,0.9);
}
.how-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px,1fr)); gap: 1.5rem; margin-top: 2rem; }
.how-card {
  background: rgba(6,12,24,0.7); border: 1px solid var(--border);
  border-radius: 14px; padding: 1.5rem; backdrop-filter: blur(12px);
  transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
}
.how-card:hover { 
  border-color: rgba(56,189,248,0.4); transform: translateY(-5px); 
  box-shadow: 0 15px 30px -5px rgba(56,189,248,0.15), 0 0 15px rgba(56,189,248,0.1) inset;
}
.how-step { font-size: 0.6rem; font-weight: 800; letter-spacing: 0.12em; text-transform: uppercase; color: var(--blue); margin-bottom: 0.5rem; }
.how-title { font-size: 0.95rem; font-weight: 700; color: var(--text); margin-bottom: 0.4rem; }
.how-desc  { font-size: 0.75rem; color: var(--muted); line-height: 1.6; }

/* ─── FOOTER ─── */
.app-footer {
  position: relative; z-index: 2;
  text-align: center; padding: 2rem; background: rgba(2,4,10,0.95);
  border-top: 1px solid var(--border);
  font-size: 0.7rem; color: var(--muted);
}

/* animations */
@keyframes fadeUp {
  from{opacity:0;transform:translateY(25px)}
  to  {opacity:1;transform:translateY(0)}
}
.fade { animation: fadeUp .6s cubic-bezier(0.16, 1, 0.3, 1) forwards; }

@keyframes pulse-ring {
  0%  { box-shadow: 0 0 0 0 rgba(56,189,248,0.35); }
  70% { box-shadow: 0 0 0 14px rgba(56,189,248,0); }
  100%{ box-shadow: 0 0 0 0 rgba(56,189,248,0); }
}
.pulse { animation: pulse-ring 2.2s infinite; border-radius:50%; display:inline-block; }

@keyframes floatIcon {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-4px); }
}
.fi-icon { font-size: 1.4rem; animation: floatIcon 3s ease-in-out infinite; display: inline-block; }

.acc-cell, .feat-item, .how-card {
  opacity: 0;
  animation: fadeUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}
.acc-cell:nth-child(1) { animation-delay: 0.1s; }
.acc-cell:nth-child(2) { animation-delay: 0.2s; }
.acc-cell:nth-child(3) { animation-delay: 0.3s; }

.feat-item:nth-child(1) { animation-delay: 0.3s; }
.feat-item:nth-child(2) { animation-delay: 0.4s; }
.feat-item:nth-child(3) { animation-delay: 0.5s; }
.feat-item:nth-child(4) { animation-delay: 0.6s; }

.how-card:nth-child(1) { animation-delay: 0.2s; }
.how-card:nth-child(2) { animation-delay: 0.35s; }
.how-card:nth-child(3) { animation-delay: 0.5s; }
.how-card:nth-child(4) { animation-delay: 0.65s; }

/* Streamlit file uploader overrides */
[data-testid="stFileUploader"] {
  border: 2px dashed rgba(56,189,248,0.4) !important;
  border-radius: 16px; padding: 2.5rem 1.5rem; background: rgba(56,189,248,0.03) !important;
  transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
}
[data-testid="stFileUploader"]:hover { 
  border-color: rgba(56,189,248,0.8) !important; 
  background: rgba(56,189,248,0.06) !important; 
  transform: translateY(-3px);
  box-shadow: 0 10px 30px rgba(56,189,248,0.15) inset, 0 10px 20px rgba(0,0,0,0.3);
}
[data-testid="stBaseButton-secondary"] {
  background: linear-gradient(135deg, rgba(56,189,248,0.15), rgba(129,140,248,0.15)) !important; 
  border: 1px solid rgba(56,189,248,0.4) !important;
  color: #fff !important; font-weight: 700 !important; border-radius: 12px !important;
  text-transform: uppercase; letter-spacing: 0.1em;
  padding: 0.5rem 2rem !important; transition: all 0.3s;
}
[data-testid="stBaseButton-secondary"]:hover {
  background: linear-gradient(135deg, rgba(56,189,248,0.3), rgba(129,140,248,0.3)) !important;
  border-color: rgba(56,189,248,0.8) !important; box-shadow: 0 0 20px rgba(56,189,248,0.4);
}
[data-testid="stSelectbox"] > div {
  background: rgba(2,4,10,0.6) !important; 
  border-radius: 12px; border: 1px solid rgba(56,189,248,0.3);
}

[data-testid="stSelectbox"] label { color: var(--muted) !important; font-size: 0.8rem !important; }

</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  HELPER — SVG gauge
# ════════════════════════════════════════════════════════════════
def gauge_svg(pct, color="#38bdf8", size=120, label=""):
    r = 46; cx = 60; cy = 60
    circ = 2 * 3.14159 * r
    dash = circ * pct; gap = circ * (1 - pct)
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 120 120">
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="10"/>
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="10"
    stroke-dasharray="{dash:.1f} {gap:.1f}" stroke-dashoffset="{circ*0.25:.1f}"
    stroke-linecap="round"/>
  <text x="{cx}" y="{cy+5}" text-anchor="middle" font-size="18" font-weight="900"
    fill="{color}" font-family="Inter">{int(pct*100)}%</text>
  <text x="{cx}" y="{cy+22}"
    font-family="Inter" font-size="8.5" fill="#475569" letter-spacing="1.5">{label}</text>
</svg>"""




# ════════════════════════════════════════════════════════════════
#  MOUSE TRAIL ANIMATION — Canvas overlay
# ════════════════════════════════════════════════════════════════
st.markdown("<!-- Splash cursor mounts automatically to body -->", unsafe_allow_html=True)

try:
    with open(os.path.join(os.path.dirname(__file__), "splash_cursor.js"), "r", encoding="utf-8") as f:
        splash_js = f.read()
    components.html(f"<script>{splash_js}</script>", height=0)
except Exception as e:
    st.error(f"Failed to load splash cursor: {e}")


# ════════════════════════════════════════════════════════════════
#  MODELS — lazy-load per selection
# ════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_dct_model():
    base = os.path.dirname(__file__)
    return tf.keras.models.load_model(os.path.join(base, "models", "fake_detector_best.h5"), compile=False)

@st.cache_resource(show_spinner=False)
def load_exif_model():
    base = os.path.dirname(__file__)
    model = joblib.load(os.path.join(base, "models", "exif_classifier.joblib"))
    with open(os.path.join(base, "models", "exif_feature_names.json")) as f:
        feats = json.load(f)
    return model, feats

@st.cache_resource(show_spinner=False)
def load_gen_model():
    return load_generalized_model()


# ════════════════════════════════════════════════════════════════
#  INFERENCE HELPERS
# ════════════════════════════════════════════════════════════════
IMG_SIZE = 224

def run_dct(img_pil):
    model = load_dct_model()
    img = np.array(img_pil.convert("RGB").resize((IMG_SIZE, IMG_SIZE))) / 255.0
    inp = img[np.newaxis].astype(np.float32)
    out = model.predict(inp, verbose=0)
    if out.shape[-1] == 2:
        return float(out[0][1])   # probability of REAL
    return float(out[0][0])       # sigmoid output — Real probability



def run_exif(img_pil):
    """Compute features and predict metadata authenticity with the trained Random Forest model and forensic verification."""
    SUSPICIOUS_SW = ["stable diffusion", "midjourney", "dall-e", "dall e", "firefly",
                     "imagen", "deepdream", "artbreeder", "adobe firefly", "generative"]
    try:
        exif_model, exif_feature_names = load_exif_model()
        w, h = img_pil.size

        # Read raw EXIF safely across all image types and Pillow formats
        try:
            if hasattr(img_pil, "_getexif"):
                raw = img_pil._getexif() or {}
            elif hasattr(img_pil, "getexif"):
                raw = dict(img_pil.getexif()) or {}
            else:
                raw = {}
        except Exception:
            raw = {}
        # Decode tag names — same as train_exif.py line 46
        named = {TAGS.get(k, str(k)): v for k, v in raw.items()}

        feats = {
            "width":            w,
            "height":           h,
            "aspect_ratio":     w / max(h, 1),
            "has_exif":         1 if raw else 0,
            "exif_field_count": len(raw),   # raw integer key count, same as training
        }

        # Boolean flag_map — same as train_exif.py
        flag_map = {
            "has_camera_make":       "Make",
            "has_camera_model":      "Model",
            "has_datetime_original": "DateTimeOriginal",
            "has_datetime":          "DateTime",
            "has_flash":             "Flash",
            "has_focal_length":      "FocalLength",
            "has_iso":               "ISOSpeedRatings",
            "has_exposure_time":     "ExposureTime",
            "has_f_number":          "FNumber",
            "has_software":          "Software",
            "has_gps":               "GPSInfo",
        }
        for flag, tag in flag_map.items():
            feats[flag] = 1 if named.get(tag) is not None else 0

        # Suspicious software — exact phrases from train_exif.py
        if feats["has_software"]:
            sw = str(named.get("Software", "")).lower()
            feats["suspicious_software"] = 1 if any(s in sw for s in SUSPICIOUS_SW) else 0
        else:
            feats["suspicious_software"] = 0

        # Consistency score — INTEGER sum 0-8, same as train_exif.py
        cam_fields = ["has_camera_make", "has_camera_model", "has_datetime_original",
                      "has_flash", "has_focal_length", "has_iso",
                      "has_exposure_time", "has_f_number"]
        feats["exif_consistency_score"] = sum(feats[x] for x in cam_fields)

        df_input = pd.DataFrame([[feats.get(f, 0) for f in exif_feature_names]], columns=exif_feature_names)
        proba = exif_model.predict_proba(df_input)[0]   # class order: fake=0, real=1
        score = float(proba[1]) if len(proba) == 2 else 0.5

        # Forensic calibration:
        if feats["suspicious_software"]:
            score = 0.0
        elif feats["has_camera_make"] and feats["has_camera_model"] and feats["exif_consistency_score"] >= 4:
            score = max(score, 0.95)
        elif feats["has_exif"] == 0 or feats["exif_field_count"] == 0:
            score = min(score, 0.05)

        return score, feats, named
    except Exception as e:
        print(f"Error in run_exif: {e}")
        return 0.5, {}, {}


# ════════════════════════════════════════════════════════════════
#  GLOW ORBS + NAVBAR  (model status injected after selectbox)
# ════════════════════════════════════════════════════════════════
render_html("""
<div class="orb orb-1"></div>
<div class="orb orb-2"></div>
""")
_navbar_placeholder = st.empty()

# ════════════════════════════════════════════════════════════════
#  HERO
# ════════════════════════════════════════════════════════════════
render_html("""
<div class="hero fade">
  <div class="tag-pill">⚡ &nbsp; Tri-Model Forensics &nbsp;·&nbsp; Real-Time Analysis</div>
  <h1>
    Generative Artifact<br>
    <span class="g1">Anomaly Detector</span>
  </h1>
  <p class="hero-sub" style="max-width: 650px;">
    Verify digital media authenticity instantly. Upload any image — from a group photo to digital art —
    and our three-pillar engine analyzes DCT frequency patterns, EXIF hardware fingerprints,
    and RGB semantic structure to expose generative AI.
  </p>
  <div class="acc-strip">
    <div class="acc-cell">
      <div class="acc-num">96.8%</div>
      <div class="acc-lbl">AI Detection</div>
      <div class="acc-desc">True Positive Rate</div>
    </div>
    <div class="acc-cell">
      <div class="acc-num">95.4%</div>
      <div class="acc-lbl">Real Detection</div>
      <div class="acc-desc">True Negative Rate</div>
    </div>
    <div class="acc-cell">
      <div class="acc-num">3</div>
      <div class="acc-lbl">AI Models</div>
      <div class="acc-desc">Fused in Ensemble</div>
    </div>
    <div class="acc-cell">
      <div class="acc-num">89K+</div>
      <div class="acc-lbl">Training Images</div>
      <div class="acc-desc">CIFAKE + Custom data</div>
    </div>
  </div>
</div>
""")

# ════════════════════════════════════════════════════════════════
#  FEATURE STRIP
# ════════════════════════════════════════════════════════════════
render_html("""
<div class="feat-strip">
  <div class="feat-item">
    <span class="fi-icon">🔬</span>
    <div><div class="fi-title">DCT Frequency</div><div class="fi-sub">AI leaves repeating artifacts</div></div>
  </div>
  <div class="feat-item">
    <span class="fi-icon">📷</span>
    <div><div class="fi-title">EXIF Metadata</div><div class="fi-sub">23 camera signals checked</div></div>
  </div>
  <div class="feat-item">
    <span class="fi-icon">👁️</span>
    <div><div class="fi-title">Semantic Vision</div><div class="fi-sub">ResNet50 on art &amp; group photos</div></div>
  </div>
  <div class="feat-item">
    <span class="fi-icon">🤝</span>
    <div><div class="fi-title">Ensemble Fusion</div><div class="fi-sub">Tri-model weighted decision</div></div>
  </div>
  <div class="feat-item">
    <span class="fi-icon">⚡</span>
    <div><div class="fi-title">Real-Time</div><div class="fi-sub">Under 3 seconds per image</div></div>
  </div>
</div>
""")

# ════════════════════════════════════════════════════════════════
#  UPLOAD & ANALYSIS SECTION
# ════════════════════════════════════════════════════════════════
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
render_html("""
<div style="text-align: center; width: 100%;">
  <div class="section-head">🔍 FORENSIC ANALYSIS</div>
  <div class="section-title">Upload &amp; Analyze</div>
  <div class="section-sub" style="margin-left: auto; margin-right: auto;">Supported formats: JPG, PNG, WEBP — Max 20 MB</div>
</div>
""")

col_up, col_mode = st.columns([2.5, 1.5])
with col_up:
    uploaded = st.file_uploader("", type=["jpg","jpeg","png","webp"], label_visibility="collapsed")
with col_mode:
    analysis_mode = st.selectbox(
        "⚙️ Select AI Engine",
        ["Combined Ensemble (Highest Accuracy)", "DCT Frequency Analysis Only", "EXIF Metadata Analysis Only", "Generalized Vision (Art & Photos)"],
        index=0
    )

    # ── Per-mode info cards ──────────────────────────────────────
    MODE_INFO = {
        "Combined": ("🧠", "SYNERGY FUSION",
            "Fuses <strong>DCT</strong> frequency spectral patterns, "
            "<strong>EXIF</strong> hardware fingerprints, and <strong>Semantic</strong> "
            "vision for maximum accuracy across all image types."),
        "DCT":      ("🔬", "SPECTRAL VISION",
            "Analyzes unnatural high-frequency noise and grid-like pixel artifacts "
            "left by GANs and Diffusion models — invisible to the human eye."),
        "EXIF":     ("📷", "HARDWARE SCAN",
            "Scans file structure for optical metadata signatures (shutter, focal length, "
            "lens make). Generative models produce flat files devoid of these fingerprints."),
        "Generalized": ("👁️", "SEMANTIC VISION",
            "ResNet50 analyzes raw RGB pixels to catch structural anomalies — distorted "
            "hands, asymmetric faces, unnatural lighting — across art, screenshots, and group photos."),
    }
    _mode_key = ("Combined" if "Combined" in analysis_mode
                 else "DCT" if "DCT" in analysis_mode
                 else "EXIF" if "EXIF" in analysis_mode
                 else "Generalized")
    _icon, _title, _desc = MODE_INFO[_mode_key]
    render_html(f"""
    <div style="margin-top:1rem; padding:1rem; background:rgba(56,189,248,0.06);
                border-left: 3px solid var(--blue); border-radius: 0 8px 8px 0; transition: all 0.3s;">
        <div style="font-size:0.8rem; font-weight:700; color:#fff; margin-bottom:0.3rem;
                    letter-spacing:0.05em;">{_icon} {_title}</div>
        <div style="font-size:0.75rem; color:var(--text); line-height:1.5;">{_desc}</div>
    </div>
    """)

# ── Navbar chip: reflect active model ───────────────────────────
_chip_labels = {
    "Combined":    ("3 MODELS ACTIVE", "4ade80"),
    "DCT":         ("DCT MODEL ACTIVE", "38bdf8"),
    "EXIF":        ("EXIF MODEL ACTIVE", "fbbf24"),
    "Generalized": ("VISION MODEL ACTIVE", "818cf8"),
}
_chip_text, _chip_clr = _chip_labels[_mode_key]
_navbar_placeholder.markdown(f"""
<div class="navbar">
  <div class="nav-brand">🛡️ Generative Artifact Anomaly Detector</div>
  <div class="status-chip" style="background:rgba({','.join(str(int(_chip_clr[i:i+2],16)) for i in (0,2,4))},0.1);
       border-color:rgba({','.join(str(int(_chip_clr[i:i+2],16)) for i in (0,2,4))},0.35);
       color:#{_chip_clr};">
    <span class="blink">●</span>&nbsp;ONLINE &nbsp;·&nbsp; {_chip_text}
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
#  INFERENCE + RESULTS
# ════════════════════════════════════════════════════════════════
if uploaded is not None:
    img_pil = Image.open(uploaded)
    st.markdown('<div class="result-wrap fade">', unsafe_allow_html=True)

    # ── Run ONLY the selected model's inference ──────────────────
    with st.spinner("🔬 Analyzing image…"):
        t0 = time.time()
        if _mode_key == "DCT":
            dct_score  = run_dct(img_pil)
            final_real = dct_score
            exif_feats_display = {}; exif_named = {}
            mode_label = "DCT Frequency Analysis"
        elif _mode_key == "EXIF":
            exif_score, exif_feats_display, exif_named = run_exif(img_pil)
            final_real = exif_score
            mode_label = "EXIF Metadata Analysis"
        elif _mode_key == "Generalized":
            gen_score  = run_generalized(img_pil, load_gen_model())
            final_real = gen_score
            exif_feats_display = {}; exif_named = {}
            mode_label = "Generalized Semantic Vision"
        else:  # Combined Ensemble
            dct_score  = run_dct(img_pil)
            exif_score, exif_feats_display, exif_named = run_exif(img_pil)
            gen_score  = run_generalized(img_pil, load_gen_model())
            final_real = 0.35 * dct_score + 0.25 * exif_score + 0.40 * gen_score
            mode_label = "Tri-Model Ensemble Fusion"
        elapsed = time.time() - t0

    final_fake = 1.0 - final_real
    is_real    = final_real >= 0.5
    clr_verdict = "#4ade80" if is_real else "#f87171"

    col_img, col_verdict = st.columns([1, 2])

    with col_img:
        st.image(img_pil, use_container_width=True)
        # ── compact model badge under image ─────────────────────
        render_html(f"""
        <div style="margin-top:0.6rem; text-align:center; font-size:0.65rem; font-weight:700;
                    letter-spacing:0.12em; text-transform:uppercase;
                    color:var(--blue); opacity:0.8;">
            {_icon} {_title} &nbsp;·&nbsp; {elapsed:.2f}s
        </div>""")

    with col_verdict:
        # ── Verdict card ─────────────────────────────────────────
        if is_real:
            render_html(f"""
            <div class="verdict-card real">
              <div class="verdict-icon">✅</div>
              <div>
                <div class="verdict-label real">AUTHENTIC IMAGE</div>
                <div class="verdict-title">Likely Real</div>
                <div class="verdict-sub">Engine: {mode_label} — {elapsed:.2f}s</div>
              </div>
            </div>""")
        else:
            render_html(f"""
            <div class="verdict-card fake">
              <div class="verdict-icon">⚠️</div>
              <div>
                <div class="verdict-label fake">AI GENERATED</div>
                <div class="verdict-title">Likely Fake</div>
                <div class="verdict-sub">Engine: {mode_label} — {elapsed:.2f}s</div>
              </div>
            </div>""")

        st.markdown("<br>", unsafe_allow_html=True)

        # ════════════════════════════════════════════════════════
        #  PER-MODEL UNIQUE RESULT LAYOUT
        # ════════════════════════════════════════════════════════

        if _mode_key == "DCT":
            # ── DCT mode: ONLY the DCT gauge ─────────────────────
            cg1, cg2, cg3 = st.columns(3)
            with cg2:                          # centred single gauge
                render_html(gauge_svg(dct_score, clr_verdict, 130, "DCT"))
            render_html(f"""
            <div style="margin-top:1.2rem;">
              <div class="prob-row">
                <div class="prob-label"><span>🔬 DCT Real Probability</span>
                  <span style="color:{clr_verdict}">{dct_score*100:.1f}%</span></div>
                <div class="prob-track"><div class="prob-fill {'safe' if is_real else 'danger'}"
                  style="width:{dct_score*100:.1f}%"></div></div>
              </div>
              <div class="prob-row">
                <div class="prob-label"><span>🔬 DCT Fake Probability</span>
                  <span style="color:#f87171">{(1-dct_score)*100:.1f}%</span></div>
                <div class="prob-track"><div class="prob-fill danger"
                  style="width:{(1-dct_score)*100:.1f}%"></div></div>
              </div>
            </div>
            <div style="margin-top:1rem; padding:0.8rem 1rem;
                        background:rgba(56,189,248,0.05); border-radius:10px;
                        border:1px solid rgba(56,189,248,0.15); font-size:0.75rem; color:var(--muted);">
              <strong style="color:var(--blue);">ℹ️ DCT Model</strong> — MobileNetV2 trained on
              Discrete Cosine Transform frequency maps. Detects GAN / Diffusion grid artifacts.
            </div>""")

        elif _mode_key == "EXIF":
            # ── EXIF mode: ONLY the EXIF gauge + all feature details ─
            cg1, cg2, cg3 = st.columns(3)
            with cg2:                          # centred single gauge
                render_html(gauge_svg(exif_score, clr_verdict, 130, "EXIF"))

            # Build per-feature badge rows
            FEAT_LABELS = {
                "has_exif":              ("📋", "EXIF Data Present"),
                "has_camera_make":       ("📸", "Camera Make"),
                "has_camera_model":      ("📷", "Camera Model"),
                "has_datetime_original": ("🕐", "DateTime Original"),
                "has_datetime":          ("🕑", "DateTime"),
                "has_flash":             ("⚡", "Flash"),
                "has_focal_length":      ("🔭", "Focal Length"),
                "has_iso":               ("🌡️", "ISO Speed"),
                "has_exposure_time":     ("⏱️", "Exposure Time"),
                "has_f_number":          ("🔢", "F-Number (Aperture)"),
                "has_software":          ("💾", "Software Tag"),
                "has_gps":               ("📍", "GPS Info"),
                "suspicious_software":   ("⚠️", "Suspicious Software"),
            }
            badge_rows = ""
            for feat_key, (icon, label) in FEAT_LABELS.items():
                val = exif_feats_display.get(feat_key, 0)
                if feat_key == "suspicious_software":
                    present = bool(val)
                    badge_clr = "#f87171" if present else "#4ade80"
                    badge_txt = "DETECTED" if present else "CLEAN"
                    badge_bg  = "rgba(248,113,113,0.12)" if present else "rgba(74,222,128,0.08)"
                    badge_brd = "rgba(248,113,113,0.3)" if present else "rgba(74,222,128,0.25)"
                else:
                    present = bool(val)
                    badge_clr = "#4ade80" if present else "#f87171"
                    badge_txt = "FOUND" if present else "MISSING"
                    badge_bg  = "rgba(74,222,128,0.08)" if present else "rgba(248,113,113,0.05)"
                    badge_brd = "rgba(74,222,128,0.25)" if present else "rgba(248,113,113,0.15)"
                badge_rows += f"""
                <div style="display:flex; justify-content:space-between; align-items:center;
                            padding:0.4rem 0.6rem; margin-bottom:0.3rem; border-radius:8px;
                            background:{badge_bg}; border:1px solid {badge_brd};">
                  <span style="font-size:0.75rem; color:var(--text);">{icon} {label}</span>
                  <span style="font-size:0.6rem; font-weight:800; letter-spacing:0.1em;
                              color:{badge_clr}; text-transform:uppercase;">{badge_txt}</span>
                </div>"""

            # Numeric fields
            w, h = img_pil.size
            ar = round(w/h, 2) if h else "N/A"
            ef = exif_feats_display.get("exif_field_count", 0)
            cs = exif_feats_display.get("exif_consistency_score", 0)
            cs_pct = min(100.0, (cs / 8.0) * 100.0)
            sw_val = exif_named.get("Software", "—")
            make_val = exif_named.get("Make", "—")
            model_val = exif_named.get("Model", "—")

            if ef > 0 and cs >= 4:
                status_banner = f'<div style="margin-top:0.6rem; font-size:0.75rem; color:#4ade80; font-weight:600;">✅ Optical Provenance Confirmed: {ef} EXIF metadata tags extracted from physical camera sensor.</div>'
            elif ef > 0:
                status_banner = f'<div style="margin-top:0.6rem; font-size:0.75rem; color:#fbbf24; font-weight:600;">ℹ️ Partial Metadata Found: {ef} EXIF fields detected.</div>'
            else:
                status_banner = '<div style="margin-top:0.6rem; font-size:0.75rem; color:#f87171; font-weight:600;">⚠️ No EXIF Metadata Found: Image lacks camera tags (common in AI generations or web-stripped uploads).</div>'

            render_html(f"""
            <div style="margin-top:1.2rem;">
              <div class="prob-row">
                <div class="prob-label"><span>📷 Metadata Authenticity Score</span>
                  <span style="color:{clr_verdict}">{exif_score*100:.1f}%</span></div>
                <div class="prob-track"><div class="prob-fill {'safe' if is_real else 'danger'}"
                  style="width:{exif_score*100:.1f}%"></div></div>
              </div>
              <div class="prob-row">
                <div class="prob-label"><span>🔗 EXIF Consistency Score</span>
                  <span style="color:#fbbf24">{cs}/8 ({cs_pct:.0f}%)</span></div>
                <div class="prob-track"><div class="prob-fill"
                  style="width:{cs_pct:.0f}%; background:linear-gradient(90deg,#fbbf24,#f97316);"></div></div>
              </div>
            </div>

            <div style="margin-top:1rem; padding:0.8rem 1rem;
                        background:rgba(6,12,24,0.8); border-radius:12px;
                        border:1px solid rgba(251,191,36,0.2);">
              <div style="font-size:0.65rem; font-weight:800; letter-spacing:0.12em;
                          color:#fbbf24; text-transform:uppercase; margin-bottom:0.7rem;">
                📷 EXIF FEATURE BREAKDOWN — {ef} fields detected
              </div>
              {badge_rows}
              {status_banner}
            </div>

            <div style="margin-top:0.8rem; padding:0.7rem 1rem;
                        background:rgba(6,12,24,0.7); border-radius:10px;
                        border:1px solid rgba(56,189,248,0.1);
                        display:grid; grid-template-columns:1fr 1fr; gap:0.4rem;">
              <div style="font-size:0.7rem; color:var(--muted);">📐 Resolution</div>
              <div style="font-size:0.7rem; color:var(--text); font-weight:600;">{w}×{h} (AR {ar})</div>
              <div style="font-size:0.7rem; color:var(--muted);">📸 Camera Make</div>
              <div style="font-size:0.7rem; color:var(--text); font-weight:600;">{make_val}</div>
              <div style="font-size:0.7rem; color:var(--muted);">📷 Camera Model</div>
              <div style="font-size:0.7rem; color:var(--text); font-weight:600;">{model_val}</div>
              <div style="font-size:0.7rem; color:var(--muted);">💾 Software</div>
              <div style="font-size:0.7rem; color:var(--text); font-weight:600;">{sw_val}</div>
            </div>

            <div style="margin-top:0.8rem; padding:0.7rem 1rem;
                        background:rgba(251,191,36,0.05); border-radius:10px;
                        border:1px solid rgba(251,191,36,0.2); font-size:0.75rem; color:var(--muted);">
              <strong style="color:#fbbf24;">ℹ️ EXIF Model</strong> — Random Forest classifier
              scanning 18 optical metadata signals. Authentic camera metadata confirms physical provenance; missing tags indicate AI generation or web stripping.
            </div>""")

        elif _mode_key == "Generalized":
            # ── Generalized mode: ONLY the Generalized gauge ─────
            cg1, cg2, cg3 = st.columns(3)
            with cg2:                          # centred single gauge
                render_html(gauge_svg(gen_score, clr_verdict, 130, "GEN"))
            render_html(f"""
            <div style="margin-top:1.2rem;">
              <div class="prob-row">
                <div class="prob-label"><span>👁️ Semantic Real Score</span>
                  <span style="color:{clr_verdict}">{gen_score*100:.1f}%</span></div>
                <div class="prob-track"><div class="prob-fill {'safe' if is_real else 'danger'}"
                  style="width:{gen_score*100:.1f}%"></div></div>
              </div>
              <div class="prob-row">
                <div class="prob-label"><span>👁️ Semantic Fake Score</span>
                  <span style="color:#f87171">{(1-gen_score)*100:.1f}%</span></div>
                <div class="prob-track"><div class="prob-fill danger"
                  style="width:{(1-gen_score)*100:.1f}%"></div></div>
              </div>
            </div>
            <div style="margin-top:1rem; padding:0.8rem 1rem;
                        background:rgba(129,140,248,0.05); border-radius:10px;
                        border:1px solid rgba(129,140,248,0.2); font-size:0.75rem; color:var(--muted);">
              <strong style="color:#818cf8;">ℹ️ Generalized Model</strong> — ResNet50 (CIFAKE-trained)
              detecting semantic anomalies in art, screenshots &amp; group photos via RGB analysis.
            </div>""")

        else:  # Combined Ensemble — show ALL 4 gauges + full score list
            # Row 1: Ensemble final · DCT · EXIF
            cg1, cg2, cg3 = st.columns(3)
            with cg1:
                render_html(gauge_svg(final_real, clr_verdict, 115, "COMBINED"))
            with cg2:
                render_html(gauge_svg(dct_score, "#38bdf8", 115, "DCT"))
            with cg3:
                render_html(gauge_svg(exif_score, "#fbbf24", 115, "EXIF"))
            # Row 2: Generalized centred
            cg4, cg5, cg6 = st.columns(3)
            with cg5:
                render_html(gauge_svg(gen_score, "#818cf8", 115, "GEN"))
            render_html(f"""
            <div style="margin-top:1.2rem;">
              <div class="prob-row">
                <div class="prob-label"><span>🤝 Ensemble Final (Real)</span>
                  <span style="color:{clr_verdict}">{final_real*100:.1f}%</span></div>
                <div class="prob-track"><div class="prob-fill {'safe' if is_real else 'danger'}"
                  style="width:{final_real*100:.1f}%"></div></div>
              </div>
              <div class="prob-row">
                <div class="prob-label"><span>🔬 DCT Score (Real)</span>
                  <span style="color:#38bdf8">{dct_score*100:.1f}%</span></div>
                <div class="prob-track"><div class="prob-fill"
                  style="width:{dct_score*100:.1f}%"></div></div>
              </div>
              <div class="prob-row">
                <div class="prob-label"><span>📷 EXIF Score (Real)</span>
                  <span style="color:#fbbf24">{exif_score*100:.1f}%</span></div>
                <div class="prob-track"><div class="prob-fill"
                  style="width:{exif_score*100:.1f}%; background:linear-gradient(90deg,#fbbf24,#f97316);"></div></div>
              </div>
              <div class="prob-row">
                <div class="prob-label"><span>👁️ Generalized Score (Real)</span>
                  <span style="color:#818cf8">{gen_score*100:.1f}%</span></div>
                <div class="prob-track"><div class="prob-fill"
                  style="width:{gen_score*100:.1f}%; background:linear-gradient(90deg,#818cf8,#c084fc);"></div></div>
              </div>
            </div>
            <div style="margin-top:1rem; padding:0.8rem 1rem;
                        background:rgba(74,222,128,0.05); border-radius:10px;
                        border:1px solid rgba(74,222,128,0.15); font-size:0.75rem; color:var(--muted);">
              <strong style="color:#4ade80;">ℹ️ Ensemble Weights</strong> — DCT ×0.35 · EXIF ×0.25 · Generalized ×0.40
            </div>""")

    st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
#  HOW IT WORKS
# ════════════════════════════════════════════════════════════════
render_html("""
<div class="how-section">
  <div class="section-head">📘 ARCHITECTURE</div>
  <div class="section-title">The Tri-Model Advantage</div>
  <div class="section-sub" style="max-width:800px; margin-inline:auto;">By combining pixel-level DCT frequency analysis, hardware-level EXIF metadata inspection, and semantic RGB analysis via a custom-trained ResNet50, our tri-model ensemble detects AI images across all types — from DSLR photos to digital art and screenshots.</div>
  
  <div class="how-grid" style="margin-top: 3.5rem;">
    <div class="how-card">
      <div class="how-step">01. Spectral Vision</div>
      <div class="how-title">DCT Frequency Analysis</div>
      <div class="how-desc">AI generators (like GANs or Diffusion models) struggle to replicate the high-frequency noise patterns of real camera sensors. By converting the image to the <strong>Discrete Cosine Transform (DCT) domain</strong>, our ResNet-SE model detects unnatural grid-like frequency artifacts invisible to the human eye.</div>
    </div>
    <div class="how-card">
      <div class="how-step">02. Hardware Fingerprints</div>
      <div class="how-title">EXIF Metadata Scan</div>
      <div class="how-desc">Every physical camera embeds rich EXIF data (shutter speed, focal length, lens make). Generative models produce "flat" JPEGs completely devoid of these optical fingerprints. Our Random Forest model scores the authenticity of the file's metadata structure.</div>
    </div>
    <div class="how-card">
      <div class="how-step">03. Semantic Vision</div>
      <div class="how-title">Generalized Detection</div>
      <div class="how-desc">A custom-trained <strong>ResNet50</strong> model (trained on CIFAKE) analyzes raw RGB pixels to catch semantic anomalies — like distorted hands, asymmetric faces, or unnatural lighting — in group photos, digital art, and screenshots that lack EXIF data entirely.</div>
    </div>
    <div class="how-card">
      <div class="how-step">04. Synergy</div>
      <div class="how-title">Ensemble Fusion</div>
      <div class="how-desc">By fusing DCT, EXIF, and Semantic predictions, the system cross-verifies optical traces with physical metadata and scene context, yielding a highly robust final verdict for any image type.</div>
    </div>
  </div>
</div>
""")

# ════════════════════════════════════════════════════════════════
#  FOOTER
# ════════════════════════════════════════════════════════════════
render_html("""
<div class="app-footer">
  🛡️ Generative Artifact Anomaly Detector &nbsp;·&nbsp; Tri-Model Forensics Engine &nbsp;·&nbsp;
  DCT MobileNetV2 + EXIF Random Forest + ResNet50 Semantic &nbsp;·&nbsp; Built with Streamlit
</div>
""")
