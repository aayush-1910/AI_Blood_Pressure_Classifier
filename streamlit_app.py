import os, re, json, joblib
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename

# ── Optional OCR imports (graceful fallback if not installed) ─────────────────
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB
ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg"}

os.makedirs("uploads", exist_ok=True)

# ── Load trained models & metadata ───────────────────────────────────────────
def load_assets():
    with open("models/metadata.json") as f:
        meta = json.load(f)
    scaler   = joblib.load("models/scaler.pkl")
    features = joblib.load("models/features.pkl")
    models = {}
    for name in meta["results"]:
        key = name.replace(" ", "_")
        models[name] = joblib.load(f"models/{key}.pkl")
    return meta, scaler, features, models

try:
    META, SCALER, FEATURES, MODELS = load_assets()
    MODELS_LOADED = True
except FileNotFoundError:
    MODELS_LOADED = False
    META, SCALER, FEATURES, MODELS = None, None, [], {}

# ── Helpers ───────────────────────────────────────────────────────────────────
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

BP_COLORS = {
    "Normal":               "#27ae60",
    "Elevated":             "#f39c12",
    "Hypertension Stage 1": "#e67e22",
    "Hypertension Stage 2": "#e74c3c",
    "Hypertensive Crisis":  "#8e44ad",
}

def extract_text_from_pdf(path):
    text = ""
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_image(path):
    img = Image.open(path)
    return pytesseract.image_to_string(img)

def parse_params_from_text(text):
    patterns = {
        "age":      r"age[:\s]+(\d+)",
        "sex":      r"sex[:\s]+([01])",
        "trestbps": r"(?:systolic|trestbps|bp)[:\s]+(\d+)",
        "chol":     r"cholesterol[:\s]+(\d+)",
        "fbs":      r"fbs[:\s]+([01])",
        "thalach":  r"(?:max heart rate|thalach)[:\s]+(\d+)",
        "exang":    r"exang[:\s]+([01])",
        "oldpeak":  r"oldpeak[:\s]+([\d.]+)",
    }
    extracted = {}
    for key, pat in patterns.items():
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            extracted[key] = float(m.group(1))
    return extracted

def run_all_models(input_vector):
    scaled = SCALER.transform([input_vector])
    predictions = {}
    for name, clf in MODELS.items():
        pred  = clf.predict(scaled)[0]
        proba = clf.predict_proba(scaled)[0]
        conf  = round(float(max(proba)) * 100, 1)
        predictions[name] = {"label": pred, "confidence": conf}
    return predictions

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    try:
        return render_template("index.html",
                               models_loaded=MODELS_LOADED,
                               features=FEATURES if MODELS_LOADED else [])
    except Exception as e:
        return f"""
        <html>
        <body style='background:#0f172a;color:white;font-family:sans-serif;padding:2rem'>
        <h2>⚠️ Template Error</h2>
        <pre style='background:#1e293b;padding:1rem;border-radius:8px;color:#f87171'>{e}</pre>
        <p>Models loaded: <b>{MODELS_LOADED}</b></p>
        <p>Make sure <code>templates/index.html</code> exists inside <code>D:\\bp_classifier\\templates\\</code></p>
        </body>
        </html>
        """

@app.route("/test")
def test():
    import os
    template_path = os.path.join(app.root_path, "templates", "index.html")
    template_exists = os.path.exists(template_path)
    return f"""
    <html>
    <body style='background:#0f172a;color:white;font-family:sans-serif;padding:2rem'>
    <h2 style='color:#4ade80'>✅ Flask is working!</h2>
    <p>Models loaded: <b style='color:{'#4ade80' if MODELS_LOADED else '#f87171'}'>{MODELS_LOADED}</b></p>
    <p>templates/index.html found: <b style='color:{'#4ade80' if template_exists else '#f87171'}'>{template_exists}</b></p>
    <p>App root: <code>{app.root_path}</code></p>
    <p>Templates folder: <code>{os.path.join(app.root_path, 'templates')}</code></p>
    </body>
    </html>
    """

@app.route("/predict", methods=["POST"])
def predict():
    if not MODELS_LOADED:
        return redirect(url_for("index"))

    params = {}
    for feat in FEATURES:
        val = request.form.get(feat, "").strip()
        try:
            params[feat] = float(val)
        except ValueError:
            params[feat] = 0.0

    input_vector = [params[f] for f in FEATURES]
    predictions  = run_all_models(input_vector)

    labels      = [v["label"] for v in predictions.values()]
    final_label = max(set(labels), key=labels.count)
    final_color = BP_COLORS.get(final_label, "#3498db")

    return render_template(
        "result.html",
        params=params,
        predictions=predictions,
        final_label=final_label,
        final_color=final_color,
        bp_colors=BP_COLORS,
        meta=META,
        input_type="manual",
    )

@app.route("/upload", methods=["POST"])
def upload():
    if not MODELS_LOADED:
        return redirect(url_for("index"))

    if "report" not in request.files:
        return redirect(url_for("index"))

    file = request.files["report"]
    if file.filename == "" or not allowed_file(file.filename):
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    ext  = filename.rsplit(".", 1)[1].lower()
    text = ""
    if ext == "pdf" and PDF_AVAILABLE:
        text = extract_text_from_pdf(filepath)
    elif ext in ("png", "jpg", "jpeg") and OCR_AVAILABLE:
        text = extract_text_from_image(filepath)

    extracted = parse_params_from_text(text)

    defaults = {"age": 54, "sex": 1, "trestbps": 131, "chol": 246,
                "fbs": 0, "thalach": 149, "exang": 0, "oldpeak": 1.0}
    params = {f: extracted.get(f, defaults[f]) for f in FEATURES}

    input_vector = [params[f] for f in FEATURES]
    predictions  = run_all_models(input_vector)

    labels      = [v["label"] for v in predictions.values()]
    final_label = max(set(labels), key=labels.count)
    final_color = BP_COLORS.get(final_label, "#3498db")

    return render_template(
        "result.html",
        params=params,
        predictions=predictions,
        final_label=final_label,
        final_color=final_color,
        bp_colors=BP_COLORS,
        meta=META,
        input_type="upload",
        extracted_fields=list(extracted.keys()),
    )

if __name__ == "__main__":
    app.run(debug=True)
