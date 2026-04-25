import os
import sqlite3
from datetime import datetime
from pathlib import Path
from functools import wraps

from flask import (
    Flask, render_template, request, redirect, url_for, flash, session, send_from_directory
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# --- Your existing detector code (MUST be alongside app.py) ---
import testing2 as t2  # uses MODEL_PATH, PREPROCESS_MODE, explain_image(), etc.  :contentReference[oaicite:4]{index=4}

# Reuse helpers from your video script for frames + Grad-CAM composite figures
from run_video_detect import (
    extract_frames_every, run_detector_on_frames, save_csv
)  # creates "output frames" with two-panel Grad-CAM figures  :contentReference[oaicite:5]{index=5}

# -------------------- Basic Config --------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
IMAGE_DIR = UPLOAD_DIR / "images"
VIDEO_DIR = UPLOAD_DIR / "videos"
ALLOWED_IMG = {"png", "jpg", "jpeg", "bmp", "webp"}
ALLOWED_VID = {"mp4", "avi", "mov", "mkv", "webm"}

DB_PATH = BASE_DIR / "site.db"
SECRET_KEY = os.environ.get("FLASK_SECRET", "dev-secret-change-me")

# -------------------- App Factory --------------------
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # up to 1 GB
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

# -------------------- DB Helpers --------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute(
        """CREATE TABLE IF NOT EXISTS users(
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               email TEXT UNIQUE NOT NULL,
               name TEXT,
               password_hash TEXT NOT NULL,
               created_at TEXT NOT NULL
           )"""
    )
    conn.commit()
    conn.close()

init_db()

# -------------------- Auth Helpers --------------------
def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped

# -------------------- Model Loading (1-time) --------------------
# Load your Keras model exactly how your testing2 expects it.
def _load_model_once():
    model_path = getattr(t2, "MODEL_PATH", "new_deepfake_detector.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path} (from testing2.py)")
    from tensorflow.keras.models import load_model
    return load_model(model_path, compile=False)

MODEL = _load_model_once()

# -------------------- Utils --------------------
def allowed_file(filename, allowed):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed

def save_upload(file_storage, dst_dir: Path):
    filename = secure_filename(file_storage.filename)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    name, ext = os.path.splitext(filename)
    new_name = f"{name}_{stamp}{ext.lower()}"
    out_path = dst_dir / new_name
    file_storage.save(str(out_path))
    return out_path

def _basic_image_prediction(img_path: Path, model):
    img_rgb = t2.load_rgb(str(img_path))
    x, _ = t2.preprocess_for_model(img_rgb)
    prob = float(model.predict(x, verbose=0)[0][0])
    p_real = prob if getattr(t2, "POSITIVE_LABEL_IS_REAL", True) else (1.0 - prob)
    label = "REAL" if p_real >= 0.5 else "FAKE"
    explanation = (
        "Prediction completed, but detailed face-region analysis is unavailable in this "
        "environment, so a basic Grad-CAM explanation was used."
    )
    return label, p_real, explanation

def _predict_image_with_fallback(img_path: Path, model):
    try:
        return t2.explain_image(str(img_path), model)
    except Exception as exc:
        app.logger.exception("Detailed image explanation failed for %s; using fallback.", img_path)
        return _basic_image_prediction(img_path, model)

# -------------------- Routes: Auth --------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        name = request.form.get("name", "").strip()
        password = request.form.get("password", "")
        if not email or not password:
            flash("Email and password are required.", "danger")
            return redirect(url_for("register"))
        pw_hash = generate_password_hash(password)
        try:
            conn = get_db()
            conn.execute(
                "INSERT INTO users(email, name, password_hash, created_at) VALUES(?,?,?,?)",
                (email, name, pw_hash, datetime.utcnow().isoformat()),
            )
            conn.commit()
            flash("Registration successful. Please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Email already registered.", "warning")
            return redirect(url_for("register"))
        finally:
            conn.close()
    return render_template("auth/register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        conn.close()
        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["user_name"] = user["name"] or email
            flash("Welcome!", "success")
            return redirect(url_for("index"))
        flash("Invalid credentials.", "danger")
    return render_template("auth/login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out.", "info")
    return redirect(url_for("login"))

# -------------------- Routes: Home --------------------
@app.route("/")
@login_required
def index():
    return render_template("index.html", user_name=session.get("user_name", "User"))

# -------------------- Routes: Image Detection --------------------
@app.route("/detect/image", methods=["GET", "POST"])
@login_required
def detect_image():
    result = None
    exp_png_rel = None
    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            flash("Please upload an image.", "warning")
            return redirect(url_for("detect_image"))
        if not allowed_file(file.filename, ALLOWED_IMG):
            flash("Unsupported image format.", "danger")
            return redirect(url_for("detect_image"))

        img_path = save_upload(file, IMAGE_DIR)

        # Run the explanation pipeline, but keep the request alive if detailed face analysis fails.
        label, p_real, explanation = _predict_image_with_fallback(img_path, MODEL)

        # Save the composite Grad-CAM figure using your helper from run_video_detect.py
        from run_video_detect import save_explain_figure  # local import to avoid cycles
        out_dir = img_path.parent / f"{Path(img_path).stem}_explain"
        out_png = save_explain_figure(img_path, MODEL, out_dir, label, float(p_real), explanation)  # :contentReference[oaicite:6]{index=6}

        # Build a static-relative path to show in template
        exp_png_rel = str(out_png).split("static/")[-1] if "static" in str(out_png) else None
        result = {
            "image_path": str(img_path).split("static/")[-1],
            "label": label,
            "p_real": float(p_real),
            "explanation": explanation,
            "figure_rel": exp_png_rel,
        }
    return render_template("detect/image.html", result=result)

# -------------------- Routes: Video Detection --------------------
# ====== add/keep these imports at top of app.py ======
import os, csv
from pathlib import Path
from flask import render_template, request, redirect, url_for, flash, send_from_directory
# =====================================================

# ====== paths (adjust only if you already define these elsewhere) ======
STATIC_DIR  = Path(app.static_folder)                         # e.g., static/
UPLOADS_DIR = STATIC_DIR / "uploads"                          # static/uploads/
VIDEO_DIR   = UPLOADS_DIR / "videos"                          # static/uploads/videos/
OUTPUT_DIR  = UPLOADS_DIR / "output"                          # static/uploads/output/
# ======================================================================

def _rel_to_uploads(p: os.PathLike | str) -> str:
    """
    Return a forward-slash path starting with 'uploads/...'
    Works on Windows/Linux. Handles absolute or relative inputs.
    """
    s = str(p)
    # Trim anything before and including 'static/' if present
    if f"static{os.sep}" in s:
        s = s.split(f"static{os.sep}", 1)[-1]
    elif "static/" in s:
        s = s.split("static/", 1)[-1]
    # Ensure forward slashes for URLs
    s = s.replace("\\", "/")
    # If it doesn't already start with uploads/, prefix it if we're inside static/
    if not s.startswith("uploads/") and "uploads/" in s:
        s = s[s.index("uploads/"):]
    return s

@app.route("/detect/video", methods=["GET", "POST"])
@login_required
def detect_video():
    gallery = []
    summary = None
    csv_rel = None

    if request.method == "POST":
        file = request.files.get("video")
        step = float(request.form.get("step", "0.5") or "0.5")  # seconds

        if not file or file.filename == "":
            flash("Please upload a video.", "warning")
            return redirect(url_for("detect_video"))
        if not allowed_file(file.filename, ALLOWED_VID):
            flash("Unsupported video format.", "danger")
            return redirect(url_for("detect_video"))

        # Save uploaded video under static/uploads/videos/<name>.<ext>
        vid_path: Path = save_upload(file, VIDEO_DIR)
        video_stem = vid_path.stem

        # Per-video output folder:
        # static/uploads/output/<video_stem>/ and its "output frames" subdir
        out_dir = OUTPUT_DIR / video_stem
        out_explain_dir = out_dir / "output frames"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_explain_dir.mkdir(parents=True, exist_ok=True)

        # 1) Extract frames into out_dir
        frames = extract_frames_every(vid_path, out_dir, step_sec=step, jpeg_quality=95)

        # 2) Run detector on frames (writes Grad-CAM composite PNGs into out_explain_dir)
        results = run_detector_on_frames(frames, MODEL, out_explain_dir)

        # 3) Write per-frame CSV beside images
        out_csv = out_dir / "frame_results.csv"
        save_csv(results, out_csv)

        # Build majority summary from results
        counts = {"REAL": 0, "FAKE": 0}
        probs = []
        for r in results:
            if r.get("pred_label") in counts:
                counts[r["pred_label"]] += 1
            if "p_real" in r:
                probs.append(float(r["p_real"]))
        final_label = "REAL" if counts["REAL"] >= counts["FAKE"] else "FAKE"
        mean_p = (sum(probs) / len(probs)) if probs else float("nan")

        # Prepare values for template
        csv_rel = _rel_to_uploads(out_csv)
        summary = {
            "video_rel": _rel_to_uploads(vid_path),
            "final_label": final_label,
            "counts": counts,
            "mean_p": float(mean_p),
        }

        # Prefer reading from CSV so paths/metadata exactly match saved files
        gallery = []
        if out_csv.exists():
            with open(out_csv, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Each row contains 'explain_file' (absolute or relative), 'pred_label', 'p_real', 'explanation'
                    fig_rel = _rel_to_uploads(row.get("explain_file", ""))

                    # Skip if no file or missing on disk
                    fig_abs = STATIC_DIR / fig_rel.replace("uploads/", "")
                    if not fig_abs.exists():
                        # Try fallback: the file path could be absolute already—check it
                        alt = Path(row.get("explain_file", "")).expanduser()
                        if not alt.exists():
                            continue
                        # If it exists but outside static/, keep URL relative based on actual location under static
                        fig_rel = _rel_to_uploads(alt)

                    gallery.append({
                        "fig_rel": fig_rel,  # "uploads/output/<video_stem>/output frames/xxx.png"
                        "pred": row.get("pred_label", ""),
                        "p_real": float(row.get("p_real", "nan")),
                        "explanation": row.get("explanation", ""),
                    })

        # If CSV missing for some reason, list images directly (no metadata)
        if not gallery:
            for p in sorted(out_explain_dir.glob("*.png")):
                fig_rel = _rel_to_uploads(p)
                gallery.append({
                    "fig_rel": fig_rel,
                    "pred": "",
                    "p_real": float("nan"),
                    "explanation": "",
                })

    return render_template("detect/video.html", gallery=gallery, summary=summary, csv_rel=csv_rel)

# Serves any file under static/uploads/… via /uploads/<path>
@app.route("/uploads/<path:filename>")
@login_required
def uploaded_file(filename):
    return send_from_directory(app.static_folder, f"uploads/{filename}")


# ================== ADD at top with other imports ==================
import os, io, csv
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import cv2, numpy as np
from PIL import Image
# ================================================================

# ================== CONFIG (paths) ==================
# Put your tampering model here (you uploaded it as tampering_detector_final.pth)
TAMPER_MODEL_PATH = Path("static/models/tampering_detector_final.pth")  # move file here
TAMPER_OUT_ROOT   = Path(app.static_folder) / "uploads" / "tampering"   # static/uploads/tampering
TAMPER_OUT_ROOT.mkdir(parents=True, exist_ok=True)
# ====================================================

# ================== LOAD Torch model ==================
# Expecting a resnet18 checkpoint dict with keys: model_state_dict, class_names
tamper_checkpoint = torch.load(TAMPER_MODEL_PATH, map_location="cpu")
tamper_class_names = tamper_checkpoint["class_names"]

_tamper_model = models.resnet18(pretrained=False)
_tamper_model.fc = torch.nn.Linear(_tamper_model.fc.in_features, len(tamper_class_names))
_tamper_model.load_state_dict(tamper_checkpoint["model_state_dict"])
_tamper_model.eval()

LAST_CONV_NAME = "layer4"
tamper_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ================== Grad-CAM helpers ==================
class _GradCAM:
    def __init__(self, model, target_layer=LAST_CONV_NAME):
        self.model = model
        self.target_activations = None
        self.target_grads = None
        layer = dict(model.named_children())[target_layer]
        self.fh = layer.register_forward_hook(self._fwd)
        try:
            self.bh = layer.register_full_backward_hook(self._bwd)
        except Exception:
            self.bh = layer.register_backward_hook(self._bwd)

    def _fwd(self, m, i, o): self.target_activations = o
    def _bwd(self, m, gi, go): self.target_grads = go[0]

    def __del__(self):
        for h in [getattr(self, "fh", None), getattr(self, "bh", None)]:
            try: h.remove()
            except Exception: pass


    def _rel_to_uploads(p: os.PathLike | str) -> str:
        """
        Return a forward-slash path that starts with 'uploads/...'
        Works on Windows/Linux; accepts absolute or relative inputs.
        """
        s = str(p)
        if f"static{os.sep}" in s:
            s = s.split(f"static{os.sep}", 1)[-1]
        elif "static/" in s:
            s = s.split("static/", 1)[-1]
        s = s.replace("\\", "/")
        if not s.startswith("uploads/") and "uploads/" in s:
            s = s[s.index("uploads/"):]
        return s


    def generate(self, scores, target_index=None):
        if target_index is None:
            target_index = scores.argmax(dim=1).item()
        self.model.zero_grad()
        target = scores[:, target_index]
        target.backward(retain_graph=True)
        A = self.target_activations        # [1,K,H,W]
        G = self.target_grads              # [1,K,H,W]
        w = G.mean(dim=(2, 3), keepdim=True)
        cam = (w * A).sum(dim=1, keepdim=True)
        cam = F.relu(cam).squeeze().detach().cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam  # HxW in [0,1]

def _overlay_heatmap(frame_bgr, cam_2d):
    H, W, _ = frame_bgr.shape
    cam = cv2.resize(cam_2d, (W, H), interpolation=cv2.INTER_CUBIC)
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(frame_bgr, 0.6, heat, 0.4, 0)

def _cam_bbox(cam_2d, thresh=0.6):
    m = (cam_2d >= thresh).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return (x, y, x + w, y + h)

def _describe_loc(b, Hc, Wc):
    x1, y1, x2, y2 = b
    cx, cy = (x1 + x2)/2.0, (y1 + y2)/2.0
    horiz = "left" if cx < Wc/3 else ("center" if cx < 2*Wc/3 else "right")
    vert  = "upper" if cy < Hc/3 else ("middle" if cy < 2*Hc/3 else "lower")
    return f"{vert}-{horiz} area"

def _ela_score(rgb, quality=90):
    pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality); buf.seek(0)
    pil2 = Image.open(buf).convert("RGB")
    diff = np.abs(np.asarray(pil, dtype=np.int16) - np.asarray(pil2, dtype=np.int16)).astype(np.uint8)
    return float(diff.mean())

def _edge_var(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return float(np.var(mag))

def _explain_from_cam_roi(frame_bgr, cam_2d, bbox_cam, thresh=0.6):
    Hf, Wf, _ = frame_bgr.shape
    Hc, Wc = cam_2d.shape
    if bbox_cam is None:
        return {"text":"Evidence dispersed; no dominant hotspot.","bbox_px":None}

    x1c,y1c,x2c,y2c = bbox_cam
    x1 = int(round(x1c * (Wf / Wc)));  y1 = int(round(y1c * (Hf / Hc)))
    x2 = int(round(x2c * (Wf / Wc)));  y2 = int(round(y2c * (Hf / Hc)))
    x1,y1 = max(0,x1), max(0,y1); x2,y2 = min(Wf-1,x2), min(Hf-1,y2)
    if x2<=x1 or y2<=y1: return {"text":"Hotspot too small.","bbox_px":None}

    roi = frame_bgr[y1:y2, x1:x2]
    loc = _describe_loc(bbox_cam, Hc, Wc)
    ela_all = _ela_score(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    ela_roi = _ela_score(cv2.cvtColor(roi,        cv2.COLOR_BGR2RGB))
    e_all   = _edge_var(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
    e_roi   = _edge_var(cv2.cvtColor(roi,        cv2.COLOR_BGR2GRAY))

    ela_spike  = (ela_roi > (ela_all * 1.25)) and (ela_roi - ela_all > 1.0)
    edge_spike = (e_roi > (e_all * 1.35))
    reasons = []
    if ela_spike:  reasons.append("ELA spike (recompression anomaly)")
    if edge_spike: reasons.append("edge inconsistency (sharp/blocky boundary)")
    text = (f"Strong {' and '.join(reasons)} in {loc} → likely local edit/splice."
            if reasons else f"Model attention focused in {loc}; artifacts not decisive but suspicious region present.")
    return {"text": text, "bbox_px": (x1,y1,x2,y2)}

def _save_frame_and_heat(video_stem, tkey, frame_bgr, heat_bgr=None):
    fdir = TAMPER_OUT_ROOT / video_stem / "frames"
    hdir = TAMPER_OUT_ROOT / video_stem / "heatmaps"
    fdir.mkdir(parents=True, exist_ok=True)
    hdir.mkdir(parents=True, exist_ok=True)

    fpath = fdir / f"{tkey}.jpg"
    cv2.imwrite(str(fpath), frame_bgr)

    hpath = None
    if heat_bgr is not None:
        hpath = hdir / f"{tkey}.jpg"
        cv2.imwrite(str(hpath), heat_bgr)

    # return URLs relative to /uploads route, e.g. "uploads/tampering/VideoName/heatmaps/xxx.jpg"
    rel_f = _rel_to_uploads(fpath)
    rel_h = _rel_to_uploads(hpath) if hpath else None
    return rel_f, rel_h

def _analyze_video_tampering(vpath: Path, fps=2.0, cam_thresh=0.6):
    cap = cv2.VideoCapture(str(vpath))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(actual_fps / max(fps,1e-3))), 1)
    gc = _GradCAM(_tamper_model, LAST_CONV_NAME)

    labels = []; idx = 0; tsec = 0.0
    vstem = vpath.stem

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok: break
        if idx % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            x = tamper_transform(pil).unsqueeze(0).requires_grad_()
            with torch.enable_grad():
                scores = _tamper_model(x)
            probs = F.softmax(scores, dim=1)[0].detach().cpu().numpy()
            pred_i = int(np.argmax(probs)); pred_name = tamper_class_names[pred_i]

            heat_bgr, why = None, None
            if pred_name != "real":
                cam = gc.generate(scores, target_index=pred_i)
                bbox = _cam_bbox(cam, cam_thresh)
                expl = _explain_from_cam_roi(frame, cam, bbox, cam_thresh)
                heat = _overlay_heatmap(frame, cam)
                if expl.get("bbox_px"):
                    x1,y1,x2,y2 = expl["bbox_px"]; cv2.rectangle(heat,(x1,y1),(x2,y2),(0,255,255),2)
                heat_bgr, why = heat, expl["text"]

            tkey = f"t_{int(round(tsec*1000)):06d}"
            f_web, h_web = _save_frame_and_heat(vstem, tkey, frame, heat_bgr)
            topk = [(tamper_class_names[i], float(probs[i])) for i in probs.argsort()[-3:][::-1]]
            labels.append({
                "time": round(tsec, 2),
                "label": pred_name,
                "topk": topk,
                "frame_path": f_web,
                "heatmap_path": h_web,
                "explanation": why
            })
            tsec += 1.0 / fps
        idx += 1
    cap.release()
    return labels

# ================== ROUTE: /tampering (GET/POST) ==================
@app.route("/tampering", methods=["GET", "POST"])
@login_required
def tampering_page():
    labels = []; majority = None; video_rel = None
    if request.method == "POST":
        file = request.files.get("video")
        if not file or file.filename == "":
            flash("Please upload a video.", "warning")
            return redirect(url_for("tampering_page"))
        if not allowed_file(file.filename, ALLOWED_VID):
            flash("Unsupported video format.", "danger")
            return redirect(url_for("tampering_page"))

        # save video under static/uploads/tampering/<name>
        vpath = TAMPER_OUT_ROOT / Path(file.filename).name
        vpath.parent.mkdir(parents=True, exist_ok=True)
        file.save(str(vpath))
        video_rel = str(vpath).split("static/")[-1].replace("\\","/")

        # analyze
        labels = _analyze_video_tampering(vpath, fps=2.0, cam_thresh=0.6)
        preds = [x["label"] for x in labels]
        majority = "real" if all(p == "real" for p in preds) else "tampered"

    return render_template("tampering/page.html",
                           labels=labels,
                           majority=majority,
                           video_rel=video_rel)


# -------------------- Main --------------------
if __name__ == "__main__":
    app.run(debug=False)
