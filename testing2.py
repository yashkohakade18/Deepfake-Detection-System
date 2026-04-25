# testing.py

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess

# =========================
# Config
# =========================
IMG_SIZE = (299, 299)
MODEL_PATH = "new_deepfake_detector.h5"   # hardcoded model path
TEST_IMAGE = r"E:\TECNOBIJ 2025-2026\deepppp\datsset\Celeb-DF-v2\Celeb-synthesis\id50_id54_0002\frame_0007_03s000ms.jpg"
# Focus only on strong CAM (red/orange) regions
# Text explanation should consider only strong CAM (red/orange)
FOCUS_TOP_PERCENT = 85    # keep top 15% activations
ABSOLUTE_MIN = 0.60       # floor for strong activations (0..1)
MORPH_OPEN_K = 3          # clean small speckles; 0 = off
           # denoise mask (3x3), set 0 to disable

# IMPORTANT: pick ONE to match your pipeline
# - 'xception'  -> matches your Flask app/predict.py
# - 'rescale'   -> matches your train.py ImageDataGenerator(rescale=1/255)
PREPROCESS_MODE = "xception"   # <<< set to 'xception' to match Flask, 'rescale' to match training

# Toggle alignment/CLAHE (turn OFF to match Flask exactly)
FACE_ALIGN = False             # Flask does no face-crop
USE_CLAHE  = False             # Flask does no CLAHE

# Label semantics for your sigmoid head
POSITIVE_LABEL_IS_REAL = True  # sigmoid=1 means REAL

# Grad-CAM params
GRADCAM_THRESHOLD = 0.30
HEATMAP_MIN_VALID = 0.05
LAST_CONV_CANDIDATES = [
    "block14_sepconv2_act", "block14_sepconv2",
    "block14_sepconv1_act", "block13_sepconv2_act", "block13_sepconv2"
]

# =========================
# OpenCV face helpers (Python 3.12-safe fallback for explanations)
# =========================
_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# --- Context gating (outside-face -> REAL) ---
OUTSIDE_DOMINANCE_FACTOR = 2.0   # outside must be > 2x inside to trigger the gate
FORCE_REAL_WHEN_OUTSIDE_DOMINATES = True  # set False if you only want to annotate, not override
MIN_FORCED_REAL_PROB = 0.75      # when forcing REAL, ensure p_real is at least this


def _detect_primary_face_box(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    return int(x), int(y), int(w), int(h)


def _region_masks_from_face_box(img_rgb, face_box):
    h, w = img_rgb.shape[:2]
    x, y, bw, bh = face_box
    masks = {}

    face_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(
        face_mask,
        (x + bw // 2, y + bh // 2),
        (max(1, int(bw * 0.48)), max(1, int(bh * 0.58))),
        0,
        0,
        360,
        1,
        -1,
    )

    def rect_mask(rx1, ry1, rx2, ry2):
        mask = np.zeros((h, w), dtype=np.uint8)
        x1 = max(0, min(w, int(rx1)))
        y1 = max(0, min(h, int(ry1)))
        x2 = max(0, min(w, int(rx2)))
        y2 = max(0, min(h, int(ry2)))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
        return mask

    left_eye = rect_mask(x + 0.12 * bw, y + 0.22 * bh, x + 0.42 * bw, y + 0.46 * bh)
    right_eye = rect_mask(x + 0.58 * bw, y + 0.22 * bh, x + 0.88 * bw, y + 0.46 * bh)
    mouth = rect_mask(x + 0.22 * bw, y + 0.62 * bh, x + 0.78 * bw, y + 0.90 * bh)
    nose_cheeks = rect_mask(x + 0.22 * bw, y + 0.36 * bh, x + 0.78 * bw, y + 0.72 * bh)
    forehead = rect_mask(x + 0.14 * bw, y + 0.02 * bh, x + 0.86 * bw, y + 0.28 * bh)
    jawline = rect_mask(x + 0.10 * bw, y + 0.72 * bh, x + 0.90 * bw, y + 0.98 * bh)

    masks["eyes"] = (left_eye | right_eye) & face_mask
    masks["mouth"] = mouth & face_mask
    masks["nose/cheeks"] = nose_cheeks & face_mask
    masks["forehead"] = forehead & face_mask
    masks["jawline"] = jawline & face_mask
    masks["face"] = face_mask
    return masks

def map_gradcam_regions_mediapipe(img_rgb_uint8, gradcam_mask_uint8, overlap_thresh=100):
    try:
        face_box = _detect_primary_face_box(img_rgb_uint8)
        if face_box is None:
            return ["no face landmarks detected"]
        region_masks = _region_masks_from_face_box(img_rgb_uint8, face_box)
        regions_hit = []
        gmask = gradcam_mask_uint8.astype(bool)
        for name, mask in region_masks.items():
            if name == "face":
                continue
            overlap = np.sum(np.logical_and(mask.astype(bool), gmask))
            if overlap > overlap_thresh:
                regions_hit.append(name)
        if not regions_hit:
            return ["no specific region"]
        preferred = ["eyes", "mouth", "nose/cheeks", "forehead", "jawline"]
        ordered = [r for r in preferred if r in regions_hit]
        return ordered if ordered else regions_hit
    except Exception:
        return ["mediapipe_error"]


def face_hull_mask(img_rgb):
    """Return uint8 face-area mask (0/255). None if detection fails."""
    face_box = _detect_primary_face_box(img_rgb)
    if face_box is None:
        return None
    return (_region_masks_from_face_box(img_rgb, face_box)["face"] * 255).astype(np.uint8)

def largest_face_component(mask_uint8, face_mask_uint8=None, min_area_ratio=0.002):
    """Keep largest connected component (centroid inside face); drop tiny noise."""
    m = (mask_uint8 > 0).astype(np.uint8)
    if m.max() == 0:
        return mask_uint8
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
    H, W = m.shape[:2]
    best_idx, best_area = -1, -1
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area_ratio * H * W:
            continue
        cx, cy = centroids[i]
        if face_mask_uint8 is not None and face_mask_uint8[int(cy), int(cx)] == 0:
            continue
        if area > best_area:
            best_area = area
            best_idx = i
    out = np.zeros_like(m)
    if best_idx > 0:
        out[labels == best_idx] = 1
    else:
        out = m
    return (out * 255).astype(np.uint8)


def humanize_regions(regions):
    if not regions or "no specific region" in regions:
        return "Manipulation detected, but not localized to a clear facial feature."
    if "no face landmarks detected" in regions:
        return "Manipulation detected, but the face could not be localized reliably."
    if "mediapipe_error" in regions:
        return "Manipulation detected, but face-region analysis encountered an error."
    nice = {
        "eyes": "around the eyes (blink edges and eyelid contours look inconsistent)",
        "mouth": "near the lips and smile lines (texture boundaries look irregular)",
        "nose/cheeks": "across the mid-face (nose and cheek textures don't blend naturally)",
        "forehead": "on the forehead (skin tone and shading look uneven)",
        "jawline": "along the jawline (edge transitions appear unnaturally sharp)"
    }
    parts = [nice.get(r, r) for r in regions]
    return "Manipulation signs are visible " + "; ".join(parts) + "."

# =========================
# Preprocessing / alignment
# =========================
def crop_face_mediapipe(img_rgb):
    face_box = _detect_primary_face_box(img_rgb)
    if face_box is None:
        return None
    h, w = img_rgb.shape[:2]
    x, y, bw, bh = face_box
    mx = int(0.15 * bw)
    my = int(0.20 * bh)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(w, x + bw + mx)
    y2 = min(h, y + bh + my)
    if x2 <= x1 or y2 <= y1: return None
    return img_rgb[y1:y2, x1:x2]

def apply_clahe(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    L2 = clahe.apply(L)
    return cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2RGB)

def preprocess_for_model(img_rgb):
    """Return (batch_tensor_for_model, display_img_rgb) following PREPROCESS_MODE & toggles."""
    # (optional) align
    if FACE_ALIGN:
        face = crop_face_mediapipe(img_rgb) or img_rgb
    else:
        face = img_rgb
    # (optional) CLAHE
    if USE_CLAHE:
        face = apply_clahe(face)
    # resize
    face = cv2.resize(face, IMG_SIZE, interpolation=cv2.INTER_AREA)
    disp = face.copy()

    # to float32
    arr = face.astype(np.float32)
    if PREPROCESS_MODE == "rescale":
        arr = arr / 255.0
    elif PREPROCESS_MODE == "xception":
        # expects RGB in [0..255]
        arr = xception_preprocess(arr.copy())
    else:
        raise ValueError("PREPROCESS_MODE must be 'rescale' or 'xception'")

    bat = np.expand_dims(arr, 0)
    return bat, disp

def load_rgb(path):
    img = Image.open(path).convert("RGB")
    return np.array(img)

# =========================
# Grad-CAM for binary sigmoid (use logits)
# =========================
def pick_last_conv_layer(model):
    for lname in LAST_CONV_CANDIDATES:
        try:
            _ = model.get_layer(lname); return lname
        except Exception:
            continue
    for layer in reversed(model.layers):
        if "conv" in layer.name:
            return layer.name
    raise ValueError("No suitable conv layer found")

def gradcam_binary(img_batch, model, last_conv, force_class=None):
    conv_layer = model.get_layer(last_conv)
    grad_model = tf.keras.Model(model.inputs, [conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_out, prob = grad_model(img_batch, training=False)  # prob in [0,1] for positive class
        p = prob[:, 0]
        # logit (more stable grads than probability)
        z = tf.math.log(p/(1.0 - p + 1e-8) + 1e-8)
        if force_class is None:
            target = (z if p>=0.5 else -z) if POSITIVE_LABEL_IS_REAL else (-z if p>=0.5 else z)
        elif force_class.upper() == "REAL":
            target = z if POSITIVE_LABEL_IS_REAL else -z
        else:  # FAKE
            target = -z if POSITIVE_LABEL_IS_REAL else z
    grads = tape.gradient(target, conv_out)
    weights = tf.reduce_mean(grads, axis=(0,1,2))
    cam = tf.tensordot(conv_out[0], weights, axes=(2,0))
    cam = tf.nn.relu(cam)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy()

def strong_cam_mask(hm, top_percent=85, absolute_min=0.60, morph_k=3):
    """Return uint8 mask (0/255) keeping only strongest activations."""
    thr_p = np.percentile(hm, top_percent)
    thr = max(thr_p, absolute_min)
    m = (hm >= thr).astype(np.uint8) * 255
    if morph_k and morph_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    return m



# =========================
# Explain + visualize
# =========================
def explain_image(img_path, model, force_class=None):
    # ---- local defaults in case the new config flags aren't defined globally ----
    ODF = globals().get('OUTSIDE_DOMINANCE_FACTOR', 2.0)            # outside dominance factor
    FORCE_REAL = globals().get('FORCE_REAL_WHEN_OUTSIDE_DOMINATES', True)
    MIN_FORCED_REAL_PROB = float(globals().get('MIN_FORCED_REAL_PROB', 0.75))

    img_rgb = load_rgb(img_path)
    x, display_face = preprocess_for_model(img_rgb)

    # prediction
    prob = float(model.predict(x, verbose=0)[0][0])  # sigmoid for positive class
    p_real = prob if POSITIVE_LABEL_IS_REAL else (1.0 - prob)
    pred_label = "REAL" if p_real >= 0.5 else "FAKE"
    conf = p_real if pred_label == "REAL" else (1.0 - p_real)

    # Grad-CAM
    last_conv = pick_last_conv_layer(model)
    heatmap = gradcam_binary(x, model, last_conv, force_class=force_class)

    if heatmap is None or np.max(heatmap) < HEATMAP_MIN_VALID:
        msg = ("No manipulation found — features look natural."
               if pred_label == "REAL" else
               "Manipulation detected but regions unclear.")
        plt.imshow(display_face); plt.title(f"{pred_label} ({conf:.2%})\n{msg}")
        plt.axis("off"); plt.show()
        return pred_label, p_real, msg

    # Resize CAM to image size
    H, W = display_face.shape[:2]
    hm = cv2.resize(heatmap, (W, H))

    # --- keep the Grad-CAM view as before (no change) ---
    jet = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(display_face[..., ::-1], 0.6, jet, 0.4, 0)[..., ::-1]

    # ===== make TEXT follow the PICTURE (use global strong CAM, not face-renorm) =====
    thr_global = max(np.percentile(hm, FOCUS_TOP_PERCENT), ABSOLUTE_MIN)
    strong_global = (hm >= thr_global).astype(np.uint8) * 255
    if MORPH_OPEN_K and MORPH_OPEN_K > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_OPEN_K, MORPH_OPEN_K))
        strong_global = cv2.morphologyEx(strong_global, cv2.MORPH_OPEN, k)

    # Split strong mask into inside-face vs outside-face
    fmask = face_hull_mask(display_face)  # 0/255 or None
    override_reason = None  # <-- declare early to avoid editor warning

    if fmask is not None:
        strong_face    = cv2.bitwise_and(strong_global, fmask)
        strong_outside = cv2.bitwise_and(strong_global, cv2.bitwise_not(fmask))
        # keep only the most plausible in-face blob
        strong_face = largest_face_component(strong_face, face_mask_uint8=fmask, min_area_ratio=0.002)
        cnt_face    = int(np.count_nonzero(strong_face))
        cnt_outside = int(np.count_nonzero(strong_outside))
        face_area   = np.count_nonzero(fmask)
        strong_ratio = (cnt_face / (face_area + 1e-8)) if face_area > 0 else 0.0

        # === Context gate: if Grad-CAM is dominated by outside (neck/background), force REAL ===
        if FORCE_REAL and (cnt_outside > ODF * max(cnt_face, 1)):
            if pred_label != "REAL":
                override_reason = "Outside-face Grad-CAM dominates (neck/background). Forcing REAL."
            pred_label = "REAL"
            p_real = max(p_real, MIN_FORCED_REAL_PROB)
            conf = p_real
    else:
        strong_face    = strong_global
        cnt_face       = int(np.count_nonzero(strong_face))
        cnt_outside    = 0
        strong_ratio   = float(cnt_face) / (H * W)
        # No face mask -> cannot apply outside/neck dominance logic

    # ---------------- Explanation text ----------------
    if pred_label == "REAL" and force_class is None and not override_reason:
        explanation = "No manipulation found — facial features appear natural and consistent."
    else:
        if fmask is not None and cnt_outside > 2 * max(cnt_face, 1):
            base_expl = ("Model’s strongest evidence lies outside the face region "
                         "(e.g., neck/background). This can reflect context/dataset bias; "
                         "interpret with caution.")
        else:
            regions = map_gradcam_regions_mediapipe(display_face, strong_face)
            base_expl = humanize_regions(regions) + f" High-confidence face area ≈ {strong_ratio:.1%}."

        # Append override reason if applied
        if override_reason:
            explanation = base_expl + f" Override applied: {override_reason}"
        else:
            explanation = base_expl

    # === RENDER: images on top row, full-width text at bottom ===
    import textwrap
    from matplotlib.gridspec import GridSpec

    wrapped_expl = "\n".join(textwrap.wrap(explanation, width=110))

    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[3.0, 1.1])

    ax_img1 = fig.add_subplot(gs[0, 0])
    ax_img2 = fig.add_subplot(gs[0, 1])
    ax_text = fig.add_subplot(gs[1, :])

    ax_img1.imshow(display_face)
    ax_img1.set_title("Aligned Face" if FACE_ALIGN else "Input Face")
    ax_img1.axis("off")

    ax_img2.imshow(overlay)
    ax_img2.set_title("Grad-CAM (manipulation areas)")
    ax_img2.axis("off")

    ax_text.axis("off")
    ax_text.text(
        0.5, 0.5,
        f"Pred: {pred_label} ({conf:.2%})  •  Preproc={PREPROCESS_MODE}  •  Align={FACE_ALIGN}\n"
        f"{wrapped_expl}\n(CAM layer: {last_conv})",
        ha="center", va="center", fontsize=11, wrap=True
    )

    fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.06, hspace=0.25, wspace=0.08)
    plt.show()

    return pred_label, p_real, explanation

# =========================
# Main
# =========================
if __name__=="__main__":
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    if not os.path.exists(TEST_IMAGE):
        raise FileNotFoundError(f"Test image not found at {TEST_IMAGE}")

    model = load_model(MODEL_PATH, compile=False)
    label, prob, explanation = explain_image(TEST_IMAGE, model)
    print("Final Result:", label, prob, explanation)
