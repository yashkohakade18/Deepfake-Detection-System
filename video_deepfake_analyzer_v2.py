# -*- coding: utf-8 -*-
"""
Video Deepfake Analyzer v2 (stabilized, fixed)
Fixes: preprocessing match, motion noise, background bias, flicker.
Outputs: annotated mp4 + CSV (per-frame).
"""

import os, csv, argparse, textwrap
import cv2
import numpy as np
import tensorflow as tf

# === Import your exact image pipeline ===
from testing2 import (  # must be in same folder
    MODEL_PATH, HEATMAP_MIN_VALID, POSITIVE_LABEL_IS_REAL,
    FOCUS_TOP_PERCENT, ABSOLUTE_MIN, MORPH_OPEN_K,
    PREPROCESS_MODE, FACE_ALIGN,
    preprocess_for_model, pick_last_conv_layer, gradcam_binary,
    face_hull_mask, map_gradcam_regions_mediapipe, humanize_regions,
    crop_face_mediapipe
)

# --------------------real.mp4
# Tunables for VIDEO
# --------------------
FACE_ALIGN_FOR_VIDEO = True     # turn ON for video (more stable)
MIN_FACE_RATIO = 0.05           # Reduced min fraction of frame area for face
PROB_EMA_ALPHA = 0.25           # temporal smoothing factor
DRAW_TEXT = True
HEAT_ALPHA = 0.35               # CAM overlay strength

# Custom re-labeling rule
def apply_custom_rule(label, p_real):
    p_fake = 1.0 - p_real
    if label == "FAKE":
        if (p_fake < 0.96) or (p_real > 0.05):
            return "REAL"
    return label

def argp():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--out_video", default=None)
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--min_face_ratio", type=float, default=MIN_FACE_RATIO, 
                   help="Minimum face ratio to process frame (default: 0.05)")
    return ap.parse_args()

def ensure_outnames(in_path, out_video, out_csv):
    stem = os.path.splitext(os.path.basename(in_path))[0]
    if out_video is None: out_video = f"{stem}_annotated.mp4"
    if out_csv is None:   out_csv   = f"{stem}_results.csv"
    return out_video, out_csv

def detect_face_ratio(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    m = face_hull_mask(rgb)
    if m is None: return 0.0
    return float(np.count_nonzero(m)) / (m.shape[0] * m.shape[1])

def preprocess_video_frame(frame_bgr):
    """Force RGB -> reuse your image preprocess. Optionally crop face for video."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if FACE_ALIGN_FOR_VIDEO:
        face = crop_face_mediapipe(rgb)
        if face is None:
            face = rgb
        x, disp = preprocess_for_model(face)
        return x, disp, rgb
    else:
        x, disp = preprocess_for_model(rgb)
        return x, disp, rgb

def draw_banner(img_bgr, label, p_real, explanation, fps):
    H, W = img_bgr.shape[:2]
    bar_h = max(40, H // 16)
    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (0,0), (W, bar_h), (0,0,0), -1)
    img_bgr[:] = cv2.addWeighted(overlay, 0.5, img_bgr, 0.5, 0)

    if label == "REAL":
        txt = f"Pred: REAL ({p_real:.1%})"
        color = (60,220,60)
    else:
        txt = f"Pred: FAKE ({(1.0-p_real):.1%})"
        color = (40,40,230)
    cv2.putText(img_bgr, txt, (12, bar_h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    flags = f"Preproc={PREPROCESS_MODE} • FaceAlignVideo={FACE_ALIGN_FOR_VIDEO}"
    (tw, _), _ = cv2.getTextSize(flags, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(img_bgr, flags, (W - tw - 12, bar_h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240,240,240), 1, cv2.LINE_AA)

    if explanation:
        exp = textwrap.shorten(" ".join(explanation.split()), width=110, placeholder="…")
        cv2.putText(img_bgr, exp, (12, min(H-12, bar_h + 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245,245,245), 1, cv2.LINE_AA)

def overlay_heatmap(frame_bgr, heatmap_01, alpha=HEAT_ALPHA):
    H, W = frame_bgr.shape[:2]
    hm = cv2.resize(heatmap_01, (W, H))
    jet = cv2.applyColorMap(np.uint8(255*hm), cv2.COLORMAP_JET)
    return cv2.addWeighted(frame_bgr, 1.0, jet, float(alpha), 0)

def analyse_single_frame(frame_bgr, model, last_conv):
    x, disp_face, rgb_full = preprocess_video_frame(frame_bgr)
    prob_pos = float(model.predict(x, verbose=0)[0][0])
    p_real   = prob_pos if POSITIVE_LABEL_IS_REAL else (1.0 - prob_pos)
    base_lbl = "REAL" if p_real >= 0.5 else "FAKE"

    heat = gradcam_binary(x, model, last_conv, force_class=None)
    if heat is None or np.max(heat) < HEATMAP_MIN_VALID:
        explanation = ("No manipulation found — features look natural."
                       if base_lbl=="REAL" else "Manipulation detected but regions unclear.")
        return base_lbl, p_real, explanation, frame_bgr

    Hs, Ws = disp_face.shape[:2]
    hm_small = cv2.resize(heat, (Ws, Hs))
    thr = max(np.percentile(hm_small, FOCUS_TOP_PERCENT), ABSOLUTE_MIN)
    strong = (hm_small >= thr).astype(np.uint8) * 255
    if MORPH_OPEN_K and MORPH_OPEN_K > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_OPEN_K, MORPH_OPEN_K))
        strong = cv2.morphologyEx(strong, cv2.MORPH_OPEN, k)

    fmask = face_hull_mask(disp_face)
    if fmask is not None:
        strong_face    = cv2.bitwise_and(strong, fmask)
        strong_outside = cv2.bitwise_and(strong, cv2.bitwise_not(fmask))
        cnt_face       = int(np.count_nonzero(strong_face))
        cnt_outside    = int(np.count_nonzero(strong_outside))
        face_area      = np.count_nonzero(fmask)
        strong_ratio   = (cnt_face/(face_area+1e-8)) if face_area>0 else 0.0
    else:
        strong_ratio   = float(np.count_nonzero(strong)) / (Hs*Ws)

    if base_lbl == "REAL":
        explanation = "No manipulation found — facial features appear natural and consistent."
    else:
        if fmask is not None and cnt_outside > 2 * max(int(np.count_nonzero(strong_face)), 1):
            explanation = ("Model's strongest evidence lies outside the face region "
                           "(neck/background) — may indicate dataset/context bias.")
        else:
            regions = map_gradcam_regions_mediapipe(disp_face, strong)
            base_text = humanize_regions(regions)
            explanation = f"{base_text} High-confidence face area ≈ {strong_ratio:.1%}."

    annotated = overlay_heatmap(frame_bgr, heat, alpha=HEAT_ALPHA)
    return base_lbl, p_real, explanation, annotated

def main():
    args = argp()
    in_path = args.video
    out_video, out_csv = ensure_outnames(in_path, args.out_video, args.out_csv)
    stride = max(1, int(args.stride))
    min_face_ratio = max(0.01, float(args.min_face_ratio))  # Ensure reasonable minimum

    if not os.path.exists(in_path): raise FileNotFoundError(in_path)
    if not os.path.exists(args.model): raise FileNotFoundError(args.model)

    model = tf.keras.models.load_model(args.model, compile=False)
    last_conv = pick_last_conv_layer(model)

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open video: {in_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video, fourcc, fps, (W, H))
    if not writer.isOpened(): raise RuntimeError(f"Cannot open writer: {out_video}")

    fcsv = open(out_csv, "w", newline="", encoding="utf-8")
    cw = csv.writer(fcsv)
    cw.writerow(["frame_index","timestamp_ms","label_raw","p_real_raw",
                 "label_after_rule","p_real_ema","explanation"])

    ema = None
    idx = 0
    processed = 0

    while True:
        ok, frame = cap.read()
        if not ok: break
        
        if (idx % stride) == 0:
            fr = detect_face_ratio(frame)
            if fr < min_face_ratio:
                # Instead of skipping, try to process anyway but with a warning
                print(f"Warning: Face ratio {fr:.3f} < {min_face_ratio} at frame {idx}, attempting analysis anyway")
                
                # Try to process the frame despite low face ratio
                try:
                    label_raw, p_real_raw, explanation, annotated = analyse_single_frame(frame, model, last_conv)
                    if ema is None: ema = p_real_raw
                    else: ema = PROB_EMA_ALPHA * p_real_raw + (1.0 - PROB_EMA_ALPHA) * ema

                    label_after = "REAL" if ema >= 0.5 else "FAKE"
                    label_after = apply_custom_rule(label_after, ema)

                    if DRAW_TEXT:
                        draw_banner(annotated, label_after, ema, explanation + " (Low face ratio)", fps)

                    writer.write(annotated)
                    ts = int((idx/fps)*1000.0)
                    cw.writerow([idx, ts, label_raw, f"{p_real_raw:.6f}", label_after, f"{ema:.6f}", explanation + " (Low face ratio)"])
                    processed += 1
                except Exception as e:
                    # If analysis fails, fall back to original behavior
                    print(f"Analysis failed for frame {idx}: {str(e)}")
                    writer.write(frame)
                    ts = int((idx/fps)*1000.0)
                    cw.writerow([idx, ts, "UNKNOWN", "", "UNKNOWN", "", f"face too small/absent: {str(e)}"])
            else:
                # Normal processing
                label_raw, p_real_raw, explanation, annotated = analyse_single_frame(frame, model, last_conv)
                if ema is None: ema = p_real_raw
                else: ema = PROB_EMA_ALPHA * p_real_raw + (1.0 - PROB_EMA_ALPHA) * ema

                label_after = "REAL" if ema >= 0.5 else "FAKE"
                label_after = apply_custom_rule(label_after, ema)

                if DRAW_TEXT:
                    draw_banner(annotated, label_after, ema, explanation, fps)

                writer.write(annotated)
                ts = int((idx/fps)*1000.0)
                cw.writerow([idx, ts, label_raw, f"{p_real_raw:.6f}", label_after, f"{ema:.6f}", explanation])
                processed += 1
        else:
            writer.write(frame)
        idx += 1

    cap.release(); writer.release(); fcsv.close()
    print(f"[OK] Annotated: {out_video}")
    print(f"[OK] CSV      : {out_csv}")
    print(f"[Info] Frames : {total_frames} | processed: {processed} | stride: {stride}")

if __name__ == "__main__":
    main()