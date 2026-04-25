# video_analyzer.py
import os
import csv
import argparse
from pathlib import Path
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Reuse your working pipeline (NO changes to testing2.py needed)
import testing2 as t2

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def analyze_frame_with_testing2(frame_bgr, model, last_conv_layer: str):
    """
    Minimal detector that uses ONLY testing2.py functions:
      - t2.preprocess_for_model
      - model.predict
      - (optional) t2.gradcam_binary + region text (kept lightweight)
    Returns (pred_label, p_real, explanation)
    """
    # BGR -> RGB (testing2 expects RGB)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Preprocess exactly like your image flow
    x, display_face = t2.preprocess_for_model(frame_rgb)

    # Predict using your trained model (sigmoid head)
    prob = float(model.predict(x, verbose=0)[0][0])
    p_real = prob if t2.POSITIVE_LABEL_IS_REAL else (1.0 - prob)
    pred_label = "REAL" if p_real >= 0.5 else "FAKE"

    # Lightweight explanation (no plots): if FAKE, compute a brief region text
    explanation = "No manipulation found — facial features appear natural and consistent."
    if pred_label == "FAKE":
        try:
            heatmap = t2.gradcam_binary(x, model, last_conv_layer, force_class=None)
            if heatmap is not None and float(np.max(heatmap)) >= t2.HEATMAP_MIN_VALID:
                H, W = display_face.shape[:2]
                hm = cv2.resize(heatmap, (W, H))

                # strong region mask (same thresholds from testing2)
                thr_global = max(np.percentile(hm, t2.FOCUS_TOP_PERCENT), t2.ABSOLUTE_MIN)
                strong_global = (hm >= thr_global).astype(np.uint8) * 255
                if getattr(t2, "MORPH_OPEN_K", 0) and t2.MORPH_OPEN_K > 0:
                    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (t2.MORPH_OPEN_K, t2.MORPH_OPEN_K))
                    strong_global = cv2.morphologyEx(strong_global, cv2.MORPH_OPEN, k)

                fmask = t2.face_hull_mask(display_face)  # 0/255 or None
                if fmask is not None:
                    strong_face = cv2.bitwise_and(strong_global, fmask)
                    strong_face = t2.largest_face_component(
                        strong_face, face_mask_uint8=fmask, min_area_ratio=0.002
                    )
                    regions = t2.map_gradcam_regions_mediapipe(display_face, strong_face)
                else:
                    regions = t2.map_gradcam_regions_mediapipe(display_face, strong_global)

                explanation = t2.humanize_regions(regions)
            else:
                explanation = "Manipulation detected but regions unclear."
        except Exception:
            explanation = "Manipulation detected (explanation unavailable)."

    return pred_label, p_real, explanation

def main():
    ap = argparse.ArgumentParser(description="Make frames from a video and auto-detect using testing2.py")
    ap.add_argument("--video", required=True, help="Path to input video")
    ap.add_argument("--every_n_frames", type=int, default=10, help="Sample every Nth frame")
    ap.add_argument("--every_secs", type=float, default=None, help="Sample every S seconds (overrides --every_n_frames)")
    ap.add_argument("--out_dir", default="video_frames_out", help="Directory to save frames + CSV")
    ap.add_argument("--save_all_frames", action="store_true",
                    help="If set, save every sampled frame image to disk")
    ap.add_argument("--model", default=t2.MODEL_PATH, help="Model path (defaults to testing2.MODEL_PATH)")
    args = ap.parse_args()

    in_path = Path(args.video)
    assert in_path.exists(), f"Video not found: {in_path}"

    out_root = Path(args.out_dir)
    frames_dir = out_root / f"{in_path.stem}_frames"
    ensure_dir(out_root)
    if args.save_all_frames:
        ensure_dir(frames_dir)

    # Load your model once
    model = load_model(args.model, compile=False)
    last_conv = t2.pick_last_conv_layer(model)

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Sampling strategy
    if args.every_secs and args.every_secs > 0:
        step = max(1, int(round(fps * args.every_secs)))
    else:
        step = max(1, int(args.every_n_frames))

    # CSV setup
    csv_path = out_root / f"{in_path.stem}_per_frame.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "time_sec", "pred_label", "p_real", "explanation"])

        idx = -1
        processed = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % step != 0:
                continue

            tsec = idx / fps

            # Detect using testing2’s pipeline
            pred_label, p_real, explanation = analyze_frame_with_testing2(frame, model, last_conv)

            # Optionally save the sampled frame
            if args.save_all_frames:
                frame_name = frames_dir / f"frame_{idx:06d}.jpg"
                cv2.imwrite(str(frame_name), frame)

            # Write CSV row
            w.writerow([idx, f"{tsec:.2f}", pred_label, f"{p_real:.6f}", explanation])
            processed += 1

    cap.release()
    print("Done.")
    print(f"CSV saved at: {csv_path}")
    if args.save_all_frames:
        print(f"Frames saved under: {frames_dir}")

if __name__ == "__main__":
    main()
