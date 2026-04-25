# run_video_detect.py
import os
import math
import csv
import argparse
from pathlib import Path

# Headless rendering for matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cv2
import numpy as np

# --- your existing detector code (must be alongside this script) ---
import testing2 as t2  # contains MODEL_PATH, preprocess, explain_image, etc.

def _basic_frame_prediction(frame_path: Path, model):
    img_rgb = t2.load_rgb(str(frame_path))
    x, _ = t2.preprocess_for_model(img_rgb)
    prob = float(model.predict(x, verbose=0)[0][0])
    p_real = prob if getattr(t2, "POSITIVE_LABEL_IS_REAL", True) else (1.0 - prob)
    label = "REAL" if p_real >= 0.5 else "FAKE"
    explanation = (
        "Prediction completed, but detailed face-region analysis is unavailable in this "
        "environment, so a basic Grad-CAM explanation was used."
    )
    return label, p_real, explanation

def _predict_frame_with_fallback(frame_path: Path, model):
    try:
        return t2.explain_image(str(frame_path), model)
    except Exception as exc:
        print(f"[WARN] Detailed explanation failed for {frame_path.name}: {exc}")
        return _basic_frame_prediction(frame_path, model)

# ----------------- Helpers -----------------
def _stamp_from_seconds(t: float) -> str:
    s = int(math.floor(t))
    ms = int(round((t - s) * 1000))
    return f"{s:02d}s{ms:03d}ms"

def extract_frames_every(video_path: Path, out_dir: Path, step_sec: float = 0.5, jpeg_quality: int = 95):
    """
    Extract frames every `step_sec` seconds across entire video and save as JPGs.
    Returns list of saved file paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

    saved_paths = []

    if fps > 0 and total_frames > 0:
        duration_sec = total_frames / fps
        times = np.arange(0.0, duration_sec + 1e-6, step_sec, dtype=float)
        idxs = np.unique(np.clip(np.rint(times * fps).astype(int), 0, total_frames - 1))

        for i, idx in enumerate(idxs, start=1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            t = idx / fps
            out_file = out_dir / f"frame_{i:04d}_{_stamp_from_seconds(t)}.jpg"
            cv2.imwrite(str(out_file), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
            saved_paths.append(out_file)
    else:
        # Fallback: time-based read
        step_ms = step_sec * 1000.0
        next_target_ms = 0.0
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if pos_ms >= next_target_ms - 1e-3:
                i += 1
                out_file = out_dir / f"frame_{i:04d}_{_stamp_from_seconds(pos_ms/1000.0)}.jpg"
                cv2.imwrite(str(out_file), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
                saved_paths.append(out_file)
                next_target_ms += step_ms

    cap.release()
    return saved_paths

def build_gradcam_overlay_rgb(display_face_rgb, heatmap):
    """Return RGB overlay image. If heatmap invalid, return face with 'No Grad-CAM' note."""
    H, W = display_face_rgb.shape[:2]
    hm_valid_min = float(getattr(t2, "HEATMAP_MIN_VALID", 0.05))
    if heatmap is None or float(np.max(heatmap)) < hm_valid_min:
        overlay_bgr = cv2.cvtColor(display_face_rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(overlay_bgr, "No Grad-CAM", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2, cv2.LINE_AA)
        return cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    hm = cv2.resize(heatmap, (W, H))
    jet = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)  # BGR
    face_bgr = cv2.cvtColor(display_face_rgb, cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.addWeighted(face_bgr, 0.6, jet, 0.4, 0)
    return cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

def save_explain_figure(frame_path: Path, model, out_dir: Path, label: str, p_real: float, explanation: str):
    """
    Save a composite figure:
      Left: Input face (after t2.preprocess_for_model)
      Right: Grad-CAM overlay
      Bottom: text with Pred, Preproc, Align, explanation, CAM layer
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Preprocess & heatmap using the same pipeline as testing2
    img_rgb = t2.load_rgb(str(frame_path))
    x, display_face = t2.preprocess_for_model(img_rgb)
    last_conv = t2.pick_last_conv_layer(model)
    heatmap = t2.gradcam_binary(x, model, last_conv, force_class=None)
    overlay_rgb = build_gradcam_overlay_rgb(display_face, heatmap)

    # Confidence consistent with label
    conf = float(p_real if label == "REAL" else (1.0 - p_real))

    # Compose the figure (like your screenshot)
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[3.0, 1.1])

    ax_img1 = fig.add_subplot(gs[0, 0])
    ax_img2 = fig.add_subplot(gs[0, 1])
    ax_text = fig.add_subplot(gs[1, :])

    ax_img1.imshow(display_face)
    ax_img1.set_title("Input Face" if not getattr(t2, "FACE_ALIGN", False) else "Aligned Face")
    ax_img1.axis("off")

    ax_img2.imshow(overlay_rgb)
    ax_img2.set_title("Grad-CAM (manipulation areas)")
    ax_img2.axis("off")

    ax_text.axis("off")
    ax_text.text(
        0.5, 0.5,
        f"Pred: {label} ({conf:.2%})  •  Preproc={t2.PREPROCESS_MODE}  •  Align={getattr(t2, 'FACE_ALIGN', False)}\n"
        f"{explanation}\n(CAM layer: {last_conv})",
        ha="center", va="center", fontsize=11, wrap=True
    )

    fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.06, hspace=0.25, wspace=0.08)

    out_path = out_dir / f"{frame_path.stem}_explain.png"
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    return out_path

# ------------- Detector runner -------------
def run_detector_on_frames(frame_paths, model, out_explain_dir: Path):
    """
    Run t2.explain_image on each frame + save composite figure to 'output frames'.
    Returns list of dicts with results.
    """
    results = []
    for fp in frame_paths:
        try:
            label, p_real, expl = _predict_frame_with_fallback(fp, model)

            # Save the composite figure like the screenshot (two panels + text)
            exp_path = save_explain_figure(fp, model, out_explain_dir, label, float(p_real), expl)

            results.append({
                "frame": str(fp),
                "explain_file": str(exp_path),
                "pred_label": label,
                "p_real": float(p_real),
                "explanation": expl
            })
            print(f"[OK] {fp.name}: {label} (p_real={p_real:.3f})  | figure: {exp_path.name}")
        except Exception as e:
            print(f"[WARN] Failed on {fp.name}: {e}")
    return results

def majority_vote(results):
    """Return ('REAL' or 'FAKE', counts dict, mean_p_real)."""
    if not results:
        return "UNKNOWN", {"REAL": 0, "FAKE": 0}, float("nan")
    counts = {"REAL": 0, "FAKE": 0}
    probs = []
    for r in results:
        lbl = r.get("pred_label", "UNKNOWN")
        if lbl in counts:
            counts[lbl] += 1
        if "p_real" in r:
            probs.append(r["p_real"])
    final = "REAL" if counts["REAL"] >= counts["FAKE"] else "FAKE"
    mean_p = float(np.mean(probs)) if probs else float("nan")
    return final, counts, mean_p

def save_csv(results, out_csv_path: Path):
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame_file", "explain_file", "pred_label", "p_real", "explanation"])
        for r in results:
            w.writerow([r["frame"], r["explain_file"], r["pred_label"], f"{r['p_real']:.6f}", r["explanation"]])

# ------------- CLI -------------
def main():
    ap = argparse.ArgumentParser(
        description="Extract frames every N seconds (default 0.5s), run detector, and save composite Grad-CAM figures.")
    ap.add_argument("video", help="Path to input video (e.g., E:\\Downloads\\clip.mp4)")
    ap.add_argument("-s", "--step", type=float, default=0.5, help="Interval in seconds between frames (default: 0.5)")
    ap.add_argument("-q", "--quality", type=int, default=95, help="JPEG quality 0-100 for raw frames (default: 95)")
    args = ap.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Main output dir: folder named after the video (no extension)
    out_dir = video_path.parent / video_path.stem
    # Composite figure subfolder with the exact name you asked
    out_explain_dir = out_dir / "output frames"

    print(f"Extracting frames every {args.step}s from: {video_path}")
    print(f"Saving raw frames to: {out_dir}")
    print(f"Saving explanation figures to: {out_explain_dir}")

    frames = extract_frames_every(video_path, out_dir, step_sec=args.step, jpeg_quality=args.quality)
    if not frames:
        raise RuntimeError("No frames were saved. Aborting.")

    # Load model ONCE using your existing config in testing2.py
    model_path = t2.MODEL_PATH
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path} (from testing2.py)")
    print(f"Loading model: {model_path}")
    model = t2.load_model(model_path, compile=False) if hasattr(t2, "load_model") else None
    if model is None:
        from tensorflow.keras.models import load_model
        model = load_model(model_path, compile=False)

    # Run detector (+ save composite figures)
    print("Running detector on frames...")
    results = run_detector_on_frames(frames, model, out_explain_dir)

    # Aggregate + save CSV
    out_csv = out_dir / "frame_results.csv"
    save_csv(results, out_csv)

    final_label, counts, mean_p = majority_vote(results)
    print("\n=== SUMMARY ===")
    print(f"Frames analyzed: {len(results)}")
    print(f"REAL: {counts['REAL']} | FAKE: {counts['FAKE']}")
    print(f"Mean p_real: {mean_p:.3f}")
    print(f"Video-level verdict (majority): {final_label}")
    print(f"Per-frame CSV: {out_csv}")

if __name__ == "__main__":
    main()
