import cv2
import numpy as np
from pathlib import Path

# --- Input video path (change if needed)
video_path = r"E:\Downloads\WIN_20250912_12_08_24_Pro.mp4"

# --- Create output folder with video name (without extension)
video_file = Path(video_path)
out_dir = video_file.parent / video_file.stem
out_dir.mkdir(parents=True, exist_ok=True)

# --- Open video
cap = cv2.VideoCapture(str(video_file))
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {video_file}")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

# --- Number of frames to extract
n = 5 if total_frames >= 5 else total_frames

# --- Get evenly spaced frame indices
idxs = np.unique(np.rint(np.linspace(0, total_frames - 1, n)).astype(int))

print(f"Total frames in video: {total_frames}, extracting {len(idxs)} frames...")

# --- Save frames
for i, target_idx in enumerate(idxs, start=1):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(target_idx))
    ret, frame = cap.read()
    if not ret:
        continue

    out_file = out_dir / f"frame_{i:02d}.jpg"
    cv2.imwrite(str(out_file), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"Saved: {out_file}")

cap.release()
print(f"✅ Done! All frames saved in: {out_dir}")
