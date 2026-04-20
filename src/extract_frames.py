import argparse
from pathlib import Path
import cv2


def main():
    parser = argparse.ArgumentParser(description="Extract frames from a single video.")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--out", required=True, help="Output folder for extracted frames")
    parser.add_argument("--every", type=int, default=1, help="Save every N frames (default: 1)")
    args = parser.parse_args()

    video_path = Path(args.video)
    out_dir = Path(args.out)
    every_n_frames = max(1, int(args.every))

    if not video_path.is_file():
        raise RuntimeError(f"Video not found: {video_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_n_frames == 0:
            out_file = out_dir / f"frame_{saved:06d}.jpg"
            ok = cv2.imwrite(str(out_file), frame)
            if ok:
                saved += 1

        frame_idx += 1

    cap.release()
    print(f"Saved {saved} frames to {out_dir} (every {every_n_frames} frames)")

    if saved == 0:
        print(" Warning: 0 frames saved. Check video decode / --every value.")


if __name__ == "__main__":
    main()
