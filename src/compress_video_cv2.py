import argparse
from pathlib import Path

import cv2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compress MP4 video with OpenCV.")
    parser.add_argument("--input", required=True, help="Input video path.")
    parser.add_argument("--output", required=True, help="Output video path.")
    parser.add_argument("--width", type=int, default=1280, help="Output width.")
    parser.add_argument("--height", type=int, default=720, help="Output height.")
    parser.add_argument("--fps", type=float, default=24.0, help="Output FPS.")
    parser.add_argument("--codec", default="mp4v", help="FourCC codec, e.g. mp4v.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    src = Path(args.input)
    dst = Path(args.output)

    if not src.exists():
        raise FileNotFoundError(f"Input video not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {src}")

    fps_in = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps_in <= 0.0:
        fps_in = args.fps
    sample_every = max(1, int(round(fps_in / max(args.fps, 1e-6))))

    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    out = cv2.VideoWriter(str(dst), fourcc, args.fps, (args.width, args.height))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(
            f"Failed to create output video with codec '{args.codec}'. "
            "Try a different codec (e.g., mp4v)."
        )

    idx = 0
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % sample_every == 0:
            resized = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)
            out.write(resized)
            written += 1
        idx += 1
        if idx % 300 == 0:
            print(f"[Compress] read={idx}/{frame_count} written={written}")

    cap.release()
    out.release()

    size_mb = dst.stat().st_size / (1024.0 * 1024.0)
    print(f"[Compress] Done: {dst}")
    print(f"[Compress] Output size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
