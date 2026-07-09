"""Visual live-stream detection preview (detection-only).

Runs YOLO + ByteTrack on a live source and draws bounding boxes in an OpenCV
window. Use this for a quick, lightweight detection check.

For the full live pipeline (teams, possession, homography) with periodic
visual checks, prefer::

    python main_live.py --source <url> --no-ws --preview --preview-every 30

    python live_preview.py --source data/test.mp4
    python live_preview.py --source https://example.com/stream.m3u8
    python live_preview.py --source 0          # webcam
    python live_preview.py --source <url> --save output_videos/live_preview.avi
"""

import argparse
import time

import cv2

from live import ResilientCapture
from trackers import Tracker


def _draw_tracks(tracker, frame, tracks):
    """Overlay detections using the same markers as the offline pipeline."""
    out = frame.copy()
    for track_id, player in tracks["players"].items():
        out = tracker.draw_ellipse(out, player["bbox"], (0, 0, 255), track_id)
    for track_id, referee in tracks["referees"].items():
        out = tracker.draw_ellipse(out, referee["bbox"], (0, 255, 255), track_id)
    for _, ball in tracks["ball"].items():
        out = tracker.draw_traingle(out, ball["bbox"], (0, 255, 0))
    return out


def _overlay_counts(frame, tracks, fps):
    n_players = len(tracks["players"])
    n_refs = len(tracks["referees"])
    n_ball = len(tracks["ball"])
    lines = [
        f"players={n_players}  referees={n_refs}  ball={'yes' if n_ball else 'no'}",
        f"fps={fps:.1f}  [q] quit",
    ]
    y = 30
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 0), 1, cv2.LINE_AA)
        y += 32
    return frame


def main():
    parser = argparse.ArgumentParser(
        description="Visual YOLO detection preview for live streams")
    parser.add_argument("--source", required=True,
                        help="HLS .m3u8, RTSP, local file, or webcam index (e.g. 0)")
    parser.add_argument("--model", default="models/best.pt",
                        help="Path to fine-tuned YOLO weights")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Stop after N frames (for quick smoke tests)")
    parser.add_argument("--resize", default="1920x1080",
                        help="Resize frames before inference, WIDTHxHEIGHT "
                             "(matches live pipeline calibration; use 0x0 to skip)")
    parser.add_argument("--save", default=None,
                        help="Optional path to write an annotated AVI")
    parser.add_argument("--conf", type=float, default=None,
                        help="Override YOLO confidence threshold (default: 0.1)")
    parser.add_argument("--device", default=None,
                        help="YOLO device override (mps/cpu/cuda). Same as main_live.py.")
    parser.add_argument("--half", action="store_true",
                        help="FP16 inference.")
    parser.add_argument("--headless", action="store_true",
                        help="Skip the OpenCV window (use with --save for recording)")
    args = parser.parse_args()

    resize = None
    if args.resize and args.resize != "0x0":
        w, h = args.resize.lower().split("x")
        resize = (int(w), int(h))

    tracker = Tracker(args.model, device=args.device, half=args.half, verbose=False)
    if args.conf is not None:
        tracker.model.overrides["conf"] = args.conf

    print(f"Opening source: {args.source}")
    capture = ResilientCapture(args.source).start()

    writer = None
    processed = 0
    t0 = time.monotonic()
    fps = 0.0

    try:
        while True:
            if args.max_frames is not None and processed >= args.max_frames:
                break

            frame = capture.read(timeout=5.0)
            if frame is None:
                continue

            if resize is not None and (frame.shape[1], frame.shape[0]) != resize:
                frame = cv2.resize(frame, resize)

            tracks = tracker.track_frame(frame)
            annotated = _draw_tracks(tracker, frame, tracks)

            elapsed = time.monotonic() - t0
            if elapsed > 0:
                fps = (processed + 1) / elapsed
            annotated = _overlay_counts(annotated, tracks, fps)

            if args.save is not None:
                if writer is None:
                    h, w = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"XVID")
                    writer = cv2.VideoWriter(args.save, fourcc, 24, (w, h))
                writer.write(annotated)

            if not args.headless:
                cv2.imshow("Football detection preview", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            n_players = len(tracks["players"])
            n_ball = len(tracks["ball"])
            print(f"frame {processed:>5}  players={n_players}  "
                  f"referees={len(tracks['referees'])}  "
                  f"ball={'yes' if n_ball else 'no'}  fps={fps:.1f}")
            processed += 1

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        capture.stop()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        if args.save:
            print(f"Saved annotated video to {args.save}")


if __name__ == "__main__":
    main()