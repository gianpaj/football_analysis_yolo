"""Live football-analysis entry point.

Mirrors ``main.py`` but for a live feed: pulls frames from an HLS ``.m3u8`` URL
(or RTSP / webcam index / local file), runs the same detection / tracking /
analytics incrementally, and streams per-frame JSON stats over a WebSocket for a
downstream consumer to subscribe to.

Fast path on Apple Silicon Mac (biggest single win):
    python main_live.py --source <url> --device mps --imgsz 640 --inference-every 1

Even faster (lower update rate):
    python main_live.py --source <url> --device mps --imgsz 640 --inference-every 2

Examples:
    python main_live.py --source <m3u8-url> --model models/best.pt --ws-port 8765
    python main_live.py --source 0 --no-ws --preview   # visual detection check
    python main_live.py --source <url> --preview --preview-every 15
"""

import argparse
import platform
import time

import cv2

from live import LiveFootballAnalyzer, ResilientCapture, StatsBroadcaster
from live.preview import draw_stats_frame


def _default_device():
    """Best-effort default for YOLO device on this machine.

    Prefers 'mps' on Apple Silicon Macs (biggest live speedup).
    Falls back to letting Ultralytics auto-pick otherwise.
    """
    if platform.system() == "Darwin":
        # Apple Silicon Macs expose arm64 + MPS support in recent torch.
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
    # Let Ultralytics decide (cuda if present, else cpu, or mps if somehow missed).
    return None


def main():
    parser = argparse.ArgumentParser(description="Live football analysis over WebSocket")
    parser.add_argument("--source", required=True,
                        help="HLS .m3u8 URL, RTSP URL, local file, or webcam index (e.g. 0)")
    parser.add_argument("--model", default="models/best.pt", help="Path to YOLO model")
    parser.add_argument("--ws-host", default="0.0.0.0", help="WebSocket bind host")
    parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket port")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Stop after N frames (for testing)")
    parser.add_argument("--no-ws", action="store_true",
                        help="Disable the WebSocket server (print stats to stdout instead)")
    parser.add_argument("--ignore-region", type=float, nargs=4, action="append",
                        metavar=("X1", "Y1", "X2", "Y2"),
                        help="Graphics region to ignore, as fractions 0..1 of the "
                             "frame (e.g. --ignore-region 0.04 0.03 0.36 0.10 for a "
                             "top-left scorebug). Repeatable.")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Global detection confidence floor (default 0.25). "
                             "Raise to suppress phantom detections on a noisy feed.")
    parser.add_argument("--ball-conf", type=float, default=0.4,
                        help="Stricter confidence floor for the ball class only "
                             "(default 0.4) — the noisiest class on a broadcast feed.")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Inference resolution override (e.g. 640 or 832). "
                             "Lower values are much faster on CPU/MPS; 640 is a good "
                             "live default, 1280 helps tiny ball recall.")
    parser.add_argument("--device", default=None,
                        help="YOLO device: 'mps' (Apple Silicon), 'cpu', 'cuda', or "
                             "GPU index. Omit for auto (mps preferred on Mac).")
    parser.add_argument("--half", action="store_true",
                        help="Use FP16 half precision for inference (faster on supported devices).")
    parser.add_argument("--yolo-verbose", action="store_true",
                        help="Enable verbose YOLO logging per frame (noisy, off by default).")
    parser.add_argument("--inference-every", type=int, default=1, metavar="N",
                        help="Run full YOLO detection + analytics every N frames "
                             "(default 1 = every frame). Use 2 or 3 on slow hardware "
                             "to trade update rate for higher FPS/latency.")
    parser.add_argument("--preview", action="store_true",
                        help="Show an OpenCV window with detection overlays")
    parser.add_argument("--preview-every", type=int, default=30, metavar="N",
                        help="Refresh the preview window every N frames (default: 30)")
    args = parser.parse_args()

    broadcaster = None
    if not args.no_ws:
        broadcaster = StatsBroadcaster(host=args.ws_host, port=args.ws_port).start()
        print(f"WebSocket server listening on ws://{args.ws_host}:{args.ws_port}")

    device = args.device if args.device is not None else _default_device()

    analyzer = LiveFootballAnalyzer(
        args.model,
        broadcaster=broadcaster,
        ignore_regions=args.ignore_region,
        conf=args.conf,
        ball_conf=args.ball_conf,
        imgsz=args.imgsz,
        device=device,
        half=args.half,
        yolo_verbose=args.yolo_verbose,
        inference_stride=args.inference_every,
    )

    print(f"Opening source: {args.source}")
    print(f"Using device: {device or 'auto (ultralytics default)'}  half={args.half}")
    capture = ResilientCapture(args.source).start()

    preview_every = max(1, args.preview_every)

    # FPS measurement (processing rate, not source rate)
    t0 = time.monotonic()
    frame_count = 0
    last_fps_print = 0

    try:
        run_iter = analyzer.run(
            capture, max_frames=args.max_frames, return_frames=args.preview)

        for item in run_iter:
            if args.preview:
                stats, frame = item
            else:
                stats = item

            frame_count += 1
            elapsed = time.monotonic() - t0
            fps = frame_count / elapsed if elapsed > 0 else 0.0

            if broadcaster is None:
                print(stats)
            else:
                ball = "ball" if stats["ball"] else "no-ball"
                inf = "det" if stats.get("inference", True) else "skip"
                print(f"frame {stats['frame']:>6}  players={len(stats['players'])}  "
                      f"{ball}  tactical={stats['tactical']}  "
                      f"camera_stable={stats['camera_stable']}  "
                      f"cut={stats['scene_cut']}  {inf}  fps={fps:.1f}")

            # Periodic FPS hint even without WS
            if broadcaster is None and (frame_count - last_fps_print) >= 30:
                print(f"[info] processed {frame_count} frames @ {fps:.1f} fps (device={args.device or 'auto'})")
                last_fps_print = frame_count

            if args.preview and stats["frame"] % preview_every == 0:
                annotated = draw_stats_frame(analyzer.tracker, frame, stats)
                cv2.imshow("Live football analysis", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        capture.stop()
        if broadcaster is not None:
            broadcaster.stop()
        if args.preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
