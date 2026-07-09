"""Live football-analysis entry point.

Mirrors ``main.py`` but for a live feed: pulls frames from an HLS ``.m3u8`` URL
(or RTSP / webcam index / local file), runs the same detection / tracking /
analytics incrementally, and streams per-frame JSON stats over a WebSocket for a
downstream consumer to subscribe to.

Detection recall vs speed (imgsz is the biggest lever on a 1920-wide feed):
    --imgsz 1280   default; good recall of distant players
    --imgsz 1536   best recall, slower
    --imgsz 960    faster, drops some far players
    --inference-every 2   run YOLO every other frame to claw back FPS

Player recall is driven by --imgsz and --conf; the ball is gated separately by
--ball-conf, so keep --conf low (0.15) for player recall without phantom balls.

Named sources (presets): pass a preset name to --source instead of a URL and
the matching broadcast URL, HTTP headers, and graphics ignore-regions are
applied automatically. See SOURCE_PRESETS below / run with --list-sources.

Examples:
    python main_live.py --source rtve --model models/best.pt   # RTVE La 1 (World Cup)
    python main_live.py --source <m3u8-url> --model models/best.pt --ws-port 8765
    python main_live.py --source 0 --no-ws --preview   # visual detection check
    # headless/remote preview to a file (no display needed):
    python main_live.py --source rtve --preview-file latest-frame.jpg --preview-every 15
    # Apple Silicon, tuned for FPS:
    python main_live.py --source <url> --device mps --imgsz 960 --inference-every 2
"""

import argparse
import platform
import time

import cv2

from live import (LiveFootballAnalyzer, ResilientCapture, StatsBroadcaster,
                  build_ffmpeg_options)
from live.preview import draw_stats_frame, write_frame_atomic


# Named broadcast sources. Pass the key to --source and the URL, HTTP headers,
# and default graphics ignore-regions (scorebug / channel logo, as fractions of
# the frame) are filled in. Ignore-regions are overridden if you pass your own
# --ignore-region. Coordinates are for the standard channel graphics; retune if
# the broadcast overlay changes.
SOURCE_PRESETS = {
    # RTVE La 1 (Spain) — free-to-air public channel carrying FIFA World Cup
    # 2026 matches. Clear (non-DRM) HLS rendition, 1280x720. Note: the RTVE
    # live stream is normally geo-restricted to Spain. The DRM manifest (the
    # site's default) is FairPlay-encrypted and NOT decodable by ffmpeg/OpenCV;
    # this is the clear rendition (drop the "_drm" suffix).
    "rtve-la1": {
        "url": "https://rtvelivestream.rtve.es/rtvesec/la1/la1_main_dvr.m3u8",
        "headers": {"Referer": "https://www.rtve.es/",
                    "User-Agent": "Mozilla/5.0"},
        # Top-left scorebug ("MM:SS  FRA 0-0 MAR") and top-right "1" channel
        # logo. Both sit over crowd/stands, above pitch level, so masking them
        # can't drop real players.
        "ignore_regions": [[0.04, 0.03, 0.40, 0.13],
                           [0.90, 0.07, 1.00, 0.18]],
        "note": "RTVE La 1 — FIFA World Cup 2026 (geo-restricted to Spain)",
    },
    # Catalunya regional feed of the same channel.
    "rtve-cat": {
        "url": "https://rtvelivestream.rtve.es/rtvesec/cat/la1_cat_main_dvr.m3u8",
        "headers": {"Referer": "https://www.rtve.es/",
                    "User-Agent": "Mozilla/5.0"},
        "ignore_regions": [[0.04, 0.03, 0.40, 0.13],
                           [0.90, 0.07, 1.00, 0.18]],
        "note": "RTVE La 1 Catalunya (geo-restricted to Spain)",
    },
}
# Convenience alias.
SOURCE_PRESETS["rtve"] = SOURCE_PRESETS["rtve-la1"]


def resolve_source(source):
    """Expand a preset name in ``source`` to (url, preset_dict).
    Returns (source, None) unchanged when it isn't a known preset."""
    preset = SOURCE_PRESETS.get(source)
    if preset is None:
        return source, None
    return preset["url"], preset


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
    parser.add_argument("--source",
                        help="HLS .m3u8 URL, RTSP URL, local file, webcam index "
                             "(e.g. 0), or a preset name (e.g. 'rtve'). "
                             "Run --list-sources to see presets.")
    parser.add_argument("--list-sources", action="store_true",
                        help="List the named source presets and exit.")
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
    parser.add_argument("--conf", type=float, default=0.15,
                        help="Global detection confidence floor (default 0.15). "
                             "Kept low for player recall on wide broadcast shots; "
                             "the ball is gated separately via --ball-conf, so this "
                             "doesn't reintroduce phantom balls.")
    parser.add_argument("--ball-conf", type=float, default=0.4,
                        help="Stricter confidence floor for the ball class only "
                             "(default 0.4) — the noisiest class on a broadcast feed.")
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="Inference resolution (default 1280). Biggest lever "
                             "for detecting distant/small players on a 1920-wide "
                             "broadcast frame: 640 misses far players, 1536 catches "
                             "the most. Lower it (or use --inference-every 2) if FPS "
                             "suffers on your hardware.")
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
                        help="Show an OpenCV window with detection overlays "
                             "(needs a local display).")
    parser.add_argument("--preview-file", default=None, metavar="PATH",
                        help="Write the annotated frame to PATH every "
                             "--preview-every frames (atomic overwrite). Works "
                             "headless/over SSH — serve or sync the file to preview "
                             "remotely, e.g. --preview-file latest-frame.jpg")
    parser.add_argument("--preview-every", type=int, default=30, metavar="N",
                        help="Refresh the preview window/file every N frames "
                             "(default: 30).")
    args = parser.parse_args()

    if args.list_sources:
        print("Named source presets (pass the name to --source):\n")
        seen = {}
        for name, preset in SOURCE_PRESETS.items():
            seen.setdefault(id(preset), []).append(name)
        for names in seen.values():
            preset = SOURCE_PRESETS[names[0]]
            print(f"  {', '.join(sorted(names)):<20} {preset['note']}")
            print(f"  {'':<20} {preset['url']}\n")
        return

    if not args.source:
        parser.error("--source is required (a URL, webcam index, or preset name; "
                     "see --list-sources)")

    # Expand a preset name (e.g. 'rtve') into its URL + headers + ignore-regions.
    source, preset = resolve_source(args.source)
    ffmpeg_options = None
    ignore_regions = args.ignore_region
    if preset is not None:
        print(f"Source preset '{args.source}': {preset['note']}")
        ffmpeg_options = build_ffmpeg_options(headers=preset.get("headers"))
        # Only apply the preset's graphics masks if the user didn't pass their own.
        if not ignore_regions:
            ignore_regions = preset.get("ignore_regions")

    broadcaster = None
    if not args.no_ws:
        broadcaster = StatsBroadcaster(host=args.ws_host, port=args.ws_port).start()
        print(f"WebSocket server listening on ws://{args.ws_host}:{args.ws_port}")

    device = args.device if args.device is not None else _default_device()

    analyzer = LiveFootballAnalyzer(
        args.model,
        broadcaster=broadcaster,
        ignore_regions=ignore_regions,
        conf=args.conf,
        ball_conf=args.ball_conf,
        imgsz=args.imgsz,
        device=device,
        half=args.half,
        yolo_verbose=args.yolo_verbose,
        inference_stride=args.inference_every,
    )

    print(f"Opening source: {source}")
    print(f"Using device: {device or 'auto (ultralytics default)'}  half={args.half}")
    capture = ResilientCapture(source, ffmpeg_options=ffmpeg_options).start()

    preview_every = max(1, args.preview_every)
    want_frames = args.preview or bool(args.preview_file)

    # FPS measurement (processing rate, not source rate)
    t0 = time.monotonic()
    frame_count = 0
    last_fps_print = 0

    try:
        run_iter = analyzer.run(
            capture, max_frames=args.max_frames, return_frames=want_frames)

        for item in run_iter:
            if want_frames:
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

            if want_frames and stats["frame"] % preview_every == 0:
                annotated = draw_stats_frame(analyzer.tracker, frame, stats)

                if args.preview_file:
                    if not write_frame_atomic(args.preview_file, annotated):
                        print(f"[warn] failed to encode preview frame to "
                              f"{args.preview_file}")

                if args.preview:
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
