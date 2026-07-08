"""Live football-analysis entry point.

Mirrors ``main.py`` but for a live feed: pulls frames from an HLS ``.m3u8`` URL
(or RTSP / webcam index / local file), runs the same detection / tracking /
analytics incrementally, and streams per-frame JSON stats over a WebSocket for a
downstream consumer to subscribe to. No on-screen display or recording.

    python main_live.py --source <m3u8-url> --model models/best.pt --ws-port 8765
    python main_live.py --source 0   # webcam smoke test
"""

import argparse

from live import LiveFootballAnalyzer, ResilientCapture, StatsBroadcaster


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
    args = parser.parse_args()

    broadcaster = None
    if not args.no_ws:
        broadcaster = StatsBroadcaster(host=args.ws_host, port=args.ws_port).start()
        print(f"WebSocket server listening on ws://{args.ws_host}:{args.ws_port}")

    analyzer = LiveFootballAnalyzer(args.model, broadcaster=broadcaster)

    print(f"Opening source: {args.source}")
    capture = ResilientCapture(args.source).start()

    try:
        for stats in analyzer.run(capture, max_frames=args.max_frames):
            if broadcaster is None:
                print(stats)
            else:
                ball = "ball" if stats["ball"] else "no-ball"
                print(f"frame {stats['frame']:>6}  players={len(stats['players'])}  "
                      f"{ball}  camera_stable={stats['camera_stable']}  "
                      f"cut={stats['scene_cut']}")
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        capture.stop()
        if broadcaster is not None:
            broadcaster.stop()


if __name__ == "__main__":
    main()
