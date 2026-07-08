"""Benchmark YOLO detector backbones for this project.

Answers the practical question "is it worth swapping the yolov8x base for a
YOLO11/12 one?" by comparing candidates on the two axes that matter here:

  * accuracy  -> val mAP on the football dataset (with the ball-class AP called
                 out separately, since the tiny ball is the real bottleneck)
  * latency   -> median / p90 per-frame `model.predict(frame, conf=...)` time,
                 measured exactly how `Tracker.track_frame` calls it in the live
                 pipeline, so the FPS number reflects real streaming throughput.

Because a bare COCO base (e.g. yolo11m.pt) has the wrong classes, val mAP is
only meaningful on weights fine-tuned on the football data. Pass --train to
fine-tune each base first, or point --models at already-trained best.pt files.
Latency needs no training and works on any base.

Examples
--------
Latency-only shootout on the pretrained bases (downloads them on first use):
    python scripts/bench_models.py \
        --models yolov8x.pt yolo11m.pt yolo11l.pt yolo11x.pt \
        --source data/test.mp4 --latency-frames 200

Full accuracy + latency, fine-tuning each base on the football set:
    python scripts/bench_models.py \
        --models yolov8x.pt yolo11m.pt yolo11l.pt \
        --data models/football-players-detection-1/data.yaml \
        --train --epochs 100 --imgsz 640 --device 0 \
        --source data/test.mp4 --out bench_results.json

Compare already-trained weights (skip training):
    python scripts/bench_models.py \
        --models models/football_yolo_v8x/weights/best.pt \
                 models/football_yolo_11l/weights/best.pt \
        --data models/football-players-detection-1/data.yaml \
        --source data/test.mp4
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np


def _log(msg):
    print(msg, flush=True)


def fine_tune(base, data, epochs, imgsz, device, project, name):
    """Fine-tune `base` on `data`; return the path to the best.pt produced."""
    from ultralytics import YOLO

    _log(f"  training {base} for {epochs} epochs (imgsz={imgsz}, device={device})...")
    model = YOLO(base)
    model.train(data=data, epochs=epochs, imgsz=imgsz, device=device,
                project=project, name=name, exist_ok=True, verbose=False)
    # Read the actual best.pt path from the trainer rather than reconstructing
    # it: Ultralytics resolves the save dir via its runs_dir setting, so a
    # relative `project` can land under runs/detect/ instead of where we'd guess.
    best = getattr(model.trainer, "best", None)
    if best is None or not Path(best).exists():
        best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    return str(best)


def eval_map(weights, data, imgsz, device):
    """Run val on `weights`; return dict of mAP metrics incl. per-class ball AP."""
    from ultralytics import YOLO

    model = YOLO(weights)
    res = model.val(data=data, imgsz=imgsz, device=device, verbose=False)
    box = res.box

    names = model.names  # {idx: name}
    ball_ap = None
    ball_idx = next((i for i, n in names.items() if str(n).lower() == "ball"), None)
    if ball_idx is not None:
        try:
            # box.maps is per-class mAP50-95, indexed by class id present in val.
            ball_ap = float(box.maps[ball_idx])
        except (IndexError, TypeError):
            ball_ap = None

    return {
        "map50_95": round(float(box.map), 4),
        "map50": round(float(box.map50), 4),
        "ball_map50_95": None if ball_ap is None else round(ball_ap, 4),
    }


def load_frames(source, want, calibration_size):
    """Return a list of BGR frames to benchmark predict() on.

    Mirrors the live pipeline: every frame is resized to the calibration
    resolution (1920x1080) before it reaches the model. Falls back to synthetic
    noise frames when no readable source is given, so latency can still be
    measured without a video."""
    frames = []
    if source is not None and Path(str(source)).exists() or (
            isinstance(source, str) and source.isdigit()):
        cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
        while len(frames) < want:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames.append(cv2.resize(frame, calibration_size))
        cap.release()

    if not frames:
        _log("  no readable --source; using synthetic frames for latency only")
        rng = np.random.RandomState(0)
        frames = [rng.randint(0, 255, (calibration_size[1], calibration_size[0], 3),
                               dtype=np.uint8) for _ in range(min(want, 60))]
    return frames


def bench_latency(weights, frames, conf, imgsz, device, warmup=5):
    """Time single-frame predict() exactly like Tracker.track_frame does."""
    from ultralytics import YOLO

    model = YOLO(weights)

    # Warm up (first calls include lazy CUDA init / graph build).
    for i in range(min(warmup, len(frames))):
        model.predict(frames[i], conf=conf, imgsz=imgsz, device=device, verbose=False)

    times_ms = []
    for frame in frames:
        t0 = time.perf_counter()
        model.predict(frame, conf=conf, imgsz=imgsz, device=device, verbose=False)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    times_ms.sort()
    median = statistics.median(times_ms)
    p90 = times_ms[min(len(times_ms) - 1, int(0.9 * len(times_ms)))]
    return {
        "frames": len(times_ms),
        "median_ms": round(median, 2),
        "p90_ms": round(p90, 2),
        "mean_ms": round(statistics.fmean(times_ms), 2),
        "fps_median": round(1000.0 / median, 2) if median > 0 else None,
    }


def main():
    ap = argparse.ArgumentParser(description="Benchmark YOLO bases for football detection")
    ap.add_argument("--models", nargs="+", required=True,
                    help="Model bases or trained .pt weights (e.g. yolov8x.pt yolo11m.pt)")
    ap.add_argument("--data", default=None,
                    help="data.yaml for val mAP (omit to skip accuracy, latency-only)")
    ap.add_argument("--source", default="data/test.mp4",
                    help="Video/webcam-index for latency frames (synthetic if unreadable)")
    ap.add_argument("--train", action="store_true",
                    help="Fine-tune each base on --data before evaluating")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640,
                    help="imgsz for train/val/predict (try 1280 for better ball recall)")
    ap.add_argument("--device", default="0", help="'0' for GPU, 'cpu' otherwise")
    ap.add_argument("--conf", type=float, default=0.1,
                    help="predict confidence (matches Tracker's conf=0.1)")
    ap.add_argument("--latency-frames", type=int, default=200)
    ap.add_argument("--calibration-width", type=int, default=1920)
    ap.add_argument("--calibration-height", type=int, default=1080)
    ap.add_argument("--out", default=None, help="Write results JSON to this path")
    args = ap.parse_args()

    calibration_size = (args.calibration_width, args.calibration_height)
    frames = load_frames(args.source, args.latency_frames, calibration_size)

    results = []
    for base in args.models:
        _log(f"\n=== {base} ===")
        row = {"model": base}

        weights = base
        if args.train:
            if not args.data:
                ap.error("--train requires --data")
            name = "bench_" + Path(base).stem
            weights = fine_tune(base, args.data, args.epochs, args.imgsz,
                                args.device, project="models", name=name)
            row["trained_weights"] = weights

        if args.data:
            try:
                row["accuracy"] = eval_map(weights, args.data, args.imgsz, args.device)
                _log(f"  mAP50-95={row['accuracy']['map50_95']}  "
                     f"mAP50={row['accuracy']['map50']}  "
                     f"ball_mAP50-95={row['accuracy']['ball_map50_95']}")
            except Exception as e:  # noqa: BLE001 - report and keep going
                row["accuracy"] = {"error": str(e)}
                _log(f"  val skipped/failed: {e}")

        try:
            row["latency"] = bench_latency(weights, frames, args.conf,
                                           args.imgsz, args.device)
            lat = row["latency"]
            _log(f"  latency: median={lat['median_ms']}ms  p90={lat['p90_ms']}ms  "
                 f"~{lat['fps_median']} FPS over {lat['frames']} frames")
        except Exception as e:  # noqa: BLE001
            row["latency"] = {"error": str(e)}
            _log(f"  latency failed: {e}")

        results.append(row)

    # Summary table.
    def _label(path):
        # For a trained .../<run>/weights/best.pt show <run>; else the stem.
        p = Path(path)
        if p.name.endswith(".pt") and p.parent.name == "weights":
            return p.parent.parent.name
        return p.stem

    _log("\n" + "=" * 66)
    _log(f"{'model':<24}{'mAP50-95':>10}{'ball mAP':>10}{'median ms':>12}{'FPS':>10}")
    _log("-" * 66)
    for r in results:
        acc = r.get("accuracy") or {}
        lat = r.get("latency") or {}
        m = acc.get("map50_95", "-")
        b = acc.get("ball_map50_95", "-")
        ms = lat.get("median_ms", "-")
        fps = lat.get("fps_median", "-")
        _log(f"{_label(r['model']):<24}{str(m):>10}{str(b):>10}{str(ms):>12}{str(fps):>10}")
    _log("=" * 66)

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        _log(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
