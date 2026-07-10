"""Benchmark SAM 3 (Hugging Face) as a zero-shot text-prompted football detector.

Answers "is SAM 3 good enough to pre-label frames for the dataset?" on the same
two axes `scripts/bench_models.py` uses for the YOLO bases, so the rows can sit
next to each other:

  * accuracy  -> zero-shot mAP50-95 / mAP50 on the football val split, with the
                 ball-class AP called out separately (the tiny ball is the real
                 bottleneck; the fine-tuned YOLOv8x only reaches ~0.40 recall).
  * latency   -> median / p90 for a full per-frame detection of all four
                 classes, on calibration-resolution (1920x1080) frames, timed
                 the way `Tracker.track_frame` times `model.predict`.

SAM 3 is prompted with one text concept per forward pass, so the four classes
are covered by encoding the frame once (`get_vision_features`) and re-running
only the cheap text/decoder half per concept. That is the honest equivalent of
YOLO's single `predict()` call: one image encode, all classes out.

SAM 3 is inference-only here -- there is no fine-tuning path, and at ~840M
params it is not a live-detector candidate. The number that decides its fate is
the zero-shot ball AP.

Setup
-----
    uv sync --extra sam3          # transformers + accelerate
    hf auth login                 # facebook/sam3 is a gated repo (accept the
                                  # licence at huggingface.co/facebook/sam3)

Examples
--------
Accuracy + latency with the default prompts:
    python scripts/bench_sam3_hf.py \
        --data models/football-players-detection-1/data.yaml \
        --source data/test.mp4 --out bench_sam3.json

Prompt sweep for the ball concept (accuracy only, quick):
    for p in ball "soccer ball" football; do
        python scripts/bench_sam3_hf.py --skip-latency \
            --data models/football-players-detection-1/data.yaml \
            --prompts "ball=$p"
    done

Latency only, no dataset needed (falls back to synthetic frames):
    python scripts/bench_sam3_hf.py --skip-accuracy --source data/test.mp4

Check the accuracy harness itself (scores the ground truth against itself,
must print ~1.0; needs no model download):
    python scripts/bench_sam3_hf.py --selftest-map \
        --data models/football-players-detection-1/data.yaml
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import supervision as sv
import yaml
from PIL import Image
from supervision.metrics import MeanAveragePrecision, MetricTarget

# `scripts/` is not a package; reuse the frame loader from the YOLO benchmark so
# both benchmarks time the exact same frames at the same calibration size.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_models import load_frames  # noqa: E402

# Text concept per class name. Prompt wording is the main accuracy variable for
# a zero-shot detector, so every one of these is overridable via --prompts.
DEFAULT_PROMPTS = {
    "ball": "ball",
    "goalkeeper": "goalkeeper",
    "player": "football player",
    "referee": "referee",
}

# data.yaml class order for the Roboflow football dataset, used when no --data
# is given (latency-only runs still need class ids to tag detections with).
DEFAULT_CLASS_NAMES = ["ball", "goalkeeper", "player", "referee"]

GATED_REPO_HELP = """
`{model_id}` is a gated Hugging Face repo. One-time setup:
  1. Accept the licence at https://huggingface.co/{model_id}
  2. `hf auth login` (or export HF_TOKEN=<your token>)
"""


def _log(msg):
    print(msg, flush=True)


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------


def resolve_device(requested):
    """Pick cuda > mps > cpu unless the caller named a device."""
    import torch

    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(requested, device):
    """fp16 on accelerators, fp32 on CPU (fp16 on CPU is slower, not faster)."""
    import torch

    if requested and requested != "auto":
        return getattr(torch, requested)
    return torch.float32 if device == "cpu" else torch.float16


def load_model(model_id, device, dtype, image_size=None):
    """Load SAM 3 + its processor, translating the gated-repo error into help."""
    try:
        from transformers import Sam3Model, Sam3Processor
    except ImportError as e:
        raise SystemExit(
            "SAM 3 needs transformers + accelerate, which are an optional extra:\n"
            "    uv sync --extra sam3"
        ) from e

    model_kwargs = {}
    processor_kwargs = {}
    if image_size:
        # Documented escape hatch for faster/cheaper inference. The model is
        # trained at 1008px; smaller inputs trade accuracy for speed.
        from transformers import Sam3Config

        config = Sam3Config.from_pretrained(model_id)
        config.image_size = image_size
        model_kwargs["config"] = config
        processor_kwargs["size"] = {"height": image_size, "width": image_size}

    try:
        try:
            model = Sam3Model.from_pretrained(model_id, dtype=dtype, **model_kwargs)
        except TypeError:  # transformers < 4.56 spells it torch_dtype
            model = Sam3Model.from_pretrained(model_id, torch_dtype=dtype, **model_kwargs)
        processor = Sam3Processor.from_pretrained(model_id, **processor_kwargs)
    except Exception as e:  # noqa: BLE001 - the gated case needs a human hint
        msg = str(e)
        if "gated" in msg.lower() or "401" in msg or "restricted" in msg.lower():
            raise SystemExit(GATED_REPO_HELP.format(model_id=model_id)) from e
        raise

    model.to(device)
    model.eval()
    return model, processor


def _target_sizes(inputs):
    """`original_sizes` comes back as a tensor with return_tensors='pt'."""
    sizes = inputs.get("original_sizes")
    return sizes.tolist() if hasattr(sizes, "tolist") else sizes


class Sam3Detector:
    """Text-prompted detector with a YOLO-shaped `detect()` -> sv.Detections."""

    def __init__(self, model, processor, prompts, conf, share_vision=True):
        self.model = model
        self.processor = processor
        self.prompts = prompts  # {class_id: text concept}
        self.conf = conf
        self.share_vision = share_vision

    def _postprocess(self, outputs, target_sizes):
        return self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.conf,
            mask_threshold=0.5,
            target_sizes=target_sizes,
        )

    def _forward_shared_vision(self, image):
        """One image encode, one cheap text/decoder pass per concept."""
        import torch

        img_inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)
        target_sizes = _target_sizes(img_inputs)
        with torch.no_grad():
            vision_embeds = self.model.get_vision_features(pixel_values=img_inputs.pixel_values)

        per_concept = []
        for text in self.prompts.values():
            text_inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model(
                    vision_embeds=vision_embeds,
                    input_ids=text_inputs["input_ids"],
                    attention_mask=text_inputs.get("attention_mask"),
                )
            per_concept.append(self._postprocess(outputs, target_sizes)[0])
        return per_concept

    def _forward_batched(self, image):
        """Same frame batched once per concept; re-encodes the image N times."""
        import torch

        texts = list(self.prompts.values())
        inputs = self.processor(
            images=[image] * len(texts), text=texts, return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return self._postprocess(outputs, _target_sizes(inputs))

    def detect(self, frame_bgr):
        """Detect every prompted concept in one BGR frame."""
        image = Image.fromarray(frame_bgr[:, :, ::-1])  # cv2 BGR -> RGB
        forward = self._forward_shared_vision if self.share_vision else self._forward_batched
        per_concept = forward(image)

        boxes, scores, class_ids = [], [], []
        for class_id, result in zip(self.prompts, per_concept):
            found = np.asarray(result["boxes"].float().cpu(), dtype=np.float32).reshape(-1, 4)
            if not len(found):
                continue
            boxes.append(found)
            scores.append(np.asarray(result["scores"].float().cpu(), dtype=np.float32).ravel())
            class_ids.append(np.full(len(found), class_id, dtype=int))

        if not boxes:
            return sv.Detections.empty()
        return sv.Detections(
            xyxy=np.concatenate(boxes),
            confidence=np.concatenate(scores),
            class_id=np.concatenate(class_ids),
        )


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------


def resolve_split(data_yaml, split):
    """Return (images_dir, labels_dir, class_names) for `split` in a data.yaml.

    Roboflow writes the split paths with a leading `../` that is wrong for this
    layout (README 6.2), and `val` lives in a directory called `valid`. Try the
    plausible spellings rather than making the caller sed the yaml first."""
    yaml_path = Path(data_yaml).resolve()
    if not yaml_path.is_file():
        raise SystemExit(
            f"No data.yaml at {yaml_path}.\n"
            "Download the dataset first -- see README 6.2 'Download the training dataset' --\n"
            "or pass --skip-accuracy to benchmark latency only."
        )
    spec = yaml.safe_load(yaml_path.read_text())
    names = spec.get("names")
    if isinstance(names, dict):  # {0: ball, ...}
        names = [names[i] for i in sorted(names)]

    root = yaml_path.parent
    if spec.get("path"):
        root = (root / spec["path"]).resolve()

    entry = spec.get(split)
    candidates = []
    if entry:
        entry = entry[0] if isinstance(entry, list) else entry
        candidates += [Path(entry), root / entry, root / str(entry).lstrip("./").lstrip("../")]
    # `val` is stored under valid/ in the Roboflow export.
    for guess in ({"val": "valid"}.get(split, split), split):
        candidates.append(root / guess / "images")

    images_dir = next((c for c in candidates if c.is_dir()), None)
    if images_dir is None:
        raise SystemExit(
            f"No images dir for split '{split}' under {root}.\n"
            "Download the dataset first -- see README 6.2 'Download the training dataset'."
        )

    # Only the trailing `images` segment is the split's image dir; a parent
    # directory called `images` must survive.
    labels_dir = images_dir.parent / "labels" if images_dir.name == "images" else images_dir
    if not labels_dir.is_dir():
        raise SystemExit(f"Found images at {images_dir} but no labels dir at {labels_dir}")
    return images_dir, labels_dir, names or DEFAULT_CLASS_NAMES


def load_yolo_labels(label_path, width, height):
    """YOLO txt (normalized `cls cx cy w h`, or a polygon) -> absolute xyxy."""
    if not label_path.exists():
        return sv.Detections.empty()

    boxes, class_ids = [], []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        vals = [float(v) for v in parts[1:]]
        if len(vals) == 4:
            cx, cy, w, h = vals
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        else:  # segmentation polygon -> its bounding box
            xs, ys = vals[0::2], vals[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        boxes.append([x1 * width, y1 * height, x2 * width, y2 * height])
        class_ids.append(class_id)

    if not boxes:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.asarray(boxes, dtype=np.float32),
        class_id=np.asarray(class_ids, dtype=int),
    )


def iter_split(images_dir, labels_dir, max_images=None):
    """Yield (image_path, ground_truth Detections) pairs, ordered."""
    paths = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if max_images:
        paths = paths[:max_images]
    for path in paths:
        with Image.open(path) as im:  # lazy: reads the header, not the pixels
            width, height = im.size
        yield path, load_yolo_labels(labels_dir / f"{path.stem}.txt", width, height)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def summarize_map(predictions, targets, class_names):
    """supervision mAP -> the same keys `bench_models.eval_map` reports."""
    result = (
        MeanAveragePrecision(metric_target=MetricTarget.BOXES)
        .update(predictions, targets)
        .compute()
    )

    # ap_per_class is (n_matched_classes, n_iou_thresholds) over IoU 0.50:0.95.
    ap_per_class = np.asarray(result.ap_per_class, dtype=float)
    matched = list(np.asarray(result.matched_classes).ravel())
    per_class = {}
    for row, class_id in enumerate(matched):
        name = class_names[class_id] if class_id < len(class_names) else str(class_id)
        per_class[name] = {
            "map50_95": round(float(ap_per_class[row].mean()), 4),
            "map50": round(float(ap_per_class[row][0]), 4),
        }

    ball = per_class.get("ball")
    return {
        "map50_95": round(float(result.map50_95), 4),
        "map50": round(float(result.map50), 4),
        "ball_map50_95": None if ball is None else ball["map50_95"],
        "per_class": per_class,
        "images": len(targets),
    }


def eval_zero_shot(detector, images_dir, labels_dir, class_names, max_images=None):
    """Run `detector` over the split and score it against the YOLO labels."""
    import cv2

    predictions, targets = [], []
    for i, (path, ground_truth) in enumerate(iter_split(images_dir, labels_dir, max_images)):
        frame = cv2.imread(str(path))
        if frame is None:
            _log(f"  unreadable image, skipping: {path.name}")
            continue
        predictions.append(detector.detect(frame))
        targets.append(ground_truth)
        if (i + 1) % 10 == 0:
            _log(f"  {i + 1} images...")

    if not targets:
        raise SystemExit(f"No readable images in {images_dir}")
    return summarize_map(predictions, targets, class_names)


def selftest_map(images_dir, labels_dir, class_names, max_images=None):
    """Score the ground truth against itself: the harness must report ~1.0."""
    predictions, targets = [], []
    for _, ground_truth in iter_split(images_dir, labels_dir, max_images):
        as_prediction = sv.Detections(
            xyxy=ground_truth.xyxy.copy(),
            class_id=ground_truth.class_id.copy(),
            confidence=np.ones(len(ground_truth), dtype=np.float32),
        )
        predictions.append(as_prediction)
        targets.append(ground_truth)
    return summarize_map(predictions, targets, class_names)


def bench_latency(detector, frames, warmup=5):
    """Time a full per-frame, all-concepts detect() -- never one concept x 4."""
    for frame in frames[:warmup]:
        detector.detect(frame)

    times_ms = []
    for frame in frames:
        t0 = time.perf_counter()
        detector.detect(frame)
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


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------


def build_prompts(overrides, class_names):
    """Merge `--prompts name=text` over the defaults; return {class_id: text}."""
    by_name = dict(DEFAULT_PROMPTS)
    for override in overrides or []:
        if "=" not in override:
            raise SystemExit(f"--prompts expects name=text pairs, got '{override}'")
        name, text = override.split("=", 1)
        if name not in class_names:
            raise SystemExit(f"Unknown class '{name}'; dataset classes: {class_names}")
        by_name[name] = text

    prompts = {}
    for class_id, name in enumerate(class_names):
        if name in by_name:
            prompts[class_id] = by_name[name]
    if not prompts:
        raise SystemExit(f"No prompts matched the dataset classes {class_names}")
    return prompts


def main():
    ap = argparse.ArgumentParser(description="Zero-shot SAM 3 benchmark for football detection")
    ap.add_argument("--model-id", default="facebook/sam3",
                    help="HF repo id (e.g. yonigozlan/sam3-litetext-s0 for the lite variant)")
    ap.add_argument("--data", default="models/football-players-detection-1/data.yaml",
                    help="data.yaml for zero-shot mAP (with --skip-accuracy it is unused)")
    ap.add_argument("--split", default="val", help="data.yaml split key to score")
    ap.add_argument("--max-images", type=int, default=None,
                    help="Cap the scored images for quick iterations")
    ap.add_argument("--prompts", nargs="+", default=None, metavar="NAME=TEXT",
                    help='Override a class prompt, e.g. --prompts ball="soccer ball"')
    ap.add_argument("--conf", type=float, default=0.1,
                    help="Score threshold (matches Tracker's conf=0.1)")
    ap.add_argument("--source", default="data/test.mp4",
                    help="Video/webcam-index for latency frames (synthetic if unreadable)")
    ap.add_argument("--latency-frames", type=int, default=50,
                    help="Fewer than the YOLO benchmark's 200 -- SAM 3 is slow")
    ap.add_argument("--calibration-width", type=int, default=1920)
    ap.add_argument("--calibration-height", type=int, default=1080)
    ap.add_argument("--image-size", type=int, default=None,
                    help="Override SAM 3's 1008px working resolution (faster, less accurate)")
    ap.add_argument("--device", default="auto", help="'cuda' / 'mps' / 'cpu' (default: best available)")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--batched-prompts", action="store_true",
                    help="Batch the frame once per concept instead of sharing one image encode")
    ap.add_argument("--skip-accuracy", action="store_true")
    ap.add_argument("--skip-latency", action="store_true")
    ap.add_argument("--selftest-map", action="store_true",
                    help="Score the ground truth against itself (~1.0) and exit; loads no model")
    ap.add_argument("--out", default=None, help="Write results JSON to this path")
    args = ap.parse_args()

    if args.skip_accuracy and args.skip_latency:
        ap.error("--skip-accuracy and --skip-latency leave nothing to run")

    # The harness self-test exercises the dataset + mAP path with no model at all.
    if args.selftest_map:
        images_dir, labels_dir, class_names = resolve_split(args.data, args.split)
        acc = selftest_map(images_dir, labels_dir, class_names, args.max_images)
        _log(f"ground-truth self-test over {acc['images']} images: "
             f"mAP50-95={acc['map50_95']}  mAP50={acc['map50']}")
        if acc["map50_95"] < 0.99:
            raise SystemExit("self-test failed: scoring the labels against themselves is not ~1.0")
        _log("self-test passed")
        return

    class_names = DEFAULT_CLASS_NAMES
    images_dir = labels_dir = None
    if not args.skip_accuracy:
        images_dir, labels_dir, class_names = resolve_split(args.data, args.split)

    prompts = build_prompts(args.prompts, class_names)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    _log(f"=== {args.model_id} on {device} ({str(dtype).split('.')[-1]}) ===")
    for class_id, text in prompts.items():
        _log(f"  {class_names[class_id]:<12} <- {text!r}")

    model, processor = load_model(args.model_id, device, dtype, args.image_size)
    detector = Sam3Detector(model, processor, prompts, args.conf,
                            share_vision=not args.batched_prompts)

    row = {
        "model": args.model_id,
        "device": device,
        "dtype": str(dtype).split(".")[-1],
        "conf": args.conf,
        "image_size": args.image_size or 1008,
        "prompts": {class_names[c]: t for c, t in prompts.items()},
    }

    if not args.skip_accuracy:
        _log(f"\nzero-shot mAP on '{args.split}' ({images_dir})")
        try:
            row["accuracy"] = eval_zero_shot(detector, images_dir, labels_dir,
                                             class_names, args.max_images)
            acc = row["accuracy"]
            _log(f"  mAP50-95={acc['map50_95']}  mAP50={acc['map50']}  "
                 f"ball_mAP50-95={acc['ball_map50_95']}")
            for name, scores in acc["per_class"].items():
                _log(f"    {name:<12} mAP50-95={scores['map50_95']:<8} mAP50={scores['map50']}")
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001 - report and still time the model
            # Name the type: `str(KeyError('x'))` is just "'x'", which tells a
            # reader nothing about what went wrong.
            row["accuracy"] = {"error": f"{type(e).__name__}: {e}"}
            _log(f"  accuracy failed: {type(e).__name__}: {e}")

    if not args.skip_latency:
        frames = load_frames(args.source, args.latency_frames,
                             (args.calibration_width, args.calibration_height))
        _log(f"\nlatency over {len(frames)} frames "
             f"({args.calibration_width}x{args.calibration_height}, {len(prompts)} concepts/frame)")
        try:
            row["latency"] = bench_latency(detector, frames)
            lat = row["latency"]
            _log(f"  median={lat['median_ms']}ms  p90={lat['p90_ms']}ms  "
                 f"~{lat['fps_median']} FPS over {lat['frames']} frames")
        except Exception as e:  # noqa: BLE001
            row["latency"] = {"error": f"{type(e).__name__}: {e}"}
            _log(f"  latency failed: {type(e).__name__}: {e}")

    acc, lat = row.get("accuracy") or {}, row.get("latency") or {}
    _log("\n" + "=" * 66)
    _log(f"{'model':<24}{'mAP50-95':>10}{'ball mAP':>10}{'median ms':>12}{'FPS':>10}")
    _log("-" * 66)
    _log(f"{Path(args.model_id).name:<24}{str(acc.get('map50_95', '-')):>10}"
         f"{str(acc.get('ball_map50_95', '-')):>10}{str(lat.get('median_ms', '-')):>12}"
         f"{str(lat.get('fps_median', '-')):>10}")
    _log("=" * 66)

    if args.out:
        Path(args.out).write_text(json.dumps([row], indent=2))
        _log(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
