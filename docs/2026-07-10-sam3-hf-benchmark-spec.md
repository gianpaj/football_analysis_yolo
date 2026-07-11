# Design Spec — SAM 3 Zero-Shot Benchmark via Hugging Face

> Status: **implemented** (`scripts/bench_sam3_hf.py`) · Owner: gianpaj · Last updated: 2026-07-10
>
> Benchmarks Meta's **SAM 3** (`facebook/sam3`, via 🤗 `transformers`) as a
> zero-shot, text-prompted detector on the football dataset, producing numbers
> directly comparable to `scripts/bench_models.py` (val mAP + per-frame
> latency). Primary question: **is SAM 3 good enough to pre-label frames for
> the dataset?** Secondary: how far is it from live-pipeline latency?

---

## 1. Goals / Non-goals

### Goals

- **Zero-shot accuracy**: mAP50-95 / mAP50 on the football val split, with
  ball-class AP called out separately (same axes as `bench_models.py`).
- **Latency**: median / p90 / FPS for a full per-frame detection pass (all four
  classes), measured the same way `bench_latency` measures YOLO — warmup, then
  timed single-frame calls on calibration-resolution (1920×1080) frames.
- **Prompt tuning**: class-name→text-concept mapping is a CLI knob, so prompt
  variants ("ball" vs "soccer ball") can be compared cheaply.
- **Comparable output**: same summary-table format and a JSON schema compatible
  with `bench_results.json` rows, so results can sit next to YOLO numbers.

### Non-goals

- Fine-tuning SAM 3 (inference-only here; no training path exists in scope).
- Replacing the live detector — SAM 3 is ~840M params; it will not hit
  streaming FPS and we don't pretend otherwise.
- Mask-quality evaluation. The dataset has box labels only; we evaluate
  **boxes** (SAM 3 returns boxes alongside masks).
- Integrating SAM 3 into `Tracker` or the pre-labeling scripts (that's a
  follow-up once the numbers justify it).

---

## 2. Background & constraints

- `scripts/bench_models.py` covers YOLO-family bases; it works unchanged for
  YOLO26 but **not** for SAM 3, because Ultralytics exposes the SAM family as
  inference-only (`model.val()` and `model.train()` unsupported) and
  text-prompted detection needs a dedicated predictor class.
- The **Hugging Face route** is preferred over the Ultralytics SAM 3 wrapper:
  native `Sam3Model`/`Sam3Processor` support in `transformers`, fp16 +
  `device_map="auto"` for free, weights pulled from the Hub, and a lighter
  `SAM3-LiteText` drop-in variant that shares the same processor/API.
- Dataset: Roboflow `football-players-detection` v1 at
  `models/football-players-detection-1/` (see README §6.2 for download; **not
  committed** — the script must fail with a helpful message when absent).
  Classes in `data.yaml` order: `0=ball, 1=goalkeeper, 2=player, 3=referee`.
- `supervision` is already a project dependency → use it for mAP; only
  `transformers` + `accelerate` are new, and only for this script.
- `facebook/sam3` is a **gated repo**: requires accepting Meta's license on the
  model page and `hf auth login` (or `HF_TOKEN`) before first download.

---

## 3. Design overview

One new standalone script, **`scripts/bench_sam3_hf.py`**. No changes to
`bench_models.py` beyond (optionally) importing its `load_frames` helper.

```
bench_sam3_hf.py
  ├─ load model+processor (facebook/sam3, fp16, device auto: cuda>mps>cpu)
  ├─ CONCEPTS: {class_id: text prompt} from --prompts (default below)
  ├─ detect(frame) ── one batched forward pass, all 4 concepts
  ├─ accuracy: iterate val split → detect() → supervision mAP
  ├─ latency: load_frames() (reused from bench_models) → timed detect() loop
  └─ report: summary table + optional JSON out
```

### 3.1 Model loading

```python
model = Sam3Model.from_pretrained(args.model_id, torch_dtype=dtype, device_map="auto")
processor = Sam3Processor.from_pretrained(args.model_id)
```

- `--model-id` defaults to `facebook/sam3`; also accepts the SAM3-LiteText
  checkpoint (same processor/prompting interface) for a lighter comparison run.
- dtype: fp16 on CUDA, **fp32 on MPS and CPU** (`--dtype float16` to try the
  faster path). fp16-on-MPS is the more common way SAM 3 fails on a Mac —
  half-precision op gaps and silent NaNs — so the Mac default is the safe one.
- This machine is macOS → MPS is the expected local device; CUDA on the Ubuntu
  box used for training (see frame-extraction docs). The script sets
  `PYTORCH_ENABLE_MPS_FALLBACK=1` at import so any op MPS hasn't implemented
  (RoPE / windowed-attention corners) routes to CPU instead of crashing the run.

### 3.2 Per-frame detection (`detect`)

HF SAM 3 takes **one text concept per image** per forward pass. Batching the
same frame four times would work, but it re-runs the ViT image encoder — the
dominant cost of an 840M-param model — once per class. Instead, encode the
frame **once** and re-run only the cheap text/decoder half per concept. That
is both faster and the honest equivalent of YOLO's single `predict()` call:
one image encode, all classes out.

```python
DEFAULT_PROMPTS = {"ball": "ball", "goalkeeper": "goalkeeper",
                   "player": "football player", "referee": "referee"}

image = Image.fromarray(frame_bgr[:, :, ::-1])           # cv2 BGR → RGB
img_inputs = processor(images=image, return_tensors="pt").to(model.device)
with torch.no_grad():
    vision_embeds = model.get_vision_features(pixel_values=img_inputs.pixel_values)

for class_id, text in prompts.items():
    text_inputs = processor(text=text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(vision_embeds=vision_embeds,
                        input_ids=text_inputs["input_ids"],
                        attention_mask=text_inputs.get("attention_mask"))
    result = processor.post_process_instance_segmentation(
        outputs, threshold=args.conf, mask_threshold=0.5,
        target_sizes=img_inputs.get("original_sizes").tolist())[0]
    # result: boxes (xyxy px), scores, masks → tag each box with class_id
```

`--batched-prompts` selects the naive batch-the-frame-N-times path, so the two
can be timed against each other. `detect()` returns an `sv.Detections` — the
same object the ground-truth loader produces, which is what the metric eats.

### 3.3 Accuracy — zero-shot mAP on the val split

- Parse `--data` (`data.yaml`) → resolve the `val` images/labels dirs (mind the
  README-documented Roboflow path quirk; resolve relative to the yaml file).
- For each val image (optionally capped by `--max-images`): run `detect()`,
  read the YOLO-format label file (normalized `cls cx cy w h` → absolute xyxy).
- Feed predictions + ground truth into `supervision`'s `MeanAveragePrecision`;
  report `map50_95`, `map50`, and per-class AP for `ball` — same keys as
  `bench_models.eval_map` so rows are directly comparable.
- `--conf` default **0.1** to match `Tracker`'s confidence and the YOLO
  benchmark. mAP integrates over confidence, so a low threshold is the right
  setting for a fair accuracy number.

### 3.4 Latency

- Reuse `load_frames()` from `bench_models.py` (import via
  `sys.path.insert(0, Path(__file__).parent)` — `scripts/` is not a package):
  same 1920×1080 calibration resize, same synthetic-frame fallback.
- Same protocol as `bench_latency`: 5 warmup calls, then time `detect(frame)`
  per frame; report `median_ms`, `p90_ms`, `mean_ms`, `fps_median`, `frames`.
- The timed unit is the **full 4-concept pass** including pre/post-processing —
  never one concept × 4.

### 3.5 CLI

| flag | default | notes |
|---|---|---|
| `--model-id` | `facebook/sam3` | any SAM3-compatible HF repo (e.g. LiteText) |
| `--data` | `models/football-players-detection-1/data.yaml` | omit → skip accuracy |
| `--split` | `val` | `val` / `test` / `train` |
| `--max-images` | all | cap val images for quick iterations |
| `--prompts` | see §3.2 | `class=prompt` pairs, e.g. `--prompts ball="soccer ball"` |
| `--conf` | `0.1` | post-process score threshold |
| `--source` | `data/test.mp4` | latency frames (synthetic fallback) |
| `--latency-frames` | `50` | lower than YOLO's 200 — SAM 3 is slow |
| `--skip-latency` / `--skip-accuracy` | off | run one axis only |
| `--device` | auto | `cuda` > `mps` > `cpu` |
| `--dtype` | auto (fp16 CUDA / fp32 MPS+CPU) | `float16` to force half precision |
| `--image-size` | `1008` | SAM 3's trained resolution; lower = faster, less accurate |
| `--batched-prompts` | off | re-encode the frame per concept instead of sharing one encode |
| `--selftest-map` | off | score the labels against themselves (~1.0), load no model |
| `--out` | none | write results JSON |

### 3.6 Output

Same summary-table shape as `bench_models.py` (`model / mAP50-95 / ball mAP /
median ms / FPS`), one row per prompt-set run. JSON rows carry `model`,
`prompts`, `accuracy{...}`, `latency{...}` with the field names above.

---

## 4. Dependencies & setup

- Add an optional dependency group in `pyproject.toml` so the main install
  stays lean: `[project.optional-dependencies] sam3 = ["transformers>=4.57",
  "accelerate"]` → `uv sync --extra sam3`.
- One-time: accept the `facebook/sam3` license on huggingface.co, then
  `hf auth login` (or export `HF_TOKEN`). The script should catch the gated-repo
  error and print these exact instructions.
- Checkpoint is multi-GB (~840M params); document expected ≥16 GB unified/GPU
  memory for comfortable fp16 inference at 1008px.

---

## 5. Risks & open questions

- **Cross-concept duplicates**: "goalkeeper" and "football player" prompts can
  both fire on the keeper (separate forward passes, no cross-concept NMS).
  mAP is per-class so scoring stays well-defined, but pre-labeling use would
  need a class-priority rule (e.g. goalkeeper beats player on IoU > 0.9).
  Measure first; mitigation is out of scope for the benchmark.
- **Prompt sensitivity** is the main accuracy variable — hence `--prompts` as a
  first-class flag. Sweep at least: `ball` vs `soccer ball` vs `football`,
  `football player` vs `soccer player` vs `person`.
- **Tiny-ball failure mode**: SAM 3 runs at ~1008px internally; the ball is
  often <10px at 1080p. Expect weak ball AP; that per-class number is exactly
  what decides the pre-labeling question (current YOLOv8x ball recall ≈ 0.40).
- **Throughput realism**: benchmark timings on MPS and CUDA will differ wildly;
  JSON output should record the resolved device so numbers aren't mixed up.
- **API drift**: `Sam3Processor.post_process_instance_segmentation` signature
  is new (transformers ≥ 4.57); pin the minimum version in the extra.
- Open: is the small **val** split statistically enough for prompt sweeps, or
  should sweeps run on `train` (unused by SAM 3, so fair game) for more samples?

---

## 6. Implementation plan

1. ✅ `scripts/bench_sam3_hf.py` skeleton: CLI, model loading, `detect()`.
2. ✅ Accuracy path: data.yaml/split parsing, YOLO-label loader, supervision mAP,
   ball AP extraction. Sanity-check the harness by feeding ground truth back in
   as predictions (must score mAP ≈ 1.0) — exposed as `--selftest-map`.
3. ✅ Latency path: import `load_frames`, timing loop, device recording.
4. ✅ Summary table + JSON out; README §7.3 pointer.
5. ⬜ Runs: default prompts on val (accuracy), 50-frame latency on `data/test.mp4`,
   then a prompt sweep for `ball`. **Blocked on** the gated-repo licence and a
   local copy of the dataset (neither is checked in).

### Acceptance criteria

- ✅ Script runs latency-only with no dataset present (synthetic-frame fallback)
  and prints actionable errors for missing dataset / gated HF repo / missing
  `--extra sam3` install.
- ✅ Ground-truth self-test scores mAP50-95 ≥ 0.99 (verified: `1.0`).
- ✅ One command produces a table row comparable to a `bench_models.py` row, and
  `--out` JSON round-trips both axes with the resolved device and prompt set.

### What is and isn't verified

`detect()` was exercised against a stub implementing the documented SAM 3
surface (both prompt paths, BGR→RGB, empty-concept handling), and every call it
makes was checked against `modeling_sam3.py` / `processing_sam3.py` upstream.
The dataset + mAP path is verified end to end on a synthetic YOLO-format split.
**Nothing has been run against the real weights** — that needs the licence
acceptance and the Roboflow download, i.e. step 5 above.
