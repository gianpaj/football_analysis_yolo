# AGENTS.md — Football Analysis YOLO

> Guidance for AI agents working in this repository.

## Project Overview

Computer vision pipeline that converts broadcast football video into structured, frame-level analytics. It detects players, referees, and the ball; tracks entities across frames; compensates for camera motion; maps positions to field coordinates; clusters teams by jersey color; and computes per-player speed and distance.

There are two execution paths sharing the same core modules:

| Entry point | Path | Behaviour |
|---|---|---|
| `main.py` | **Offline batch** | Loads a whole recorded file, runs all stages over the full frame list, writes an annotated video to `output_videos/output_video.avi` |
| `main_live.py` | **Live streaming** | Pulls frames from HLS/RTSP/webcam, runs the same analytics incrementally one frame at a time, broadcasts per-frame JSON stats over a WebSocket (`live/` package) |

Secondary entry points: `live_preview.py` (detection-only visual check on a live source), `yolo_inference.py` (bare YOLO smoke test), `scripts/bench_models.py` (detector backbone benchmark: val mAP + per-frame latency).

---

## Architecture

The pipeline is split into single-responsibility modules:

| Module | Class | Responsibility |
|---|---|---|
| `trackers/tracker.py` | `Tracker` | YOLO detection + ByteTrack multi-object tracking; batch (`get_object_tracks`) and streaming (`track_frame`, `update_ball_position`, `reset`) APIs |
| `camera_movement_estimator/` | `CameraMovementEstimator` | Lucas-Kanade optical flow for camera compensation; batch (`get_camera_movement`) and streaming (`update`, `reset`) APIs |
| `view_transformer/` | `ViewTransformer` | Perspective homography → field coordinates (metres) |
| `speed_and_distance_estimator/` | `SpeedAndDistance_Estimator` | Per-player kinematics (km/h, cumulative metres); batch (fixed 24 FPS) and streaming (`update`/`reset`, real wall-clock timestamps) APIs |
| `team_assigner/` | `TeamAssigner` | KMeans clustering on jersey colour; `reset_player_cache()`, `is_fitted()`, `live=` flag for the live path |
| `player_ball_assigner/` | `PlayerBallAssigner` | Nearest-player ball possession heuristic |
| `utils/video_utils.py` | — | OpenCV video I/O helpers |
| `utils/bbox_utils.py` | — | Bounding-box geometry (centre, foot, distance) |

Live-only modules (`live/` package):

| Module | Class | Responsibility |
|---|---|---|
| `live/capture.py` | `ResilientCapture` | Threaded FFmpeg capture (HLS/RTSP/webcam/file); keeps only the freshest frame (`maxsize=1` queue), reopens with backoff on failure |
| `live/scene_cut.py` | `SceneCutDetector` | Broadcast hard-cut detection: HSV-histogram divergence AND large pixel difference (both required, to reject fast pans and fades) |
| `live/shot_gate.py` | `ShotTypeGate` | Debounced tactical-wide-shot classifier (player count + max box height + grass-green ratio); gates all position-derived output |
| `live/broadcaster.py` | `StatsBroadcaster` | asyncio `websockets` server on its own thread; one JSON message per processed frame |
| `live/pipeline.py` | `LiveFootballAnalyzer` | Per-frame orchestrator wiring all of the above together |
| `live/preview.py` | — | Overlay drawing for `main_live.py --preview` (window) and `--preview-file` (headless, atomic image overwrite) |

### Data flow (offline, `main.py`)

```
Video frames
  → Tracker (YOLO + ByteTrack)
  → CameraMovementEstimator (optical flow)
  → ViewTransformer (pixel → metres)
  → SpeedAndDistance_Estimator
  → TeamAssigner (KMeans)
  → PlayerBallAssigner
  → Annotation & render → output_videos/output_video.avi
```

### Data flow (live, `main_live.py`)

Per frame, in `LiveFootballAnalyzer.process_frame`:

```
ResilientCapture (freshest frame)
  → resize to 1920×1080 (keeps hardcoded pixel constants valid)
  → SceneCutDetector — on a cut: reset tracker, team cache, speed buffers,
    optical flow, shot gate; start a cooldown (camera_stable=False)
  → CameraMovementEstimator.update (every frame, cheap)
  → [every Nth frame, per --inference-every]
      Tracker.track_frame → drop detections in --ignore-region rects
      → ShotTypeGate — non-tactical shot (close-up/replay/graphic):
        null pitch positions, freeze possession, starve ball tracker
      → per-player team / transformed position / streaming speed
      → Tracker.update_ball_position (hold + constant-velocity extrapolation,
        gives up after 10 missed frames)
      → PlayerBallAssigner → running possession tallies
  → stats dict → StatsBroadcaster (WebSocket) and/or stdout
```

### Core track format

Each processing stage enriches the same nested dict:

```python
tracks = {
    "players": [   # index = frame number
        {
            track_id: {
                "bbox": [x1, y1, x2, y2],
                "position": (x, y),               # foot point
                "position_adjusted": (x, y),      # after camera compensation
                "position_transformed": (x, y),   # field coords in metres
                "speed": float,                   # km/h
                "distance": float,                # cumulative metres
                "team": int,                      # 1 or 2
                "team_color": [B, G, R],
                "has_ball": bool,
            }
        },
        ...
    ],
    "referees": [...],
    "ball": [...],
}
```

### Live stats format

The live path emits one JSON-serialisable dict per frame instead of enriching a batch dict:

```python
stats = {
    "timestamp": float,          # monotonic seconds
    "frame": int,
    "scene_cut": bool,           # hard camera cut detected this frame
    "camera_stable": bool,       # False during post-cut cooldown
    "tactical": bool,            # ShotTypeGate: usable wide shot?
    "inference": bool,           # fresh YOLO ran (False on --inference-every skip frames)
    "players": {                 # keys are str(track_id)
        "12": {
            "bbox": [x1, y1, x2, y2],
            "position": [x, y] | None,   # pitch metres; None unless camera_stable AND tactical
            "team": 1 | 2 | None,        # None until team colours fitted
            "team_color": [B, G, R] | None,
            "speed": float | None,       # km/h, from wall-clock timestamps
            "distance": float | None,    # cumulative metres
            "has_ball": bool,
        },
    },
    "referees": {...},           # bbox + position only
    "ball": {"bbox": [...], "position": [...] | None} | None,  # None when lost
    "possession": {"team": 1|2|None, "team_1_pct": float|None, "team_2_pct": float|None},
}
```

---

## Running the Pipeline

### Prerequisites

```bash
uv sync    # or: pip install ultralytics supervision opencv-python numpy pandas scikit-learn roboflow websockets
```

`pyproject.toml` also defines console scripts: `football-analysis` (offline) and `football-analysis-live` (live).

Required assets (not in repo — obtain separately):

| Path | Description |
|---|---|
| `models/best.pt` | YOLO weights fine-tuned for football |
| `data/test.mp4` | Input broadcast video |

Output is written to `output_videos/output_video.avi` (directory created at runtime).

### Basic run (offline)

```bash
python main.py
```

### Live run

```bash
# webcam smoke test, stats to stdout
python main_live.py --source 0 --no-ws

# broadcast HLS feed → WebSocket on :8765
python main_live.py --source <m3u8-url> --model models/best.pt --ws-port 8765

# fast path on Apple Silicon; visual overlay check
python main_live.py --source <url> --device mps --imgsz 640 --no-ws --preview --preview-every 15
```

Device auto-selection prefers MPS on Apple Silicon (see `_default_device()` in `main_live.py`). See `python main_live.py --help` for the full flag list (`--conf`, `--ball-conf`, `--imgsz`, `--device`, `--half`, `--inference-every`, `--ignore-region`, `--preview`, `--preview-file`).

### Quick YOLO-only inference test

```bash
python yolo_inference.py        # offline, single image/video
python live_preview.py --source 0   # detection-only overlay on a live source
```

### Benchmark detector backbones

```bash
python scripts/bench_models.py --models yolov8x.pt yolo11m.pt --source data/test.mp4
```

Compares val mAP (ball-class AP separated out) and median/p90 per-frame `predict` latency, measured exactly how `Tracker.track_frame` calls it. mAP is only meaningful on football-fine-tuned weights (`--train` or trained `best.pt` paths); latency works on any base.

---

## Stub Cache System

Tracking and optical flow are expensive. Intermediate results can be serialised to disk so later runs skip recomputation:

```python
# Read cached result (fast, reproducible)
tracks = tracker.get_object_tracks(
    frames, read_from_stub=True, stub_path="stubs/track_stubs.pkl"
)

# Recompute from scratch
tracks = tracker.get_object_tracks(
    frames, read_from_stub=False, stub_path="stubs/track_stubs.pkl"
)
```

Stub files:

| File | Contents |
|---|---|
| `stubs/track_stubs.pkl` | Cached `get_object_tracks()` output |
| `stubs/camera_movement_stub.pkl` | Cached `get_camera_movement()` output |

When modifying detection/tracking logic, always set `read_from_stub=False` to regenerate the stubs; otherwise the old cached data will be used and changes will have no visible effect.

---

## Key Configuration Values

These are currently hardcoded (or default-valued) — know where they live before tuning:

| Value | Location | Notes |
|---|---|---|
| Frame rate | `speed_and_distance_estimator/speed_and_distance_estimator.py` | Assumed 24 FPS — **offline path only**; the streaming `update()` uses real timestamps |
| Speed window (offline) | same file | 5 frames (~0.2 s at 24 FPS) |
| Speed window (streaming) | same file | 0.2 s min / 2.0 s max wall-clock span per sample window |
| Field dimensions | `view_transformer/view_transformer.py` | 68 m × 23.32 m |
| Pixel-to-field vertices | same file | Hardcoded for one specific camera angle |
| Ball assignment threshold | `player_ball_assigner/player_ball_assigner.py` | 70 px from foot point |
| Optical flow mask | `camera_movement_estimator/camera_movement_estimator.py` | Columns 0–20 and 900–1050 (sidelines) |
| Goalkeeper team override | `team_assigner/team_assigner.py` | Player ID 91 → Team 1 — **disabled on the live path** via `get_player_team(..., live=True)` |
| Detection batch size | `trackers/tracker.py` | 20 frames per YOLO call (offline `detect_frames`) |
| Detection confidence | `trackers/tracker.py` / `main_live.py` | Offline: `conf=0.1`. Live defaults: `--conf 0.25`, `--ball-conf 0.4` (stricter ball-only floor) |
| Ball hold / extrapolation | `trackers/tracker.py` | Live: hold + constant-velocity extrapolation for up to `_max_ball_hold_frames=10` missed frames; `max_ball_jump=250` px trajectory-consistency gate for candidate selection |
| Calibration resolution | `live/pipeline.py` | Live frames resized to 1920×1080 (`calibration_size`) so the pixel constants above stay valid |
| Scene-cut thresholds | `live/scene_cut.py` | Histogram correlation < 0.5 AND mean-abs-diff > 40 on a 64×64 downscale |
| Post-cut cooldown | `live/pipeline.py` | `cut_cooldown_frames=15` (`camera_stable=False` during cooldown) |
| Shot-type gate | `live/shot_gate.py` | ≥ 6 players, max box height ≤ 0.5 × frame, grass-green ratio ≥ 0.25, 3-frame debounce |
| Team-fit trigger | `live/pipeline.py` | `min_players_for_team_fit=6` on a tactical shot before KMeans team colours are fitted (once) |

---

## Development Guidelines

### Adding a new pipeline stage

1. Create a new directory with an `__init__.py` and a single class file (live-only components go in `live/` instead).
2. Expose the enriched `tracks` dict (or a new parallel data structure) from the stage's main method.
3. Wire the call into `main.py` between the existing stages that it depends on. If the stage should also run live, add a streaming (per-frame) method and wire it into `LiveFootballAnalyzer.process_frame`; if it carries inter-frame state, add it to the scene-cut reset contract (see below).
4. Add a stub-caching wrapper if the stage is expensive (offline path only).

### Modifying detection / tracking

- The `Tracker` class wraps both YOLO detection and ByteTrack; changes to either live there.
- **Both paths share `Tracker`** — the offline path uses `get_object_tracks()` (batched, stub-cached); the live path uses `track_frame()` (one frame, ByteTrack state carried on the instance). Keep their per-frame logic in sync when changing detection handling (e.g. the goalkeeper→player class remap exists in both).
- Ball positions can be sparse. Offline: `interpolate_ball_positions()` fills gaps using Pandas `interpolate()` + `bfill()` (needs future frames). Live: `update_ball_position()` instead holds/extrapolates the last bbox and reports the ball lost after 10 missed frames. Changes to ball detection must account for both.
- Live ball selection: `_select_ball()` gates candidates on trajectory consistency (within `max_ball_jump` px of a constant-velocity prediction) before picking by confidence — don't reintroduce "keep last candidate" behaviour.
- Tracker constructor knobs (`conf`, `ball_conf`, `imgsz`, `device`, `half`, `verbose`) default to the original offline behaviour; only `main_live.py` overrides them.

### Modifying team assignment

- `TeamAssigner` uses two sequential KMeans passes (player pixels → player colour → cluster players into teams). Crops are downscaled to ≤ 40×30 before KMeans for speed.
- The goalkeeper hard-code (`player_id == 91`) applies to the sample offline clip only; it is disabled via `live=True` on the live path. It must be revisited whenever a different video is used — the ID is track-specific, not universal.
- On the live path, team colours are fitted **once**, only on a tactical shot with ≥ 6 players (`is_fitted()` guards re-fitting); the per-track team cache is flushed on every scene cut via `reset_player_cache()` because ByteTrack re-issues IDs after a reset.

### The live scene-cut reset contract

Any component that carries inter-frame state **must** expose a `reset()` (or equivalent) and be wired into `LiveFootballAnalyzer._apply_scene_cut_reset()`. A broadcast hard cut invalidates optical flow, ByteTrack IDs, team-cache entries, speed buffers, and shot-gate state; a stateful component that survives a cut will silently corrupt downstream stats (e.g. a speed spike from pairing pre-cut and post-cut positions). When adding live-pipeline state, add it to the reset contract in the same change.

### Shot-type gating (live)

- `ShotTypeGate` decides per frame whether output is tactically meaningful. On non-tactical frames the orchestrator nulls pitch positions, freezes possession accumulation, and feeds the ball tracker no detections (so phantom balls on close-ups can't poison trajectory state).
- The gate is a cheap heuristic (player count, max box height fraction, grass-green HSV ratio) with a 3-frame debounce — no extra model. Tune thresholds in `ShotTypeGate.__init__` rather than adding per-frame special cases downstream.

### Camera calibration for a new video

- Edit the `pixel_vertices` and `target_vertices` arrays in `ViewTransformer.__init__()`.
- The four pixel vertices must be coplanar field corners visible in the camera frame.
- After changing these, regenerate both stubs (`read_from_stub=False`).

### Coordinate system

- **Image coords:** pixels, origin at top-left.
- **Adjusted coords:** pixels, camera motion subtracted (field-relative but still pixel scale).
- **Transformed coords:** metres, origin at the nearest field corner used for calibration.

---

## Known Limitations

- **Fixed 24 FPS assumption (offline path only)** — `main.py` will silently compute wrong speeds for other frame rates. The live path computes speed from wall-clock timestamps and is unaffected.
- **Hardcoded homography** — perspective mapping breaks for any camera angle other than the one used during calibration. The live path *gates* position-derived stats on non-tactical/unstable shots rather than recalibrating per camera; live frames are resized to 1920×1080 to keep the pixel constants valid.
- **Team colour clustering** — can fail under poor lighting or when jerseys are similar colours. On the live path, the one-time fit locks in whatever colours were visible on the first qualifying tactical shot.
- **Ball possession** — nearest-foot heuristic fails in dense contact situations; no learned interaction model.
- **Live shot gating is heuristic** — the tactical/non-tactical decision (player count + box size + green ratio) can misfire on unusual shots (e.g. goal-mouth scrambles with few visible players).
- **HLS latency** — an HLS source inherently trails the live event by ~10–40 s.
- **No evaluation metrics** — tracking stability, possession accuracy, and speed error are not quantified (`scripts/bench_models.py` covers detector mAP/latency only).

---

## Future Work (from README)

- Learned ball-possession model instead of nearest-neighbour heuristic
- Automatic homography calibration per stadium/camera (would let the live path recalibrate across camera cuts instead of gating)
- Dynamic frame-rate handling for the offline path (done for live)
- Quantitative evaluation metrics
- Service packaging (REST API + job queue + artifact store) — the live WebSocket broadcaster is a first step
- Downstream ML: event classification, tactical pattern detection
