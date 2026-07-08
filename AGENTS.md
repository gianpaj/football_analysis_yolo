# AGENTS.md — Football Analysis YOLO

> Guidance for AI agents working in this repository.

## Project Overview

Computer vision pipeline that converts broadcast football video into structured, frame-level analytics. It detects players, referees, and the ball; tracks entities across frames; compensates for camera motion; maps positions to field coordinates; clusters teams by jersey color; and computes per-player speed and distance.

**Entry point:** `main.py` — orchestrates all stages end-to-end.

---

## Architecture

The pipeline is split into single-responsibility modules:

| Module | Class | Responsibility |
|---|---|---|
| `trackers/tracker.py` | `Tracker` | YOLO detection + ByteTrack multi-object tracking |
| `camera_movement_estimator/` | `CameraMovementEstimator` | Lucas-Kanade optical flow for camera compensation |
| `view_transformer/` | `ViewTransformer` | Perspective homography → field coordinates (metres) |
| `speed_and_distance_estimator/` | `SpeedAndDistance_Estimator` | Per-player kinematics (km/h, cumulative metres) |
| `team_assigner/` | `TeamAssigner` | KMeans clustering on jersey colour |
| `player_ball_assigner/` | `PlayerBallAssigner` | Nearest-player ball possession heuristic |
| `utils/video_utils.py` | — | OpenCV video I/O helpers |
| `utils/bbox_utils.py` | — | Bounding-box geometry (centre, foot, distance) |

### Data flow

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

---

## Running the Pipeline

### Prerequisites

```bash
pip install ultralytics supervision opencv-python numpy pandas scikit-learn
```

Required assets (not in repo — obtain separately):

| Path | Description |
|---|---|
| `models/best.pt` | YOLO weights fine-tuned for football |
| `data/test.mp4` | Input broadcast video |

Output is written to `output_videos/output_video.avi` (directory created at runtime).

### Basic run

```bash
python main.py
```

### Quick YOLO-only inference test

```bash
python yolo_inference.py
```

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

These are currently hardcoded — know where they live before tuning:

| Value | Location | Notes |
|---|---|---|
| Frame rate | `speed_and_distance_estimator/speed_and_distance_estimator.py` | Assumed 24 FPS |
| Speed window | same file | 5 frames (~0.2 s at 24 FPS) |
| Field dimensions | `view_transformer/view_transformer.py` | 68 m × 23.32 m |
| Pixel-to-field vertices | same file | Hardcoded for one specific camera angle |
| Ball assignment threshold | `player_ball_assigner/player_ball_assigner.py` | 70 px from foot point |
| Optical flow mask | `camera_movement_estimator/camera_movement_estimator.py` | Columns 0–20 and 900–1050 (sidelines) |
| Goalkeeper team override | `team_assigner/team_assigner.py` | Player ID 91 → Team 1 |
| Detection batch size | `trackers/tracker.py` | 20 frames per YOLO call |

---

## Development Guidelines

### Adding a new pipeline stage

1. Create a new directory with an `__init__.py` and a single class file.
2. Expose the enriched `tracks` dict (or a new parallel data structure) from the stage's main method.
3. Wire the call into `main.py` between the existing stages that it depends on.
4. Add a stub-caching wrapper if the stage is expensive.

### Modifying detection / tracking

- The `Tracker` class wraps both YOLO detection and ByteTrack; changes to either live there.
- Ball positions can be sparse — `interpolate_ball_positions()` fills gaps using Pandas `interpolate()` + `bfill()`. Any changes to ball detection must account for this.

### Modifying team assignment

- `TeamAssigner` uses two sequential KMeans passes (player pixels → player colour → cluster players into teams).
- The goalkeeper hard-code (`player_id == 91`) must be revisited whenever a different video is used — the ID is track-specific, not universal.

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

- **Fixed 24 FPS assumption** — will silently compute wrong speeds for other frame rates.
- **Hardcoded homography** — perspective mapping breaks for any camera angle other than the one used during calibration.
- **Team colour clustering** — can fail under poor lighting or when jerseys are similar colours.
- **Ball possession** — nearest-foot heuristic fails in dense contact situations; no learned interaction model.
- **No evaluation metrics** — tracking stability, possession accuracy, and speed error are not quantified.

---

## Future Work (from README)

- Learned ball-possession model instead of nearest-neighbour heuristic
- Automatic homography calibration per stadium/camera
- Dynamic frame-rate handling
- Quantitative evaluation metrics
- Service packaging (REST API + job queue + artifact store)
- Downstream ML: event classification, tactical pattern detection
