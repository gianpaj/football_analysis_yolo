"""Per-frame live orchestrator.

Runs the same detection / tracking / analytics as the offline ``main.py`` but
incrementally, one frame at a time, and emits a JSON-serialisable stats dict per
frame (no on-screen drawing).

Speed knobs (passed from main_live.py):
- device / half: YOLO acceleration (use "mps" + half=True on Apple Silicon)
- imgsz: lower (640) = much higher FPS
- inference_stride: run detection/analytics only every N frames
"""

import time

import cv2
import numpy as np

from trackers import Tracker
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from utils import get_center_of_bbox, get_foot_position

from .scene_cut import SceneCutDetector
from .shot_gate import ShotTypeGate


def _to_native(value):
    """Best-effort convert numpy scalars/arrays to JSON-native types."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


class LiveFootballAnalyzer:
    def __init__(self, model_path, broadcaster=None,
                 calibration_size=(1920, 1080),
                 min_players_for_team_fit=6,
                 cut_cooldown_frames=15,
                 ignore_regions=None,
                 conf=0.25, ball_conf=0.4, imgsz=None,
                 device=None, half=False, yolo_verbose=False,
                 inference_stride=1):
        self.calibration_size = calibration_size
        self.min_players_for_team_fit = min_players_for_team_fit
        self.cut_cooldown_frames = cut_cooldown_frames
        self.broadcaster = broadcaster
        self.inference_stride = max(1, int(inference_stride))

        # Fixed broadcast graphics (scorebug, watermark, logos) sit in constant
        # screen regions and make the detector fire phantom balls/players there.
        # ``ignore_regions`` is a list of [x1, y1, x2, y2] as fractions (0..1) of
        # the calibration frame; any detection whose centre lands inside one is
        # dropped. Broadcast-specific, so it's opt-in (default: no masking).
        # Example for a top-left scorebug + top-right watermark:
        #     ignore_regions=[[0.04, 0.03, 0.36, 0.10], [0.72, 0.0, 1.0, 0.14]]
        self._ignore_rects_px = []
        if ignore_regions:
            width, height = calibration_size
            for x1, y1, x2, y2 in ignore_regions:
                self._ignore_rects_px.append(
                    (x1 * width, y1 * height, x2 * width, y2 * height))

        # Live feeds want a stricter confidence floor than the offline default
        # (0.1) — it's the single biggest lever against phantom detections — and
        # a stricter one still for the noisy ball class.
        self.tracker = Tracker(model_path, conf=conf, ball_conf=ball_conf,
                               imgsz=imgsz, device=device, half=half,
                               verbose=yolo_verbose)
        # CameraMovementEstimator needs a seed frame; created lazily on frame 0.
        self.camera_movement_estimator = None
        self.view_transformer = ViewTransformer()
        self.speed_estimator = SpeedAndDistance_Estimator()
        self.team_assigner = TeamAssigner()
        self.player_ball_assigner = PlayerBallAssigner()
        self.scene_cut_detector = SceneCutDetector()
        self.shot_gate = ShotTypeGate(min_players=min_players_for_team_fit)

        self._prev_frame = None
        self._frame_index = -1
        self._cooldown_remaining = 0

        # Running team-ball-control tallies.
        self._team_possession_frames = {1: 0, 2: 0}
        self._last_possession_team = None

        # For inference_stride > 1 we carry forward the last full tracks + stats
        # on the cheap frames so ByteTrack keeps receiving frames only when we
        # actually detect (we still advance camera flow on every frame).
        self._last_tracks = None
        self._last_players_out = None
        self._last_referees_out = None
        self._last_ball_out = None
        self._last_possession = None
        self._last_tactical = False

    # -- scene-cut reset contract -------------------------------------------
    def _apply_scene_cut_reset(self, frame):
        self.tracker.reset()
        self.team_assigner.reset_player_cache()
        self.speed_estimator.reset()
        if self.camera_movement_estimator is not None:
            self.camera_movement_estimator.reset(frame)
        self.shot_gate.reset()
        self._cooldown_remaining = self.cut_cooldown_frames

    # -- per-frame processing -----------------------------------------------
    def process_frame(self, frame, timestamp=None):
        if timestamp is None:
            timestamp = time.monotonic()

        # 1. Resize to calibration resolution so the hardcoded pixel constants
        #    in CameraMovementEstimator / ViewTransformer stay valid.
        if (frame.shape[1], frame.shape[0]) != self.calibration_size:
            frame = cv2.resize(frame, self.calibration_size)

        self._frame_index += 1

        if self.camera_movement_estimator is None:
            self.camera_movement_estimator = CameraMovementEstimator(frame)

        # 2. Scene-cut detection + reset contract.
        scene_cut = self.scene_cut_detector.is_cut(self._prev_frame, frame)
        if scene_cut:
            self._apply_scene_cut_reset(frame)
            # Force a fresh detection on the first frame after a cut.
            self._last_tracks = None

        # Decide whether this frame runs the expensive YOLO + full analytics.
        # We still always run cheap per-frame work: resize, scene cut, optical flow.
        do_inference = (
            self._last_tracks is None or
            (self._frame_index % self.inference_stride == 0)
        )

        # 4. Camera movement + position adjustment (cheap, run every frame).
        dx, dy = self.camera_movement_estimator.update(frame)

        camera_stable = self._cooldown_remaining == 0

        if do_inference:
            # 3. Detect + track this frame, then drop detections that fall inside
            #    fixed graphics regions (scorebug/watermark) so they can't become
            #    phantom balls/players or skew the shot-type gate below.
            tracks = self.tracker.track_frame(frame)
            tracks = self._filter_ignored(tracks)
            self._last_tracks = tracks

            # 3b. Shot-type gate: is this a usable tactical wide shot? On a close-up /
            #     replay / graphic, tactical output is meaningless and the detector
            #     hallucinates (phantom balls, fragmented players), so gate it off.
            tactical, _shot_info = self.shot_gate.update(frame, tracks["players"])
            self._last_tactical = tactical

            # Pitch positions are only trusted on a stable, tactical wide shot.
            positions_trusted = camera_stable and tactical

            players_out = self._build_object_stats(
                frame, tracks["players"], dx, dy, timestamp,
                positions_trusted, tactical, is_player=True)
            referees_out = self._build_object_stats(
                frame, tracks["referees"], dx, dy, timestamp,
                positions_trusted, tactical, is_player=False)
            self._last_players_out = players_out
            self._last_referees_out = referees_out

            # 6. Ball hold/extrapolation. On a non-tactical shot the ball detections
            #    are unreliable phantoms, so feed the tracker nothing (it holds then
            #    reports the ball lost) rather than poisoning its trajectory state.
            ball_input = tracks["ball"] if tactical else {}
            ball_track = self.tracker.update_ball_position(ball_input)
            ball_out = None
            if ball_track is not None:
                ball_position = self._transformed_position(
                    get_center_of_bbox(ball_track["bbox"]), dx, dy, positions_trusted)
                ball_out = {
                    "bbox": [float(v) for v in ball_track["bbox"]],
                    "position": ball_position,
                }
            self._last_ball_out = ball_out

            # 8. Ball possession + running team control (only on tactical shots).
            possession = self._assign_possession(tracks["players"], ball_track,
                                                  players_out, tactical)
            self._last_possession = possession
        else:
            # Light frame (stride): reuse previous heavy results. We only advance
            # the cheap state (camera flow already done, ball extrapolation below).
            tracks = self._last_tracks or {"players": {}, "referees": {}, "ball": {}}
            tactical = self._last_tactical

            # Pitch positions trusted flag uses the *last* tactical decision.
            positions_trusted = camera_stable and tactical

            # Advance ball hold/extrapolation even on skipped detection frames.
            # Feed *no* detection so the missing-frame counter and extrapolation
            # advance as expected.
            ball_input = {}
            ball_track = self.tracker.update_ball_position(ball_input)
            ball_out = None
            if ball_track is not None:
                ball_position = self._transformed_position(
                    get_center_of_bbox(ball_track["bbox"]), dx, dy, positions_trusted)
                ball_out = {
                    "bbox": [float(v) for v in ball_track["bbox"]],
                    "position": ball_position,
                }
            self._last_ball_out = ball_out

            # Carry the last computed outputs (bboxes/speeds/teams are from the
            # previous inference frame). This is the explicit speed vs freshness
            # tradeoff controlled by --inference-every.
            players_out = self._last_players_out or {}
            referees_out = self._last_referees_out or {}
            possession = self._last_possession or {
                "team": None, "team_1_pct": None, "team_2_pct": None
            }

            # Keep the shot gate alive on every frame (very cheap) so it can
            # react quickly when the camera cuts to a close-up or back to wide.
            self.shot_gate.update(frame, tracks.get("players", {}))

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        self._prev_frame = frame

        stats = {
            "timestamp": timestamp,
            "frame": self._frame_index,
            "scene_cut": bool(scene_cut),
            "camera_stable": bool(camera_stable),
            "tactical": bool(tactical),
            "players": players_out,
            "referees": referees_out,
            "ball": ball_out,
            "possession": possession,
            "inference": bool(do_inference),   # tells consumers whether fresh detection ran
        }
        return stats

    def _filter_ignored(self, tracks):
        """Drop detections whose bbox centre lands in a configured graphics
        region. No-op when no ignore regions are set."""
        if not self._ignore_rects_px:
            return tracks
        out = {"players": {}, "referees": {}, "ball": {}}
        for kind, detections in tracks.items():
            for track_id, info in detections.items():
                x1, y1, x2, y2 = info["bbox"]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                if not any(rx1 <= cx <= rx2 and ry1 <= cy <= ry2
                           for rx1, ry1, rx2, ry2 in self._ignore_rects_px):
                    out[kind][track_id] = info
        return out

    def _transformed_position(self, foot_or_center, dx, dy, positions_trusted):
        """Adjust a pixel position for camera movement and map to pitch coords.
        Returns ``[x, y]`` in pitch metres, or ``None`` when untrusted."""
        if not positions_trusted:
            return None
        adjusted = np.array([foot_or_center[0] - dx, foot_or_center[1] - dy])
        transformed = self.view_transformer.transform_point(adjusted)
        if transformed is None:
            return None
        return [float(v) for v in np.array(transformed).squeeze().tolist()]

    def _build_object_stats(self, frame, object_tracks, dx, dy, timestamp,
                            positions_trusted, tactical, is_player):
        # 7. One-time team-colour fit — only on a tactical shot with enough
        #    players (a close-up at startup must not lock in wrong colours).
        if (is_player and tactical and not self.team_assigner.is_fitted()
                and len(object_tracks) >= self.min_players_for_team_fit):
            self.team_assigner.assign_team_color(frame, object_tracks)

        out = {}
        for track_id, info in object_tracks.items():
            bbox = info["bbox"]
            entry = {"bbox": [float(v) for v in bbox]}

            position = self._transformed_position(
                get_foot_position(bbox), dx, dy, positions_trusted)
            entry["position"] = position

            if is_player:
                # Team: null until colours fitted.
                if self.team_assigner.is_fitted():
                    team = self.team_assigner.get_player_team(
                        frame, bbox, track_id, live=True)
                    entry["team"] = int(team)
                    entry["team_color"] = [
                        _to_native(c) for c in self.team_assigner.team_colors[team]]
                else:
                    entry["team"] = None
                    entry["team_color"] = None

                # 9. Streaming speed/distance from real elapsed time.
                speed = self.speed_estimator.update(track_id, position, timestamp)
                if speed is not None:
                    entry["speed"] = speed["speed"]
                    entry["distance"] = speed["distance"]
                else:
                    entry["speed"] = None
                    entry["distance"] = None

                entry["has_ball"] = False

            out[str(track_id)] = entry
        return out

    def _assign_possession(self, player_tracks, ball_track, players_out, tactical):
        possession = {
            "team": None,
            "team_1_pct": None,
            "team_2_pct": None,
        }

        # Only accumulate possession on tactical shots — a close-up / replay
        # isn't observing the match, so carrying possession forward through it
        # would inflate whichever team last held the ball.
        if tactical:
            assigned_team = None
            if ball_track is not None and player_tracks:
                assigned_player = self.player_ball_assigner.assign_ball_to_player(
                    player_tracks, ball_track["bbox"])
                if assigned_player != -1:
                    key = str(assigned_player)
                    if key in players_out:
                        players_out[key]["has_ball"] = True
                        assigned_team = players_out[key].get("team")

            # Carry possession forward when nobody is assigned this frame,
            # matching main.py's team_ball_control[-1] behaviour — but without
            # the IndexError when no team has ever had the ball.
            if assigned_team is None:
                assigned_team = self._last_possession_team
            else:
                self._last_possession_team = assigned_team

            if assigned_team in (1, 2):
                self._team_possession_frames[assigned_team] += 1

        total = self._team_possession_frames[1] + self._team_possession_frames[2]
        if total > 0:
            possession["team"] = self._last_possession_team
            possession["team_1_pct"] = 100.0 * self._team_possession_frames[1] / total
            possession["team_2_pct"] = 100.0 * self._team_possession_frames[2] / total

        return possession

    # -- run loop ------------------------------------------------------------
    def run(self, capture, max_frames=None, read_timeout=5.0, return_frames=False):
        """Pull frames from a ``ResilientCapture`` and process until stopped.
        Yields each stats dict (also broadcast if a broadcaster was given).
        When ``return_frames`` is True, yields ``(stats, frame)`` tuples instead."""
        processed = 0
        while True:
            if max_frames is not None and processed >= max_frames:
                break
            frame = capture.read(timeout=read_timeout)
            if frame is None:
                continue
            stats = self.process_frame(frame)
            if self.broadcaster is not None:
                self.broadcaster.broadcast(stats)
            processed += 1
            if return_frames:
                yield stats, frame
            else:
                yield stats
