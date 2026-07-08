"""Per-frame live orchestrator.

Runs the same detection / tracking / analytics as the offline ``main.py`` but
incrementally, one frame at a time, and emits a JSON-serialisable stats dict per
frame (no on-screen drawing). See ``~/.claude/plans/eager-marinating-falcon.md``
section 3 for the flow and the scene-cut reset contract.
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
                 cut_cooldown_frames=15):
        self.calibration_size = calibration_size
        self.min_players_for_team_fit = min_players_for_team_fit
        self.cut_cooldown_frames = cut_cooldown_frames
        self.broadcaster = broadcaster

        self.tracker = Tracker(model_path)
        # CameraMovementEstimator needs a seed frame; created lazily on frame 0.
        self.camera_movement_estimator = None
        self.view_transformer = ViewTransformer()
        self.speed_estimator = SpeedAndDistance_Estimator()
        self.team_assigner = TeamAssigner()
        self.player_ball_assigner = PlayerBallAssigner()
        self.scene_cut_detector = SceneCutDetector()

        self._prev_frame = None
        self._frame_index = -1
        self._cooldown_remaining = 0

        # Running team-ball-control tallies.
        self._team_possession_frames = {1: 0, 2: 0}
        self._last_possession_team = None

    # -- scene-cut reset contract -------------------------------------------
    def _apply_scene_cut_reset(self, frame):
        self.tracker.reset()
        self.team_assigner.reset_player_cache()
        self.speed_estimator.reset()
        if self.camera_movement_estimator is not None:
            self.camera_movement_estimator.reset(frame)
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

        # 3. Detect + track this frame.
        tracks = self.tracker.track_frame(frame)

        # 4. Camera movement + position adjustment.
        dx, dy = self.camera_movement_estimator.update(frame)

        camera_stable = self._cooldown_remaining == 0

        players_out = self._build_object_stats(
            frame, tracks["players"], dx, dy, timestamp,
            camera_stable, is_player=True)
        referees_out = self._build_object_stats(
            frame, tracks["referees"], dx, dy, timestamp,
            camera_stable, is_player=False)

        # 6. Ball hold/extrapolation.
        ball_track = self.tracker.update_ball_position(tracks["ball"])
        ball_out = None
        if ball_track is not None:
            ball_position = self._transformed_position(
                get_center_of_bbox(ball_track["bbox"]), dx, dy, camera_stable)
            ball_out = {
                "bbox": [float(v) for v in ball_track["bbox"]],
                "position": ball_position,
            }

        # 8. Ball possession + running team control.
        possession = self._assign_possession(tracks["players"], ball_track,
                                              players_out)

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        self._prev_frame = frame

        stats = {
            "timestamp": timestamp,
            "frame": self._frame_index,
            "scene_cut": bool(scene_cut),
            "camera_stable": bool(camera_stable),
            "players": players_out,
            "referees": referees_out,
            "ball": ball_out,
            "possession": possession,
        }
        return stats

    def _transformed_position(self, foot_or_center, dx, dy, camera_stable):
        """Adjust a pixel position for camera movement and map to pitch coords.
        Returns ``[x, y]`` in pitch metres, or ``None`` when untrusted."""
        if not camera_stable:
            return None
        adjusted = np.array([foot_or_center[0] - dx, foot_or_center[1] - dy])
        transformed = self.view_transformer.transform_point(adjusted)
        if transformed is None:
            return None
        return [float(v) for v in np.array(transformed).squeeze().tolist()]

    def _build_object_stats(self, frame, object_tracks, dx, dy, timestamp,
                            camera_stable, is_player):
        # 7. One-time team-colour fit once enough players are visible.
        if (is_player and not self.team_assigner.is_fitted()
                and len(object_tracks) >= self.min_players_for_team_fit):
            self.team_assigner.assign_team_color(frame, object_tracks)

        out = {}
        for track_id, info in object_tracks.items():
            bbox = info["bbox"]
            entry = {"bbox": [float(v) for v in bbox]}

            position = self._transformed_position(
                get_foot_position(bbox), dx, dy, camera_stable)
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

    def _assign_possession(self, player_tracks, ball_track, players_out):
        possession = {
            "team": None,
            "team_1_pct": None,
            "team_2_pct": None,
        }

        assigned_team = None
        if ball_track is not None and player_tracks:
            assigned_player = self.player_ball_assigner.assign_ball_to_player(
                player_tracks, ball_track["bbox"])
            if assigned_player != -1:
                key = str(assigned_player)
                if key in players_out:
                    players_out[key]["has_ball"] = True
                    assigned_team = players_out[key].get("team")

        # Carry possession forward when nobody is assigned this frame, matching
        # main.py's team_ball_control[-1] behaviour — but without the IndexError
        # when no team has ever had the ball.
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
    def run(self, capture, max_frames=None, read_timeout=5.0):
        """Pull frames from a ``ResilientCapture`` and process until stopped.
        Yields each stats dict (also broadcast if a broadcaster was given)."""
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
            yield stats
