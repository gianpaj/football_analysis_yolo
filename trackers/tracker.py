from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys 
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position, measure_distance

class Tracker:
    def __init__(self, model_path, max_ball_jump=250,
                 conf=0.1, ball_conf=None, imgsz=None,
                 device=None, half=False, verbose=False,
                 track_activation_threshold=None, lost_track_buffer=30):
        self.model = YOLO(model_path)
        device = self._normalize_device(device)
        if device is not None:
            self.model.to(device)
        # ByteTrack. The offline path (track_activation_threshold=None) keeps
        # supervision's stock defaults so main.py is unchanged. The live pipeline
        # passes track_activation_threshold=conf: ByteTrack's default activation
        # gate is 0.25, well above the live --conf floor (0.15), so players
        # detected between the floor and 0.25 never spawn a track and vanish from
        # the output even though YOLO found them. Matching it to conf keeps them.
        if track_activation_threshold is not None:
            self.tracker = sv.ByteTrack(
                track_activation_threshold=track_activation_threshold,
                lost_track_buffer=lost_track_buffer)
        else:
            self.tracker = sv.ByteTrack()

        # Inference thresholds. ``conf`` is the global detection confidence
        # floor; ``ball_conf`` is an optional *higher* floor applied only to the
        # ball class (the noisiest — grass/kit/logos trigger phantom balls, so a
        # broadcast feed wants it gated harder than players/refs). ``imgsz``
        # overrides the inference resolution (larger = better tiny-ball recall,
        # lower FPS). Defaults preserve the original behaviour so the offline
        # ``main.py`` path is unchanged.
        self._conf = conf
        self._ball_conf = ball_conf if ball_conf is not None else conf
        self._imgsz = imgsz
        self._device = device
        self._half = bool(half)
        self._verbose = bool(verbose)

        # Streaming ball hold/extrapolation state (used by the live pipeline).
        self._last_ball_bbox = None          # last known ball bbox
        self._prev_ball_bbox = None          # ball bbox before the last one (for velocity)
        self._ball_missing_frames = 0        # consecutive frames without a fresh detection
        self._max_ball_hold_frames = 10      # after this many missed frames, report ball lost
        self._max_ball_jump = max_ball_jump  # max plausible ball centre move between frames (px)

    @staticmethod
    def _normalize_device(device):
        """Normalise a device spec for ``torch.nn.Module.to``.

        A bare GPU index ("0", "1") is valid for Ultralytics' predict() but not
        for torch's ``.to()`` (which needs "cuda:0"). Map it so the documented
        ``--device 0`` form works. "cuda", "cpu", "mps", "cuda:0" pass through.
        """
        if isinstance(device, int):
            return f"cuda:{device}"
        if isinstance(device, str) and device.isdigit():
            return f"cuda:{device}"
        return device

    def track_frame(self, frame):
        """Detect + track a single frame for the live pipeline.

        Mirrors the per-frame logic inside ``get_object_tracks`` but for one
        frame at a time. ``sv.ByteTrack`` is stateful across calls, so track IDs
        stay consistent frame-to-frame without any batching.

        Returns a dict ``{"players": {id: {"bbox": ...}}, "referees": {...},
        "ball": {1: {"bbox": ...}} or {}}`` for this frame only.
        """
        predict_kwargs = {
            "conf": self._conf,
            "verbose": self._verbose,
        }
        if self._imgsz is not None:
            predict_kwargs["imgsz"] = self._imgsz
        if self._device is not None:
            predict_kwargs["device"] = self._device
        if self._half:
            predict_kwargs["half"] = True
        detection = self.model.predict(frame, **predict_kwargs)[0]

        cls_names = detection.names
        cls_names_inv = {v: k for k, v in cls_names.items()}

        detection_supervision = sv.Detections.from_ultralytics(detection)

        # Convert GoalKeeper to player object
        for object_ind, class_id in enumerate(detection_supervision.class_id):
            if cls_names[class_id] == "goalkeeper":
                detection_supervision.class_id[object_ind] = cls_names_inv["player"]

        detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

        frame_tracks = {"players": {}, "referees": {}, "ball": {}}

        for frame_detection in detection_with_tracks:
            bbox = frame_detection[0].tolist()
            cls_id = frame_detection[3]
            track_id = frame_detection[4]

            if cls_id == cls_names_inv['player']:
                frame_tracks["players"][track_id] = {"bbox": bbox}

            if cls_id == cls_names_inv['referee']:
                frame_tracks["referees"][track_id] = {"bbox": bbox}

        # Collect every ball candidate (bbox + confidence) and pick the best one,
        # rather than arbitrarily keeping the last in iteration order. On a busy
        # frame the detector emits several phantom balls (grass/kit/logos); the
        # naive "keep last" would grab a random one.
        # The ball class gets a stricter confidence floor than players/refs
        # (``_ball_conf``): on a broadcast feed the detector fires low-confidence
        # phantom balls on grass, kit and logos, and a single wrong ball skews
        # possession. Drop anything below the ball floor before trajectory
        # selection even runs.
        ball_candidates = []
        for frame_detection in detection_supervision:
            cls_id = frame_detection[3]
            if cls_id == cls_names_inv['ball']:
                bbox = frame_detection[0].tolist()
                conf = frame_detection[2]
                conf = float(conf) if conf is not None else 0.0
                if conf >= self._ball_conf:
                    ball_candidates.append((bbox, conf))

        selected_ball = self._select_ball(ball_candidates)
        if selected_ball is not None:
            frame_tracks["ball"][1] = {"bbox": selected_ball}

        return frame_tracks

    def _predicted_ball_center(self):
        """Constant-velocity prediction of this frame's ball centre from the last
        two known detections, or ``None`` if the ball hasn't been seen yet."""
        if self._last_ball_bbox is None:
            return None
        last = get_center_of_bbox(self._last_ball_bbox)
        if self._prev_ball_bbox is None:
            return last
        prev = get_center_of_bbox(self._prev_ball_bbox)
        return (last[0] + (last[0] - prev[0]), last[1] + (last[1] - prev[1]))

    def _select_ball(self, candidates):
        """Choose the best ball from ``[(bbox, conf), ...]``.

        Prefer the highest-confidence candidate, but first gate on trajectory
        consistency: keep only candidates within ``_max_ball_jump`` px of the
        predicted position so a phantom ball that teleports across the frame
        can't win on confidence alone. If every candidate is far from the
        prediction (genuine reappearance after occlusion), fall back to plain
        highest confidence."""
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0][0]

        predicted = self._predicted_ball_center()
        if predicted is not None:
            near = [(bbox, conf) for bbox, conf in candidates
                    if measure_distance(get_center_of_bbox(bbox), predicted) <= self._max_ball_jump]
            pool = near if near else candidates
        else:
            pool = candidates

        best_bbox, _ = max(pool, key=lambda bc: bc[1])
        return best_bbox

    def update_ball_position(self, ball_track_this_frame):
        """Streaming replacement for ``interpolate_ball_positions``.

        Live feeds have no future frames to back-fill from, so instead of
        interpolating we hold the last known ball bbox (with a simple
        constant-velocity extrapolation from the last two detections) when the
        ball isn't detected this frame. After ``_max_ball_hold_frames`` missed
        frames we give up and report the ball as lost (returns ``None``).

        ``ball_track_this_frame`` is the ``{1: {"bbox": ...}}`` dict from
        ``track_frame`` (may be empty). Returns ``{"bbox": [...]}`` or ``None``.
        """
        detected = ball_track_this_frame.get(1)

        if detected is not None:
            bbox = detected["bbox"]
            self._prev_ball_bbox = self._last_ball_bbox
            self._last_ball_bbox = bbox
            self._ball_missing_frames = 0
            return {"bbox": list(bbox)}

        # No detection this frame.
        self._ball_missing_frames += 1

        if self._last_ball_bbox is None or self._ball_missing_frames > self._max_ball_hold_frames:
            return None

        # Constant-velocity extrapolation when we have two prior detections,
        # otherwise just hold the last known bbox.
        if self._prev_ball_bbox is not None:
            velocity = [c - p for c, p in zip(self._last_ball_bbox, self._prev_ball_bbox)]
            extrapolated = [b + v * self._ball_missing_frames
                            for b, v in zip(self._last_ball_bbox, velocity)]
            return {"bbox": extrapolated}

        return {"bbox": list(self._last_ball_bbox)}

    def reset(self):
        """Reset all inter-frame state. Called on a broadcast scene cut, where
        ByteTrack IDs and the ball hold state are meaningless across the cut."""
        self.tracker.reset()
        self._last_ball_bbox = None
        self._prev_ball_bbox = None
        self._ball_missing_frames = 0

    def add_position_to_tracks(sekf,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self,ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        batch_size=20 
        detections = [] 
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(
                frames[i:i+batch_size],
                conf=0.1,
                verbose=self._verbose,
                device=self._device,
                half=self._half,
            )
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert GoalKeeper to player object
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks
    
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame

    def draw_traingle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame

    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        # Draw a semi-transparent rectaggle 
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900,970), (255,255,255), -1 )
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%",(1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%",(1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame

    def draw_annotations(self,video_frames, tracks,team_ball_control):
        output_video_frames= []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"],color, track_id)

                if player.get('has_ball',False):
                    frame = self.draw_traingle(frame, player["bbox"],(0,0,255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"],(0,255,255))
            
            # Draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"],(0,255,0))


            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames