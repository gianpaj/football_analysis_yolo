import cv2
from collections import deque
import sys
sys.path.append('../')
from utils import measure_distance ,get_foot_position

class SpeedAndDistance_Estimator():
    def __init__(self):
        self.frame_window=5
        self.frame_rate=24

        # Streaming state: per-track rolling buffer of (timestamp, transformed
        # position) samples, plus accumulated distance per track.
        self._min_window_seconds = 0.20     # need at least this much time span to emit
        self._max_window_seconds = 2.0      # discard samples older than this
        self._stream_buffers = {}           # track_id -> deque[(t, (x, y))]
        self._stream_total_distance = {}    # track_id -> metres

    def reset(self):
        """Flush all per-track buffers. Called on a scene cut so the first
        post-cut sample doesn't pair with a stale pre-cut position and yield a
        garbage speed spike."""
        self._stream_buffers = {}
        self._stream_total_distance = {}

    def update(self, track_id, position_transformed, timestamp):
        """Streaming speed/distance for a single player track.

        Keeps a rolling deque of ``(timestamp, position)`` samples and computes
        speed from the *actual* elapsed wall-clock time between the oldest and
        newest buffered sample — not an assumed constant fps. Returns
        ``{"speed": km_per_hour, "distance": total_metres}`` once the buffered
        window spans enough time, otherwise ``None``.
        """
        if position_transformed is None:
            return None

        buf = self._stream_buffers.setdefault(track_id, deque())
        buf.append((timestamp, tuple(position_transformed)))

        # Drop samples older than the max window.
        while len(buf) > 1 and (timestamp - buf[0][0]) > self._max_window_seconds:
            buf.popleft()

        if len(buf) < 2:
            return None

        start_time, start_position = buf[0]
        end_time, end_position = buf[-1]
        time_elapsed = end_time - start_time

        if time_elapsed < self._min_window_seconds:
            return None

        distance_covered = measure_distance(start_position, end_position)
        speed_meters_per_second = distance_covered / time_elapsed
        speed_km_per_hour = speed_meters_per_second * 3.6

        self._stream_total_distance[track_id] = (
            self._stream_total_distance.get(track_id, 0) + distance_covered)

        # Reset the window start to the newest sample so accumulated distance
        # doesn't double count overlapping windows.
        buf.clear()
        buf.append((end_time, end_position))

        return {"speed": speed_km_per_hour,
                "distance": self._stream_total_distance[track_id]}

    def add_speed_and_distance_to_tracks(self,tracks):
        total_distance= {}

        for object, object_tracks in tracks.items():
            if object == "ball" or object == "referees":
                continue 
            number_of_frames = len(object_tracks)
            for frame_num in range(0,number_of_frames, self.frame_window):
                last_frame = min(frame_num+self.frame_window,number_of_frames-1 )

                for track_id,_ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]:
                        continue

                    start_position = object_tracks[frame_num][track_id]['position_transformed']
                    end_position = object_tracks[last_frame][track_id]['position_transformed']

                    if start_position is None or end_position is None:
                        continue
                    
                    distance_covered = measure_distance(start_position,end_position)
                    time_elapsed = (last_frame-frame_num)/self.frame_rate
                    speed_meteres_per_second = distance_covered/time_elapsed
                    speed_km_per_hour = speed_meteres_per_second*3.6

                    if object not in total_distance:
                        total_distance[object]= {}
                    
                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0
                    
                    total_distance[object][track_id] += distance_covered

                    for frame_num_batch in range(frame_num,last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        tracks[object][frame_num_batch][track_id]['speed'] = speed_km_per_hour
                        tracks[object][frame_num_batch][track_id]['distance'] = total_distance[object][track_id]
    
    def draw_speed_and_distance(self,frames,tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            for object, object_tracks in tracks.items():
                if object == "ball" or object == "referees":
                    continue 
                for _, track_info in object_tracks[frame_num].items():
                   if "speed" in track_info:
                       speed = track_info.get('speed',None)
                       distance = track_info.get('distance',None)
                       if speed is None or distance is None:
                           continue
                       
                       bbox = track_info['bbox']
                       position = get_foot_position(bbox)
                       position = list(position)
                       position[1]+=40

                       position = tuple(map(int,position))
                       cv2.putText(frame, f"{speed:.2f} km/h",position,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
                       cv2.putText(frame, f"{distance:.2f} m",(position[0],position[1]+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
            output_frames.append(frame)
        
        return output_frames