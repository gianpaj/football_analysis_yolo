"""Shot-type gating for the live pipeline.

A broadcast constantly cuts to shots that carry no tactical value — close-ups of
a single player, replays, crowd/bench reactions, full-screen graphics. On those
frames the detector produces garbage (phantom balls on grass/kit/logos,
fragmented player boxes) and none of the position-derived stats (pitch position,
speed, possession) are meaningful because there's no valid pitch context.

``SceneCutDetector`` catches the *transition* between shots; this gate answers
the persistent question "is the current frame a usable tactical wide shot?" so
the orchestrator can null tactical output for the whole duration of a non-tactical
shot, not just the cut frame.

Classification is a cheap heuristic over per-frame detection statistics plus a
grass-green pixel ratio — no extra model. A tactical wide shot has *many*
*small* players over *lots of green*; a close-up has few players, or one huge
box, or (for graphics/crowd) little green. A short debounce prevents a single
noisy frame from flipping the state.
"""

import cv2


class ShotTypeGate:
    def __init__(self, min_players=6, max_box_height_frac=0.5,
                 min_green_ratio=0.25, debounce_frames=3, downscale_width=128):
        self.min_players = min_players
        self.max_box_height_frac = max_box_height_frac
        self.min_green_ratio = min_green_ratio
        self.debounce_frames = debounce_frames
        self.downscale_width = downscale_width

        self._state = None      # committed tactical/non-tactical state
        self._pending = None    # candidate new state awaiting debounce
        self._pending_count = 0

    def _green_ratio(self, frame):
        """Fraction of pixels that look like grass (green in HSV). Downscaled
        for speed — the ratio is scale-invariant."""
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (self.downscale_width,
                                   max(1, int(h * self.downscale_width / w))))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
        return float(mask.mean()) / 255.0

    def _classify_raw(self, frame, player_tracks):
        """Single-frame tactical/not decision, before debouncing."""
        frame_h = frame.shape[0]
        n_players = len(player_tracks)

        max_h_frac = 0.0
        for info in player_tracks.values():
            bbox = info["bbox"]
            max_h_frac = max(max_h_frac, (bbox[3] - bbox[1]) / frame_h)

        green = self._green_ratio(frame)

        tactical = (n_players >= self.min_players
                    and max_h_frac <= self.max_box_height_frac
                    and green >= self.min_green_ratio)

        info = {
            "n_players": n_players,
            "max_box_height_frac": round(max_h_frac, 3),
            "green_ratio": round(green, 3),
        }
        return tactical, info

    def update(self, frame, player_tracks):
        """Update the gate with a new frame. Returns ``(is_tactical, info)``.

        The returned bool is the *debounced* state: a candidate flip must persist
        for ``debounce_frames`` consecutive frames before it commits."""
        raw, info = self._classify_raw(frame, player_tracks)

        if self._state is None:
            # First frame: adopt the raw decision immediately.
            self._state = raw
            self._pending = raw
            self._pending_count = 0
            return self._state, info

        if raw == self._state:
            self._pending_count = 0
        else:
            if raw == self._pending:
                self._pending_count += 1
            else:
                self._pending = raw
                self._pending_count = 1
            if self._pending_count >= self.debounce_frames:
                self._state = raw
                self._pending_count = 0

        return self._state, info

    def reset(self):
        """Clear state so the next frame re-seeds. Called on a scene cut — the
        new shot must be re-classified from scratch."""
        self._state = None
        self._pending = None
        self._pending_count = 0
