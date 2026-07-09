"""Broadcast camera-cut detection.

A real TV broadcast hard-cuts between physical cameras. Such a cut invalidates
all inter-frame state (optical flow, track IDs, team cache, speed buffers), so
we need to detect it cheaply. We combine an HSV colour-histogram correlation
with a downscaled mean-absolute-difference: a cut is flagged when the histograms
diverge *and* the pixels differ a lot. Using both cuts down false positives from
fast pans (pixels move but colour distribution holds) and from lighting flicker.
"""

import cv2
import numpy as np


class SceneCutDetector:
    def __init__(self, hist_corr_threshold=0.5, mad_threshold=40.0, small_size=64,
                 max_dt=0.4):
        # Lower correlation => more different. Higher MAD => more different.
        self.hist_corr_threshold = hist_corr_threshold
        self.mad_threshold = mad_threshold
        self.small_size = small_size
        # Only judge a cut between frames close in time. When consecutive
        # *processed* frames arrive far apart (a slow/bursty live feed where the
        # freshest-frame queue drops everything in between), normal play motion
        # alone makes the frames look completely different, so the both-signals
        # test still false-fires. Above this gap (seconds) we can't tell a real
        # hard cut from accumulated motion, and a wrong reset (which clears all
        # track IDs) is far more damaging than a missed one — so we skip.
        self.max_dt = max_dt

    def _histogram(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    def _small(self, frame):
        # Downscaled colour frame: more sensitive than luma-only (two very
        # different colours can share the same grayscale value).
        return cv2.resize(frame, (self.small_size, self.small_size))

    def is_cut(self, prev_frame, frame, dt=None):
        """True if ``frame`` looks like a hard cut from ``prev_frame``.

        A cut needs *both* a diverging colour-histogram (low correlation) and a
        large pixel difference. Requiring both rejects fast pans (pixels move a
        lot but the colour distribution holds) and gentle fades/colour grades
        (distribution shifts but pixels barely move).

        ``dt`` is the wall-clock gap (seconds) since ``prev_frame`` was
        processed; when it exceeds ``max_dt`` the frames are too far apart to
        judge reliably, so we report no cut (see ``max_dt`` rationale)."""
        if prev_frame is None or frame is None:
            return False

        if dt is not None and dt > self.max_dt:
            return False

        corr = cv2.compareHist(self._histogram(prev_frame),
                               self._histogram(frame), cv2.HISTCMP_CORREL)

        mad = float(np.mean(np.abs(
            self._small(prev_frame).astype(np.int16)
            - self._small(frame).astype(np.int16))))

        return corr < self.hist_corr_threshold and mad > self.mad_threshold
