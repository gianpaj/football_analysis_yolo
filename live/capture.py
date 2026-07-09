"""Resilient frame capture for live feeds.

Wraps ``cv2.VideoCapture(..., cv2.CAP_FFMPEG)``, which handles an ``.m3u8``
HLS URL, RTSP, a webcam index, or a local file uniformly. The blocking read
loop runs on a background thread and pushes frames into a ``maxsize=1`` queue:
we always keep the *freshest* frame and drop stale ones so downstream
processing never falls further behind the live wall clock. If reads start
failing (HLS playlist hiccups, network drops), the capture is reopened with a
capped backoff.
"""

import os
import threading
import time
import queue

import cv2


def build_ffmpeg_options(headers=None, extra=None):
    """Build an ``OPENCV_FFMPEG_CAPTURE_OPTIONS`` string.

    ``headers`` is a dict of HTTP headers (e.g. ``{"Referer": "https://..."}``)
    passed to FFmpeg's HLS/HTTP demuxer — needed by some broadcast CDNs that
    gate on Referer/User-Agent. ``extra`` is a dict of raw FFmpeg AVOptions.
    OpenCV's format is ``key;value`` pairs joined by ``|``; multiple HTTP
    headers go in the single ``headers`` value, CRLF-separated.
    Returns ``None`` when there's nothing to set.
    """
    opts = []
    if headers:
        blob = "".join(f"{k}: {v}\r\n" for k, v in headers.items())
        opts.append(f"headers;{blob}")
    for k, v in (extra or {}).items():
        opts.append(f"{k};{v}")
    return "|".join(opts) if opts else None


class ResilientCapture:
    def __init__(self, source, reopen_after_failures=30,
                 backoff_initial=0.5, backoff_max=10.0,
                 ffmpeg_options=None):
        # A bare digit string (e.g. "0") is a webcam index, not a path/URL.
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        self.source = source
        self.reopen_after_failures = reopen_after_failures
        self.backoff_initial = backoff_initial
        self.backoff_max = backoff_max
        # FFmpeg demuxer options (HTTP headers etc.) for cv2's FFMPEG backend.
        # OpenCV reads this env var when VideoCapture(CAP_FFMPEG) is constructed,
        # so we set it right before opening.
        self.ffmpeg_options = ffmpeg_options

        self._queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread = None
        self._cap = None

    def _open(self):
        if self.ffmpeg_options:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = self.ffmpeg_options
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        return cap if cap.isOpened() else None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        backoff = self.backoff_initial
        consecutive_failures = 0

        while not self._stop.is_set():
            if self._cap is None:
                self._cap = self._open()
                if self._cap is None:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, self.backoff_max)
                    continue
                backoff = self.backoff_initial
                consecutive_failures = 0

            ok, frame = self._cap.read()
            if not ok or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= self.reopen_after_failures:
                    self._cap.release()
                    self._cap = None
                    consecutive_failures = 0
                else:
                    # brief pause avoids hammering a temporarily stalled stream
                    time.sleep(min(backoff, 0.1))
                continue

            consecutive_failures = 0
            self._offer(frame)

        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _offer(self, frame):
        """Put the newest frame in the queue, evicting a stale one if present."""
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                pass

    def read(self, timeout=5.0):
        """Block up to ``timeout`` seconds for the freshest frame.
        Returns the frame, or ``None`` on timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False
