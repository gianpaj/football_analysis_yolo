"""Rendering helpers for the live pipeline's optional visual preview.

Two consumers:

* ``main_live.py --preview`` shows an OpenCV window (needs a local display).
* ``main_live.py --preview-file PATH`` writes an annotated image to disk every
  N frames so you can watch a **headless / remote** run (SSH, cloud box) by
  serving or syncing that one file — no X display required.

``draw_stats_frame`` reuses the offline ``Tracker`` drawing primitives so the
live overlay looks like the recorded output. Detection bboxes in ``stats`` are
in calibration space (1920x1080); the raw capture frame may be any resolution,
so we resize to the calibration size before drawing to keep overlays aligned.
"""

import os

import cv2
import numpy as np


def _bgr_int(color, default=(0, 0, 255)):
    """Coerce a stored team colour (list of floats, maybe None) to an int BGR
    tuple that OpenCV drawing calls accept."""
    if not color:
        return default
    try:
        return tuple(int(round(float(c))) for c in color[:3])
    except (TypeError, ValueError):
        return default


def draw_stats_frame(tracker, frame, stats, calibration_size=(1920, 1080)):
    """Return an annotated copy of ``frame`` for the given per-frame ``stats``.

    ``tracker`` supplies the ellipse/triangle primitives (shared with the
    offline annotator). Draws players (team-coloured), referees, the ball, and a
    compact status banner (frame #, shot-type gate, possession, ball state).
    """
    if (frame.shape[1], frame.shape[0]) != calibration_size:
        frame = cv2.resize(frame, calibration_size)
    else:
        frame = frame.copy()

    # Players — team colour when known, red fallback before colours are fitted.
    for track_id, player in stats.get("players", {}).items():
        bbox = player.get("bbox")
        if not bbox:
            continue
        color = _bgr_int(player.get("team_color"))
        tid = int(track_id) if str(track_id).isdigit() else None
        frame = tracker.draw_ellipse(frame, bbox, color, tid)
        if player.get("has_ball"):
            frame = tracker.draw_traingle(frame, bbox, (0, 0, 255))

    # Referees — yellow.
    for _, referee in stats.get("referees", {}).items():
        bbox = referee.get("bbox")
        if bbox:
            frame = tracker.draw_ellipse(frame, bbox, (0, 255, 255))

    # Ball — green triangle.
    ball = stats.get("ball")
    if ball and ball.get("bbox"):
        frame = tracker.draw_traingle(frame, ball["bbox"], (0, 255, 0))

    _draw_banner(frame, stats)
    return frame


def _draw_banner(frame, stats):
    """Compact top-left status banner: frame #, gate, possession, ball."""
    tactical = stats.get("tactical", False)
    possession = stats.get("possession") or {}
    t1 = possession.get("team_1_pct")
    t2 = possession.get("team_2_pct")

    lines = [
        f"frame {stats.get('frame', '?')}   "
        f"{'DET' if stats.get('inference', True) else 'skip'}",
        f"gate: {'TACTICAL' if tactical else 'non-tactical (gated)'}",
        f"ball: {'yes' if stats.get('ball') else 'no'}    "
        f"players: {len(stats.get('players', {}))}",
    ]
    if t1 is not None and t2 is not None:
        lines.append(f"possession  T1 {t1:.0f}%  |  T2 {t2:.0f}%")

    # Semi-transparent black box behind the text for legibility.
    overlay = frame.copy()
    box_h = 30 * len(lines) + 20
    cv2.rectangle(overlay, (20, 20), (720, 20 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    gate_color = (0, 255, 0) if tactical else (0, 165, 255)
    y = 55
    for i, line in enumerate(lines):
        color = gate_color if i == 1 else (255, 255, 255)
        cv2.putText(frame, line, (35, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2, cv2.LINE_AA)
        y += 30


def write_frame_atomic(path, image, quality=85):
    """Encode ``image`` and write it to ``path`` atomically.

    Encodes to a temp file in the same directory, then ``os.replace`` — so a
    remote viewer polling ``path`` never reads a half-written image. Extension
    of ``path`` picks the format (``.jpg`` recommended for small, fast frames;
    ``.png`` for lossless). Returns True on success.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    ext = os.path.splitext(path)[1] or ".jpg"
    params = []
    if ext.lower() in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]

    ok, buf = cv2.imencode(ext, image, params)
    if not ok:
        return False

    tmp = os.path.join(directory, f".{os.path.basename(path)}.tmp")
    with open(tmp, "wb") as f:
        f.write(buf.tobytes())
    os.replace(tmp, path)
    return True
