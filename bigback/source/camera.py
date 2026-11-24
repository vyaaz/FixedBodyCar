"""Real backlot source: YOLO detection + tracking, no spot grid.

Fulfills the seam described in base.py — returns the same RawObservation
shape SimSource does, so nothing downstream (state machine, identity,
timers, notifier) needs to change. Unlike the simulator's fixed spot ids
0..NUM_SPOTS-1, the key here is YOLO's own track id (via model.track(...,
persist=True), which keeps a car's id stable frame to frame) — there's no
grid to map detections onto, timing just follows however many cars are
actually in frame. If a track gets lost and reappears with a new id (e.g.
after occlusion), IdentityDB's embedding match still recognizes it as the
same car and TimerService's existing car_id-keyed clock carries on unaffected.

Identity matching uses a real Re-ID embedding (reid.py's VehicleDINO), not a
pixel color — see embed() below, which pipeline.py calls on a background
thread (it's too slow, ~6-18s on CPU, to run inline). `RawObservation.color`
is still a plain mean RGB, but it's cosmetic only now (panel swatch,
synthetic violation photo), decoupled from how cars are actually told apart.
"""

import os
import threading
import time
from typing import Dict, List, Set, Tuple

# RTSP over UDP drops/corrupts frames on the backlot Wi-Fi; force TCP. Must be
# set before cv2 opens the stream. Harmless for webcam/file sources.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
# Quiet libav's per-packet decode warnings (stray cu_qp_delta / missing POC
# ref messages are expected noise on a real RTSP link, not fatal — a genuinely
# dead stream still surfaces as our own RuntimeError from a failed read()).
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

import cv2
from ultralytics import YOLO

from .. import config, notifier
from ..types import Color, RawObservation
from .base import FrameSource
from .reid import VehicleReID

Box = Tuple[int, int, int, int]  # x1, y1, x2, y2 in pixel coords


class _DirectReader:
    """Plain synchronous read, for a video-file source."""

    def __init__(self, cap: cv2.VideoCapture):
        self._cap = cap

    def read(self):
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError("camera/video source ended or disconnected")
        return frame

    def release(self) -> None:
        self._cap.release()


class _LatestFrameReader:
    """Drains a capture on its own thread and hands out only the newest frame.

    YOLO on CPU runs at ~1 FPS while the Reolink pushes 30; read straight off
    cv2.VideoCapture and you get ever-staler buffered frames until the RTSP
    connection stalls outright. This thread keeps the buffer empty so step()
    always sees a fresh frame. Used for live streams/webcams only — a video
    file wants to be consumed frame by frame, not raced through.

    This is meant to run unattended for 72h+ at a stretch, so a dropped link
    (wifi hiccup, camera reboot) reconnects with backoff instead of killing
    the whole app — step() just keeps returning the last good frame while
    this reconnects in the background, and an email goes out on both the
    drop and the recovery so it isn't a silent gap in the violation clock.
    """

    def __init__(self, source):
        self._source = source
        self._lock = threading.Lock()
        self._frame = None
        self._stop = False
        self._cap = self._open()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self._source)
        if not cap.isOpened():
            raise RuntimeError(f"could not open video source {self._source!r}")
        return cap

    def _loop(self) -> None:
        while not self._stop:
            ok, frame = self._cap.read()
            if ok:
                with self._lock:
                    self._frame = frame
                continue
            self._reconnect()

    def _reconnect(self) -> None:
        down_since = time.time()
        print("[camera] stream dropped, reconnecting...")
        notifier.send_email(
            subject="BigBack: camera stream down",
            body="The backlot RTSP stream dropped. Attempting to reconnect automatically "
                 "— violation clocks keep running off wall-clock time regardless.",
        )
        self._cap.release()

        backoff = list(config.RECONNECT_BACKOFF_SECONDS)
        attempt = 0
        while not self._stop:
            try:
                self._cap = self._open()
                break
            except RuntimeError:
                delay = backoff[min(attempt, len(backoff) - 1)]
                attempt += 1
                time.sleep(delay)
        else:
            return  # release() was called mid-reconnect; give up quietly

        outage_min = (time.time() - down_since) / 60
        print(f"[camera] reconnected after {outage_min:.1f} min")
        notifier.send_email(
            subject="BigBack: camera stream back up",
            body=f"The backlot RTSP stream reconnected after {outage_min:.1f} min down.",
        )

    def read(self):
        # block briefly for the very first frame after connect
        for _ in range(100):
            with self._lock:
                if self._frame is not None:
                    return self._frame
            time.sleep(0.05)
        raise RuntimeError("no frame from source within 5s of connecting")

    def release(self) -> None:
        self._stop = True
        self._thread.join(timeout=1.0)
        self._cap.release()


class CameraSource(FrameSource):
    def __init__(self, video_source=None):
        source = config.CAMERA_SOURCE if video_source is None else video_source
        is_stream = isinstance(source, int) or (isinstance(source, str) and "://" in source)

        if is_stream:
            self._reader = _LatestFrameReader(source)  # owns connect + reconnect
        else:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise RuntimeError(f"could not open video source {source!r}")
            self._reader = _DirectReader(cap)

        self._model = YOLO(config.YOLO_MODEL)
        self._reid = VehicleReID()
        self._known_ids: Set[int] = set()  # every track id ever seen, so a leave can be confirmed
        self.last_frame = None  # latest BGR frame, for viz_cv to draw on
        self.last_boxes: List[Tuple[int, Box]] = []  # (track_id, box) currently visible, for viz_cv

    def step(self, now_hours: float) -> Dict[int, RawObservation]:
        frame = self._reader.read()

        n = max(1, config.CAMERA_DOWNSCALE)
        if n > 1:
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (w // n, h // n), interpolation=cv2.INTER_AREA)
        self.last_frame = frame  # everything downstream works in this (downscaled) space

        results = self._model.track(frame, classes=config.YOLO_CLASSES, conf=config.YOLO_CONF,
                                     imgsz=config.YOLO_IMGSZ, tracker=config.YOLO_TRACKER,
                                     persist=True, verbose=False)

        current: Dict[int, RawObservation] = {}
        self.last_boxes = []
        for box in results[0].boxes:
            if box.id is None:
                continue  # tracker hasn't assigned this detection a stable id yet
            track_id = int(box.id[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            self.last_boxes.append((track_id, (int(x1), int(y1), int(x2), int(y2))))
            current[track_id] = RawObservation(occupied=True, color=_mean_color(frame, x1, y1, x2, y2))

        self._known_ids |= current.keys()
        return {tid: current.get(tid, RawObservation(occupied=False)) for tid in self._known_ids}

    def crop_for(self, track_id: int):
        """BGR pixel crop of a currently-visible car, for a real violation
        photo. None if the id isn't in this tick's frame (e.g. mid-debounce
        gap) — notifier falls back to its synthetic swatch in that case."""
        if self.last_frame is None:
            return None
        for tid, (x1, y1, x2, y2) in self.last_boxes:
            if tid == track_id:
                crop = self.last_frame[max(0, y1):y2, max(0, x1):x2]
                return crop if crop.size else None
        return None

    def embed(self, crop):
        """VehicleDINO Re-ID embedding for an already-grabbed crop (from
        crop_for()) — the actual identity-matching feature (see
        identity.py/reid.py). Takes ~6-18s on CPU, so pipeline.py runs this
        on a background thread rather than calling it inline; this method
        itself just forwards to the model, no lookup of live tracking state,
        so it's safe to call from another thread after last_boxes has moved
        on to a later tick."""
        return self._reid.embed(crop)

    def release(self) -> None:
        self._reader.release()


def _mean_color(frame, x1: float, y1: float, x2: float, y2: float) -> Color:
    """Plain whole-box mean RGB — cosmetic only now (panel swatch, synthetic
    violation photo). Real car identity is decided by VehicleReID.embed() in
    embed_for() below, not this; a pixel mean isn't discriminative enough to
    tell two same-colored cars apart (see CLAUDE.md)."""
    crop = frame[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
    if crop.size == 0:
        return (128, 128, 128)
    b, g, r = crop.reshape(-1, 3).mean(axis=0)
    return (int(r), int(g), int(b))
