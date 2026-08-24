"""Real backlot source: YOLO detection + tracking, no spot grid.

Fulfills the seam described in base.py — returns the same RawObservation
shape SimSource does, so nothing downstream (state machine, identity,
timers, notifier) needs to change. Unlike the simulator's fixed spot ids
0..NUM_SPOTS-1, the key here is YOLO's own track id (via model.track(...,
persist=True), which keeps a car's id stable frame to frame) — there's no
grid to map detections onto, timing just follows however many cars are
actually in frame. If a track gets lost and reappears with a new id (e.g.
after occlusion), IdentityDB's color match still recognizes it as the same
car and TimerService's existing car_id-keyed clock carries on unaffected.

A detected car's color is the mean RGB of its bounding-box crop, standing
in for an embedding exactly like the simulator's box color does — swap
`_mean_color` for a real embedding model later and IdentityDB is unaffected.
"""

from typing import Dict, List, Set, Tuple

import cv2
from ultralytics import YOLO

from .. import config
from ..types import Color, RawObservation
from .base import FrameSource

Box = Tuple[int, int, int, int]  # x1, y1, x2, y2 in pixel coords


class CameraSource(FrameSource):
    def __init__(self, video_source=None):
        source = config.CAMERA_SOURCE if video_source is None else video_source
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open video source {source!r}")

        self._model = YOLO(config.YOLO_MODEL)
        self._known_ids: Set[int] = set()  # every track id ever seen, so a leave can be confirmed
        self.last_frame = None  # latest BGR frame, for viz_cv to draw on
        self.last_boxes: List[Tuple[int, Box]] = []  # (track_id, box) currently visible, for viz_cv

    def step(self, now_hours: float) -> Dict[int, RawObservation]:
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError("camera/video source ended or disconnected")
        self.last_frame = frame

        results = self._model.track(frame, classes=config.YOLO_CLASSES, conf=config.YOLO_CONF,
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

    def release(self) -> None:
        self._cap.release()


def _mean_color(frame, x1: float, y1: float, x2: float, y2: float) -> Color:
    crop = frame[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
    if crop.size == 0:
        return (128, 128, 128)
    b, g, r = crop.reshape(-1, 3).mean(axis=0)
    return (int(r), int(g), int(b))  # RGB, matches the Color convention used everywhere else
