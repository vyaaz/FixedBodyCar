"""Live view for a real camera/video source. Draws a bounding box directly
around each currently-detected car (no fixed grid — box coords come from
pipeline.source.last_boxes, this tick's actual YOLO detections) with
OpenCV, yellow if over the 72h limit, same convention as the matplotlib
sim view. Also renders the Identified Cars / Over 72h Limit lists as a side
panel image concatenated onto the frame — the cv2 equivalent of viz.py's
matplotlib panels.
"""

import cv2
import numpy as np

from . import config

EMPTY_COLOR_BGR = (200, 200, 200)
VIOLATION_COLOR_BGR = (0, 212, 255)  # yellow (BGR) for spots/cars over the 72h limit
PANEL_WIDTH = 340


def run(pipeline, window_name: str = "BigBack — live") -> None:
    try:
        while True:
            snaps = pipeline.tick()
            frame = pipeline.source.last_frame
            if frame is None:
                continue
            frame = frame.copy()

            by_id = {snap.spot_id: snap for snap in snaps}
            for track_id, (x1, y1, x2, y2) in getattr(pipeline.source, "last_boxes", []):
                snap = by_id.get(track_id)
                if snap is None:
                    continue
                edge = VIOLATION_COLOR_BGR if snap.violating else (0, 200, 0)
                thickness = 4 if snap.violating else 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), edge, thickness)
                label = f"{snap.car_id} {snap.elapsed_hours:.1f}h/{config.VIOLATION_HOURS:.0f}h"
                if snap.violating:
                    label += " !OVER"
                cv2.putText(frame, label, (x1 + 4, max(16, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, edge, 2, cv2.LINE_AA)

            panel = _build_panel(frame.shape[0], pipeline)
            cv2.imshow(window_name, np.hstack([frame, panel]))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
        release = getattr(pipeline.source, "release", None)
        if release:
            release()


def _bgr(color_rgb):
    r, g, b = color_rgb
    return (int(b), int(g), int(r))


def _build_panel(height: int, pipeline) -> np.ndarray:
    panel = np.full((height, PANEL_WIDTH, 3), 255, dtype=np.uint8)
    y = 24
    cv2.putText(panel, "Identified Cars", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    y += 28

    cars = pipeline.identity.all_cars()
    if not cars:
        cv2.putText(panel, "(none yet)", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)
        y += 22

    for car in cars:
        if y > height - 80:
            cv2.putText(panel, "...", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)
            y += 22
            break
        clock = pipeline.timers.clock_for(car.car_id)
        if clock and clock.spot_id is not None:
            elapsed = pipeline.timers.elapsed_hours(car.car_id, pipeline.now)
            status = f"spot {clock.spot_id} - {elapsed:.1f}h" + (" !" if clock.flagged else "")
        else:
            status = "off-lot"
        cv2.rectangle(panel, (10, y - 12), (28, y + 2), _bgr(car.color), -1)
        cv2.rectangle(panel, (10, y - 12), (28, y + 2), (0, 0, 0), 1)
        cv2.putText(panel, f"{car.car_id} {status}", (34, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 1, cv2.LINE_AA)
        y += 22

    y += 18
    cv2.putText(panel, "Over 72h Limit", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 90, 180), 2, cv2.LINE_AA)
    y += 28

    violators = [(c, pipeline.timers.clock_for(c.car_id)) for c in cars]
    violators = [(c, cl) for c, cl in violators if cl and cl.flagged]
    if not violators:
        cv2.putText(panel, "(none)", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)

    for car, clock in violators:
        if y > height - 20:
            cv2.putText(panel, "...", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)
            break
        where = f"spot {clock.spot_id}" if clock.spot_id is not None else "off-lot"
        elapsed = pipeline.now - clock.park_start
        cv2.rectangle(panel, (10, y - 12), (28, y + 2), _bgr(car.color), -1)
        cv2.rectangle(panel, (10, y - 12), (28, y + 2), VIOLATION_COLOR_BGR, 2)
        cv2.putText(panel, f"{car.car_id} {where} {elapsed:.1f}h", (34, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 1, cv2.LINE_AA)
        y += 22

    return panel
