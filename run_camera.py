"""Entry point for a live/real feed: python run_camera.py

Defaults to webcam index 0 (config.CAMERA_SOURCE) — point it at the real
backlot camera (RTSP url or video path) by editing that config value, or
pass a source explicitly: CameraSource(video_source=...).

Clock speed follows config.CAMERA_DEMO_SPEED: compressed (20s == 72h, same
as run_sim.py) while True, real wall-clock hours once flipped to False.
"""

import time

from bigback import config, viz_cv
from bigback.pipeline import Pipeline
from bigback.source.camera import CameraSource

if __name__ == "__main__":
    start = time.time()
    if config.CAMERA_DEMO_SPEED:
        clock = lambda: (time.time() - start) * config.SIM_HOURS_PER_REAL_SECOND
    else:
        clock = lambda: (time.time() - start) / 3600.0
    pipeline = Pipeline(source=CameraSource(), clock=clock)
    viz_cv.run(pipeline)
