"""Entry point for a live/real feed: python run_camera.py

Defaults to webcam index 0 (config.CAMERA_SOURCE) — point it at the real
backlot camera (RTSP url or video path) by editing that config value, or
pass a source explicitly: CameraSource(video_source=...).

Clock speed follows config.CAMERA_DEMO_SPEED: compressed (20s == 72h, same
as run_sim.py) while True, real wall-clock hours once flipped to False.

Known cars and their clocks persist across restarts (config.STATE_PATH,
re-saved every tick) — without this, every restart would silently reset
every currently-parked car's violation clock back to zero.
"""

import json
import time

from bigback import config, viz_cv
from bigback.identity import IdentityDB
from bigback.pipeline import Pipeline
from bigback.source.camera import CameraSource
from bigback.source.reid import cosine_distance
from bigback.timer_service import TimerService

if __name__ == "__main__":
    saved = None
    if config.STATE_PATH.exists():
        try:
            saved = json.loads(config.STATE_PATH.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"[state] couldn't read {config.STATE_PATH}: {e} -- starting fresh")

    # Reusing the OLD start epoch (not time.time()) is what makes a restart
    # invisible to the clock: `now` is hours-since-start, so keeping the same
    # start means `now` picks up right where it left off instead of jumping
    # back to ~0, which would make every parked car look like it just arrived.
    start = saved["start_epoch"] if saved else time.time()
    if config.CAMERA_DEMO_SPEED:
        clock = lambda: (time.time() - start) * config.SIM_HOURS_PER_REAL_SECOND
    else:
        clock = lambda: (time.time() - start) / 3600.0

    identity = IdentityDB(threshold=config.CAMERA_IDENTITY_MATCH_THRESHOLD, distance_fn=cosine_distance)
    timers = TimerService()
    if saved:
        identity.load_dict(saved["identity"])
        timers.load_dict(saved["timers"])
        print(f"[state] restored {len(identity.all_cars())} known car(s) from {config.STATE_PATH}")

    pipeline = Pipeline(source=CameraSource(), clock=clock, identity=identity, timers=timers,
                         persist_path=config.STATE_PATH, persist_meta={"start_epoch": start})
    viz_cv.run(pipeline)  # closes the pipeline (incl. a final state save) in its own finally block
