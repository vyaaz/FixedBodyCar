"""All the knobs. Change these; don't hunt magic numbers through the codebase."""

import os
from pathlib import Path
from urllib.parse import quote

GRID_ROWS = 3
GRID_COLS = 4
NUM_SPOTS = GRID_ROWS * GRID_COLS

VIOLATION_HOURS = 72.0        # max continuous park time
COOLDOWN_HOURS = 24.0          # must sit off-lot this long before the clock resets (rule: 5-10h)
DEBOUNCE_TICKS = 2            # ticks a raw reading must hold before it's trusted

IDENTITY_MATCH_THRESHOLD = 30  # max color-distance to call two sightings "the same car"

TICK_INTERVAL_MS = 400        # wall-clock ms between animation frames
# run_sim.py's clock. False => the sim runs on real wall-clock time, so 72h means 72 real hours
# (the timer shows true HH:MM:SS). True => compress time for a demo: 20 real seconds == 72 sim
# hours, so a 72h violation plays out over ~20 real seconds.
SIM_DEMO_SPEED = False
# only used when SIM_DEMO_SPEED is True:
SIM_HOURS_PER_TICK = VIOLATION_HOURS * TICK_INTERVAL_MS / 20_000

# run_camera.py's clock. False => real wall-clock time, so 72h means 72 real hours and the live
# timer shows true HH:MM:SS. True => compress time the same way the simulator does (20s == 72h) so
# a violation/email is reachable without literally waiting 3 real days.
CAMERA_DEMO_SPEED = False
SIM_HOURS_PER_REAL_SECOND = VIOLATION_HOURS / 20  # only used when CAMERA_DEMO_SPEED is True

VIOLATIONS_DIR = Path(__file__).resolve().parent.parent / "violations"  # saved photo per violation
DETECTIONS_DIR = Path(__file__).resolve().parent.parent / "detections"  # saved photo per newly-tagged car
EMAIL_RECIPIENT = "vyaas.mohan@gmail.com"  # default "To" for violation-notice emails

# camera stream reconnect backoff (seconds), cycled then held at the last value
RECONNECT_BACKOFF_SECONDS = [2, 5, 10, 30]

# --- CameraSource (real YOLO detection + tracking, no spot grid) ---
# The backlot Reolink. IP/port/stream path live here; credentials come from
# BIGBACK_RTSP_USER / BIGBACK_RTSP_PASS in the environment (same opt-in pattern
# as the email creds — nothing secret is committed). VLC prompts for the same
# user/pass; use the camera's account, not the Reolink app login.
RTSP_HOST = "10.1.10.182"
RTSP_PORT = 554
RTSP_PATH = "h264Preview_01_main"   # Reolink main stream; "h264Preview_01_sub" is the low-res substream
# _RTSP_USER = os.environ.get("BIGBACK_RTSP_USER", "")
# _RTSP_PASS = os.environ.get("BIGBACK_RTSP_PASS", "")
_RTSP_USER = "admin"
_RTSP_PASS = "unicorn"


def _rtsp_url() -> str:
    cred = f"{quote(_RTSP_USER, safe='')}:{quote(_RTSP_PASS, safe='')}@" if _RTSP_USER else ""
    return f"rtsp://{cred}{RTSP_HOST}:{RTSP_PORT}/{RTSP_PATH}"


# webcam index / file path / RTSP url all work; default is the Reolink stream above.
CAMERA_SOURCE = "rtsp://admin:unicorn@10.1.10.182:554/h264Preview_01_main"
YOLO_MODEL = "yolo11s.pt"     # auto-downloads on first run if not already cached; nano missed 2/9 real cars
                               # from the steep backlot angle (~4 FPS on CPU vs nano's ~7 FPS) — bump to
                               # yolo11m/l/x for even more margin at the cost of speed
YOLO_CLASSES = [2]            # COCO class ids to detect; 2=car, 3=motorcycle, 5=bus, 7=truck
YOLO_CONF = 0.25              # minimum detection confidence
# ultralytics' default tracker (botsort.yaml) runs GMC, which estimates
# *camera* motion between consecutive frames via feature matching — pointless
# on our fixed security cam, and actively breaks down when slow CPU inference
# means processed frames are a second+ apart (not enough matching points ->
# tracking resets -> the same car keeps getting a new track id -> identity.py
# re-registers it as a "new" car each time). bytetrack.yaml skips GMC
# entirely and tracks on motion/IOU, which suits a static camera better.
YOLO_TRACKER = "bytetrack.yaml"

# The Reolink main stream is 2560x1440. CAMERA_DOWNSCALE shrinks each frame by
# this integer factor right after capture — every stage downstream (detection,
# viz boxes, violation crops) then works in that smaller space, so coords stay
# consistent. 2 => 1280x720, a good balance of range vs. CPU cost; 1 = full res.
CAMERA_DOWNSCALE = 2
# Inference size YOLO letterboxes each frame to before detecting. Default 640
# throws away most of a 1440p feed's detail, so distant backlot cars vanish;
# 960 keeps enough to see them without the full cost of native res.
YOLO_IMGSZ = 960

# viz_cv's live window is scaled to fit within this box (detection/crop
# resolution, set by CAMERA_DOWNSCALE above, is untouched — this only shrinks
# what's drawn on screen). Lower these if the window still covers too much
# of your display.
DISPLAY_MAX_WIDTH = 1000
DISPLAY_MAX_HEIGHT = 650
