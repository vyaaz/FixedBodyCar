"""All the knobs. Change these; don't hunt magic numbers through the codebase."""

from pathlib import Path

GRID_ROWS = 3
GRID_COLS = 4
NUM_SPOTS = GRID_ROWS * GRID_COLS

VIOLATION_HOURS = 72.0        # max continuous park time
COOLDOWN_HOURS = 24.0          # must sit off-lot this long before the clock resets (rule: 5-10h)
DEBOUNCE_TICKS = 2            # ticks a raw reading must hold before it's trusted

IDENTITY_MATCH_THRESHOLD = 30  # max color-distance to call two sightings "the same car"

TICK_INTERVAL_MS = 400        # wall-clock ms between animation frames
# 20 real seconds == 72 sim hours (demo speed), so a 72h violation plays out over ~20 real seconds.
SIM_HOURS_PER_TICK = VIOLATION_HOURS * TICK_INTERVAL_MS / 20_000

# run_camera.py's clock: while testing, compress time the same way the simulator does (20s == 72h)
# so a violation/email is reachable without literally waiting 3 real days. Flip to False once this
# is actually mounted and tracking real cars, so 72h means 72 real hours.
CAMERA_DEMO_SPEED = True
SIM_HOURS_PER_REAL_SECOND = VIOLATION_HOURS / 20  # only used when CAMERA_DEMO_SPEED is True

VIOLATIONS_DIR = Path(__file__).resolve().parent.parent / "violations"  # saved photo per violation
EMAIL_RECIPIENT = "vyaas.mohan@gmail.com"  # default "To" for violation-notice emails

# --- CameraSource (real YOLO detection + tracking, no spot grid) ---
CAMERA_SOURCE = 1  # webcam index (0 was front-facing) — file path or RTSP url also work
YOLO_MODEL = "yolo11s.pt"     # auto-downloads on first run if not already cached; nano missed 2/9 real cars
                               # from the steep backlot angle (~4 FPS on CPU vs nano's ~7 FPS) — bump to
                               # yolo11m/l/x for even more margin at the cost of speed
YOLO_CLASSES = [2]            # COCO class ids to detect; 2=car, 3=motorcycle, 5=bus, 7=truck
YOLO_CONF = 0.25              # minimum detection confidence
