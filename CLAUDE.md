# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

BigBack is a CV-based parking tracker for a Theta Xi backlot. It tracks per-spot occupancy, times how long each car has been parked, and flags 72h+ violations for a fine notice.

Rules being enforced:
- Max 72h parked continuously (`VIOLATION_HOURS`).
- After moving, a car must stay off-lot before re-parking resets its clock (`COOLDOWN_HOURS`, currently 24.0 in `bigback/config.py`; rule of thumb is 5-10h, tune as needed). Re-parking before the cooldown is served just continues the existing clock instead of resetting it.
- Violation -> fine notice (console + optional email with a photo) to the brother who owns the car.

There are two runnable modes:

- **`python run_sim.py`** — V1, simulation only. `bigback/source/simulator.py` fabricates colored boxes wandering on/off a grid in place of a real camera + occupancy model, and a car's color stands in for an appearance embedding. The sim clock follows `SIM_DEMO_SPEED` in `config.py`: `False` (default) runs on real wall-clock time, so 72h means 72 real hours and the live timer shows true HH:MM:SS; `True` uses the derived `SIM_HOURS_PER_TICK` compression so a 72h violation plays out in ~20 real seconds for a quick demo.
- **`python run_camera.py`** — real detection, no grid. `bigback/source/camera.py` runs YOLO (`ultralytics`) with its built-in tracker (`model.track(..., persist=True)`) on a live video source (webcam/RTSP/file, `config.CAMERA_SOURCE`), so each car keeps a stable track id frame-to-frame; that id is used directly as the pipeline key (in place of a spot id) and each detection's mean crop color is the embedding stand-in — same `RawObservation` shape the simulator produces. This mode tracks real wall-clock time via `Pipeline`'s `clock` hook (see below), not the simulator's compressed tick.

Everything downstream of the source (debounce, identity matching, timers, fines) is identical in both modes — that's the point of the `FrameSource` seam.

## Commands

```
pip install -r requirements.txt
python run_sim.py       # simulated backlot, matplotlib view
python run_camera.py    # real YOLO detection off a video source, OpenCV view
```

There is no test suite, lint config, or build step yet — this is a small prototype. `yolov8n.pt` auto-downloads to the repo root on first `run_camera.py` run (needs network access once).

## Architecture

Pipeline (`bigback/pipeline.py`) wires everything together and is the only file whose shape changes between simulated and real sources. Each `tick()`:

1. **Source** (`bigback/source/`) — `FrameSource` is the abstract seam (`base.py`), returning one `RawObservation` (occupied + color) keyed by an int id per tick. `SimSource` (`simulator.py`) uses fixed spot ids 0..NUM_SPOTS-1 on a fabricated grid. `CameraSource` (`camera.py`) has no grid at all — the key is YOLO's own track id, which persists across frames for as long as a car stays in view; if a track is lost and a new id shows up later, `IdentityDB`'s color match still recognizes it as the same car so the existing car_id's clock keeps running. Either source plugs into `Pipeline(source=...)` — nothing else in the pipeline needs to change. `CameraSource.last_boxes` and `.last_frame` are read by `viz_cv.py` to draw a box directly around each live detection.
2. **State machine** (`state_machine.py`) — debounces raw per-key readings into trustworthy `ParkEvent`/`LeaveEvent` transitions; a transition only fires once a new reading holds for `DEBOUNCE_TICKS` in a row. Dict-based rather than a fixed-size array, since camera mode's keys (track ids) are dynamic and unbounded, unlike the simulator's fixed spot range. Also tracks each key's last-seen color (`color_at`) independent of the current tick's raw reading, since confirmed occupancy lags the raw feed by up to `DEBOUNCE_TICKS` — reading color straight off the raw observation instead caused a `None`-color crash right after a car left but before the leave was confirmed.
3. **Identity** (`identity.py`) — `IdentityDB.match_or_register` matches a car's color/embedding against known cars via distance under `IDENTITY_MATCH_THRESHOLD`. No match -> auto-registers a new `car-N` id (in prod this is where the chairman would get a photo to confirm/tag before the DB grows). The pipeline passes `exclude=timers.active_car_ids()` — car_ids already parked under another key can't be a match (a car isn't in two spots at once), so several similarly-colored cars on the lot at the same time each get their own id instead of collapsing onto the first one's; a car that has *left* stays matchable so a returning/re-tracked car re-links by color. Everything downstream only ever sees a `car_id`, never raw appearance data. `all_cars()`/`get()` back the "Identified Cars" panel in both viz modes.
4. **Timer service** (`timer_service.py`) — owns the 72h/cooldown rule. Keyed by `car_id`, not `spot_id`, so a car's clock survives being re-tagged to a different spot or briefly dropping out of view.
5. **Notifier** (`notifier.py`) — on a violation, saves a snapshot photo (real crop in camera mode, a rendered color swatch in sim mode) to `violations/` and calls `send_email`. Email only actually sends if `BIGBACK_EMAIL_USER`/`BIGBACK_EMAIL_PASS` (Gmail + App Password) are set in the environment; otherwise it just prints what it would have sent, so nothing goes out until you opt in.
6. **Viz** — `viz.py` (matplotlib, for `run_sim.py`) and `viz_cv.py` (OpenCV, for `run_camera.py`) are independent renderers for the same `SpotSnapshot` data plus a "Identified Cars" / "Over 72h Limit" side panel pulled from `pipeline.identity`/`pipeline.timers`. Neither can tell whether the source is simulated or real.

`bigback/types.py` holds the two shared dataclasses (`RawObservation`, `SpotSnapshot`) that decouple every stage from the others' internals. `bigback/config.py` centralizes every tunable constant (grid size for sim mode, timing thresholds, debounce counts, YOLO model/classes/confidence) — change knobs there, not inline.

Design intent to preserve: no license plate OCR (unreliable at security-cam angles/lighting) and no full vehicle re-ID model — human-in-the-loop (chairman confirming unmatched cars) is the intended way to keep identity matching simple and robust rather than adding a heavier ML model. `IDENTITY_MATCH_THRESHOLD` was tuned against the simulator's discrete color palette and will likely need retuning against real footage's continuous color variance.
