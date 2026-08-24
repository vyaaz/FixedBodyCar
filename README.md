# BigBack — Backlot Parking Tracker

CV system for Theta Xi backlot. Tracks per-spot occupancy, times parked cars, flags 72h+ violations.

## Rule
- Max 72h parked continuously.
- After moving, car must stay off-lot 5-10h (cooldown) before re-parking resets clock.
- Violation -> fine notice to brother.

## Pipeline
1. **Camera** - continuous feed of backlot, fixed ROI per spot.
2. **Occupancy model** - per-spot binary classifier/detector: empty vs occupied. No per-car training needed (generalizes to any vehicle).
3. **State machine** - debounced empty<->occupied transitions per spot.
   - empty->occupied: start timer, take snapshot.
   - occupied->empty: stop/reset timer.
4. **Identity match** - compare snapshot against known-car embedding DB.
   - High confidence -> auto-tag brother.
   - Low/no match -> notify chairman with photo -> human confirms/tags -> added to DB (system learns over time).
5. **Timer service** - tracks per (spot, car) pair, not just per spot. Enforces 72h limit and 5-10h cooldown before reset.
6. **Fine notice** - on 72h+ violation, notify brother + log.

## Notes
- No license plate OCR (unreliable at security-cam angles/lighting).
- No full vehicle re-ID needed — human-in-the-loop handles unknown cars, keeps model simple/robust.

## V1 — simulation

No camera yet, so v1 replaces steps 1-2 (camera + occupancy model) with a
simulator that spawns colored boxes onto a 4x3 grid and moves them in and
out. Everything downstream — debounce, identity matching, timers, fines —
is the real logic, just fed synthetic data. Sim time runs ~1.5h per tick so
72h violations show up within a short animation instead of 3 real days.

```
pip install -r requirements.txt
python run_sim.py
```

```
bigback/
  config.py            tunable constants (grid size, 72h/cooldown, thresholds)
  types.py              shared dataclasses (RawObservation, SpotSnapshot)
  source/
    base.py             FrameSource interface — the seam for a real camera
    simulator.py         v1: fabricates per-spot occupancy + a color-as-embedding
  state_machine.py       debounces raw readings into park/leave events
  identity.py             matches a car's color against known cars (embedding stand-in)
  timer_service.py        72h limit + cooldown-before-reset, keyed by car
  notifier.py              fine notices (prints for now)
  pipeline.py                wires it all together, one tick at a time
  viz.py                      live matplotlib view of the grid
run_sim.py                     entry point
```

**To go live later:** implement `CameraSource(FrameSource)` in `source/`
that reads each spot's ROI and runs the real occupancy model + embedding
extractor, returning the same `RawObservation` shape `SimSource` does. Swap
it into `Pipeline(source=...)`. Nothing else changes.
