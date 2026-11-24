"""Shared data shapes. Keeping these separate is what lets every stage of the
pipeline (perception, state machine, identity, timers, viz) stay ignorant of
each other's internals."""

from dataclasses import dataclass
from typing import Optional, Tuple

# (r, g, b) for the simulator's flat box colors. CameraSource's real embedding
# is several RGB bands concatenated (see camera.py's _color_embedding) — still
# fixed-length within a single run, which is all IdentityDB's distance math needs.
Color = Tuple[int, ...]


@dataclass
class RawObservation:
    """One spot, one tick, straight out of perception.

    In v1 the simulator fabricates this. In prod: `occupied` comes from the
    per-spot binary classifier, `color` from a crop/embedding of the car —
    same shape either way, so nothing downstream needs to change.
    """
    occupied: bool
    color: Optional[Color] = None


@dataclass
class SpotSnapshot:
    """Everything the UI needs to draw one spot, for one frame."""
    spot_id: int
    occupied: bool
    car_id: Optional[str] = None
    color: Optional[Color] = None
    elapsed_hours: float = 0.0
    violating: bool = False
