"""Debounces raw per-key readings into trustworthy park/leave events.

A transition only fires once it's held for DEBOUNCE_TICKS in a row, so one
flaky reading doesn't start or stop a timer. Dict-based rather than a fixed
NUM_SPOTS-sized array, since the key isn't always a physical parking spot:
SimSource uses fixed spot ids 0..NUM_SPOTS-1, but CameraSource uses YOLO
track ids, which are arbitrary and grow over a session.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from . import config
from .types import Color, RawObservation


@dataclass
class ParkEvent:
    spot_id: int
    color: Optional[Color]


@dataclass
class LeaveEvent:
    spot_id: int


Event = Union[ParkEvent, LeaveEvent]


class StateMachine:
    def __init__(self):
        self._confirmed: Dict[int, bool] = {}
        self._pending: Dict[int, Optional[bool]] = {}
        self._pending_count: Dict[int, int] = {}
        self._last_color: Dict[int, Optional[Color]] = {}

    def is_occupied(self, spot_id: int) -> bool:
        return self._confirmed.get(spot_id, False)

    def color_at(self, spot_id: int) -> Optional[Color]:
        """Last color seen for this spot/track. Confirmed occupancy lags the
        raw feed by up to DEBOUNCE_TICKS, so a key can still read as occupied
        for a tick or two after the raw color drops to None (car left) —
        callers that display a color for an occupied spot should use this,
        not the raw observation's color, to avoid a stale-true/empty-color
        mismatch."""
        return self._last_color.get(spot_id)

    def update(self, obs: Dict[int, RawObservation]) -> List[Event]:
        events: List[Event] = []
        for spot_id, o in obs.items():
            if o.color is not None:
                self._last_color[spot_id] = o.color

            confirmed = self._confirmed.get(spot_id, False)
            if o.occupied == confirmed:
                self._pending[spot_id] = None
                self._pending_count[spot_id] = 0
                continue

            if self._pending.get(spot_id) == o.occupied:
                self._pending_count[spot_id] = self._pending_count.get(spot_id, 0) + 1
            else:
                self._pending[spot_id] = o.occupied
                self._pending_count[spot_id] = 1

            if self._pending_count[spot_id] >= config.DEBOUNCE_TICKS:
                self._confirmed[spot_id] = o.occupied
                self._pending[spot_id] = None
                self._pending_count[spot_id] = 0
                events.append(ParkEvent(spot_id, o.color) if o.occupied else LeaveEvent(spot_id))
        return events
