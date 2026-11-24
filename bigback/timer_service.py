"""Owns the 72h / cooldown rule. Keyed by car_id (not spot_id) so a car's
clock survives it being re-tagged to a different spot or briefly dropping out
of view — only a full cooldown period off-lot resets it.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from . import config


def hms(hours: float) -> str:
    """Format an elapsed duration (in hours, the unit clocks are kept in) as
    HH:MM:SS for the live view. Hours can exceed 24 and just keep counting up
    (e.g. 80:14:03), matching how the 72h rule is phrased."""
    total_seconds = int(round(max(hours, 0.0) * 3600))
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


@dataclass
class CarClock:
    car_id: str
    spot_id: Optional[int]
    park_start: float
    left_at: Optional[float] = None
    flagged: bool = False


class TimerService:
    def __init__(self):
        self._clocks: Dict[str, CarClock] = {}
        self._spot_to_car: Dict[int, str] = {}

    def on_park(self, spot_id: int, car_id: str, now: float) -> None:
        clock = self._clocks.get(car_id)
        if clock is None:
            clock = CarClock(car_id, spot_id, park_start=now)
        else:
            if clock.left_at is not None and now - clock.left_at >= config.COOLDOWN_HOURS:
                print(f"[timer] {car_id} cooldown served ({now - clock.left_at:.1f}h off-lot) -> clock reset")
                clock.park_start = now  # full cooldown served -> fresh clock
                clock.flagged = False
            elif clock.left_at is not None:
                print(f"[timer] {car_id} re-parked after {now - clock.left_at:.2f}h off-lot "
                      f"(<{config.COOLDOWN_HOURS}h cooldown) -> continuing clock, "
                      f"{now - clock.park_start:.1f}h elapsed since original park_start")
            clock.spot_id = spot_id
            clock.left_at = None
        self._clocks[car_id] = clock
        self._spot_to_car[spot_id] = car_id

    def reassign(self, old_id: str, new_id: str, now: float) -> None:
        """Folds old_id's clock into new_id's. Used when a car starts under a
        provisional id (so its clock isn't blocked on the slow async Re-ID
        embedding — see Pipeline._handle_park) and identity.py later
        resolves it to a different, already-known car_id — same physical
        car, recognized from an earlier sighting. Whichever clock started
        earliest keeps its park_start/flagged (that's the real violation
        history); old_id's spot/left_at always wins since it reflects what's
        actually happening right now."""
        old = self._clocks.pop(old_id, None)
        if old is None:
            return
        existing = self._clocks.get(new_id)
        if existing is None:
            old.car_id = new_id
            self._clocks[new_id] = old
        else:
            if old.park_start < existing.park_start:
                existing.park_start = old.park_start
                existing.flagged = existing.flagged or old.flagged
            existing.spot_id = old.spot_id
            existing.left_at = old.left_at
        for spot, car_id in list(self._spot_to_car.items()):
            if car_id == old_id:
                self._spot_to_car[spot] = new_id

    def on_leave(self, spot_id: int, now: float) -> None:
        car_id = self._spot_to_car.pop(spot_id, None)
        if car_id and car_id in self._clocks:
            self._clocks[car_id].spot_id = None
            self._clocks[car_id].left_at = now

    def check_violations(self, now: float) -> List[CarClock]:
        newly_violating = []
        for clock in self._clocks.values():
            if clock.spot_id is None or clock.flagged:
                continue
            if now - clock.park_start >= config.VIOLATION_HOURS:
                clock.flagged = True
                newly_violating.append(clock)
        return newly_violating

    def elapsed_hours(self, car_id: str, now: float) -> float:
        return now - self._clocks[car_id].park_start

    def car_at(self, spot_id: int) -> Optional[str]:
        return self._spot_to_car.get(spot_id)

    def active_car_ids(self) -> Set[str]:
        """car_ids currently parked somewhere — i.e. occupying a key right now."""
        return set(self._spot_to_car.values())

    def is_flagged(self, car_id: str) -> bool:
        return self._clocks[car_id].flagged

    def clock_for(self, car_id: str) -> Optional[CarClock]:
        return self._clocks.get(car_id)

    def to_dict(self) -> dict:
        """Plain-JSON snapshot for disk persistence (see Pipeline._persist and
        run_camera.py). `park_start`/`left_at` are hours since whatever epoch
        the live `clock` callable used — meaningless on their own after a
        restart, which is why run_camera.py also persists/restores that
        epoch (`start_epoch`) alongside this."""
        return {
            "clocks": [
                {"car_id": c.car_id, "spot_id": c.spot_id, "park_start": c.park_start,
                 "left_at": c.left_at, "flagged": c.flagged}
                for c in self._clocks.values()
            ],
            "spot_to_car": {str(spot): car_id for spot, car_id in self._spot_to_car.items()},
        }

    def load_dict(self, data: dict) -> None:
        self._clocks = {
            c["car_id"]: CarClock(c["car_id"], c["spot_id"], c["park_start"], c["left_at"], c["flagged"])
            for c in data.get("clocks", [])
        }
        self._spot_to_car = {int(spot): car_id for spot, car_id in data.get("spot_to_car", {}).items()}
