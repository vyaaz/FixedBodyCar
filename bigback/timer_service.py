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
