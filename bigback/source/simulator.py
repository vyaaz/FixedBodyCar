"""Stands in for camera + occupancy model: makes colored 'cars' wander in and
out of a grid of spots so the rest of the pipeline has real traffic to handle.
A car's color is a stand-in for an appearance embedding — see identity.py.
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from .. import config
from ..types import Color, RawObservation
from .base import FrameSource

_PALETTE: List[Color] = [
    (220, 50, 50), (50, 130, 220), (60, 180, 75), (240, 180, 30),
    (160, 60, 200), (250, 130, 50), (50, 200, 200), (200, 90, 160),
]


@dataclass
class _SimCar:
    color: Color
    spot: Optional[int] = None
    leave_at: Optional[float] = None    # sim-hour this car vacates its spot
    return_at: Optional[float] = None   # sim-hour it's allowed to park again


class SimSource(FrameSource):
    def __init__(self, num_spots: int = config.NUM_SPOTS, num_cars: int = 6, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self._num_spots = num_spots
        colors = self._rng.sample(_PALETTE, k=min(num_cars, len(_PALETTE)))
        self._cars = [_SimCar(color=c) for c in colors]

    def step(self, now_hours: float) -> Dict[int, RawObservation]:
        taken = {c.spot for c in self._cars if c.spot is not None}

        for car in self._cars:
            if car.spot is None:
                if car.return_at is not None and now_hours < car.return_at:
                    continue
                free = [s for s in range(self._num_spots) if s not in taken]
                if not free:
                    continue
                car.spot = self._rng.choice(free)
                taken.add(car.spot)
                # mostly short stays, occasionally long enough to trip the 72h rule
                stay = self._rng.choice([self._rng.uniform(4, 40), self._rng.uniform(60, 100)])
                car.leave_at = now_hours + stay
            elif now_hours >= car.leave_at:
                taken.discard(car.spot)
                car.spot = None
                # sometimes returns before the cooldown clears, to exercise that rule too
                car.return_at = now_hours + self._rng.uniform(1, config.COOLDOWN_HOURS * 1.5)

        obs = {s: RawObservation(occupied=False) for s in range(self._num_spots)}
        for car in self._cars:
            if car.spot is not None:
                obs[car.spot] = RawObservation(occupied=True, color=car.color)
        return obs
