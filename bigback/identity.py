"""Matches an appearance cue against known cars.

v1 uses raw RGB distance on the sim's box color as a stand-in embedding.
Swap `_distance` for cosine/L2 over a real embedding model's output and
IdentityDB works unchanged — everything downstream only ever sees a car_id.
Unmatched cars get auto-registered here; in prod that's where the chairman
gets a photo to confirm/tag before the DB grows.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from . import config
from .types import Color


@dataclass
class KnownCar:
    car_id: str
    color: Color


class IdentityDB:
    def __init__(self):
        self._cars: Dict[str, KnownCar] = {}
        self._next_id = 1

    def match_or_register(self, color: Color) -> str:
        best_id, best_dist = None, float("inf")
        for car in self._cars.values():
            d = _distance(car.color, color)
            if d < best_dist:
                best_id, best_dist = car.car_id, d

        if best_id is not None and best_dist <= config.IDENTITY_MATCH_THRESHOLD:
            return best_id

        car_id = f"car-{self._next_id}"
        self._next_id += 1
        self._cars[car_id] = KnownCar(car_id, color)
        print(f"[identity] unrecognized car -> notify chairman -> tagged {car_id}")
        return car_id

    def get(self, car_id: str) -> Optional[KnownCar]:
        return self._cars.get(car_id)

    def all_cars(self) -> List[KnownCar]:
        return sorted(self._cars.values(), key=lambda c: c.car_id)


def _distance(a: Color, b: Color) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
