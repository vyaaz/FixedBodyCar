"""Matches an appearance cue against known cars.

v1 uses raw RGB distance on the sim's box color as a stand-in embedding.
Swap `_distance` for cosine/L2 over a real embedding model's output and
IdentityDB works unchanged — everything downstream only ever sees a car_id.
Unmatched cars get auto-registered here; in prod that's where the chairman
gets a photo to confirm/tag before the DB grows.
"""

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

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

    def match_or_register(self, color: Color, exclude: Optional[Iterable[str]] = None) -> Tuple[str, bool]:
        """`exclude` is the set of car_ids already parked *right now* under
        another key. A car can't be in two spots at once, so those are never
        match candidates — without this, several similarly-colored cars in
        the lot at the same time all collapse onto the first one's car_id
        (mean crop color is a weak cue; see IDENTITY_MATCH_THRESHOLD note in
        CLAUDE.md). A car that has left is not excluded, so a returning /
        re-tracked car still re-links by color.

        Returns (car_id, is_new) — is_new tells the caller whether to fire
        the "notify chairman" photo email (only on first sighting, not on
        every re-match)."""
        excluded = set(exclude or ())
        best_id, best_dist = None, float("inf")
        for car in self._cars.values():
            if car.car_id in excluded:
                continue
            d = _distance(car.color, color)
            if d < best_dist:
                best_id, best_dist = car.car_id, d

        if best_id is not None and best_dist <= config.IDENTITY_MATCH_THRESHOLD:
            print(f"[identity] color-matched {color} to {best_id} "
                  f"(dist={best_dist:.1f}, threshold={config.IDENTITY_MATCH_THRESHOLD})")
            return best_id, False

        car_id = f"car-{self._next_id}"
        self._next_id += 1
        self._cars[car_id] = KnownCar(car_id, color)
        return car_id, True

    def get(self, car_id: str) -> Optional[KnownCar]:
        return self._cars.get(car_id)

    def all_cars(self) -> List[KnownCar]:
        return sorted(self._cars.values(), key=lambda c: c.car_id)


def _distance(a: Color, b: Color) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
