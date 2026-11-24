"""Matches an appearance cue against known cars.

Sim mode compares raw RGB distance on the simulator's flat box colors.
Camera mode compares cosine distance over VehicleDINO's Re-ID embedding
(bigback/source/reid.py) — a plain mean pixel color isn't discriminative
enough to tell two same-colored real cars apart (see CLAUDE.md). Either way
IdentityDB only ever deals in an opaque `Color` vector + a pluggable
distance function; everything downstream only ever sees a car_id.
Unmatched cars get auto-registered here; in prod that's where the chairman
gets a photo to confirm/tag before the DB grows.
"""

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from . import config
from .types import Color

DistanceFn = Callable[[Color, Color], float]


@dataclass
class KnownCar:
    car_id: str
    embedding: Color        # compared via `distance_fn` to recognize this car again
    display_color: Color    # plain RGB; cosmetic only (panel swatch, synthetic violation photo)


class IdentityDB:
    def __init__(self, threshold: Optional[float] = None, distance_fn: Optional[DistanceFn] = None):
        """`threshold` defaults to config.IDENTITY_MATCH_THRESHOLD (tuned for
        the simulator's flat RGB distance). `distance_fn` defaults to
        Euclidean RGB distance; run_camera.py passes cosine distance over
        VehicleDINO embeddings instead, with its own
        CAMERA_IDENTITY_MATCH_THRESHOLD — the two modes' embeddings live on
        completely different scales, so neither the threshold nor the
        distance function can be shared between them."""
        self._cars: Dict[str, KnownCar] = {}
        self._next_id = 1
        self._threshold = config.IDENTITY_MATCH_THRESHOLD if threshold is None else threshold
        self._distance_fn = distance_fn or _euclidean

    def match_or_register(self, embedding: Color, display_color: Color,
                           exclude: Optional[Iterable[str]] = None) -> Tuple[str, bool]:
        """`exclude` is the set of car_ids already parked *right now* under
        another key. A car can't be in two spots at once, so those are never
        match candidates — without this, several cars that look alike to the
        embedding and are in the lot at the same time all collapse onto the
        first one's car_id. A car that has left is not excluded, so a
        returning / re-tracked car still re-links by its embedding.

        Returns (car_id, is_new) — is_new tells the caller whether to fire
        the "notify chairman" photo email (only on first sighting, not on
        every re-match)."""
        excluded = set(exclude or ())
        best_id, best_dist = None, float("inf")
        for car in self._cars.values():
            if car.car_id in excluded:
                continue
            d = self._distance_fn(car.embedding, embedding)
            if d < best_dist:
                best_id, best_dist = car.car_id, d

        if best_id is not None and best_dist <= self._threshold:
            print(f"[identity] matched {best_id} (dist={best_dist:.3f}, threshold={self._threshold})")
            return best_id, False

        car_id = f"car-{self._next_id}"
        self._next_id += 1
        self._cars[car_id] = KnownCar(car_id, embedding, display_color)
        return car_id, True

    def get(self, car_id: str) -> Optional[KnownCar]:
        return self._cars.get(car_id)

    def all_cars(self) -> List[KnownCar]:
        return sorted(self._cars.values(), key=lambda c: c.car_id)

    def to_dict(self) -> dict:
        """Plain-JSON snapshot for disk persistence — see TimerService's
        matching methods and run_camera.py's periodic save/load. Without
        this, every restart forgets every known car, so a car that was
        already parked gets re-registered as brand new and its clock
        silently restarts from zero."""
        return {
            "next_id": self._next_id,
            "cars": [
                {"car_id": c.car_id, "embedding": list(c.embedding), "display_color": list(c.display_color)}
                for c in self._cars.values()
            ],
        }

    def load_dict(self, data: dict) -> None:
        """Restores a snapshot from to_dict() into this (already-constructed,
        so threshold/distance_fn stay as configured) instance."""
        self._next_id = data.get("next_id", 1)
        self._cars = {
            c["car_id"]: KnownCar(c["car_id"], tuple(c["embedding"]), tuple(c["display_color"]))
            for c in data.get("cars", [])
        }


def _euclidean(a: Color, b: Color) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
