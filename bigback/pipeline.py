"""Wires source -> state machine -> identity -> timers -> notifier.

This is the one file that changes shape when a real camera replaces the
simulator: pass a CameraSource into Pipeline() instead of the default
SimSource. Everything else is untouched.
"""

from typing import Callable, Dict, List, Optional

from . import config, notifier
from .identity import IdentityDB
from .source.base import FrameSource
from .source.simulator import SimSource
from .state_machine import ParkEvent, StateMachine
from .timer_service import TimerService
from .types import RawObservation, SpotSnapshot


class Pipeline:
    def __init__(self, source: Optional[FrameSource] = None, clock: Optional[Callable[[], float]] = None):
        """`clock`, if given, is a zero-arg callable returning the current
        sim-hours; used by run_camera.py to track real wall-clock time
        instead of SimSource's fixed per-tick compression, since a live
        feed has no fixed tick cadence to compress against."""
        self.source = source or SimSource()
        self.state_machine = StateMachine()
        self.identity = IdentityDB()
        self.timers = TimerService()
        self._clock = clock
        self.now = 0.0

    def tick(self) -> List[SpotSnapshot]:
        self.now = self._clock() if self._clock else self.now + config.SIM_HOURS_PER_TICK
        obs = self.source.step(self.now)

        for event in self.state_machine.update(obs):
            if isinstance(event, ParkEvent):
                car_id, is_new = self.identity.match_or_register(
                    event.color, exclude=self.timers.active_car_ids())
                self.timers.on_park(event.spot_id, car_id, self.now)
                if is_new:
                    crop_for = getattr(self.source, "crop_for", None)
                    photo = crop_for(event.spot_id) if crop_for else None
                    notifier.new_car_notice(car_id, event.color, photo)
            else:
                self.timers.on_leave(event.spot_id, self.now)

        for clock in self.timers.check_violations(self.now):
            car = self.identity.get(clock.car_id)
            crop_for = getattr(self.source, "crop_for", None)
            photo = crop_for(clock.spot_id) if crop_for else None
            notifier.fine_notice(clock.car_id, clock.spot_id, self.now - clock.park_start,
                                  car.color if car else None, photo=photo)

        return self._snapshot(obs)

    def _snapshot(self, obs: Dict[int, RawObservation]) -> List[SpotSnapshot]:
        snaps = []
        for spot_id in obs:
            car_id = self.timers.car_at(spot_id)
            snaps.append(SpotSnapshot(
                spot_id=spot_id,
                occupied=self.state_machine.is_occupied(spot_id),
                car_id=car_id,
                color=self.state_machine.color_at(spot_id),
                elapsed_hours=self.timers.elapsed_hours(car_id, self.now) if car_id else 0.0,
                violating=self.timers.is_flagged(car_id) if car_id else False,
            ))
        return snaps
