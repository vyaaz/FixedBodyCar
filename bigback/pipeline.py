"""Wires source -> state machine -> identity -> timers -> notifier.

This is the one file that changes shape when a real camera replaces the
simulator: pass a CameraSource into Pipeline() instead of the default
SimSource. Everything else is untouched.
"""

import concurrent.futures
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from . import config, notifier
from .identity import IdentityDB
from .source.base import FrameSource
from .source.simulator import SimSource
from .state_machine import ParkEvent, StateMachine
from .timer_service import TimerService
from .types import Color, RawObservation, SpotSnapshot


class Pipeline:
    def __init__(self, source: Optional[FrameSource] = None, clock: Optional[Callable[[], float]] = None,
                 identity: Optional[IdentityDB] = None, timers: Optional[TimerService] = None,
                 persist_path: Optional[Path] = None, persist_meta: Optional[dict] = None):
        """`clock`, if given, is a zero-arg callable returning the current
        sim-hours; used by run_camera.py to track real wall-clock time
        instead of SimSource's fixed per-tick compression, since a live
        feed has no fixed tick cadence to compress against.

        `identity`/`timers`, if given, override the defaults — run_camera.py
        passes pre-loaded ones (see its own load/save) so a crash or restart
        doesn't silently wipe every car's clock back to zero. `persist_path`,
        if given, makes every tick save identity+timers state there (cheap —
        a handful of cars); `persist_meta` is merged into that same file
        (run_camera.py uses it for the clock's epoch, which isn't Pipeline's
        to own but still needs to survive a restart alongside everything
        else)."""
        self.source = source or SimSource()
        self.state_machine = StateMachine()
        self.identity = identity or IdentityDB()
        self.timers = timers or TimerService()
        self._clock = clock
        self.now = 0.0
        self._persist_path = persist_path
        self._persist_meta = persist_meta or {}
        # VehicleDINO's embed() takes ~6-18s on CPU — run it off the main
        # thread so the live feed/detection never blocks on it, even when
        # several cars park in the same tick (e.g. a lot full of cars at
        # startup). `_pending` maps the provisional car_id each such car
        # gets immediately to the in-flight embedding job for it.
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._pending: Dict[str, Tuple[concurrent.futures.Future, Color, object]] = {}

    def tick(self) -> List[SpotSnapshot]:
        self.now = self._clock() if self._clock else self.now + config.SIM_HOURS_PER_TICK
        obs = self.source.step(self.now)

        self._resolve_pending()

        for event in self.state_machine.update(obs):
            if isinstance(event, ParkEvent):
                self._handle_park(event)
            else:
                self.timers.on_leave(event.spot_id, self.now)

        for clock in self.timers.check_violations(self.now):
            car = self.identity.get(clock.car_id)
            crop_for = getattr(self.source, "crop_for", None)
            photo = crop_for(clock.spot_id) if crop_for else None
            notifier.fine_notice(clock.car_id, clock.spot_id, self.now - clock.park_start,
                                  car.display_color if car else None, photo=photo)

        self._persist()
        return self._snapshot(obs)

    def _handle_park(self, event: ParkEvent) -> None:
        embed = getattr(self.source, "embed", None)
        crop_for = getattr(self.source, "crop_for", None)
        crop = crop_for(event.spot_id) if crop_for else None

        if embed is None or crop is None:
            # sim mode (no real crop to embed), or a rare mid-debounce miss
            # in camera mode -> nothing to wait on, resolve immediately.
            embedding = embed(crop) if (embed and crop is not None) else event.color
            car_id, is_new = self.identity.match_or_register(
                embedding, display_color=event.color, exclude=self.timers.active_car_ids())
            self.timers.on_park(event.spot_id, car_id, self.now)
            if is_new:
                notifier.new_car_notice(car_id, event.color, crop)
            return

        # camera mode: start the fine clock right away under a provisional,
        # per-spot id so it never waits on the slow embedding call;
        # _resolve_pending() reconciles this with the real car_id a few
        # ticks later, once the embedding's actually ready.
        temp_id = f"car-pending-{event.spot_id}"
        self.timers.on_park(event.spot_id, temp_id, self.now)
        future = self._executor.submit(embed, crop)
        self._pending[temp_id] = (future, event.color, crop)

    def _resolve_pending(self) -> None:
        done_ids = [temp_id for temp_id, (future, _, _) in self._pending.items() if future.done()]
        for temp_id in done_ids:
            future, display_color, photo = self._pending.pop(temp_id)
            embedding = future.result()
            car_id, is_new = self.identity.match_or_register(
                embedding, display_color=display_color, exclude=self.timers.active_car_ids())
            if car_id != temp_id:
                self.timers.reassign(temp_id, car_id, self.now)
            if is_new:
                notifier.new_car_notice(car_id, display_color, photo)

    def _persist(self) -> None:
        if self._persist_path is None:
            return
        data = {**self._persist_meta, "identity": self.identity.to_dict(), "timers": self.timers.to_dict()}
        tmp = self._persist_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(self._persist_path)  # atomic on both POSIX and Windows -> never a half-written file

    def close(self) -> None:
        self._persist()
        self._executor.shutdown(wait=False, cancel_futures=True)
        release = getattr(self.source, "release", None)
        if release:
            release()

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
