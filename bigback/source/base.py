"""The seam. Everything upstream of this file is replaceable.

To go from simulation to a real backlot, write a CameraSource(FrameSource)
that reads each spot's ROI, runs it through an occupancy model, and returns
the same dict shape SimSource does. Nothing below this file needs to know
the difference.
"""

from abc import ABC, abstractmethod
from typing import Dict

from ..types import RawObservation


class FrameSource(ABC):
    @abstractmethod
    def step(self, now_hours: float) -> Dict[int, RawObservation]:
        """Return one RawObservation per spot for this tick."""
        raise NotImplementedError
