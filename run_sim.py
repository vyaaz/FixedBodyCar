"""Entry point: python run_sim.py

Clock speed follows config.SIM_DEMO_SPEED: real wall-clock hours (72h == 72
real hours, true HH:MM:SS timer) while False, compressed (20s == 72h) once
flipped to True for a quick demo.
"""

import time

from bigback import config, viz
from bigback.pipeline import Pipeline

if __name__ == "__main__":
    if config.SIM_DEMO_SPEED:
        pipeline = Pipeline()  # SimSource + per-tick time compression
    else:
        start = time.time()
        pipeline = Pipeline(clock=lambda: (time.time() - start) / 3600.0)
    anim = viz.run(pipeline)
