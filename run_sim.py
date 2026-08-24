"""Entry point: python run_sim.py"""

from bigback import viz
from bigback.pipeline import Pipeline

if __name__ == "__main__":
    pipeline = Pipeline()
    anim = viz.run(pipeline)
