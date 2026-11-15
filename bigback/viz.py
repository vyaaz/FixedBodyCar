"""Live view: draws the spot grid as colored rectangles, plus two side
panels — every car identity.py has ever tagged, and the subset currently
over the 72h limit. Only consumes SpotSnapshot objects (grid) and the
pipeline's identity/timer state (panels), so it can't tell whether Pipeline
is backed by the simulator or a real camera feed.
"""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from . import config
from .timer_service import hms

EMPTY_COLOR = (0.9, 0.9, 0.9)
VIOLATION_COLOR = "#FFD400"  # yellow highlight for spots/cars over the 72h limit


def run(pipeline, interval_ms: int = config.TICK_INTERVAL_MS):
    fig = plt.figure(figsize=(config.GRID_COLS * 2 + 3, config.GRID_ROWS * 2))
    gs = fig.add_gridspec(nrows=2, ncols=2, width_ratios=[config.GRID_COLS, 2], hspace=0.35)
    ax_grid = fig.add_subplot(gs[:, 0])
    ax_roster = fig.add_subplot(gs[0, 1])
    ax_violations = fig.add_subplot(gs[1, 1])

    ax_grid.set_xlim(0, config.GRID_COLS)
    ax_grid.set_ylim(0, config.GRID_ROWS)
    ax_grid.set_aspect("equal")
    ax_grid.axis("off")
    ax_grid.set_title("BigBack v1 — simulated backlot")

    boxes, labels = [], []
    for spot_id in range(config.NUM_SPOTS):
        row, col = divmod(spot_id, config.GRID_COLS)
        rect = patches.Rectangle((col + 0.05, row + 0.05), 0.9, 0.9,
                                  facecolor=EMPTY_COLOR, edgecolor="black")
        ax_grid.add_patch(rect)
        label = ax_grid.text(col + 0.5, row + 0.5, "", ha="center", va="center", fontsize=7)
        boxes.append(rect)
        labels.append(label)

    def frame(_):
        for snap in pipeline.tick():
            rect, label = boxes[snap.spot_id], labels[snap.spot_id]
            if snap.occupied:
                rect.set_facecolor(tuple(c / 255 for c in snap.color))
                rect.set_edgecolor(VIOLATION_COLOR if snap.violating else "black")
                rect.set_linewidth(4 if snap.violating else 1)
                timer = f"{hms(snap.elapsed_hours)} / {hms(config.VIOLATION_HOURS)}"
                label.set_text(f"{snap.car_id}\n{timer}" + (" ⚠" if snap.violating else ""))
            else:
                rect.set_facecolor(EMPTY_COLOR)
                rect.set_edgecolor("black")
                rect.set_linewidth(1)
                label.set_text("")
        _draw_roster(ax_roster, pipeline)
        _draw_violations(ax_violations, pipeline)
        return boxes + labels

    anim = FuncAnimation(fig, frame, interval=interval_ms, cache_frame_data=False)
    plt.show()
    return anim  # caller must keep a reference alive or the GC kills the animation


def _draw_roster(ax, pipeline):
    """Side panel standing in for a 'tab' of every car identity.py has
    tagged so far — matplotlib has no real tab widget, so this redraws as a
    live list each frame instead."""
    ax.clear()
    ax.axis("off")
    ax.set_title("Identified Cars", fontsize=10)

    cars = pipeline.identity.all_cars()
    if not cars:
        ax.text(0.05, 0.95, "(none yet)", transform=ax.transAxes, va="top", fontsize=9)
        return

    for i, car in enumerate(cars):
        y = 0.95 - i * 0.11
        if y < 0:
            ax.text(0.05, 0.02, "...", transform=ax.transAxes, fontsize=9)
            break

        clock = pipeline.timers.clock_for(car.car_id)
        if clock and clock.spot_id is not None:
            elapsed = pipeline.timers.elapsed_hours(car.car_id, pipeline.now)
            status = f"spot {clock.spot_id} · {hms(elapsed)}"
            if clock.flagged:
                status += " ⚠"
        else:
            status = "off-lot"

        _draw_entry(ax, y, car, status)


def _draw_violations(ax, pipeline):
    """List of cars currently flagged as over the 72h limit — a car stays
    listed here until it serves a full cooldown off-lot and re-parks, same
    as TimerService.flagged."""
    ax.clear()
    ax.axis("off")
    ax.set_title("Over 72h Limit", fontsize=10, color="#B8860B")

    violators = []
    for car in pipeline.identity.all_cars():
        clock = pipeline.timers.clock_for(car.car_id)
        if clock and clock.flagged:
            violators.append((car, clock))

    if not violators:
        ax.text(0.05, 0.95, "(none)", transform=ax.transAxes, va="top", fontsize=9)
        return

    for i, (car, clock) in enumerate(violators):
        y = 0.95 - i * 0.11
        if y < 0:
            ax.text(0.05, 0.02, "...", transform=ax.transAxes, fontsize=9)
            break

        where = f"spot {clock.spot_id}" if clock.spot_id is not None else "off-lot"
        elapsed = pipeline.now - clock.park_start
        status = f"{where} · {hms(elapsed)} ⚠"
        _draw_entry(ax, y, car, status, edgecolor=VIOLATION_COLOR)


def _draw_entry(ax, y, car, status, edgecolor="black"):
    ax.add_patch(patches.Rectangle(
        (0.02, y - 0.025), 0.06, 0.06, transform=ax.transAxes,
        facecolor=tuple(c / 255 for c in car.color), edgecolor=edgecolor, linewidth=2, clip_on=False,
    ))
    ax.text(0.12, y, f"{car.car_id}\n{status}", transform=ax.transAxes, va="center", fontsize=8)
