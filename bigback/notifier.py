"""Fine notices. On a 72h+ violation this saves a snapshot photo of the car
and emails it.

Email actually sends only if BIGBACK_EMAIL_USER / BIGBACK_EMAIL_PASS (a
Gmail address + App Password: https://myaccount.google.com/apppasswords)
are set in the environment. Without them it just prints what it would have
sent — no credentials are hardcoded here, and nothing goes out until you
opt in.
"""

import os
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from . import config
from .types import Color


def fine_notice(car_id: str, spot_id: Optional[int], elapsed_hours: float, color: Optional[Color] = None,
                 photo=None) -> None:
    print(f"[FINE] {car_id} at spot {spot_id} parked {elapsed_hours:.1f}h (>72h) -> notice sent")
    photo_path = _save_photo(config.VIOLATIONS_DIR, f"{car_id}_spot{spot_id}", color, photo)
    send_email(
        subject=f"BigBack violation: {car_id}",
        body=(f"{car_id} has been parked at spot {spot_id} for {elapsed_hours:.1f}h, "
              f"over the {config.VIOLATION_HOURS:.0f}h limit."),
        attachment_path=photo_path,
    )


def new_car_notice(car_id: str, color: Optional[Color] = None, photo=None) -> None:
    """A car didn't match anything in IdentityDB and got auto-tagged. Emails
    a photo so the chairman can confirm/relabel it — the human-in-the-loop
    step CLAUDE.md describes in place of a heavier re-ID model."""
    print(f"[identity] unrecognized car -> notify chairman -> tagged {car_id}")
    photo_path = _save_photo(config.DETECTIONS_DIR, car_id, color, photo)
    send_email(
        subject=f"BigBack: new car tagged {car_id}",
        body=(f"A car that didn't match any known car was just auto-tagged {car_id}. "
              "Confirm or relabel it if you know whose it is."),
        attachment_path=photo_path,
    )


def send_email(subject: str, body: str, attachment_path: Optional[Path] = None) -> None:
    user = os.environ.get("BIGBACK_EMAIL_USER")
    password = os.environ.get("BIGBACK_EMAIL_PASS")
    recipient = os.environ.get("BIGBACK_EMAIL_TO", config.EMAIL_RECIPIENT)

    if not user or not password:
        print(f"[EMAIL] (stub) would send to {recipient}: {subject!r}"
              + (f" with attachment {attachment_path.name}" if attachment_path else ""))
        print("        -> set BIGBACK_EMAIL_USER / BIGBACK_EMAIL_PASS (Gmail App Password) to actually send")
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = recipient
    msg.set_content(body)

    if attachment_path and attachment_path.exists():
        msg.add_attachment(attachment_path.read_bytes(), maintype="image", subtype="png",
                            filename=attachment_path.name)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(user, password)
        smtp.send_message(msg)
    print(f"[EMAIL] sent to {recipient}: {subject!r}")


def _save_photo(directory: Path, name: str, color: Optional[Color], photo=None) -> Path:
    """In camera mode `photo` is a real BGR pixel crop of the car (from
    CameraSource.crop_for) and gets saved as-is. In sim mode — or if the car
    wasn't in view the instant the event fired — there's no real frame, so
    this renders a swatch of the car's tracked color as a stand-in."""
    directory.mkdir(exist_ok=True)
    path = directory / f"{name}.png"

    if photo is not None:
        cv2.imwrite(str(path), photo)
        return path

    fig, ax = plt.subplots(figsize=(2, 2))
    ax.add_patch(patches.Rectangle((0, 0), 1, 1, facecolor=tuple(c / 255 for c in (color or (128, 128, 128)))))
    ax.text(0.5, 0.5, name, ha="center", va="center", fontsize=14, color="white", weight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path
