"""One-off detector sanity check: python test_image.py <path/to/image.jpg> [conf]

Runs YOLO directly against a single image and saves the annotated result —
useful for checking detection quality (model, classes, confidence) before
wiring up a live feed. Doesn't touch Pipeline/state machine/tracking at all.
"""

import sys

import cv2
from ultralytics import YOLO

from bigback import config

if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("usage: python test_image.py <path/to/image.jpg> [conf]")
        raise SystemExit(1)

    image_path = sys.argv[1]
    conf = float(sys.argv[2]) if len(sys.argv) == 3 else config.YOLO_CONF

    model = YOLO(config.YOLO_MODEL)
    results = model.predict(source=image_path, classes=config.YOLO_CLASSES, conf=conf, verbose=True)

    boxes = results[0].boxes
    print(f"\n{len(boxes)} detection(s) at conf >= {conf} (classes={config.YOLO_CLASSES})")
    for box in boxes:
        print(f"  class={int(box.cls[0])} conf={float(box.conf[0]):.2f} xyxy={box.xyxy[0].tolist()}")

    out_path = "detect_result.png"
    cv2.imwrite(out_path, results[0].plot())
    print(f"annotated image saved to {out_path}")
