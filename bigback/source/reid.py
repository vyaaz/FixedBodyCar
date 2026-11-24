"""Vehicle re-identification embedding, replacing the old mean-color identity
cue entirely. A whole-box (or even banded) mean RGB collapses every
same-colored car onto nearly the same point in color space — two black cars
are indistinguishable there no matter how tight the match threshold is, which
is what was causing unrelated cars to inherit each other's parking clock.

wms2537/VehicleDINO (DINOv2 ViT-B/14 + LoRA, Apache-2.0, ONNX) is trained
specifically for vehicle re-identification and outputs a 256-dim L2-normalized
embedding meant for cosine-similarity matching. It's heavier than a pixel
mean, but identity.py only calls this once per confirmed ParkEvent
(bigback/pipeline.py), not once per tracked frame, so it doesn't reintroduce
the frame-rate problem that made bigback/source/camera.py switch trackers.

https://huggingface.co/wms2537/VehicleDINO — trained/evaluated on VeRi-776
(street-level traffic-cam footage), not a steep downward security-cam angle,
so real-world match distances should be sanity-checked against the console's
[identity] dist= logging rather than assumed from VeRi-776's reported mAP.
"""

import urllib.request
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import onnxruntime as ort

_MODEL_URL = "https://huggingface.co/wms2537/VehicleDINO/resolve/main/vehicledino_dinov2_int8.onnx"
_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "vehicledino_dinov2_int8.onnx"
_INPUT_SIZE = 560
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _ensure_model() -> Path:
    if not _MODEL_PATH.exists():
        print(f"[reid] downloading VehicleDINO weights (~139MB, one-time) -> {_MODEL_PATH}")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[reid] download complete")
    return _MODEL_PATH


class VehicleReID:
    """One instance per CameraSource; wraps the VehicleDINO ONNX session.

    pipeline.py runs embed() on a background thread so it doesn't block the
    live feed, but by default onnxruntime happily grabs every CPU core for
    a single inference — on a 4-core machine that starves the main thread's
    concurrent YOLO tracking just as badly as blocking would have, just
    spread across more ticks instead of one. Capping it to 2 threads leaves
    the other cores for YOLO; embed() itself takes somewhat longer, which is
    fine since nothing is waiting on it synchronously."""

    def __init__(self):
        path = _ensure_model()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        self._session = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])

    def embed(self, crop_bgr) -> Tuple[float, ...]:
        """`crop_bgr` is an OpenCV BGR pixel crop (e.g. CameraSource.crop_for's
        output). Returns a 256-dim L2-normalized embedding as a plain tuple,
        so it's a drop-in `Color` for IdentityDB."""
        resized = cv2.resize(crop_bgr, (_INPUT_SIZE, _INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
        chw = normalized.transpose(2, 0, 1)[None, ...].astype(np.float32)  # (1, 3, 560, 560)
        (embeds,) = self._session.run(["reid_embeds"], {"images": chw})
        return tuple(float(x) for x in embeds.reshape(-1))


def cosine_distance(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    """0 = same direction (identical), 2 = opposite. VehicleDINO's embeddings
    are already L2-normalized, so the dot product alone is cosine similarity."""
    dot = sum(x * y for x, y in zip(a, b))
    return 1.0 - dot
