"""Configured hand input."""

from config import INPUT_SOURCE
from .camera import first_camera


def create_source(kind=None, device=None, calibrate=True):
    """Build one input source.

    ``kind`` defaults to ``config.INPUT_SOURCE``. ``device`` is the camera the
    console picked; None falls back to the first selectable camera. ``calibrate``
    False imports a saved MANUS calibration instead of running the interactive
    terminal wizard, because a web button cannot answer its Enter prompts.
    """

    kind = INPUT_SOURCE if kind is None else kind
    if kind == "wilor":
        from .wilor.source import WilorSource
        return WilorSource(device or first_camera())
    if kind == "mediapipe":
        from .mediapipe.source import MediaPipeSource
        return MediaPipeSource(device or first_camera())
    if kind == "manus":
        from .manus.calibration import (
            ensure_calibration, import_saved_calibration,
        )
        from .manus.source import ManusSource
        source = ManusSource()
        try:
            path = (
                ensure_calibration(source.transport) if calibrate
                else import_saved_calibration(source.transport)
            )
        except BaseException:
            source.close()
            raise
        if path is not None:
            source.mark_calibrated(path)
        return source
    raise ValueError(f"Unsupported input source: {kind!r}")

