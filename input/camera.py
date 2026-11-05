"""Latest-frame OpenCV camera."""

import glob
import os
from pathlib import Path
import threading
import time

import cv2

# DirectShow is the most compatible Windows webcam backend; Linux stays on V4L2.
_BACKEND = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L2

# V4L2 exposes one extra "metadata" video node per camera. It shares the
# camera's name and reads back index 1, and cannot capture frames.
_NAME_DIR = "/sys/class/video4linux"


def camera_devices(pattern="/dev/video*", names=_NAME_DIR):
    """List the capture nodes a camera picker can offer, without opening any.

    Returns ``{device: label}``. Nodes V4L2 reports with a non-zero ``index``
    are metadata nodes and are skipped; a platform without ``/dev/video*``
    falls back to plain indices.
    """

    choices = {}
    for path in sorted(glob.glob(pattern)):
        node = Path(names, Path(path).name)
        try:
            if node.joinpath("index").read_text().strip() != "0":
                continue
        except OSError:
            pass  # no sysfs entry: trust the node
        try:
            label = node.joinpath("name").read_text().strip()
        except OSError:
            label = ""
        choices[path] = f"{path} · {label}" if label else path
    return choices or {index: f"camera {index}" for index in range(4)}


def first_camera():
    """Return the first selectable camera device, or None when there is none."""
    return next(iter(camera_devices()), None)


class OpenCVCamera:
    def __init__(self, device, width, height, fps, fourcc=None):
        if device is None:
            raise ValueError("no camera selected")
        # A V4L2 node has exactly one owner: a second open fails, which looks
        # like a missing camera unless the message says so.
        self.device = device
        self.capture = cv2.VideoCapture(device, _BACKEND)
        if not self.capture.isOpened():
            raise RuntimeError(
                f"Cannot open camera {device!r} "
                "(another program may already hold it)"
            )
        if fourcc is not None:
            self.capture.set(
                cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc)
            )
        for prop, value in ((cv2.CAP_PROP_FRAME_WIDTH, width),
                            (cv2.CAP_PROP_FRAME_HEIGHT, height),
                            (cv2.CAP_PROP_FPS, fps),
                            (cv2.CAP_PROP_BUFFERSIZE, 1)):
            self.capture.set(prop, value)
        actual = tuple(round(self.capture.get(prop)) for prop in (
            cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS
        ))
        self.description = f"camera {device!r} {actual[0]}x{actual[1]}@{actual[2]}"
        self.condition = threading.Condition()
        self.frame = self.error = None
        self.timestamp = 0.0
        self.sequence = 0
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            ok, frame = self.capture.read()
            if not ok:
                with self.condition:
                    self.error = RuntimeError("Camera frame capture failed")
                    self.condition.notify_all()
                return
            with self.condition:
                self.frame = frame
                # Use the common monotonic acquisition clock consumed by the
                # input and output One Euro filters.
                self.timestamp = time.monotonic()
                self.sequence += 1
                self.condition.notify_all()

    def read(self, after, timeout=2.0):
        deadline = time.monotonic() + timeout
        with self.condition:
            while self.sequence <= after and self.error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for a camera frame")
                self.condition.wait(remaining)
            if self.error is not None:
                raise self.error
            return self.sequence, self.frame.copy(), self.timestamp

    def close(self):
        self.running = False
        self.capture.release()
        self.thread.join(timeout=1.0)
