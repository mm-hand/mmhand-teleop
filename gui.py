"""NiceGUI console: one connection bar, one status bar, one 2x2 grid.

The tracking loop, the retarget worker and the two 3D panels run off the page;
the page only reads snapshots, swaps devices and pushes the latest arrays into
its own scenes. Every snapshot value is a single attribute assignment, so the
page can read a slightly stale value without any lock.
"""

import threading
import time
from pathlib import Path

import cv2
from fastapi.responses import Response
from nicegui import app, native, run, ui
import numpy as np

import config as C
import scene
from config import INPUT_SOURCE, WEB_FPS, WEB_PORT
from input import create_source
from input.camera import OpenCVCamera, camera_devices, first_camera
from retarget import RetargetWorker
from ros import RosOutput

INPUT_KINDS = {kind: kind for kind in ("mediapipe", "wilor", "manus")}
BLACK = cv2.imencode(".jpg", np.zeros((180, 320, 3), np.uint8))[1].tobytes()
FRAME_PERIOD = 1 / WEB_FPS
SCENE_PERIOD = 0.05  # 3D panels, Hz-limited by the scene's own 20 fps loop

CSS = """
body{background:#F5F7FA;color:#1e293b}
.nicegui-content{padding:0;gap:0;height:100dvh;overflow:hidden}
.shell{display:flex;flex-direction:column;height:100dvh;width:100%;overflow:hidden}
.connection-bar{min-height:44px;padding:4px 12px;background:white;gap:8px;flex-shrink:0}
.connection-bar .q-field{font-size:12px}
.connection-bar .q-field__control,.connection-bar .q-field__marginal{height:32px;min-height:32px}
.status-bar{min-height:32px;padding:4px 12px;background:#eef2f7;gap:14px;flex-shrink:0}
.status-chip{padding:4px 8px;border-radius:5px;color:#1e293b;font-weight:500;white-space:nowrap}
.q-btn{text-transform:none;border-radius:6px;font-size:12px}
.grid{flex:1;min-height:0;display:grid;grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;gap:6px;padding:6px}
.panel{display:flex;flex-direction:column;min-height:0;overflow:hidden;background:#181d25;border:1px solid #343b48}
.panel-head{min-height:26px;padding:2px 10px;gap:8px;color:#dff;font-size:12px;flex-shrink:0}
.panel-status{margin-left:auto;color:#9bd}
.panel-body{flex:1;min-height:0;position:relative;overflow:hidden;display:flex}
.panel-body>*,.panel-body iframe{width:100%;height:100%;display:block;border:0}
.panel-body img{object-fit:contain}
"""


def status_bg(value):
    """Status chip colour, using the MM-Hand console's red/yellow/green rules."""
    head = str(value).upper().split(" ")[0]
    if head in ("RUNNING", "LIVE", "TRACKING", "LEFT", "RIGHT", "READY", "ON",
                "CONNECTING"):
        return "#bbf7d0"
    if head in ("WAITING", "STALE", "OFFLINE", "OFF", "UNAVAILABLE"):
        return "#fde68a"
    if head in ("INVALID", "LOST", "ERROR", "FAILED"):
        return "#fecaca"
    return "#e2e8f0"


def jpeg(image):
    ok, encoded = cv2.imencode(".jpg", image, (cv2.IMWRITE_JPEG_QUALITY, 82))
    return encoded.tobytes() if ok else None


def age_text(age, on):
    """Render one connection's Rx age; age is seconds, None means no frame yet."""
    if not on:
        return "OFF"
    return "—" if age is None else f"{age * 1000:.0f} ms"


class Teleop:
    """Own the devices, the solver and the scenes; the page only talks to this."""

    def __init__(self, kind=INPUT_SOURCE, sim=False, ros=False):
        self.worker = RetargetWorker()
        self.source = self.palm = self.ros = self.sim = self.robot = None
        self.human_points = None
        self.status = "INPUT OFF"
        self.retarget_ms = None
        self.input_at = self.palm_at = None
        self.preview = self.palm_preview = BLACK
        self.preview_seq = self.palm_seq = 0
        self.kind, self.use_sim, self.use_ros = kind, sim, ros
        self._stop = threading.Event()
        self._closed = False

    def start(self):
        """Start every configured subsystem; one failure never stops the console."""
        if self.use_sim:
            try:
                from simulation.process import GraspSimulationProcess
                self.sim = GraspSimulationProcess()
            except Exception as error:
                self.status = f"ERROR · sim: {error}"
        try:
            self.connect_input(self.kind)
        except Exception:  # the reason is already in self.status
            pass
        if self.use_ros:
            try:
                self.set_ros(True)
            except Exception as error:
                self.status = f"ERROR · ROS 2: {error}"
        for name, target in (("teleop-track", self._loop),
                             ("teleop-palm", self._palm_loop)):
            threading.Thread(target=target, name=name, daemon=True).start()

    def connect_input(self, kind, device=None):
        device = device or first_camera()
        palm = self.palm
        if palm is not None and palm.device == device:
            raise ValueError(f"{device} is already the palm camera")
        self.disconnect_input()
        try:
            self.source = create_source(kind, device, calibrate=False)
        except Exception as error:
            self.status = f"ERROR · {kind}: {error}"
            raise
        self.kind, self.status = kind, "CONNECTING"

    def disconnect_input(self):
        source, self.source = self.source, None
        self.worker.pause()
        self.robot = self.human_points = None
        if source is not None:
            source.close()

    def connect_palm(self, device):
        if not device:
            raise ValueError("choose a palm camera first")
        camera = getattr(self.source, "camera", None)
        if camera is not None and camera.device == device:
            raise ValueError(f"{device} is already the gesture camera")
        self.disconnect_palm()
        try:
            self.palm = OpenCVCamera(
                device, *C.PALM_CAMERA_SIZE, C.PALM_CAMERA_FPS,
                fourcc=C.PALM_CAMERA_FOURCC,
            )
        except Exception as error:
            self.status = f"ERROR · palm: {error}"
            raise

    def disconnect_palm(self):
        palm, self.palm = self.palm, None
        if palm is not None:
            palm.close()

    def set_ros(self, enabled):
        if enabled == (self.ros is not None):
            return
        if enabled:
            self.ros = RosOutput(signal_handlers=False)
        else:
            ros, self.ros = self.ros, None
            ros.close()

    def snapshot(self):
        now = time.monotonic()
        return {
            "status": self.status,
            "input_on": self.source is not None,
            "palm_on": self.palm is not None,
            "ros_on": self.ros is not None,
            "retarget_ms": self.retarget_ms,
            "preview_seq": self.preview_seq,
            "palm_seq": self.palm_seq,
            "input_age": None if self.input_at is None else now - self.input_at,
            "palm_age": None if self.palm_at is None else now - self.palm_at,
        }

    def _loop(self):
        """Read, retarget, publish and render; failures return the input to idle."""
        while not self._stop.is_set():
            source = self.source
            if source is None:
                if not self.status.startswith("ERROR"):  # keep the reason visible
                    self.status = "INPUT OFF"
                self.human_points = None
                self._stop.wait(FRAME_PERIOD)
                continue
            try:
                frame = source.read()
            except (TimeoutError, RuntimeError, ValueError) as error:
                self.status = f"ERROR · {type(error).__name__}: {error}"
                self.disconnect_input()
                continue
            if frame is None:
                self._sim_step()
                self._stop.wait(0.002)
                continue
            self.status, self.input_at = frame.status, time.monotonic()
            try:
                self._track(frame)
            except (RuntimeError, ValueError) as error:
                self.status = f"ERROR · {type(error).__name__}: {error}"
            self._sim_step()

    def _track(self, frame):
        ros, worker = self.ros, self.worker
        if frame.points is not None and ros is not None:
            ros.hand(frame)
        if frame.ready and frame.handedness == "Left":
            worker.submit(
                frame.points, frame.timestamp, frame.finger_pad_directions,
                frame.initial_joint_angles,
            )
        else:
            worker.pause()
            self.robot = None
        output = worker.poll()
        if output is not None:
            self.robot, self.retarget_ms = output[0], output[2]
            if ros is not None:
                ros.joints(np.degrees(self.robot))
        self.human_points = scene.display_points(
            frame.points, frame.finger_pad_directions
        )
        if frame.preview is not None:
            encoded = jpeg(frame.preview)
            if encoded is not None:
                self.preview, self.preview_seq = encoded, self.preview_seq + 1

    def _sim_step(self):
        if self.sim is not None and not self.sim.update(self.robot):
            self.sim = None

    def _palm_loop(self):
        after = 0
        while not self._stop.is_set():
            palm = self.palm
            if palm is None:
                after = 0
                self._stop.wait(0.05)
                continue
            try:
                after, frame, _ = palm.read(after)
            except (TimeoutError, RuntimeError) as error:
                self.status = f"ERROR · palm {type(error).__name__}: {error}"
                self.disconnect_palm()
                after = 0
                continue
            self.palm_at = time.monotonic()
            encoded = jpeg(frame)
            if encoded is not None:
                self.palm_preview, self.palm_seq = encoded, self.palm_seq + 1

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        for thread in threading.enumerate():
            if thread.name in ("teleop-track", "teleop-palm"):
                thread.join(1.0)
        if self.sim is not None:
            sim, self.sim = self.sim, None
            sim.close()
        self.disconnect_input()
        self.disconnect_palm()
        self.worker.close()
        self.set_ros(False)


def panel(title, controls=None):
    """Start one dark grid panel; returns its status label and its body slot."""
    with ui.element("div").classes("panel"):
        with ui.row().classes("panel-head items-center w-full"):
            ui.label(title)
            if controls is not None:
                controls()
            status = ui.label().classes("panel-status")
        body = ui.element("div").classes("panel-body")
    return status, body


def build_page(teleop):
    """Build the console page for one browser client."""
    ui.colors(primary="#2563eb", positive="#16a34a", warning="#ea880c",
              negative="#dc2626")
    ui.add_css(CSS)
    devices = camera_devices()
    shown = [0, 0]  # last image seq per panel
    seen = [None, None]  # last human_points / robot array per scene

    with ui.element("div").classes("shell"):
        with ui.row().classes("connection-bar items-center w-full"):
            ui.label("MM-Hand Teleop").classes(
                "font-semibold text-base whitespace-nowrap")
            kind = ui.select(
                INPUT_KINDS, label="Input", value=teleop.kind,
            ).props("dense outlined").classes("w-32")
            device = ui.select(
                devices, label="Camera", value=next(iter(devices), None),
            ).props("dense outlined").classes("w-56")
            connect = ui.button("Connect", on_click=lambda: toggle_input()) \
                .props("dense outline")
            input_age = ui.label().classes("text-xs whitespace-nowrap")
            palm_device = ui.select(
                devices, label="Palm camera", value=None,
            ).props("dense outlined").classes("w-56")
            palm_connect = ui.button("Connect", on_click=lambda: toggle_palm()) \
                .props("dense outline")
            palm_age = ui.label().classes("text-xs whitespace-nowrap")
            ui.space()
            ros = ui.switch("ROS output", on_change=lambda e: toggle_ros(e.value))
        with ui.row().classes("status-bar items-center w-full"):
            chips = [ui.label().classes("status-chip text-xs")
                     for _ in range(4)]

        kind.on_value_change(lambda e: device.set_enabled(e.value != "manus"))
        device.set_enabled(teleop.kind != "manus")

        async def toggle_input():
            if teleop.source is not None:
                await run.io_bound(teleop.disconnect_input)
                return
            connect.disable()  # one attempt at a time: a camera node has one owner
            try:
                await run.io_bound(teleop.connect_input, kind.value, device.value)
            except Exception as error:
                ui.notify(f"{kind.value}: {error}", type="negative", timeout=10000)
            finally:
                connect.set_enabled(True)

        async def toggle_palm():
            if teleop.palm is not None:
                await run.io_bound(teleop.disconnect_palm)
                return
            palm_connect.disable()
            try:
                await run.io_bound(teleop.connect_palm, palm_device.value)
            except Exception as error:
                ui.notify(f"palm camera: {error}", type="negative", timeout=10000)
            finally:
                palm_connect.set_enabled(True)

        async def toggle_ros(enabled):
            try:
                await run.io_bound(teleop.set_ros, enabled)
            except Exception as error:
                ui.notify(f"ROS 2: {error}", type="negative", timeout=10000)
                ros.set_value(teleop.ros is not None)

        with ui.element("div").classes("grid"):
            input_status, body = panel("Gesture camera")
            with body:
                # QImg fades every new source in over 0.28 s and never reaches full
                # opacity at WEB_FPS; interactive_image is a plain, flicker-free img.
                preview = ui.interactive_image("/preview.jpg") \
                    .classes("w-full h-full")
            palm_status, body = panel("MMHand palm stereo")
            with body:
                palm_image = ui.interactive_image("/palm.jpg") \
                    .classes("w-full h-full")
            human_status, body = panel("Input hand")
            with body:
                human = scene.HumanScene()
            robot_status, body = panel("Retargeted MMHand")
            with body:
                robot = scene.RobotScene(teleop.worker.model)

    def refresh_images():
        for index, (image, key, route) in enumerate((
            (preview, "preview_seq", "/preview.jpg"),
            (palm_image, "palm_seq", "/palm.jpg"),
        )):
            seq = getattr(teleop, key)
            if seq != shown[index]:
                shown[index] = seq
                image.set_source(f"{route}?{seq}")

    def refresh_scenes():
        # The tracking thread replaces these arrays; keep the identity so an
        # idle console sends nothing at all.
        if teleop.human_points is not seen[0]:
            seen[0] = teleop.human_points
            human.update(seen[0])
        if teleop.robot is not seen[1]:
            seen[1] = teleop.robot
            robot.update(seen[1])

    def refresh_status():
        snap = teleop.snapshot()
        input_status.set_text(age_text(snap["input_age"], snap["input_on"]))
        palm_status.set_text(age_text(snap["palm_age"], snap["palm_on"]))
        human_status.set_text("" if snap["input_on"] else "OFF")
        robot_status.set_text(
            "waiting" if snap["retarget_ms"] is None
            else f'{snap["retarget_ms"]:.1f} ms')
        input_age.set_text(f'Rx age: {age_text(snap["input_age"], snap["input_on"])}')
        palm_age.set_text(f'Rx age: {age_text(snap["palm_age"], snap["palm_on"])}')
        connect.set_text("Disconnect" if snap["input_on"] else "Connect")
        palm_connect.set_text("Disconnect" if snap["palm_on"] else "Connect")
        for chip, (name, value) in zip(chips, (
            ("Input", snap["status"]),
            ("Palm", "LIVE" if snap["palm_on"] else "OFFLINE"),
            ("ROS", "ON" if snap["ros_on"] else "OFF"),
            ("Retarget", "waiting" if snap["retarget_ms"] is None
             else f'{snap["retarget_ms"]:.1f} ms'),
        )):
            chip.set_text(f"{name}: {value}")
            chip.style(f"background-color:{status_bg(value)}")

    ui.timer(FRAME_PERIOD, refresh_images)
    ui.timer(SCENE_PERIOD, refresh_scenes)
    ui.timer(0.25, refresh_status, immediate=True)


def run_console(teleop, show=True):
    """Start the teleop stack, serve the console, and open its browser tab."""
    port = native.find_open_port(WEB_PORT, WEB_PORT + 50)
    app.add_static_files(
        "/mmhand-meshes", Path(C.URDF_PATH).parent.parent / "meshes"
    )

    @app.get("/preview.jpg")
    def _preview():
        return Response(teleop.preview, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.get("/palm.jpg")
    def _palm():
        return Response(teleop.palm_preview, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    try:
        @ui.page("/")
        def index():
            build_page(teleop)

        teleop.start()  # devices + loops; the page only reads snapshots
        ui.run(host="127.0.0.1", port=port, show=show, reload=False,
               title="MM-Hand Teleop", favicon="🤏")
    finally:
        teleop.close()
    return 0
