"""GUI runtime checks: snapshot state machine, page build and scene builders."""

import threading
import time
import types
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
from tempfile import TemporaryDirectory

from nicegui import client
from nicegui.elements.interactive_image import InteractiveImage
from nicegui.elements.select import Select
import numpy as np

import gui
import scene
from gui import Teleop, age_text, status_bg
from input.camera import camera_devices, first_camera
from input.frame import InitialJointAngles, InputFrame
from retarget import RobotModel

POINTS = np.zeros((21, 3), float)
ANGLES = InitialJointAngles(np.zeros((4, 4)), np.zeros(2))
FRAME = InputFrame(
    1.0, POINTS, "Left", True, "TRACKING",
    initial_joint_angles=ANGLES, preview=np.zeros((36, 64, 3), np.uint8),
)


class FakeSource:
    def __init__(self, frames=(), error=None):
        self.frames, self.error, self.closed = list(frames), error, False

    def read(self):
        if self.error is not None:
            raise self.error
        return self.frames.pop(0) if self.frames else None

    def close(self):
        self.closed = True


class FakePage:
    path = "/"

    def resolve_markdown(self):
        return False


class FakeWorker:
    model = None  # set by SceneAndPageTests to a real RobotModel

    def __init__(self):
        self.submitted, self.paused, self.result = [], False, None

    def submit(self, points, timestamp=None, finger_pad_directions=None,
               initial_joint_angles=None):
        self.submitted.append(points)

    def poll(self):
        result, self.result = self.result, None
        return result

    def pause(self):
        self.paused = True

    def close(self):
        pass


class FakeRos:
    def __init__(self):
        self.hands, self.joints_seen = [], []

    def hand(self, frame):
        self.hands.append(frame)

    def joints(self, degrees):
        self.joints_seen.append(np.asarray(degrees))

    def close(self):
        pass


def make_teleop(**fields):
    """Build a Teleop without devices or threads."""
    teleop = Teleop.__new__(Teleop)
    teleop.worker = FakeWorker()
    teleop.source = teleop.palm = teleop.ros = teleop.sim = teleop.robot = None
    teleop.human_points = None
    teleop.status = "INPUT OFF"
    teleop.retarget_ms = None
    teleop.input_at = teleop.palm_at = None
    teleop.preview = teleop.palm_preview = gui.BLACK
    teleop.preview_seq = teleop.palm_seq = 0
    teleop.kind, teleop.use_sim, teleop.use_ros = "mediapipe", False, False
    teleop._stop = threading.Event()
    teleop._closed = False
    teleop.__dict__.update(fields)
    return teleop


class SnapshotTests(unittest.TestCase):
    def test_idle_snapshot_reports_off(self):
        teleop = make_teleop()
        snap = teleop.snapshot()
        self.assertEqual(snap["status"], "INPUT OFF")
        self.assertFalse(snap["input_on"])
        self.assertFalse(snap["palm_on"])
        self.assertIsNone(snap["retarget_ms"])

    def test_track_publishes_joints_and_a_new_preview(self):
        teleop = make_teleop(ros=FakeRos())
        teleop.worker.result = (
            np.zeros(21), {"total": 1.0}, 4.2, {"solve_total": 4.2},
        )
        teleop._track(FRAME)
        self.assertEqual(teleop.retarget_ms, 4.2)
        self.assertEqual(len(teleop.ros.hands), 1)
        np.testing.assert_allclose(teleop.ros.joints_seen[0], 0)
        self.assertEqual(teleop.robot.shape, (21,))
        self.assertIsNone(teleop.human_points)  # zeroed points have no palm width
        self.assertGreater(teleop.preview_seq, 0)

    def test_disconnect_input_drops_the_source_and_pauses_the_solver(self):
        source = FakeSource()
        teleop = make_teleop(source=source)
        teleop.disconnect_input()
        self.assertIsNone(teleop.source)
        self.assertIsNone(teleop.robot)
        self.assertIsNone(teleop.human_points)
        self.assertTrue(source.closed)
        self.assertTrue(teleop.worker.paused)

    def test_snapshot_reports_ros_and_palm_state(self):
        teleop = make_teleop(ros=FakeRos())
        snap = teleop.snapshot()
        self.assertTrue(snap["ros_on"])
        self.assertFalse(snap["palm_on"])
        self.assertEqual(age_text(None, False), "OFF")
        self.assertIn("Rx age", f'Rx age: {age_text(0.0123, True)}')


class FakeCamera:
    def __init__(self, device):
        self.device = device

    def close(self):
        pass


class ConnectTests(unittest.TestCase):
    def test_failed_connect_keeps_the_reason_in_the_status_bar(self):
        teleop = make_teleop()
        with patch("gui.create_source", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                teleop.connect_input("wilor", "/dev/video0")
        self.assertEqual(teleop.status, "ERROR · wilor: boom")

    def test_failed_palm_connect_keeps_the_reason_in_the_status_bar(self):
        teleop = make_teleop()
        with patch("gui.OpenCVCamera", side_effect=RuntimeError("busy")):
            with self.assertRaises(RuntimeError):
                teleop.connect_palm("/dev/video0")
        self.assertEqual(teleop.status, "ERROR · palm: busy")

    def test_one_node_cannot_feed_both_panels(self):
        teleop = make_teleop(palm=FakeCamera("/dev/video0"))
        with self.assertRaisesRegex(ValueError, "palm camera"):
            teleop.connect_input("wilor", "/dev/video0")
        teleop = make_teleop(source=types.SimpleNamespace(
            camera=FakeCamera("/dev/video0"), close=lambda: None))
        with self.assertRaisesRegex(ValueError, "gesture camera"):
            teleop.connect_palm("/dev/video0")


class LoopTests(unittest.TestCase):
    def test_read_failure_closes_the_source_and_keeps_running(self):
        teleop = make_teleop()
        teleop.source = FakeSource(error=RuntimeError("camera unplugged"))
        thread = threading.Thread(target=teleop._loop, daemon=True)
        thread.start()
        deadline = time.monotonic() + 1.0
        while teleop.source is not None and time.monotonic() < deadline:
            time.sleep(0.005)
        teleop._stop.set()
        thread.join(1.0)
        self.assertFalse(thread.is_alive())
        self.assertIn("camera unplugged", teleop.status)

    def test_palm_frame_is_encoded_and_seq_advances(self):
        class FakePalm:
            def __init__(self):
                self.calls = 0

            def read(self, after):
                self.calls += 1
                if self.calls > 3:
                    raise TimeoutError("done")
                return after + 1, np.full((48, 128, 3), 7, np.uint8), 0.0

            def close(self):
                pass

        teleop = make_teleop()
        teleop.palm = FakePalm()
        thread = threading.Thread(target=teleop._palm_loop, daemon=True)
        thread.start()
        deadline = time.monotonic() + 1.0
        while teleop.palm is not None and time.monotonic() < deadline:
            time.sleep(0.005)
        teleop._stop.set()
        thread.join(1.0)
        self.assertGreater(teleop.palm_seq, 0)


class StartTests(unittest.TestCase):
    def test_start_connects_and_advances_the_preview(self):
        teleop = make_teleop()
        source = FakeSource([FRAME] * 3)
        with patch("gui.create_source", return_value=source), \
                patch("gui.first_camera", return_value="/dev/video0"):
            teleop.start()
            try:
                deadline = time.monotonic() + 1.0
                while teleop.preview_seq == 0 and time.monotonic() < deadline:
                    time.sleep(0.005)
                self.assertGreater(teleop.preview_seq, 0)
                self.assertEqual(teleop.status, "TRACKING")
            finally:
                teleop.close()
        self.assertTrue(source.closed)


class HelpersTests(unittest.TestCase):
    def test_camera_devices_lists_only_capture_nodes(self):
        with TemporaryDirectory() as root:
            names = Path(root) / "names"
            names.mkdir()
            for node, label, index in (
                ("video0", "Integrated Camera", "0"),
                ("video1", "Integrated Camera", "1"),  # UVC metadata node
                ("video2", "USB Camera", "0"),
            ):
                (names / node).mkdir()
                (names / node / "name").write_text(label)
                (names / node / "index").write_text(index)
                (Path(root) / node).write_text("")
            devices = camera_devices(pattern=str(Path(root) / "video*"),
                                     names=str(names))
        labels = {Path(path).name: label.split(" · ")[-1]
                  for path, label in devices.items()}
        self.assertEqual(labels, {"video0": "Integrated Camera",
                                  "video2": "USB Camera"})
        self.assertEqual(first_camera(), next(iter(camera_devices())))

    def test_status_colours_follow_the_console_rules(self):
        self.assertEqual(status_bg("TRACKING"), "#bbf7d0")
        self.assertEqual(status_bg("WAITING"), "#fde68a")
        self.assertEqual(status_bg("ERROR · RuntimeError: boom"), "#fecaca")
        self.assertEqual(status_bg("whatever"), "#e2e8f0")


class SceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = RobotModel()

    def test_display_points_are_canonical_and_reject_degenerate_input(self):
        points = np.random.default_rng(0).random((21, 3)) * 0.1
        shown = scene.display_points(points)
        self.assertEqual(shown.shape, (21, 3))
        self.assertAlmostEqual(
            np.linalg.norm(shown[17] - shown[5]), scene.HUMAN_DISPLAY_PALM_WIDTH
        )
        self.assertIsNone(scene.display_points(np.zeros((21, 3))))
        self.assertIsNone(scene.display_points(None))

    def test_robot_scene_poses_every_joint(self):
        with client.Client(FakePage()):
            robot = scene.RobotScene(self.model)
            robot.update(np.zeros(21))
            robot.update(None)  # keeps the last pose
        self.assertEqual(len(robot.motors), 21)
        self.assertEqual(
            sorted({index for index, _, _ in robot.motors}), list(range(21))
        )

    def test_human_scene_hides_empty_hands(self):
        points = np.random.default_rng(1).random((21, 3)) * 0.1
        with client.Client(FakePage()):
            human = scene.HumanScene()
            human.update(scene.display_points(points))
            human.update(None)
        self.assertFalse(human.joints.visible_)
        self.assertFalse(human.bones.visible_)
        self.assertFalse(human.tips.visible_)

    def test_human_scene_draws_bones_and_marks_fingertips(self):
        points = np.random.default_rng(1).random((21, 3)) * 0.1
        with client.Client(FakePage()):
            human = scene.HumanScene()
            human.update(scene.display_points(points))
        self.assertEqual(
            len(human.bones.args[0]),
            len(scene.SKELETON_EDGES) * scene.BONE_SAMPLES,
        )
        self.assertEqual(len(human.tips.args[0]), 5)
        self.assertTrue(human.bones.visible_)
        self.assertTrue(human.tips.visible_)


class PageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        FakeWorker.model = RobotModel()

    def test_page_builds_and_every_select_value_is_an_option(self):
        teleop = make_teleop()
        with client.Client(FakePage()) as page_client:
            gui.build_page(teleop)
        selects = [element for element in page_client.elements.values()
                   if isinstance(element, Select)]
        self.assertEqual(len(selects), 3)  # input kind, camera, palm camera
        for select in selects:  # the palm camera starts unchosen
            self.assertTrue(
                select.value is None or select.value in select.options
            )
        live_images = [element for element in page_client.elements.values()
                       if isinstance(element, InteractiveImage)]
        self.assertEqual(len(live_images), 2)  # gesture and palm previews


class ConsoleTests(unittest.TestCase):
    def test_run_console_starts_the_stack_and_opens_the_browser(self):
        teleop = make_teleop()
        teleop.start = MagicMock()
        calls = []
        with patch("gui.app.add_static_files"), \
                patch.object(gui.ui, "page",
                             lambda *args, **kwargs: lambda view: view), \
                patch.object(gui.ui, "run",
                             lambda **kwargs: calls.append(kwargs)):
            self.assertEqual(gui.run_console(teleop), 0)
        teleop.start.assert_called_once_with()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["host"], "127.0.0.1")
        self.assertTrue(calls[0]["show"])
        self.assertFalse(calls[0]["reload"])


if __name__ == "__main__":
    unittest.main()
