"""GUI runtime checks: snapshot state machine, page build and scene builders."""

import threading
import time
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
from tempfile import TemporaryDirectory

from nicegui import client
from nicegui.elements.interactive_image import InteractiveImage
from nicegui.elements.label import Label
from nicegui.elements.select import Select
import numpy as np
from scipy.spatial.transform import Rotation

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
    teleop.source = teleop.ros = teleop.sim = teleop.robot = None
    teleop.human_points = teleop.human_directions = None
    teleop.status = "INPUT OFF"
    teleop.retarget_ms = None
    teleop.losses = teleop.timings = None
    teleop.input_at = None
    teleop.preview = gui.BLACK
    teleop.preview_seq = 0
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
        self.assertIsNone(snap["retarget_ms"])
        self.assertIsNone(snap["losses"])
        self.assertIsNone(snap["timings"])

    def test_track_publishes_joints_and_a_new_preview(self):
        teleop = make_teleop(ros=FakeRos())
        teleop.worker.result = (
            np.zeros(21), {"total": 1.0}, 4.2, {"solve_total": 4.2},
        )
        teleop._track(FRAME)
        self.assertEqual(teleop.retarget_ms, 4.2)
        self.assertEqual(teleop.losses, {"total": 1.0})
        self.assertEqual(teleop.timings, {"solve_total": 4.2})
        self.assertEqual(teleop.snapshot()["losses"], {"total": 1.0})
        self.assertEqual(len(teleop.ros.hands), 1)
        np.testing.assert_allclose(teleop.ros.joints_seen[0], 0)
        self.assertEqual(teleop.robot.shape, (21,))
        self.assertIsNone(teleop.human_points)  # zeroed points have no palm width
        self.assertGreater(teleop.preview_seq, 0)

    def test_disconnect_input_drops_the_source_and_pauses_the_solver(self):
        source = FakeSource()
        teleop = make_teleop(
            source=source, robot=np.zeros(21), losses={"total": 1.0},
            timings={"solve_total": 2.0},
        )
        teleop.disconnect_input()
        self.assertIsNone(teleop.source)
        self.assertIsNone(teleop.robot)
        self.assertIsNone(teleop.human_points)
        self.assertIsNone(teleop.losses)
        self.assertIsNone(teleop.timings)
        self.assertTrue(source.closed)
        self.assertTrue(teleop.worker.paused)

    def test_snapshot_reports_ros_state(self):
        teleop = make_teleop(ros=FakeRos())
        snap = teleop.snapshot()
        self.assertTrue(snap["ros_on"])
        self.assertEqual(age_text(None, False), "OFF")
        self.assertIn("Rx age", f'Rx age: {age_text(0.0123, True)}')


class ConnectTests(unittest.TestCase):
    def test_failed_connect_keeps_the_reason_in_the_status_bar(self):
        teleop = make_teleop()
        with patch("gui.create_source", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                teleop.connect_input("wilor", "/dev/video0")
        self.assertEqual(teleop.status, "ERROR · wilor: boom")

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

    def test_loss_overlay_reports_every_weighted_term_with_its_share(self):
        losses = {name: 1.0 for name, _ in gui.LOSS_LABELS}
        losses["total"] = float(len(gui.LOSS_LABELS))
        text = gui.loss_text(losses)
        self.assertEqual(text.splitlines()[0], "Weighted retarget loss")
        for _, label in gui.LOSS_LABELS:
            self.assertIn(label, text)
        self.assertIn("12.5%", text)  # eight equal terms, so 100/8 each
        self.assertEqual(gui.loss_text(None).splitlines()[-1], "waiting")
        self.assertEqual(gui.loss_text({}).splitlines()[-1], "waiting")

    def test_timing_overlay_reports_every_stage(self):
        text = gui.timing_text({"slsqp": 1.5, "worker_queue": 12.34})
        for _, label in gui.TIMING_LABELS:
            self.assertIn(label, text)
        self.assertIn("SLSQP           1.5 ms", text)
        self.assertIn("12.3 ms", text)
        # `output_filter` is only reported by older solvers
        self.assertIn("waiting", text)
        self.assertEqual(
            gui.timing_text(None).count("waiting"), len(gui.TIMING_LABELS)
        )


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

    def test_display_hand_carries_the_pad_directions_and_the_size(self):
        points = np.random.default_rng(2).random((21, 3)) * 0.1
        directions = np.tile((0.0, 0.0, 1.0), (5, 1))
        shown, shown_directions = scene.display_hand(points, directions)
        self.assertEqual(shown.shape, (21, 3))
        self.assertAlmostEqual(
            np.linalg.norm(shown[17] - shown[5]), scene.HUMAN_DISPLAY_PALM_WIDTH
        )
        np.testing.assert_allclose(
            np.linalg.norm(shown_directions, axis=1), 1.0, atol=1e-9
        )
        self.assertIsNone(scene.display_hand(points)[1])  # no pad normals
        self.assertEqual(scene.display_hand(None), (None, None))

    def test_display_hand_is_upright_whatever_the_source_rotation(self):
        # A real hand skeleton, so "upright" means the fingertips sit above the
        # wrist; a random cloud has no anatomical top.
        points = self.model.skeleton_points(self.model.seed)
        directions = np.tile((0.0, 0.0, 1.0), (5, 1))
        shown, shown_directions = scene.display_hand(points, directions)
        rotation = Rotation.from_euler("xyz", (0.3, -1.1, 2.0)).as_matrix()
        rotated = scene.display_hand(points @ rotation.T, directions @ rotation.T)
        np.testing.assert_allclose(rotated[0], shown, atol=1e-9)
        np.testing.assert_allclose(rotated[1], shown_directions, atol=1e-9)
        self.assertGreater(shown[[4, 8, 12, 16, 20], 1].min(), shown[0, 1])
        self.assertGreater(shown[12, 1], shown[9, 1])  # tip above its MCP

    def test_robot_camera_keeps_the_seed_pose_upright_and_palm_on(self):
        position, look_at, up = scene.robot_camera_pose(self.model)
        up, position, look_at = map(np.asarray, (up, position, look_at))
        self.assertAlmostEqual(float(np.linalg.norm(up)), 1.0)
        view = (look_at - position) / np.linalg.norm(look_at - position)
        self.assertAlmostEqual(float(up @ view), 0.0, places=6)
        tips = self.model.fingertips(self.model.seed)
        finger_axis = tips.mean(0) - self.model.palm_position
        finger_axis /= np.linalg.norm(finger_axis)
        self.assertGreater(float(up @ finger_axis), 0.9)  # fingers point up
        _, pads = self.model.fingertip_pads(self.model.seed)
        self.assertGreater(float((position - look_at) @ pads.mean(0)), 0.0)
        # the old hard-coded up tipped this URDF by ~83 degrees
        hard_coded = np.array((0.9993984744, -0.0346479515, -0.0014862239))
        self.assertLess(float(up @ hard_coded), 0.5)

    def test_robot_scene_poses_every_joint(self):
        with client.Client(FakePage()):
            robot = scene.RobotScene(self.model)
            robot.update(np.zeros(21))
            robot.update(None)  # keeps the last pose
        self.assertEqual(len(robot.motors), 21)
        self.assertEqual(
            sorted({index for index, _, _ in robot.motors}), list(range(21))
        )

    def test_style_script_relights_only_the_given_scene(self):
        with client.Client(FakePage()):
            robot = scene.RobotScene(self.model)
        script = scene.style_script(robot.scene)
        self.assertIn(f'scene_{robot.scene.html_id}', script)
        self.assertNotIn("__", script)  # every placeholder replaced
        self.assertIn(str(scene.SCENE_AMBIENT), script)

    def test_human_scene_hides_empty_hands(self):
        points = np.random.default_rng(1).random((21, 3)) * 0.1
        with client.Client(FakePage()):
            human = scene.HumanScene()
            human.update(scene.display_points(points))
            human.update(None)
        self.assertFalse(human.joints.visible_)
        self.assertFalse(human.bones.visible_)
        self.assertFalse(human.tips.visible_)
        self.assertFalse(any(arrow.visible_ for arrow in human.arrows))

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

    def test_human_scene_aims_one_arrow_along_each_pad_normal(self):
        points = np.random.default_rng(3).random((21, 3)) * 0.1
        directions = np.tile((0.0, 0.0, 1.0), (5, 1))
        with client.Client(FakePage()):
            human = scene.HumanScene()
            human.update(points, directions)
        tips = np.asarray(points)[list(scene.TIP_INDICES)]
        self.assertEqual(len(human.arrows), len(scene.TIP_INDICES))
        for arrow, tip in zip(human.arrows, tips):
            self.assertTrue(arrow.visible_)
            np.testing.assert_allclose(
                (arrow.x, arrow.y, arrow.z), tip, atol=1e-12
            )
            np.testing.assert_allclose(
                np.asarray(arrow.R) @ (0.0, 1.0, 0.0), directions[0], atol=1e-9
            )
        with client.Client(FakePage()):
            # MediaPipe has no fingertip pointing, so the arrows stay hidden
            human = scene.HumanScene()
            human.update(points)
        self.assertFalse(any(arrow.visible_ for arrow in human.arrows))


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
        self.assertEqual(len(selects), 2)  # input kind, camera
        for select in selects:
            self.assertTrue(
                select.value is None or select.value in select.options
            )
        live_images = [element for element in page_client.elements.values()
                       if isinstance(element, InteractiveImage)]
        self.assertEqual(len(live_images), 1)  # gesture preview
        overlays = [element for element in page_client.elements.values()
                    if isinstance(element, Label) and "overlay" in element.classes]
        # weighted retarget loss and the solve-timing breakdown
        self.assertEqual(len(overlays), 2)
        self.assertIn("overlay-loss", overlays[0].classes + overlays[1].classes)


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
