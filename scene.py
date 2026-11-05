"""Both 3D panels, rendered inside the console page with NiceGUI's `ui.scene`.

The page owns one of each per browser; the tracking thread only publishes
arrays that `update` consumes. No server, no port, no iframe.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from nicegui import ui
from scipy.spatial.transform import Rotation

from config import URDF_PATH
from input.frame import SKELETON_EDGES
from retarget import compute_cmc_frame

# The Input hand panel always shows the palm toward the screen, fingers up,
# at a fixed anatomical size. Retargeting still consumes the raw source frame.
HUMAN_DISPLAY_PALM_WIDTH = 0.085  # m
HUMAN_CAMERA_DISTANCE = 0.45  # m
HUMAN_CAMERA_LOOK_AT_Y = 0.09  # m
HUMAN_CAMERA_FOV = 42  # degrees
ROBOT_CAMERA_DISTANCE = 0.42  # m
ROBOT_CAMERA_FOV = 42  # degrees
BACKGROUND = "#ffffff"  # the old Viser panels were white
JOINT_COLOR = "#2a9bff"
BONE_COLOR = "#7fc8ff"
# The scene adds a bright fixed ambient light (0.7*pi), so a light material clips
# to a flat silhouette. This darker silver keeps the URDF visual STLs shaded.
MESH_COLOR = "#6f747c"
TIP_COLOR = "#ff9646"
TIP_POINT_SIZE = 0.008
BONE_SAMPLES = 24  # points per skeleton edge; enough to read as a line
_DISPLAY_PERMUTATION = np.array(
    ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
)


def human_display_frame(points, directions):
    """Return the palm-on display rotation, source palm width, and CMC frame.

    The returned matrix maps source geometry into a canonical view frame:
    fingers point along +Y and the palmar side faces +Z (the camera).
    ``directions`` may be None when the source has no fingertip pointing.
    """

    points = np.asarray(points, float)
    _, cmc = compute_cmc_frame(points)
    # WiLoR's pad directions already point outward from the palmar surface, so
    # their mean is the volar normal and decides which side the camera sees.
    # Without them (MediaPipe) the CMC normal below is used instead.
    volar = (
        None if directions is None
        else np.asarray(directions, float)[1:].mean(axis=0)
    )
    length = 0.0 if volar is None else np.linalg.norm(volar)
    z_axis = cmc[:, 2].copy() if length <= 1e-8 else volar / length
    x_axis = cmc[:, 0] - z_axis * np.dot(cmc[:, 0], z_axis)
    length = np.linalg.norm(x_axis)
    if length <= 1e-8:
        raise ValueError("Degenerate human display axis")
    x_axis /= length
    y_axis = np.cross(z_axis, x_axis)
    matrix = np.column_stack((x_axis, y_axis, z_axis)) @ _DISPLAY_PERMUTATION
    palm_width = float(np.linalg.norm(points[17] - points[5]))
    if not np.isfinite(matrix).all() or not np.isfinite(palm_width) or palm_width <= 0:
        raise ValueError("Degenerate human palm width")
    return matrix, palm_width, cmc


def display_points(points, directions=None):
    """Map one source hand into the canonical display frame, or None if degenerate."""

    if points is None:
        return None
    try:
        matrix, palm_width, _ = human_display_frame(points, directions)
    except ValueError:
        return None
    points = np.asarray(points, float)
    return (points - points[1]) @ matrix * (HUMAN_DISPLAY_PALM_WIDTH / palm_width)


def bone_points(points, samples=BONE_SAMPLES):
    """Sample the skeleton edges, so the cloud reads as bones and not just joints."""

    points = np.asarray(points, float)
    return np.concatenate([
        np.linspace(points[start], points[end], samples, endpoint=False)
        for start, end in SKELETON_EDGES
    ]).astype(np.float32)


def robot_camera_pose(model):
    tips = model.fingertips(model.seed)
    look_at = np.vstack((model.palm_position, tips)).mean(0)
    palm_normal = model.palm_frame[:, 2]
    transforms = model.fk(model.seed)
    base = (
        transforms["finger_1_proximal_phalanx_1"][:3, 3]
        - transforms["finger_4_proximal_phalanx_1"][:3, 3]
    )
    normal = palm_normal - base * np.dot(palm_normal, base) / np.dot(base, base)
    normal /= np.linalg.norm(normal)
    return look_at - ROBOT_CAMERA_DISTANCE * normal, look_at, (
        0.9993984744,
        -0.0346479515,
        -0.0014862239,
    )


def _camera(position, look_at, up, fov):
    camera = ui.scene.perspective_camera(fov=fov, near=0.01)
    camera.x, camera.y, camera.z = (float(value) for value in position)
    camera.look_at_x, camera.look_at_y, camera.look_at_z = (
        float(value) for value in look_at
    )
    camera.up_x, camera.up_y, camera.up_z = (float(value) for value in up)
    return camera


def _scene(camera):
    return ui.scene(
        grid=False, camera=camera, background_color=BACKGROUND,
    ).classes("w-full h-full")


def _visuals(path=URDF_PATH):
    """Return ``{link: [(mesh basename, origin xyz, origin rpy, scale)]}``."""

    visuals = {}
    for link in ET.parse(path).getroot().findall("link"):
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            origin = visual.find("origin")
            visuals.setdefault(link.get("name"), []).append((
                Path(mesh.get("filename")).name,
                _floats(origin, "xyz"),
                _floats(origin, "rpy"),
                float((mesh.get("scale") or "1 1 1").split()[0]),
            ))
    return visuals


def _floats(element, name):
    text = element.get(name) if element is not None else None
    return [float(value) for value in (text or "0 0 0").split()]


class HumanScene:
    """Input hand: canonical display frame, joints plus sampled bones."""

    def __init__(self):
        self.scene = _scene(_camera(
            (0.0, HUMAN_CAMERA_LOOK_AT_Y, HUMAN_CAMERA_DISTANCE),
            (0.0, HUMAN_CAMERA_LOOK_AT_Y, 0.0),
            (0.0, 1.0, 0.0),
            HUMAN_CAMERA_FOV,
        ))
        self.joints = self.scene.point_cloud([], point_size=0.006) \
            .material(JOINT_COLOR)
        self.bones = self.scene.point_cloud([], point_size=0.0025) \
            .material(BONE_COLOR)
        self.tips = self.scene.point_cloud([], point_size=TIP_POINT_SIZE) \
            .material(TIP_COLOR)
        self.update(None)

    def update(self, points):
        tips = None if points is None else np.asarray(points)[[4, 8, 12, 16, 20]]
        for target, data in (
            (self.joints, points),
            (self.bones, None if points is None else bone_points(points)),
            (self.tips, tips),
        ):
            target.visible(data is not None)
            if data is not None:
                target.set_points(np.asarray(data, np.float32).tolist())


class RobotScene:
    """Retargeted MMHand: URDF visual meshes posed by the 21 joint angles."""

    def __init__(self, model):
        self.scene = _scene(_camera(*robot_camera_pose(model), ROBOT_CAMERA_FOV))
        self.visuals = _visuals()
        self.motors = []
        self._link(model, "base_link")
        self.update(model.seed)

    def _link(self, model, name):
        for mesh, xyz, rpy, scale in self.visuals.get(name, ()):
            self.scene.stl(f"/mmhand-meshes/{mesh}").move(*xyz).rotate(*rpy) \
                .scale(scale).material(MESH_COLOR)
        for joint_name in model.children.get(name, ()):
            joint = model.joints[joint_name]
            origin = joint["origin"]
            with self.scene.group().move(*origin[:3, 3].tolist()) \
                    .rotate_R(origin[:3, :3].tolist()):
                with self.scene.group() as motion:
                    if joint["index"] is not None:
                        self.motors.append((joint["index"], joint["axis"], motion))
                    self._link(model, joint["child"])

    def update(self, q):
        """Rotate every actuated joint; None keeps the last pose on screen."""
        if q is None:
            return
        q = np.asarray(q, float)
        for index, axis, motion in self.motors:
            motion.rotate_R(
                Rotation.from_rotvec(axis * q[index]).as_matrix().tolist()
            )
