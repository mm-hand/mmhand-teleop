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
# At that width a human hand spans about 0.19 m from wrist to middle fingertip;
# these constants centre it in the panel with a little margin (the thumb pulls
# the bounding box to one side, hence the small x offset).
HUMAN_CAMERA_DISTANCE = 0.42  # m
HUMAN_CAMERA_LOOK_AT_X = -0.02  # m
HUMAN_CAMERA_LOOK_AT_Y = 0.04  # m
HUMAN_CAMERA_FOV = 42  # degrees
ROBOT_CAMERA_FOV = 42  # degrees
# Fraction of the panel height the fitted robot hand occupies.
FRAME_FILL = 0.62
BACKGROUND = "#ffffff"  # the old Viser panels were white
JOINT_COLOR = "#2a9bff"
BONE_COLOR = "#7fc8ff"
# NiceGUI's scene adds a fixed ambient light (0.7*pi) plus one directional light
# (0.3*pi) and only lets `material` change colour, opacity and side. `style_script`
# rebalances those lights at runtime, so the URDF visuals read as brushed silver
# instead of a flat silhouette. This colour is also acceptable unpatched.
MESH_COLOR = "#bcc2c9"
TIP_COLOR = "#ff9646"
TIP_POINT_SIZE = 0.009
JOINT_POINT_SIZE = 0.009
BONE_POINT_SIZE = 0.006
BONE_SAMPLES = 48  # points per skeleton edge; enough to read as a solid line
# Fingertip pad arrows, matching the August dashboard's proportions.
ARROW_COLOR = TIP_COLOR
ARROW_SHAFT_LENGTH = 0.025  # m
ARROW_SHAFT_RADIUS = 0.0008  # m
ARROW_HEAD_LENGTH = 0.006  # m
ARROW_HEAD_RADIUS = 0.0025  # m
TIP_INDICES = (4, 8, 12, 16, 20)
_ARROW_AXIS = np.array((0.0, 1.0, 0.0))  # a `cylinder` is extruded along +Y
# Runtime relighting of one scene: ambient and key-light intensity, and the
# Phong shininess/specular that give the URDF meshes a highlight.
SCENE_AMBIENT = 1.2
SCENE_KEY_LIGHT = 2.2
MESH_SHININESS = 50
MESH_SPECULAR = 0.35
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


def display_hand(points, directions=None):
    """Map one source hand into the canonical display frame.

    Returns the displayed points and the matching pad directions, or
    ``(None, None)`` when the input cannot define a palm frame.
    """

    if points is None:
        return None, None
    try:
        matrix, palm_width, _ = human_display_frame(points, directions)
    except ValueError:
        return None, None
    points = np.asarray(points, float)
    scale = HUMAN_DISPLAY_PALM_WIDTH / palm_width
    directions = (
        None if directions is None else np.asarray(directions, float) @ matrix
    )
    return (points - points[1]) @ matrix * scale, directions


def display_points(points, directions=None):
    """Map one source hand into the canonical display frame, or None if degenerate."""

    return display_hand(points, directions)[0]


def bone_points(points, samples=BONE_SAMPLES):
    """Sample the skeleton edges, so the cloud reads as bones and not just joints."""

    points = np.asarray(points, float)
    return np.concatenate([
        np.linspace(points[start], points[end], samples, endpoint=False)
        for start, end in SKELETON_EDGES
    ]).astype(np.float32)


def _robot_frame(model):
    """Return the robot's own view direction and screen-up axes for the seed pose.

    The camera sits on the finger-pad side, so the palm faces the panel. The up
    axis must come from the model: the MMHand URDF keeps the fingers along the
    CMC x axis, and hard-coding an up vector tilted the whole pose by ~83 degrees.
    """

    palm_normal = model.palm_frame[:, 2]
    transforms = model.fk(model.seed)
    base = (
        transforms["link_proximal_phalanx_2"][:3, 3]
        - transforms["link_proximal_phalanx_5"][:3, 3]
    )
    view = palm_normal - base * np.dot(palm_normal, base) / np.dot(base, base)
    view /= np.linalg.norm(view)
    up = model.palm_frame[:, 0] - view * np.dot(model.palm_frame[:, 0], view)
    up /= np.linalg.norm(up)
    return view, up


def _fit_camera(points, view, up, fov, plane=0.0, fill=FRAME_FILL):
    """Return the position and look-at that centre ``points`` in the panel.

    ``points`` are measured along the ``up``/``right`` axes of a camera looking
    along ``view``; the distance follows from the requested ``fill`` fraction of
    the panel height, and the look-at point sits on the plane at depth ``plane``
    so a fingertip that pokes toward the camera cannot bias the framing.
    """

    points = np.asarray(points, float)
    right = np.cross(up, view)
    right /= np.linalg.norm(right)
    height, width = points @ up, points @ right
    look_at = (
        view * plane
        + up * 0.5 * (height.max() + height.min())
        + right * 0.5 * (width.max() + width.min())
    )
    extent = float(height.max() - height.min())
    distance = extent / (fill * 2 * np.tan(np.radians(fov) / 2))
    return look_at - distance * view, look_at


def robot_camera_pose(model):
    """Return a camera position, look-at and up that frame the seed pose upright."""

    view, up = _robot_frame(model)
    position, look_at = _fit_camera(
        model.skeleton_points(model.seed), view, up, ROBOT_CAMERA_FOV,
        plane=float(model.palm_position @ view),
    )
    return position, look_at, up


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


def arrow_rotation(direction):
    """Return the rotation that turns a cylinder's +Y axis onto ``direction``."""

    direction = np.asarray(direction, float)
    direction = direction / np.linalg.norm(direction)
    return Rotation.align_vectors(direction[None], _ARROW_AXIS[None])[0].as_matrix()


class HumanScene:
    """Input hand: canonical display frame, joints, bones, and pad arrows."""

    def __init__(self):
        self.scene = _scene(_camera(
            (HUMAN_CAMERA_LOOK_AT_X, HUMAN_CAMERA_LOOK_AT_Y, HUMAN_CAMERA_DISTANCE),
            (HUMAN_CAMERA_LOOK_AT_X, HUMAN_CAMERA_LOOK_AT_Y, 0.0),
            (0.0, 1.0, 0.0),
            HUMAN_CAMERA_FOV,
        ))
        self.joints = self.scene.point_cloud([], point_size=JOINT_POINT_SIZE) \
            .material(JOINT_COLOR)
        self.bones = self.scene.point_cloud([], point_size=BONE_POINT_SIZE) \
            .material(BONE_COLOR)
        self.tips = self.scene.point_cloud([], point_size=TIP_POINT_SIZE) \
            .material(TIP_COLOR)
        self.arrows = tuple(self._arrow() for _ in TIP_INDICES)
        self.update(None)

    def _arrow(self):
        """Build one hidden finger-pad arrow along +Y, with its origin at the tip."""

        with self.scene.group() as arrow:
            self.scene.cylinder(
                ARROW_SHAFT_RADIUS, ARROW_SHAFT_RADIUS, ARROW_SHAFT_LENGTH,
                radial_segments=8,
            ).move(0.0, ARROW_SHAFT_LENGTH / 2, 0.0).material(ARROW_COLOR)
            self.scene.cylinder(
                0.0, ARROW_HEAD_RADIUS, ARROW_HEAD_LENGTH, radial_segments=8,
            ).move(0.0, ARROW_SHAFT_LENGTH + ARROW_HEAD_LENGTH / 2, 0.0) \
                .material(ARROW_COLOR)
        arrow.visible(False)
        return arrow

    def update(self, points, directions=None):
        tips = None if points is None else np.asarray(points)[list(TIP_INDICES)]
        for target, data in (
            (self.joints, points),
            (self.bones, None if points is None else bone_points(points)),
            (self.tips, tips),
        ):
            target.visible(data is not None)
            if data is not None:
                target.set_points(np.asarray(data, np.float32).tolist())
        self._update_arrows(tips, directions)

    def _update_arrows(self, tips, directions):
        """Aim each arrow along the displayed pad normal, or hide them all."""

        if tips is None or directions is None:
            for arrow in self.arrows:
                arrow.visible(False)
            return
        for arrow, tip, direction in zip(
            self.arrows, tips, np.asarray(directions, float)
        ):
            arrow.visible(True)
            arrow.move(*(float(value) for value in tip))
            arrow.rotate_R(arrow_rotation(direction).tolist())


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


STYLE_SCRIPT = """
(() => {
  const scene = window["scene___SCENE_ID__"];
  if (!scene) return false;
  for (const child of scene.children) {
    if (child.isAmbientLight) child.intensity = __AMBIENT__;
    if (child.isDirectionalLight) child.intensity = __KEY_LIGHT__;
  }
  scene.traverse((object) => {
    const material = object.material;
    if (!material || Array.isArray(material)) return;
    if (!material.isMeshPhongMaterial || material.userData.styled) return;
    material.userData.styled = true;
    material.shininess = __SHININESS__;
    material.specular.setRGB(__SPECULAR__, __SPECULAR__, __SPECULAR__);
    material.needsUpdate = true;
  });
  return true;
})()
"""


def style_script(element):
    """Return JavaScript that relights one scene's URDF meshes.

    NiceGUI fixes its scene lighting (ambient ``0.7*pi`` plus one directional
    ``0.3*pi``) and lets ``material`` change only colour, opacity and side, so a
    light material renders as a flat silhouette. The scene component publishes
    its THREE scene as ``window["scene_" + DOM id]``; if that global is ever
    removed the script returns ``False`` and the panel keeps the default shading.
    """

    return (STYLE_SCRIPT
            .replace("__SCENE_ID__", element.html_id)
            .replace("__AMBIENT__", repr(SCENE_AMBIENT))
            .replace("__KEY_LIGHT__", repr(SCENE_KEY_LIGHT))
            .replace("__SHININESS__", repr(MESH_SHININESS))
            .replace("__SPECULAR__", repr(MESH_SPECULAR)))
