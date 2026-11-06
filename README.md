# mmhand-teleop

`mmhand-teleop` converts a monocular camera (WiLoR ONNX or MediaPipe) or a
MANUS glove into one common 21-point hand observation, retargets the left hand
to all 21 MMHand joints, and serves a browser dashboard. ROS 2 output is on by
default; `--no-ros` runs the dashboard without it. The SAPIEN grasp simulation
is optional (`--sim`).

## Installation

Python 3.12 and Git LFS are required. The WiLoR weights are stored with Git LFS.

```bash
git lfs install
git lfs pull
mamba env create -f environment.yml
conda activate mmhand
python -m pip install -r input/wilor/requirements.txt
```

`environment.yml` installs the console's `nicegui`, which brings FastAPI and
uvicorn. If you reuse an existing environment instead of creating one, install
the console dependency by hand:

```bash
python -m pip install nicegui==3.16.0
```

The requirements file installs ONNX Runtime with its CUDA and cuDNN
dependencies; WiLoR requires `CUDAExecutionProvider` and an NVIDIA GPU.

MediaPipe is optional:

```bash
python -m pip install --no-deps -r input/mediapipe/requirements.txt
```

`--no-deps` keeps the environment's OpenCV instead of installing
`opencv-contrib-python`; the requirements file lists mediapipe's non-OpenCV
runtime dependencies. If you are not using the environment above, drop
`--no-deps`. Download
[hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task)
into `input/mediapipe/assets/`. The model file is not tracked by git.

MANUS requires the bundled Core SDK bridge; follow
`input/manus/assets/README.md`. Calibrate each glove once before tracking:

```bash
python -m input.manus.calibration --side left          # import the saved .mcal
python -m input.manus.calibration --side left --force  # run the wizard again
python -m input.manus.calibration --side right         # per glove, not per hand
```

The wizard waits up to `MANUS_CALIBRATION_CONNECT_TIMEOUT` for a connected
glove, lists the official SDK calibration steps, records each pose when you
press Enter, and saves
`input/manus/calibrations/<side>_<8 hex digit glove id>.mcal` (git-ignored).
`q` or Ctrl-C aborts without writing a file. `track.py` imports the saved blob
at startup and only runs the wizard when no file exists for that glove.

SAPIEN is optional and used only with `--sim`:
`python -m pip install -r simulation/requirements.txt`.

## Running

All runtime settings live in `config.py`:

```python
INPUT_SOURCE = "wilor"   # "mediapipe", "wilor", or "manus"
```

Cameras are not configured in `config.py`: the connection bar lists the
available devices, preselects the first capture node, and leaves the palm
camera unchosen until you pick one.

```bash
python track.py          # console with ROS 2 output on
python track.py --no-ros # console with ROS 2 output off
python track.py --sim    # also open the optional SAPIEN grasp window
```

`--no-ros` and `--sim` are the only command-line arguments; every other setting
lives in `config.py`. `--sim` still opens the SAPIEN window and its keyboard
control; the console itself does not show the simulation. The console opens
in your browser automatically; if it does not, open the URL it prints (the
default is <http://127.0.0.1:8080>, and a busy port moves to the next free
one). The console has one connection bar (input source and
camera, palm camera, ROS output), one status bar, and a 2x2 grid: gesture
camera, palm stereo camera, input hand, retargeted MMHand. Both 3D panels are
drawn in the page with NiceGUI's `ui.scene`: the input hand as a canonical
joint/bone cloud, the MMHand as its URDF meshes posed by the 21 joint angles.
The console is the only server; nothing else listens on a port.

Devices start unconnected. The console connects the configured
`INPUT_SOURCE` once at startup so `python track.py` still works, then every
connection is owned by the Connect buttons. A MANUS Connect imports the saved
`.mcal` if one exists and never runs the interactive wizard, which needs a
terminal; calibrate with `python -m input.manus.calibration --side left` first.

A V4L2 node has exactly one owner, so the second open reports
`Cannot open camera ... (another program may already hold it)`: stop the
previous console process, and give the palm panel a different node than the
gesture camera. The Connect buttons stay disabled while an attempt runs, and a
failed connect leaves its reason in the status bar.

Only a ready left hand enters retargeting; a right hand is displayed only. The
first WiLoR start loads the models and takes about 10-20 s; the Connect button
stays busy meanwhile.

The Input hand 3D panel uses a canonical display: the hand is expressed in its
palm frame, palm toward the screen, and normalized by palm width, so moving or
rotating the hand does not rotate or resize the view. Retargeting still uses
the raw source coordinates. Without finger-pad normals (MediaPipe) the display
uses the palm-frame normal.

## How it works

**Input.** A camera frame goes through the WiLoR detector and reconstructor
ONNX models, producing MANO vertices. A joint regressor and fingertip vertices
yield Standard21 points and five outward finger-pad normals. One Euro filters
smooth points and directions; angles are solved from the filtered points and
packaged into `input.frame.InputFrame`. MediaPipe produces Standard21 world
landmarks in metres, with no finger-pad normals: those frames carry
`finger_pad_directions=None`. The MANUS path maps Raw25 SDK data to the same
Standard21 points, pad directions, and ergonomics-derived angles.

**Retargeting.** `retarget.Retargeter` builds a thumb-CMC-local frame from the
input points, maps the input angles to robot targets, and runs a bounded SLSQP
solve over all 21 joints with analytic Jacobians. The weighted objective
contains the thumb-tip position, the two thumb bends, the 16 four-finger joint
angles, four fingertip vectors, the thumb pad, and four finger pad directions.
The two pad terms are disabled (weight 0) on frames without
`finger_pad_directions`, so MediaPipe input never fakes fingertip pointing.
The previous solution is the next warm start; outputs are clipped to URDF
limits. Input, retargeting, visualization, ROS, and simulation contain no
device-specific branches.

**Output.** `gui.Teleop` reads one frame per loop, submits ready left-hand
frames to a background solver, and publishes the newest display snapshot as two
plain arrays (`human_points` for the input hand, `robot` for the joint angles).
The page owns its 3D scenes and draws those arrays on its own timer, so the
tracking thread never touches a UI element and a slow browser costs frames, not
acquisition or solving. ROS 2 output is published while the
console's ROS switch is on (`--no-ros` starts it off); it sends
`std_msgs/msg/Float32MultiArray`:

| Topic | Shape | Meaning |
|---|---:|---|
| `/mmhand/keypoints` | 21x3 | source-frame points, metres |
| `/mmhand/finger_pad_directions` | 5x3 | outward pad normals |
| `/mmhand/joint_angle_deg` | 21 | MMHand J00-J20, degrees |

The finger-pad topic is published only when the input source provides pointing.

## Repository layout

```text
input/
  frame.py              InputFrame, topology, CMC frame
  camera.py             latest-frame OpenCV capture
  mediapipe/source.py   Hand Landmarker points, filtering, InputFrame
  wilor/source.py       ONNX detector/reconstructor, filtering, InputFrame
  manus/adapter.py      Raw25 -> Standard21 mapping and pad directions
  manus/calibration.py  terminal calibration gate
  manus/source.py       official SDK transport
assets/mmhand/          MMHand URDF, visual meshes, collision meshes
config.py               runtime settings
one_euro.py             ndarray One Euro filter
retarget.py             URDF FK, analytic Jacobians, SLSQP retargeter
ros.py                  ROS 2 publishers
scene.py                the two `ui.scene` 3D panels
track.py                process entry point
simulation/             optional SAPIEN grasp simulation
test_*.py               unit tests
```

## Tests and licenses

```bash
python -m unittest -v test_hand_capture.py test_wilor.py test_manus_phase1.py test_mediapipe.py test_gui.py
python -m unittest -v test_simulation.py
```

WiLoR model terms are in `input/wilor/assets/WILOR_MODEL_LICENSE.txt`; related
notices are in `input/wilor/assets/THIRD_PARTY_NOTICES.md`. MANUS SDK terms ship
with `input/manus/assets/`. The MMHand URDF and meshes come from
[`mm-hand/mmhand-structure`](https://github.com/mm-hand/mmhand-structure).
