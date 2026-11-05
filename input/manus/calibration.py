"""Terminal-only MANUS Core Integrated glove calibration workflow."""

import argparse
from pathlib import Path
import time

from config import MANUS_CALIBRATION_CONNECT_TIMEOUT, MANUS_CALIBRATION_DIR


def calibration_path(directory, handedness, glove_id):
    side = str(handedness).lower()
    if side not in ("left", "right"):
        raise ValueError(f"Cannot name calibration for side {handedness!r}")
    return Path(directory) / f"{side}_{int(glove_id):08X}.mcal"


def _packets(value):
    if value is None:
        return ()
    return value if isinstance(value, (tuple, list)) else (value,)


def wait_for_glove(
    transport,
    *,
    handedness="Left",
    timeout=MANUS_CALIBRATION_CONNECT_TIMEOUT,
    clock=time.monotonic,
    sleep=time.sleep,
    output=print,
):
    deadline = clock() + float(timeout)
    output(f"MANUS calibration: waiting for a connected {handedness} glove...")
    while clock() < deadline:
        for packet in _packets(transport.read()):
            if packet.handedness != handedness:
                continue
            if packet.glove_id is None or int(packet.glove_id) <= 0:
                raise RuntimeError("MANUS frame is missing a valid glove ID")
            return packet
        sleep(0.01)
    raise TimeoutError(
        f"No {handedness} MANUS glove appeared within {float(timeout):g} seconds"
    )


def _duration_text(duration):
    duration = float(duration)
    if duration < 0:
        return f"continuous step, estimated {abs(duration):.1f}s"
    return f"{duration:.1f}s"


def _prompt_enter(prompt, input_fn):
    try:
        answer = input_fn(prompt)
    except EOFError as error:
        raise RuntimeError("MANUS calibration requires an interactive terminal") from error
    if answer.strip().lower() in ("q", "quit", "cancel"):
        raise KeyboardInterrupt


def _save_calibration(path, payload):
    payload = bytes(payload)
    if not payload:
        raise RuntimeError("MANUS returned an empty calibration file")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_calibration_wizard(
    transport,
    packet,
    path,
    *,
    input_fn=input,
    output=print,
):
    glove_id = int(packet.glove_id)
    steps = transport.calibration_steps(glove_id)
    if not steps:
        raise RuntimeError("MANUS returned no calibration steps")
    output("")
    output("No saved calibration was found for this glove.")
    output(f"Glove: {packet.handedness}  ID: 0x{glove_id:08X}")
    output(f"The official MANUS sequence contains {len(steps)} steps.")
    output("Follow each SDK instruction exactly. Enter q to cancel.")
    _prompt_enter("Press Enter to start calibration: ", input_fn)

    started = False
    try:
        transport.calibration_start(glove_id)
        started = True
        for number, step in enumerate(steps, 1):
            output("")
            output(f"Step {number}/{len(steps)}: {step.title or 'Untitled step'}")
            if step.description:
                output(step.description)
            output(f"Duration: {_duration_text(step.duration)}")
            _prompt_enter("Assume the pose, then press Enter: ", input_fn)
            output("Recording... keep following the instruction.")
            transport.calibration_run_step(glove_id, step.index)
            output("Step completed.")
        transport.calibration_finish(glove_id)
        started = False
        payload = transport.calibration_export(glove_id)
        _save_calibration(path, payload)
    except BaseException:
        if started:
            try:
                transport.calibration_stop(glove_id)
            except Exception as stop_error:
                output(f"Warning: failed to stop MANUS calibration: {stop_error}")
        raise
    output("")
    output(f"Calibration completed and saved to: {path}")
    return Path(path)


def ensure_calibration(
    transport,
    *,
    directory=MANUS_CALIBRATION_DIR,
    handedness="Left",
    timeout=MANUS_CALIBRATION_CONNECT_TIMEOUT,
    force=False,
    input_fn=input,
    output=print,
    clock=time.monotonic,
    sleep=time.sleep,
):
    packet = wait_for_glove(
        transport,
        handedness=handedness,
        timeout=timeout,
        clock=clock,
        sleep=sleep,
        output=output,
    )
    path = calibration_path(directory, packet.handedness, packet.glove_id)
    if path.is_file() and not force:
        payload = path.read_bytes()
        if not payload:
            raise RuntimeError(f"Saved MANUS calibration is empty: {path}")
        transport.calibration_import(packet.glove_id, payload)
        output(
            f"MANUS calibration loaded for {packet.handedness} glove "
            f"0x{int(packet.glove_id):08X}: {path}"
        )
        return path
    return run_calibration_wizard(
        transport, packet, path, input_fn=input_fn, output=output
    )


def import_saved_calibration(
    transport,
    *,
    directory=MANUS_CALIBRATION_DIR,
    handedness="Left",
    timeout=3.0,
    output=print,
):
    """Import the saved .mcal for the connected glove without prompting.

    The web GUI cannot answer the wizard's Enter prompts, so it uses this path.
    A missing file is not fatal: the glove still tracks, and the source reports
    "not calibrated" by itself. Returns the imported path, or None.
    """

    packet = wait_for_glove(
        transport,
        handedness=handedness,
        timeout=timeout,
        output=output,
    )
    path = calibration_path(directory, packet.handedness, packet.glove_id)
    if not path.is_file():
        output(
            f"MANUS: no saved calibration {path}; run "
            f"python -m input.manus.calibration --side {str(handedness).lower()}"
        )
        return None
    transport.calibration_import(packet.glove_id, path.read_bytes())
    output(f"MANUS calibration imported: {path}")
    return path


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the official MANUS Core Integrated terminal calibration."
    )
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--force", action="store_true", help="calibrate again")
    parser.add_argument(
        "--directory", type=Path, default=MANUS_CALIBRATION_DIR,
        help="directory containing per-glove .mcal files",
    )
    return parser.parse_args(argv)


def main(argv=None):
    from .source import OfficialSdkTransport

    args = _parse_args(argv)
    transport = OfficialSdkTransport()
    try:
        ensure_calibration(
            transport,
            directory=args.directory,
            handedness=args.side.title(),
            force=args.force,
        )
    finally:
        transport.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
