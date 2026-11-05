"""Run the interactive SAPIEN simulation outside the tracking process."""

import multiprocessing as mp
import signal
import time

import numpy as np


_JOINT_COUNT = 21
_DEFAULT_UPDATE_HZ = 60.0
_STARTUP_TIMEOUT = 60.0
_SHUTDOWN_TIMEOUT = 5.0


def _send_status(connection, message):
    try:
        connection.send(message)
    except (BrokenPipeError, EOFError, OSError):
        pass


def _run_simulation(
    command_buffer,
    command_generation,
    command_lock,
    stop_event,
    status,
    update_hz,
    factory,
    headless,
):
    """Child entry point; SAPIEN and its window stay on this main thread."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    simulation = None
    try:
        if factory is None:
            from simulation.grasp import GraspSimulation

            simulation = GraspSimulation(headless=headless)
        else:
            simulation = factory()
        _send_status(status, ("ready",))

        shared_command = np.frombuffer(
            command_buffer, dtype=np.float64, count=_JOINT_COUNT
        )
        seen_generation = 0
        period = 1.0 / update_hz
        deadline = time.monotonic()
        while not stop_event.is_set():
            command = None
            with command_lock:
                generation = command_generation.value
                if generation != seen_generation:
                    command = shared_command.copy()
                    seen_generation = generation

            if not simulation.update(command):
                _send_status(status, ("closed",))
                return

            # Do not try to catch up missed GUI frames. GraspSimulation keeps
            # its own bounded 100 Hz physics accumulator.
            deadline += period
            delay = deadline - time.monotonic()
            if delay > 0:
                if stop_event.wait(delay):
                    break
            else:
                deadline = time.monotonic()
        _send_status(status, ("closed",))
    except BaseException as error:
        _send_status(status, ("error", type(error).__name__, str(error)))
    finally:
        if simulation is not None:
            try:
                simulation.close()
            except Exception:
                pass
        status.close()


class GraspSimulationProcess:
    """Non-blocking proxy for a SAPIEN simulation in a spawned process."""

    def __init__(
        self,
        update_hz=_DEFAULT_UPDATE_HZ,
        startup_timeout=_STARTUP_TIMEOUT,
        *,
        _factory=None,
        _headless=False,
    ):
        if not np.isfinite(update_hz) or update_hz <= 0:
            raise ValueError("update_hz must be positive and finite")
        if not np.isfinite(startup_timeout) or startup_timeout <= 0:
            raise ValueError("startup_timeout must be positive and finite")

        # The parent already owns camera, console, and CUDA threads. ``fork``
        # would duplicate that state and is unsafe for both CUDA and Vulkan.
        context = mp.get_context("spawn")
        self._command = context.RawArray("d", _JOINT_COUNT)
        self._generation = context.RawValue("Q", 0)
        self._command_lock = context.Lock()
        self._stop_event = context.Event()
        self._status, child_status = context.Pipe(duplex=False)
        self._process = context.Process(
            target=_run_simulation,
            args=(
                self._command,
                self._generation,
                self._command_lock,
                self._stop_event,
                child_status,
                float(update_hz),
                _factory,
                _headless,
            ),
            name="mmhand-grasp-simulation",
            daemon=True,
        )
        self._state = "starting"
        self._error = None
        self._closed = False
        try:
            self._process.start()
            child_status.close()
            self._wait_until_ready(float(startup_timeout))
        except BaseException:
            child_status.close()
            self.close()
            raise

    def _read_status(self, timeout=0.0):
        if self._status is None:
            return
        try:
            if not self._status.poll(timeout):
                return
            message = self._status.recv()
        except (EOFError, OSError):
            return
        kind = message[0]
        if kind == "ready":
            self._state = "running"
        elif kind == "closed":
            self._state = "closed"
        elif kind == "error":
            self._state = "error"
            self._error = message[1:]

    def _raise_remote_error(self):
        name, message = self._error
        detail = f"{name}: {message}" if message else name
        raise RuntimeError(f"SAPIEN simulation failed: {detail}")

    def _wait_until_ready(self, timeout):
        deadline = time.monotonic() + timeout
        while self._state == "starting":
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"SAPIEN simulation did not start within {timeout:g}s"
                )
            self._read_status(min(remaining, 0.05))
            if self._process.exitcode is not None and self._state == "starting":
                self._read_status(0.05)
                raise RuntimeError(
                    "SAPIEN simulation exited during startup "
                    f"(exit code {self._process.exitcode})"
                )
        if self._state == "error":
            self._raise_remote_error()
        if self._state != "running":
            raise RuntimeError("SAPIEN simulation closed during startup")

    def update(self, robot_joints=None):
        """Publish the newest command without waiting for physics or rendering."""
        if self._closed:
            return False
        self._read_status()
        if self._state == "error":
            self._raise_remote_error()
        if self._state == "closed":
            return False
        if self._process.exitcode is not None:
            self._read_status(0.05)
            if self._state == "error":
                self._raise_remote_error()
            if self._state == "closed":
                return False
            if self._process.exitcode == 0:
                self._state = "closed"
                return False
            raise RuntimeError(
                "SAPIEN simulation process exited unexpectedly "
                f"(exit code {self._process.exitcode})"
            )

        if robot_joints is not None:
            command = np.asarray(robot_joints, dtype=np.float64)
            if command.shape != (_JOINT_COUNT,) or not np.isfinite(command).all():
                raise ValueError("robot_joints must contain 21 finite radians")
            with self._command_lock:
                np.frombuffer(
                    self._command, dtype=np.float64, count=_JOINT_COUNT
                )[:] = command
                self._generation.value += 1
        return True

    def close(self):
        """Stop the child and reclaim it; safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        if self._process.pid is not None:
            self._process.join(_SHUTDOWN_TIMEOUT)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(_SHUTDOWN_TIMEOUT)
        if self._status is not None:
            self._status.close()
            self._status = None
        self._process.close()
