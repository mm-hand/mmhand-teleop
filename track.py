import argparse

from gui import Teleop, run_console


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-ros", action="store_false", dest="ros",
        help="start with ROS 2 output switched off",
    )
    parser.add_argument("--sim", action="store_true")
    return parser.parse_args(argv)


def main():
    args = _parse_args()
    return run_console(Teleop(sim=args.sim, ros=args.ros))


if __name__ == "__main__":
    raise SystemExit(main())
