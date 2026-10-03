"""Command-line entry point: python3 -m pysource."""

import argparse
import csv
import json
import math
from pathlib import Path

from .rk45 import Options
from .solver import DEFAULT_TARGET, Result, simulate, solve


def summary(result: Result, target: float) -> dict:
    x, y, px, py, integral = result.final.state
    return {
        "alpha": result.alpha,
        "target": target,
        "parameter": result.parameter,
        "switch_times": result.switch_times,
        "final": {
            "t": result.final.t,
            "x": x,
            "y": y,
            "p_x": px,
            "p_y": py,
            "integral": integral,
        },
        "residual": x - target,
        "accepted_steps": result.accepted_steps,
        "rejected_steps": result.rejected_steps,
        "shooting_iterations": result.shooting_iterations,
    }


def write_csv(result: Result, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("t", "x", "y", "p_x", "p_y", "u", "integral"))
        for point in result.trajectory:
            x, y, px, py, integral = point.state
            writer.writerow((point.t, x, y, px, py, point.u, integral))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bang-bang optimal control with Dormand–Prince 5(4)."
    )
    parser.add_argument(
        "--alpha", type=float, default=11.0, help="cost weight parameter (default: 11)"
    )
    parser.add_argument(
        "--target",
        type=float,
        default=DEFAULT_TARGET,
        help="terminal x in (-0.5, 0.5) (default: -11/24)",
    )
    parser.add_argument(
        "--parameter", type=float, help="fixed costate parameter; skip shooting"
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-11,
        help="absolute local tolerance (default: 1e-11)",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-9,
        help="relative local tolerance (default: 1e-9)",
    )
    parser.add_argument(
        "--max-step",
        type=float,
        default=0.05,
        help="maximum integration step (default: 0.05)",
    )
    parser.add_argument(
        "--boundary-tol",
        type=float,
        default=1e-10,
        help="terminal x tolerance (default: 1e-10)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write CSV to this path (parent directory must exist)",
    )
    args = parser.parse_args()
    try:
        if not math.isfinite(args.target) or not -0.5 < args.target < 0.5:
            raise ValueError("target must be finite and strictly between -0.5 and 0.5")
        if not math.isfinite(args.boundary_tol) or args.boundary_tol <= 0:
            raise ValueError("boundary_tol must be finite and positive")
        options = Options(atol=args.atol, rtol=args.rtol, max_step=args.max_step)
        result = (
            solve(args.alpha, args.target, options, args.boundary_tol)
            if args.parameter is None
            else simulate(args.parameter, args.alpha, options)
        )
        if args.output is not None:
            write_csv(result, args.output)
        print(
            json.dumps(summary(result, args.target), indent=2, allow_nan=False),
            flush=True,
        )
    except (ValueError, RuntimeError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
