"""End-to-end parity tests, enabled by NUMERIC_FLOW_CPP=/path/to/binary."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

CPP = os.environ.get("NUMERIC_FLOW_CPP")


@unittest.skipUnless(CPP, "set NUMERIC_FLOW_CPP to enable C++/Python parity tests")
class ParityTests(unittest.TestCase):
    def test_solutions_and_csv(self):
        cases = [
            [],
            ["--alpha", "0"],
            ["--alpha", "5"],
            ["--alpha", "100"],
            ["--target", "0.2"],
            ["--parameter", "1.05", "--max-step", "1"],
            ["--parameter", "0"],
            ["--parameter", "1", "--alpha", "0"],
            ["--alpha", "1e30"],
            ["--alpha", "1e-309"],
            ["--alpha", "1e30", "--target", "0.499999999"],
        ]
        cases.append(["--alpha", "8.244512510892962"])
        for args in cases:
            with self.subTest(args=args), tempfile.TemporaryDirectory() as directory:
                summaries, trajectories = [], []
                for i, command in enumerate(
                    ([sys.executable, "-m", "pysource"], [CPP])
                ):
                    output = Path(directory) / f"{i}.csv"
                    run = subprocess.run(
                        [*command, *args, "--output", str(output)],
                        capture_output=True,
                        text=True,
                        timeout=20,
                    )
                    self.assertEqual(run.returncode, 0, run.stderr)
                    summaries.append(json.loads(run.stdout))
                    with output.open(encoding="utf-8", newline="") as stream:
                        trajectories.append(list(csv.DictReader(stream)))
                first, second = summaries
                self.assertEqual(set(first), set(second))
                self.assertAlmostEqual(
                    first["parameter"], second["parameter"], delta=1e-8
                )
                self.assertEqual(
                    len(first["switch_times"]), len(second["switch_times"])
                )
                for a, b in zip(first["switch_times"], second["switch_times"]):
                    self.assertAlmostEqual(a, b, delta=1e-8)
                for name in first["final"]:
                    self.assertAlmostEqual(
                        first["final"][name], second["final"][name], delta=2e-8
                    )
                # Adaptive grids can differ: validate every trajectory against
                # the exact constant-acceleration law using its own intervals.
                for rows in trajectories:
                    self.assertEqual(float(rows[0]["t"]), 0.0)
                    self.assertEqual(float(rows[-1]["t"]), 1.0)
                    for a, b in zip(rows, rows[1:]):
                        dt = float(b["t"]) - float(a["t"])
                        self.assertGreater(dt, 0)
                        expected = (
                            float(a["x"])
                            + float(a["y"]) * dt
                            + float(a["u"]) * dt**2 / 2
                        )
                        self.assertAlmostEqual(float(b["x"]), expected, delta=1e-11)

    def test_cpp_rejects_invalid_inputs(self):
        for args in (
            ("--alpha=-1",),
            ("--rtol=-1",),
            ("--atol=nan",),
            ("--parameter=inf",),
            ("--target=1",),
            ("--alpha=1oops",),
            ("--unknown",),
            ("--output",),
            ("--boundary-tol=0",),
        ):
            with self.subTest(args=args):
                run = subprocess.run(
                    [CPP, *args], capture_output=True, text=True, timeout=20
                )
                self.assertNotEqual(run.returncode, 0)
                self.assertIn("error", run.stderr.lower())
                self.assertEqual(run.stdout, "")

    def test_cpp_output_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            run = subprocess.run(
                [CPP, "--output", directory], capture_output=True, text=True, timeout=20
            )
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("error", run.stderr.lower())
            self.assertEqual(run.stdout, "")

    def test_both_report_unresolved_roots(self):
        for command in ([sys.executable, "-m", "pysource"], [CPP]):
            run = subprocess.run(
                [*command, "--parameter=1.5625", "--alpha=4.62962962962963"],
                capture_output=True,
                text=True,
                timeout=20,
            )
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("ill-conditioned", run.stderr)
            self.assertEqual(run.stdout, "")

    def test_cpp_reports_stdout_failure(self):
        with open(os.devnull, "rb") as unwritable:
            run = subprocess.run(
                [CPP, "--parameter=0", "--alpha=0"],
                stdout=unwritable,
                stderr=subprocess.PIPE,
                text=True,
                timeout=20,
            )
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("error", run.stderr)
