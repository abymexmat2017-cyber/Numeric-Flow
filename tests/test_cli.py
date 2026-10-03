import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class CliTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "pysource", *args],
            capture_output=True,
            text=True,
            timeout=20,
        )

    def test_json_and_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "trajectory.csv"
            process = self.run_cli("--output", str(output))
            self.assertEqual(process.returncode, 0, process.stderr)
            summary = json.loads(process.stdout)
            self.assertLess(abs(summary["residual"]), 1e-10)
            self.assertEqual(summary["final"]["t"], 1.0)
            with output.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(
                list(rows[0]), ["t", "x", "y", "p_x", "p_y", "u", "integral"]
            )
            self.assertEqual(float(rows[0]["t"]), 0.0)
            self.assertEqual(float(rows[-1]["t"]), 1.0)
            self.assertEqual(float(rows[-1]["x"]), summary["final"]["x"])
            self.assertEqual(len(rows), summary["accepted_steps"] + 1)

    def test_fixed_parameter(self):
        process = self.run_cli("--parameter", "0", "--alpha", "0")
        self.assertEqual(process.returncode, 0, process.stderr)
        summary = json.loads(process.stdout)
        self.assertAlmostEqual(summary["final"]["integral"], -1.0, delta=1e-13)
        self.assertEqual(summary["shooting_iterations"], 0)

    def test_bad_input_fails_without_traceback(self):
        for args in (
            ("--alpha=-1",),
            ("--atol=0",),
            ("--target=0.7",),
            ("--parameter=nan",),
            ("--alpha=inf",),
            ("--alpha=oops",),
            ("--unknown",),
            ("--output",),
            ("--boundary-tol=0",),
        ):
            with self.subTest(args=args):
                process = self.run_cli(*args)
                self.assertNotEqual(process.returncode, 0)
                self.assertIn("error", process.stderr.lower())
                self.assertNotIn("Traceback", process.stderr)
                self.assertEqual(process.stdout, "")

    def test_output_failure_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            process = self.run_cli("--output", directory)
            self.assertNotEqual(process.returncode, 0)
            self.assertNotIn("Traceback", process.stderr)
            self.assertEqual(process.stdout, "")


if __name__ == "__main__":
    unittest.main()
