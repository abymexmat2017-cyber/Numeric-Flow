import math
import unittest

from pysource.rk45 import Options, integrate
from pysource.solver import simulate, solve, switching_times


class IntegratorTests(unittest.TestCase):
    def test_exponential_and_step_rejection(self):
        result = integrate(
            lambda t, y: (y[0],),
            (1.0,),
            0.0,
            1.0,
            Options(atol=1e-12, rtol=1e-12, initial_step=1, max_step=1),
        )
        self.assertAlmostEqual(result.steps[-1].state[0], math.e, delta=2e-11)
        self.assertEqual(result.steps[-1].t, 1.0)
        self.assertGreater(result.rejected_steps, 0)

    def test_zero_error_and_short_interval(self):
        result = integrate(lambda t, y: (2.0,), (0.0,), 0.0, 1e-8, Options())
        self.assertAlmostEqual(result.steps[-1].state[0], 2e-8, delta=1e-20)
        self.assertEqual(result.steps[-1].t, 1e-8)

    def test_iteration_budget(self):
        with self.assertRaises(RuntimeError):
            integrate(
                lambda t, y: (y[0],),
                (1.0,),
                0.0,
                1.0,
                Options(max_step=0.01, max_steps=2),
            )

    def test_invalid_options_and_nonfinite_rhs(self):
        for options in (
            Options(atol=0),
            Options(rtol=-1),
            Options(max_step=0),
            Options(initial_step=math.nan),
            Options(max_steps=0),
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                integrate(lambda t, y: y, (1.0,), 0.0, 1.0, options)
        with self.assertRaises(RuntimeError):
            integrate(lambda t, y: (math.inf,), (1.0,), 0.0, 1.0, Options())


class SolverTests(unittest.TestCase):
    def test_no_switch_exact_motion(self):
        result = simulate(0.0, 0.0)
        self.assertEqual(result.switch_times, [])
        for point in result.trajectory:
            self.assertAlmostEqual(point.state[0], -(point.t**2) / 2, delta=1e-13)
            self.assertAlmostEqual(point.state[1], -point.t, delta=1e-13)
            self.assertAlmostEqual(point.state[4], -point.t, delta=1e-13)

    def test_alpha_zero_analytic_boundary_solution(self):
        result = solve(alpha=0)
        tau = 1 - math.sqrt(23 / 24)
        self.assertAlmostEqual(result.parameter, 1 / math.sqrt(23 / 24), delta=3e-10)
        self.assertEqual(len(result.switch_times), 1)
        self.assertAlmostEqual(result.switch_times[0], tau, delta=1e-10)
        self.assertAlmostEqual(result.final.state[0], -11 / 24, delta=1e-10)
        self.assertAlmostEqual(result.final.state[1], 2 * tau - 1, delta=2e-10)
        self.assertAlmostEqual(result.final.state[4], 2 * tau - 1, delta=2e-10)
        self.assertAlmostEqual(result.final.state[3], 0, delta=1e-13)

    def test_all_three_switches_are_found(self):
        # phi is positive at t=0, negative at .3, positive at .7, negative at 1.
        switches = switching_times(1.05, 11.0)
        self.assertEqual(len(switches), 3)
        for t in switches:
            self.assertAlmostEqual(1.05 * (1 - t) - 1 / (1 + 11 * t**4), 0, delta=1e-12)

    def test_endpoint_root_is_not_a_switch(self):
        self.assertEqual(switching_times(1.0, 0.0), [])
        self.assertAlmostEqual(simulate(1.0, 0.0).final.state[0], -0.5, delta=1e-13)

    def test_unresolved_multiple_roots_fail_explicitly(self):
        # Rounded zeros near a stationary inflection used to shift x by ~2e-6.
        for parameter, alpha in ((1.5625, 125 / 27), (0.8922274495253677, 11)):
            with (
                self.subTest(parameter=parameter),
                self.assertRaisesRegex(RuntimeError, "ill-conditioned"),
            ):
                switching_times(parameter, alpha)

    def test_large_alpha_does_not_exhaust_shooting_on_parameter_scale(self):
        result = solve(alpha=1e30)
        self.assertLess(abs(result.final.state[0] + 11 / 24), 1e-10)
        self.assertLess(result.parameter, 1e-28)

    def test_large_alpha_and_near_endpoint_target_can_be_bracketed(self):
        result = solve(alpha=1e30, target=0.499999999)
        self.assertLess(abs(result.final.state[0] - 0.499999999), 1e-10)

    def test_shooting_can_skip_an_ill_conditioned_trial(self):
        result = solve(alpha=8.244512510892962)
        self.assertLess(abs(result.final.state[0] + 11 / 24), 1e-10)

    def test_switches_cannot_be_skipped_by_large_steps(self):
        result = simulate(1.05, 11, Options(max_step=1, initial_step=1))
        x = y = 0.0
        for left, right in zip(result.trajectory, result.trajectory[1:]):
            dt = right.t - left.t
            x += y * dt + 0.5 * left.u * dt**2
            y += left.u * dt
            self.assertAlmostEqual(right.state[0], x, delta=1e-12)
            self.assertAlmostEqual(right.state[1], y, delta=1e-12)
        for t in result.switch_times:
            self.assertTrue(any(point.t == t for point in result.trajectory))

    def test_default_solution_and_independent_cost_quadrature(self):
        result = solve()
        self.assertLess(abs(result.final.state[0] + 11 / 24), 1e-10)
        self.assertLess(abs(result.final.state[3]), 1e-12)
        self.assertEqual(len(result.switch_times), 2)
        integral = 0.0
        endpoints = [0.0, *result.switch_times, 1.0]
        for a, b in zip(endpoints, endpoints[1:]):
            mid = (a + b) / 2
            u = 1 if result.parameter * (1 - mid) > 1 / (1 + 11 * mid**4) else -1
            n = 2000
            h = (b - a) / n
            f = lambda t: u / (1 + 11 * t**4)
            integral += (
                h
                / 3
                * (
                    f(a)
                    + f(b)
                    + sum((4 if i % 2 else 2) * f(a + i * h) for i in range(1, n))
                )
            )
        self.assertAlmostEqual(result.final.state[4], integral, delta=2e-9)

    def test_invalid_problem(self):
        for alpha in (-1, math.nan, math.inf):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                solve(alpha=alpha)
        for target in (-0.5, 0.5, 1, math.nan):
            with self.subTest(target=target), self.assertRaises(ValueError):
                solve(target=target)
        with self.assertRaises(ValueError):
            solve(boundary_tol=0)
        with self.assertRaises(ValueError):
            simulate(math.inf)


if __name__ == "__main__":
    unittest.main()
