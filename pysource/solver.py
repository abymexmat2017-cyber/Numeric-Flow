"""Switch detection and shooting for the recovered fixed-horizon problem."""

from dataclasses import dataclass, replace
import math
import sys
from typing import Callable

from .rk45 import Options, State, integrate

DEFAULT_TARGET = -11 / 24
ROOT_TOL = 2e-15


class IllConditionedSwitchingError(RuntimeError):
    """Switching times cannot be resolved reliably in double precision."""


def _validate_model(parameter: float, alpha: float) -> None:
    if not math.isfinite(parameter) or parameter < 0:
        raise ValueError("parameter must be finite and nonnegative")
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")


def _opposite(a: float, b: float) -> bool:
    return (a < 0 < b) or (b < 0 < a)


def _bisect(function: Callable[[float], float], left: float, right: float) -> float:
    """Bounded bisection with an interval-width, not residual-only, stop rule."""
    fl, fr = function(left), function(right)
    if fl == 0:
        return left
    if fr == 0:
        return right
    if not _opposite(fl, fr):
        raise RuntimeError("root is not bracketed")
    for _ in range(100):
        middle = left + (right - left) / 2
        fm = function(middle)
        if fm == 0 or right - left <= ROOT_TOL or middle in (left, right):
            return middle
        if _opposite(fl, fm):
            right = middle
        else:
            left, fl = middle, fm
    raise RuntimeError("root bisection did not converge")


def switching_times(parameter: float, alpha: float = 11.0) -> list[float]:
    """Find all interior sign changes of p*(1-t) - 1/(1+alpha*t**4).

    Critical points of g(t)=(1-t)*(1+alpha*t**4) partition [0,1] into
    monotonic intervals. This catches close root pairs missed by grid sampling.
    Roots at the horizon endpoints are not control switches. Unresolved roots
    near tangencies/stationary inflections raise rather than return bad times.
    """
    _validate_model(parameter, alpha)
    critical = [0.0]
    derivative = lambda t: alpha * (t**3 * (4 - 5 * t)) - 1
    if derivative(0.6) > 0:
        critical.extend((_bisect(derivative, 0.0, 0.6), _bisect(derivative, 0.6, 0.8)))
    elif abs(derivative(0.6)) <= 64 * sys.float_info.epsilon:
        critical.append(0.6)
    critical.append(1.0)
    phi = lambda t: parameter * (1 - t) - 1 / (1 + alpha * t**4)
    for t in critical[1:-1]:
        scale = abs(parameter * (1 - t)) + 1 / (1 + alpha * t**4)
        if abs(phi(t)) <= 64 * sys.float_info.epsilon * scale:
            raise IllConditionedSwitchingError(
                "ill-conditioned switching roots: choose parameters farther from a tangency"
            )
    candidates: list[float] = []
    for left, right in zip(critical, critical[1:]):
        fl, fr = phi(left), phi(right)
        if fl == 0 and 0 < left < 1:
            candidates.append(left)
        if _opposite(fl, fr):
            candidates.append(_bisect(phi, left, right))
        if fr == 0 and 0 < right < 1:
            candidates.append(right)
    candidates = sorted(set(candidates))
    boundaries = [0.0, *candidates, 1.0]
    return [
        t
        for i, t in enumerate(candidates, 1)
        if _opposite(phi((boundaries[i - 1] + t) / 2), phi((t + boundaries[i + 1]) / 2))
    ]


@dataclass(frozen=True)
class Point:
    t: float
    state: State  # x, y (velocity), p_x, p_y, integral
    u: float


@dataclass
class Result:
    parameter: float
    alpha: float
    switch_times: list[float]
    trajectory: list[Point]
    rejected_steps: int
    shooting_iterations: int = 0

    @property
    def final(self) -> Point:
        return self.trajectory[-1]

    @property
    def accepted_steps(self) -> int:
        return len(self.trajectory) - 1


def simulate(
    parameter: float, alpha: float = 11.0, options: Options = Options()
) -> Result:
    """Integrate one shot, stopping at every switching time."""
    _validate_model(parameter, alpha)
    options.validate()
    switches = switching_times(parameter, alpha)
    boundaries = [0.0, *switches, 1.0]
    state = (0.0, 0.0, parameter, parameter, 0.0)
    trajectory: list[Point] = []
    rejected = attempts = 0
    for left, right in zip(boundaries, boundaries[1:]):
        middle = (left + right) / 2
        u = 1.0 if parameter * (1 - middle) > 1 / (1 + alpha * middle**4) else -1.0
        if trajectory:
            trajectory[-1] = Point(left, state, u)  # Right-continuous control.
        else:
            trajectory.append(Point(left, state, u))

        def rhs(t: float, values: State) -> State:
            return (values[1], u, 0.0, -values[2], u / (1 + alpha * t**4))

        if attempts >= options.max_steps:
            raise RuntimeError("integration step budget exhausted")
        segment = integrate(
            rhs,
            state,
            left,
            right,
            replace(options, max_steps=options.max_steps - attempts),
        )
        trajectory.extend(Point(step.t, step.state, u) for step in segment.steps[1:])
        state = segment.steps[-1].state
        rejected += segment.rejected_steps
        attempts += len(segment.steps) - 1 + segment.rejected_steps
    return Result(parameter, alpha, switches, trajectory, rejected)


def solve(
    alpha: float = 11.0,
    target: float = DEFAULT_TARGET,
    options: Options = Options(),
    boundary_tol: float = 1e-10,
) -> Result:
    """Find p by bracketed shooting; terminal position increases with p."""
    _validate_model(0.0, alpha)
    options.validate()
    if not math.isfinite(target) or not -0.5 < target < 0.5:
        raise ValueError("target must be finite and strictly between -0.5 and 0.5")
    if not math.isfinite(boundary_tol) or boundary_tol <= 0:
        raise ValueError("boundary_tol must be finite and positive")
    left, right = 0.0, 1.28 / (1 + alpha)
    for _ in range(1100):
        try:
            shot = simulate(right, alpha, options)
            if shot.final.state[0] >= target:
                break
        except IllConditionedSwitchingError:
            pass  # A degenerate trial need not make the desired shot degenerate.
        if right > sys.float_info.max / 2:
            raise RuntimeError("shooting bracket overflow")
        right *= 2
    else:
        raise RuntimeError("could not bracket shooting parameter")
    for iteration in range(1, 101):
        for fraction in (0.5, 0.4, 0.6, 0.3, 0.7):
            middle = left + (right - left) * fraction
            try:
                shot = simulate(middle, alpha, options)
                break
            except IllConditionedSwitchingError:
                continue
        else:
            raise IllConditionedSwitchingError(
                "ill-conditioned switching roots throughout shooting bracket"
            )
        residual = shot.final.state[0] - target
        if abs(residual) <= boundary_tol:
            shot.shooting_iterations = iteration
            return shot
        if middle in (left, right):
            break
        if residual < 0:
            left = middle
        else:
            right = middle
    raise RuntimeError("shooting did not converge; relax boundary_tol")
