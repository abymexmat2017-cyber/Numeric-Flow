"""Explicit Dormand–Prince 5(4), with an infinity-norm error controller."""

from dataclasses import dataclass
import math
from typing import Callable

State = tuple[float, ...]
Rhs = Callable[[float, State], State]

# The lower triangle of the Dormand–Prince Butcher tableau.
A = (
    (),
    (1 / 5,),
    (3 / 40, 9 / 40),
    (44 / 45, -56 / 15, 32 / 9),
    (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
    (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
    (35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84),
)
C = (0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1)
B5 = (35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0)
B4 = (5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40)


@dataclass(frozen=True)
class Options:
    atol: float = 1e-11
    rtol: float = 1e-9
    initial_step: float = 0.01
    max_step: float = 0.05
    max_steps: int = 100_000

    def validate(self) -> None:
        for name in ("atol", "initial_step", "max_step"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.rtol) or self.rtol < 0:
            raise ValueError("rtol must be finite and nonnegative")
        if not isinstance(self.max_steps, int) or self.max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")


@dataclass(frozen=True)
class Step:
    t: float
    state: State


@dataclass
class Integration:
    steps: list[Step]
    rejected_steps: int = 0


def integrate(
    rhs: Rhs, initial: State, start: float, end: float, options: Options = Options()
) -> Integration:
    """Integrate a smooth RHS forward; steps include both endpoints.

    Discontinuities must be split into separate calls by the caller. The local
    error estimate is scaled by atol + rtol * max(abs(old), abs(new)).
    """
    options.validate()
    if not (math.isfinite(start) and math.isfinite(end)) or end < start:
        raise ValueError("integration requires finite start <= end")
    if not initial or not all(math.isfinite(v) for v in initial):
        raise ValueError("initial state must be nonempty and finite")
    state = tuple(initial)
    t = start
    result = Integration([Step(t, state)])
    h = min(options.initial_step, options.max_step)
    attempts = 0
    while t < end:
        if attempts >= options.max_steps:
            raise RuntimeError("integration step budget exhausted")
        attempts += 1
        remaining = end - t
        h = min(h, options.max_step, remaining)
        if t + h == t:
            raise RuntimeError("integration step underflow; relax tolerances")
        stages: list[State] = []
        for row, c in zip(A, C):
            stage_state = tuple(
                value + h * sum(a * k[i] for a, k in zip(row, stages))
                for i, value in enumerate(state)
            )
            derivative = tuple(rhs(t + c * h, stage_state))
            if len(derivative) != len(state) or not all(
                math.isfinite(v) for v in derivative
            ):
                raise RuntimeError(
                    "RHS returned a nonfinite or incorrectly sized state"
                )
            stages.append(derivative)
        fifth = tuple(
            value + h * sum(b * k[i] for b, k in zip(B5, stages))
            for i, value in enumerate(state)
        )
        # Estimate the difference directly, avoiding subtraction of two full states.
        difference = tuple(
            h * sum((b5 - b4) * k[i] for b5, b4, k in zip(B5, B4, stages))
            for i in range(len(state))
        )
        if not all(math.isfinite(v) for v in (*fifth, *difference)):
            raise RuntimeError("nonfinite Runge–Kutta step")
        error = max(
            abs(d) / (options.atol + options.rtol * max(abs(old), abs(new)))
            for d, old, new in zip(difference, state, fifth)
        )
        if not math.isfinite(error):
            raise RuntimeError("nonfinite error estimate")
        if error <= 1:
            t = end if h == remaining else t + h
            state = fifth
            result.steps.append(Step(t, state))
        else:
            result.rejected_steps += 1
        factor = 2.5 if error == 0 else min(2.5, max(0.1, 0.9 * error**-0.2))
        h *= factor
    return result
