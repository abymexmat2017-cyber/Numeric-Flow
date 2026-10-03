"""Numerical restoration of a bang-bang optimal-control experiment."""

from .rk45 import Options
from .solver import Result, simulate, solve, switching_times

__all__ = ["Options", "Result", "simulate", "solve", "switching_times"]
__version__ = "0.1.0"
