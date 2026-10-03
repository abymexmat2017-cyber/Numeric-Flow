"""Plot a Numeric Flow CSV; matplotlib is an optional dependency."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/trajectory.svg"))
    args = parser.parse_args()
    with args.csv.open(encoding="utf-8", newline="") as stream:
        rows = [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]
    t = [row["t"] for row in rows]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.fonttype": "none",
            "svg.hashsalt": "numeric-flow",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.6), layout="constrained")
    fig.set_facecolor("#f8fafc")
    fig.suptitle(
        "NUMERIC FLOW  /  from acceleration to trajectory",
        fontsize=18,
        fontweight="bold",
        color="#162b3d",
    )
    for axis, (field, title, color) in zip(
        axes.flat,
        [
            ("x", "Position  x(t)", "#167c80"),
            ("y", "Velocity  y(t)", "#3269a8"),
            ("u", "Control  u(t)", "#d05a38"),
            ("integral", "Accumulated cost  I(t)", "#7053a2"),
        ],
    ):
        values = [row[field] for row in rows]
        axis.set_title(title, loc="left", fontweight="bold", pad=10)
        if field == "u":
            axis.step(t, values, where="post", color=color, linewidth=2.3)
            axis.set_ylim(-1.35, 1.35)
            axis.set_yticks([-1, 0, 1])
        else:
            axis.plot(t, values, color=color, linewidth=2.3)
            axis.plot(t[-1], values[-1], "o", color=color, markersize=5)
            axis.annotate(
                f"{values[-1]:.6f}",
                (t[-1], values[-1]),
                xytext=(-8, 12),
                textcoords="offset points",
                ha="right",
                color=color,
            )
        for left, right in zip(rows, rows[1:]):
            if left["u"] > 0:
                axis.axvspan(
                    left["t"], right["t"], color="#eeb885", alpha=0.15, linewidth=0
                )
        axis.set_xlim(0, 1.02)
        axis.set_xlabel("Time  t")
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#bcc8d2")
    fig.savefig(
        args.output,
        dpi=160,
        metadata={"Date": None} if args.output.suffix == ".svg" else {},
    )
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
