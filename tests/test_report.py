import numpy as np
import pandas as pd

from research.descriptive import confusion
from research.report import plot_confusion, plot_decomposition, plot_placebo, plot_wealth


def _summary():
    names = ["C0", "C1", "H4", "D1", "D1b", "R1", "F"]
    cells = {n: {"sharpe_ratio": 0.6 + 0.05 * i} for i, n in enumerate(names)}
    return {
        "cells": {"2": cells},
        "placebo_sharpes": {"2": list(np.random.default_rng(0).normal(0.55, 0.1, 50))},
        "placebo": {"2": {"p_value_vs_C0": 0.2}},
    }


def test_figures_are_written(tmp_path):
    import matplotlib
    matplotlib.use("Agg")

    idx = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(1)
    cells = {n: pd.Series(rng.normal(0.0003, 0.01, len(idx)), index=idx) for n in ["C0", "C1", "H4"]}
    labels = pd.Series(rng.choice(["Bull", "Sideways", "Bear", "Crash"], len(idx)), index=idx)

    plot_decomposition(_summary(), str(tmp_path / "d.png"))
    plot_placebo(_summary(), str(tmp_path / "p.png"))
    plot_wealth(cells, str(tmp_path / "w.png"))
    plot_confusion(confusion(labels, labels.sample(frac=1, random_state=0).set_axis(idx)), str(tmp_path / "c.png"))

    for name in ["d", "p", "w", "c"]:
        assert (tmp_path / f"{name}.png").stat().st_size > 0
