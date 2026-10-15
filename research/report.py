"""Figures and detection diagnostics for a fully analysed window (exploratory).

    python -m research.report                    # development sample
    python -m research.run_backcast --report     # backcast, under its own config

Rebuilds the cells with the same building blocks as research/run_study.py, checks
that their Sharpe ratios match results/study/<window>/summary.json, then writes
figures and tables next to it. Nothing here changes a pre-registered result.
"""
import json
import os

import numpy as np
import pandas as pd

from config import load_config
from research.books import constant_posteriors, optimizer_targets, posteriors_of, run_cell
from research.descriptive import agreement_rate, confusion, delay_summary, detection_delays
from research.h4 import correlation_aware_targets, stock_bond_correlation, with_cash
from research.oracles import argmax_labels, hindsight_posteriors, realized_vol_posteriors
from research.run_study import BACKCAST_END, DEV_END, OUT_DIR, PRIMARY_BP
from research.stats import annual_sharpe

CFG = load_config()
WINDOW_END = {"dev": DEV_END, "backcast": BACKCAST_END}
# Stress period shaded on the wealth chart, chosen by what each sample is known for.
STRESS = {
    "dev": ("2022-01-01", "2022-12-31", "Shaded: 2022, when stocks and bonds fell together"),
    "backcast": ("2000-03-24", "2002-10-09", "Shaded: the 2000-2002 equity bear market"),
}

# Validated categorical slots (light mode) and neutral inks.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, INK, INK_2, GRID = "#a3a29c", "#0b0b0b", "#52514e", "#e6e5e0"

CELL_NAMES = {
    "C1": "Static blend",
    "C0": "Strategy (C0)",
    "H4": "Strategy + cash rule (H4)",
    "D1": "Detection oracle (D1)",
    "R1": "Response oracle (R1)",
    "F": "Both oracles (F)",
    "D1b": "Future-vol oracle (D1b)",
}
INVESTABLE = {"C0", "C1", "H4"}


def rebuild_cells(summary: dict, window: str = "dev") -> tuple[dict[str, pd.Series], pd.Series, pd.Series]:
    """Daily returns of every cell at the primary cost, plus causal and hindsight labels."""
    from data.risk_free import fetch_risk_free
    from models.hmm import load_features, walk_forward_regimes

    end = WINDOW_END[window]
    returns = pd.read_parquet(CFG["paths"]["returns"]).loc[:end]
    rf = fetch_risk_free()
    features = load_features().loc[:end]
    regimes = walk_forward_regimes(features)
    index = regimes.index[regimes.index <= end]
    score = index[1:]
    cost = PRIMARY_BP / 10_000

    targets = optimizer_targets(index, returns, rf)
    post = posteriors_of(regimes.loc[index])
    d1 = hindsight_posteriors(features, index)
    d1b = realized_vol_posteriors(returns["SPY"], index)
    maps = summary["H3"]["mappings"]
    h4_targets = correlation_aware_targets(targets, stock_bond_correlation(returns))

    runs = {
        "C0": run_cell(targets, post, returns, cost=cost),
        "C1": run_cell(targets, constant_posteriors(index), returns, cost=cost),
        "H4": run_cell(h4_targets, post, with_cash(returns, rf), cost=cost),
        "D1": run_cell(targets, d1, returns, cost=cost),
        "D1b": run_cell(targets, d1b, returns, cost=cost),
        "R1": run_cell(targets, post, returns, maps["R1"], cost=cost),
        "F": run_cell(targets, d1, returns, maps["F"], cost=cost),
    }
    cells = {name: r["portfolio_return"].reindex(score).dropna() for name, r in runs.items()}

    expected = summary["cells"][str(PRIMARY_BP)]
    for name, r in cells.items():
        got = annual_sharpe(r, rf)
        if abs(got - expected[name]["sharpe_ratio"]) > 1e-3:
            raise RuntimeError(f"{name}: rebuilt Sharpe {got:.4f} does not match summary "
                               f"{expected[name]['sharpe_ratio']:.4f}; results are not reproducible")
    return cells, argmax_labels(post), argmax_labels(d1)


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_decomposition(summary: dict, path: str) -> None:
    """Sharpe of every cell at the primary cost; oracles in gray, hatched."""
    import matplotlib.pyplot as plt

    cells = summary["cells"][str(PRIMARY_BP)]
    order = ["C1", "C0", "H4", "D1", "R1", "F", "D1b"]
    values = [cells[c]["sharpe_ratio"] for c in order]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for i, (c, v) in enumerate(zip(order, values)):
        oracle = c not in INVESTABLE
        ax.barh(i, v, height=0.55, color=GRAY if oracle else BLUE,
                hatch="//" if oracle else None, edgecolor="white", linewidth=0)
        # Labels sit beyond the bar end, on whichever side of zero the bar is.
        ax.text(v + (0.01 if v >= 0 else -0.01), i, f"{v:.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=9, color=INK)
    ax.axvline(cells["C1"]["sharpe_ratio"], color=INK_2, linewidth=1, linestyle=":")
    ax.axvline(0, color=GRID, linewidth=1)
    ax.set_yticks(range(len(order)), [CELL_NAMES[c] for c in order], color=INK)
    ax.invert_yaxis()
    span = max(max(values), 0) - min(min(values), 0)
    ax.set_xlim(min(min(values), 0) - 0.15 * span * (min(values) < 0), max(max(values), 0) + 0.15 * span)
    period = f"{summary['score_start'][:4]}-{summary['score_end'][:4]}"
    ax.set_xlabel(f"Sharpe ratio, {period}, {PRIMARY_BP} bp costs", color=INK_2, fontsize=9)
    ax.set_title("Hatched gray bars use future data and are not investable",
                 loc="left", fontsize=9, color=INK_2)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_placebo(summary: dict, path: str) -> None:
    """Placebo Sharpe distribution with the strategy and the static blend marked."""
    import matplotlib.pyplot as plt

    placebo = summary["placebo_sharpes"][str(PRIMARY_BP)]
    cells = summary["cells"][str(PRIMARY_BP)]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.hist(placebo, bins=25, color=GRAY, edgecolor="white", linewidth=1)
    top = ax.get_ylim()[1]
    # The two marked values can be almost equal: the lower one is labeled on its left,
    # the higher one on its right, so the labels never cross the other line.
    marks = (("C1", ORANGE, "--", 0.95), ("C0", BLUE, "-", 0.80))
    lower = min(("C1", "C0"), key=lambda n: cells[n]["sharpe_ratio"])
    for name, color, style, height in marks:
        align, nudge = ("right", -0.006) if name == lower else ("left", 0.006)
        v = cells[name]["sharpe_ratio"]
        ax.axvline(v, color=color, linewidth=2, linestyle=style)
        ax.text(v + nudge, top * height, f"{CELL_NAMES[name]} {v:.2f}", color=INK,
                fontsize=9, ha=align)
    p = summary["placebo"][str(PRIMARY_BP)]["p_value_vs_C0"]
    ax.set_xlabel("Sharpe ratio of strategies with shifted regime timing", color=INK_2, fontsize=9)
    ax.set_ylabel("Number of shifts", color=INK_2, fontsize=9)
    ax.set_title(f"Placebo: {len(placebo)} circular shifts, p = {p:.2f}", loc="left", fontsize=9, color=INK_2)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_wealth(cells: dict[str, pd.Series], path: str, window: str = "dev") -> None:
    """Growth of 1 for the strategy, the static blend and the cash rule, direct-labeled."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ends = {}
    for name, color in (("C0", BLUE), ("C1", ORANGE), ("H4", AQUA)):
        wealth = (1 + cells[name]).cumprod()
        ax.plot(wealth.index, wealth.values, color=color, linewidth=1.6, label=CELL_NAMES[name])
        ends[name] = (wealth.index[-1], wealth.iloc[-1])
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
    # Spread end labels so they never overlap when the curves finish close together.
    low, high = ax.get_ylim()
    gap = (high - low) * 0.07
    placed = []
    for name, (x, y) in sorted(ends.items(), key=lambda kv: kv[1][1]):
        label_y = max(y, placed[-1] + gap) if placed else y
        placed.append(label_y)
        ax.annotate(f" {CELL_NAMES[name]}", (x, y), xytext=(x, label_y), va="center",
                    fontsize=9, color=INK, annotation_clip=False)
    start, end, title = STRESS[window]
    ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=GRID, alpha=0.6, linewidth=0)
    ax.set_ylabel("Growth of 1", color=INK_2, fontsize=9)
    ax.set_title(title, loc="left", fontsize=9, color=INK_2)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    first = min(c.index[0] for c in cells.values())
    last = max(c.index[-1] for c in cells.values())
    ax.set_xlim(first, last)
    fig.tight_layout()
    fig.subplots_adjust(right=0.72)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_confusion(table: pd.DataFrame, path: str) -> None:
    """Row-normalized confusion of hindsight vs causal regimes, single-hue scale."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", BLUE])
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.imshow(table.fillna(0).values, cmap=cmap, vmin=0, vmax=100)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.iat[i, j]
            ax.text(j, i, "" if np.isnan(v) else f"{v:.0f}%", ha="center", va="center",
                    fontsize=9, color="white" if v > 60 else INK)
    ax.set_xticks(range(4), table.columns, color=INK, fontsize=9)
    ax.set_yticks(range(4), table.index, color=INK, fontsize=9)
    ax.set_xlabel("Causal regime (strategy)", color=INK_2, fontsize=9)
    ax.set_ylabel("Hindsight regime (D1)", color=INK_2, fontsize=9)
    ax.set_title("Share of each hindsight regime's days, by causal label", loc="left", fontsize=9, color=INK_2)
    for side in ax.spines.values():
        side.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(window: str = "dev") -> None:
    if window not in WINDOW_END:
        raise ValueError(f"window must be one of {list(WINDOW_END)}")
    if bool(CFG.get("universe")) != (window == "backcast"):
        raise SystemExit(f"the active config does not match the {window} window")
    out = os.path.join(OUT_DIR, window)
    with open(os.path.join(out, "summary.json")) as f:
        summary = json.load(f)
    if summary["window"] != window:
        raise SystemExit(f"{out}/summary.json is for window {summary['window']}, not {window}")
    figures = os.path.join(out, "figures")
    os.makedirs(figures, exist_ok=True)

    cells, real, hind = rebuild_cells(summary, window)
    table = confusion(real, hind)
    delays = detection_delays(real, hind)
    diagnostics = {"agreement_rate": agreement_rate(real, hind), **delay_summary(delays)}

    table.round(1).to_csv(os.path.join(out, "detection_confusion.csv"))
    delays.to_csv(os.path.join(out, "detection_delays.csv"))
    with open(os.path.join(out, "detection_diagnostics.json"), "w") as f:
        json.dump(diagnostics, f, indent=2)
    pd.DataFrame(cells).to_parquet(os.path.join(out, "daily_returns.parquet"))

    plot_decomposition(summary, os.path.join(figures, "decomposition.png"))
    plot_placebo(summary, os.path.join(figures, "placebo.png"))
    plot_wealth(cells, os.path.join(figures, "wealth.png"), window)
    plot_confusion(table, os.path.join(figures, "detection_confusion.png"))
    print(json.dumps(diagnostics, indent=2))
    print(f"Figures written to {figures}")


if __name__ == "__main__":
    main()
