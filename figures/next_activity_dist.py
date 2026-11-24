#!/usr/bin/env python3
"""Results (2), next-activity prediction: one plot, all logs merged, two lines (argmax, hard constraints).

x = distance of the predicted event from the end of the trace (n - k; 1 = the last event), distances >= CAP pooled;
y = accuracy over all prefixes of ALL logs with that distance (pooled per run index, then mean +- std over the 10 runs;
large logs weigh more: e.g. the >= 10 bucket is 94% BPI-2012). Grey dashed = argmax, purple solid = hard constraints
(C_REF / C_NEW of results_rq.py). Writes next_activity_dist.{pdf,png} and .csv (also the per-log prefix counts).

Run from anywhere:  python figures/next_activity_dist.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from results_rq import AEGIS, LOGS, INK, GRID, C_REF, C_NEW  # noqa: E402  (same style)

RES = AEGIS / "results_reference/next_activity"
CAP = 10
# Small figure with large text (user, 2026-11-18): meant to be shrunk with \scalebox / width=... and stay readable.
FIGSIZE = (3.0, 1.7)   # in
LABEL_FS, TICK_FS, LEGEND_FS = 9, 8.5, 8.5


def load():
    P = pd.concat([pd.read_csv(RES / f"{key}_CZ_AT_only/next_act_predictions_sl.csv").assign(log=name) for key, name in LOGS])
    P["dist"] = (P.n - P.k).clip(upper=CAP)
    per_run = P.groupby(["dist", "run"]).agg(argmax=("ok_raw", "mean"), hard=("ok_hard", "mean"), n=("ok_raw", "size"))
    g = per_run.groupby("dist").agg(argmax=("argmax", "mean"), hard=("hard", "mean"), argmax_std=("argmax", "std"),
                                    hard_std=("hard", "std"), n_per_run=("n", "mean"))
    g[["argmax", "hard", "argmax_std", "hard_std"]] *= 100
    counts = P[P.run == 0].groupby(["dist", "log"]).size().unstack(fill_value=0)
    return g, counts


def main():
    g, counts = load()
    g.join(counts).to_csv(HERE / "next_activity_dist.csv", float_format="%.4g")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for col, c, ls, mfc, lab in (("argmax", C_REF, "--", "white", "Argmax"), ("hard", C_NEW, "-", C_NEW, "Hard constraints")):
        ax.fill_between(g.index, g[col] - g[col + "_std"], g[col] + g[col + "_std"], color=c, alpha=0.15, lw=0)
        ax.plot(g.index, g[col], ls=ls, lw=1.3, color=c, marker="o", ms=3.5, mfc=mfc, mec=c, mew=1.0, label=lab)
    ax.set_xlim(0.7, CAP + 0.3); ax.set_ylim(0, 102)
    ax.set_xticks(range(1, CAP + 1), [str(i) for i in range(1, CAP)] + [rf"$\geq${CAP}"])
    ax.set_yticks(range(0, 101, 20))
    ax.set_xlabel("Distance from the end of the trace", fontsize=LABEL_FS, color=INK, labelpad=2)
    ax.set_ylabel("Accuracy (%)", fontsize=LABEL_FS, color=INK, labelpad=2)
    ax.tick_params(length=2.5, pad=2, labelsize=TICK_FS)
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.55, 1.04), handlelength=2.2, fontsize=LEGEND_FS, borderaxespad=0.2)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.97, bottom=0.24)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"next_activity_dist.{ext}", dpi=300)
    pd.set_option("display.width", 250)
    print(g.round(1).to_string()); print(counts.to_string())


if __name__ == "__main__":
    main()
