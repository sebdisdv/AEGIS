#!/usr/bin/env python3
"""Summary figures and tables of the Results section (RQ1, RQ2).

RQ1  rq1_summary.{pdf,png}: scatter, x = baseline SANA (BASELINE.json), y = SANA_AG (argmax, trained with the semantic
     loss); accuracy (left) and MAE (right, log-log); colour = log, marker = test set. Hollow = not significant
     (two-sided Welch t-test, 10 vs 10 runs, Holm over the 5 test sets of one (log, metric), p < 0.05; tests imported
     from plot_sl_hard.py).
RQ2  rq2_summary.{pdf,png}: scatter, x = accuracy with the argmax, y = accuracy with the hard constraints (same models).
Tables: results_tables.tex (means over the 5 test sets), printed too.

Run from anywhere:  python figures/results_rq.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
AEGIS = HERE.parent
sys.path.insert(0, str(AEGIS))
from plot_sl_hard import significance  # noqa: E402  (Welch + Holm, same as the paper text)

RES = AEGIS / "results_reference/repair"
BASE = json.load(open(AEGIS / "BASELINE.json"))
TESTS = ["even", "odd", "window", "random", "attr_level"]
LOGS = [("small_log", "Small Log"), ("large_log", "Large Log"), ("bpi_2012", "BPI-2012"), ("bpi_2013", "BPI-2013"),
        ("BPI20_RequestForPayment", "BPI-2020-RFP"), ("sp2020", "SP-2020")]          # order of tab:datasets
N_BASE, ALPHA = 10, 0.05

# Okabe-Ito (CVD-safe, as in encoding_AT.py) + one marker shape per test set (secondary encoding for print / CVD)
COL = dict(even="#0072B2", odd="#E69F00", window="#009E73", random="#D55E00", attr_level="#CC79A7")
MRK = dict(even="o", odd="s", window="^", random="D", attr_level="v")
INK, INK2, GRID = "#222222", "#555555", "#dddddd"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
                     "mathtext.fontset": "custom", "mathtext.rm": "Nimbus Roman", "mathtext.it": "Nimbus Roman:italic",
                     "mathtext.bf": "Nimbus Roman:bold", "mathtext.sf": "Nimbus Sans", "font.size": 7.5, "axes.edgecolor": "#999999",
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK, "pdf.fonttype": 42})
TEXTWIDTH = 4.80                                                                      # LNCS text width (12.2 cm)


def load():
    rows = []
    for key, name in LOGS:
        d, b = RES / f"{key}_CZ_AT_only", BASE[f"{key}_CZ_AT_only"]
        runs = {t: pd.read_csv(d / f"{t}_RESULTS_sl.csv") for t in TESTS}
        ms = lambda col, f=1.0: ([runs[t][col].mean() * f for t in TESTS], [runs[t][col].std(ddof=1) * f for t in TESTS])
        bs = lambda k: ([b[t][k]["mean"] for t in TESTS], [b[t][k]["std"] for t in TESTS])
        n = [len(runs[t]) for t in TESTS]
        sig_acc = significance(ms("acc_sl"), bs("Activity_acc"), n, N_BASE)
        sig_mae = significance(ms("time:timestamp_mae"), bs("time:timestamp_mae"), n, N_BASE)
        for i, t in enumerate(TESTS):
            r = runs[t]; g = (r.acc_sl_hard - r.acc_sl) * 100
            rows.append(dict(
                log=name, test=t, runs=len(r), n_missing=int(r.n_missing.iloc[0]),
                acc_base=b[t]["Activity_acc"]["mean"] * 100, acc_sl=r.acc_sl.mean() * 100,
                acc_hard=r.acc_sl_hard.mean() * 100, gain_hard=g.mean(), gain_hard_std=g.std(ddof=1),
                mae_base=b[t]["time:timestamp_mae"]["mean"], mae_sl=r["time:timestamp_mae"].mean(),
                p_acc=sig_acc[i]["p_holm"], p_mae=sig_mae[i]["p_holm"],
                viol_sl=(1 - r.sat_sl.mean()) * 100, viol_hard=(1 - r.sat_sl_hard.mean()) * 100,
                fallback=r.fallback.mean() * 100))
    D = pd.DataFrame(rows)
    D["d_acc"] = D.acc_sl - D.acc_base
    D["mae_red"] = (D.mae_base - D.mae_sl) / D.mae_base * 100
    D["sig_acc_better"] = (D.p_acc < ALPHA) & (D.d_acc > 0)
    D["sig_mae_better"] = (D.p_mae < ALPHA) & (D.mae_red > 0)
    D["sig_acc"] = D.p_acc < ALPHA
    D["sig_mae"] = D.p_mae < ALPHA
    return D


LOG_COL = {"Small Log": "#0072B2", "Large Log": "#E69F00", "BPI-2012": "#009E73", "BPI-2013": "#D55E00",
           "BPI-2020-RFP": "#CC79A7", "SP-2020": "#000000"}                       # Okabe-Ito, one colour per log
SANA, SANA_AG = r"$\mathsf{SANA}$", r"$\mathsf{SANA_{AEGIS}}$"         # \SANAGRAPH (baseline), \SANAA (our model)
C_REF, C_NEW = "#6b6b6b", "#6a3d9a"          # name colours: reference model (grey), new model (purple); not log colours


def scatter(ax, D, xcol, ycol, sig=None, log=False, lo=None, hi=None, better="above", label=None,
            title=None, xname=None, yname=None, ms=3.0, label_fs=7):
    """x = reference, y = new model; colour = log, marker = test set, hollow = not significant (if sig given)."""
    for _, r in D.iterrows():
        filled = True if sig is None else bool(r[sig])
        c = LOG_COL[r.log]
        ax.plot(r[xcol], r[ycol], MRK[r.test], ms=ms, mec=c, mew=0.6, mfc=c if filled else "white", alpha=0.9, zorder=3)
    ax.plot([lo, hi], [lo, hi], ls="--", color=INK2, lw=0.7, zorder=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    if log:
        ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.tick_params(length=2, pad=2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if label:
        x, y, ha, va = (0.04, 0.96, "left", "top") if better == "above" else (0.96, 0.04, "right", "bottom")
        ax.text(x, y, label, transform=ax.transAxes, ha=ha, va=va, color=C_NEW, fontsize=label_fs)
    if title:
        ax.set_title(title, fontsize=8, color=INK, pad=4)
    ax.set_xlabel(xname, color=C_REF, fontsize=8.5, labelpad=2)
    ax.set_ylabel(yname, color=C_NEW, fontsize=8.5, labelpad=2)


def legend_handles(with_sig):
    logs = [Line2D([], [], ls="none", marker="s", ms=5, mec=c, mfc=c, label=n) for n, c in LOG_COL.items()]
    tests = [Line2D([], [], ls="none", marker=MRK[t], ms=3.6, mec=INK2, mfc=INK2, mew=0.9, label=t) for t in TESTS]
    if with_sig:
        tests += [Line2D([], [], ls="none", marker="o", ms=3.6, mec=INK2, mfc="white", mew=0.9, label="not significant")]
    return logs, tests


def fig_rq1(D):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.8), gridspec_kw=dict(wspace=0.38))
    names = dict(xname=SANA, yname=SANA_AG, label=SANA_AG + " better")
    scatter(a1, D, "acc_base", "acc_sl", sig="sig_acc", lo=50, hi=101, better="above", title="Activity accuracy", **names)
    scatter(a2, D, "mae_base", "mae_sl", sig="sig_mae", log=True, lo=0.0018, hi=0.45, better="below",
            title="Timestamp MAE", **names)
    a1.set_xticks(range(50, 101, 10)); a1.set_yticks(range(50, 101, 10))
    logs, tests = legend_handles(True)
    kw = dict(frameon=False, handletextpad=0.05, columnspacing=0.8, loc="lower center")
    fig.legend(handles=logs, ncol=6, bbox_to_anchor=(0.5, 0.07), **kw)
    fig.legend(handles=tests, ncol=6, bbox_to_anchor=(0.5, 0.0), **kw)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=0.25)
    return fig


def fig_rq2(D):
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.35))
    scatter(ax, D, "acc_sl", "acc_hard", lo=62, hi=101, better="above", label="hard constraints better",
            title="Activity accuracy of " + SANA_AG, xname="argmax", yname="hard constraints")
    logs, tests = legend_handles(False)
    kw = dict(frameon=False, handletextpad=0.2, loc="center left", borderaxespad=0)
    fig.legend(handles=logs, title="Log", bbox_to_anchor=(0.53, 0.55), alignment="left", **kw)
    fig.legend(handles=tests, title="Test set", bbox_to_anchor=(0.76, 0.55), alignment="left", **kw)
    fig.subplots_adjust(left=0.1, right=0.5, top=0.92, bottom=0.16)
    return fig


MS_SUM, LABEL_FS_SUM, AXIS_FS_SUM, TICK_FS_SUM = 1.7, 4, 5.5, 5.5                                                        # smaller markers in the 3-panel figure


def fig_summary(D):
    """One figure for both RQs: (a) RQ1 accuracy, (b) RQ1 MAE, (c) RQ2 accuracy; one shared legend below."""
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.1), gridspec_kw=dict(wspace=0.5))
    names = dict(xname=SANA, yname=SANA_AG, label=SANA_AG + " better")
    scatter(a1, D, "acc_base", "acc_sl", sig="sig_acc", lo=50, hi=101, better="above",
            title="(a) Activity accuracy", ms=MS_SUM, label_fs=LABEL_FS_SUM, **names)
    scatter(a2, D, "mae_base", "mae_sl", sig="sig_mae", log=True, lo=1e-3, hi=1, better="below",
            title="(b) Timestamp MAE", ms=MS_SUM, label_fs=LABEL_FS_SUM, **names)
    mae_ticks = [1e-3, 1e-2, 1e-1, 1]
    mae_labels = [r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", "1"]                     # plain 1 (user)
    a2.set_xticks(mae_ticks, mae_labels); a2.set_yticks(mae_ticks, mae_labels)
    scatter(a3, D, "acc_sl", "acc_hard", lo=50, hi=101, better="above", label=None,
            title="(c) Act. acc w/constraints", xname="Argmax", yname="Hard constraints", ms=MS_SUM, label_fs=LABEL_FS_SUM)
    for a in (a1, a2, a3):                                              # smaller axis labels than the 2-panel figures
        a.xaxis.label.set_size(AXIS_FS_SUM); a.yaxis.label.set_size(AXIS_FS_SUM)
        a.tick_params(which="both", labelsize=TICK_FS_SUM)
    for a in (a1, a3):                                                  # same accuracy scale in (a) and (c)
        a.set_xticks(range(50, 101, 10)); a.set_yticks(range(50, 101, 10))
    logs, tests = legend_handles(False)
    tests += [Line2D([], [], ls="none", marker="o", ms=3.6, mec=INK2, mfc="white", mew=0.9,
                     label="not significant (a, b)")]
    kw = dict(frameon=False, handletextpad=0.05, columnspacing=0.8, loc="lower center")
    fig.legend(handles=logs, ncol=6, bbox_to_anchor=(0.5, 0.07), **kw)
    fig.legend(handles=tests, ncol=6, bbox_to_anchor=(0.5, 0.0), **kw)
    fig.subplots_adjust(left=0.085, right=0.965, top=0.915, bottom=0.33)
    return fig


def table_merged(D):
    """One row per log: accuracy (baseline, SANA_AG argmax, SANA_AG + hard), MAE x1e-2 (baseline, SANA_AG),
    violated (trace, rule) pairs % (argmax, hard); means over the 5 test sets; best accuracy / MAE in bold."""
    m = D.groupby("log", sort=False)[["acc_base", "acc_sl", "acc_hard", "mae_base", "mae_sl", "viol_sl",
                                      "viol_hard"]].mean()
    bold = lambda v, best: rf"\textbf{{{v}}}" if best else v
    out = []
    for name, r in m.iterrows():
        accs = [r.acc_base, r.acc_sl, r.acc_hard]
        a = " & ".join(bold(f"{v:.1f}", v == max(accs)) for v in accs)
        e = " & ".join(bold(f"{v * 100:.2f}", v == min(r.mae_base, r.mae_sl)) for v in (r.mae_base, r.mae_sl))
        out.append(f"{name:<13}& {a} & {e} & {r.viol_sl:.2f} & {r.viol_hard:.2f} \\\\")
    return "\n".join(out)


def tables(D):
    g = D.groupby("log", sort=False)
    m = g[["acc_base", "acc_sl", "mae_base", "mae_sl", "acc_hard", "gain_hard", "viol_sl", "viol_hard",
           "fallback"]].mean()
    k = g[["sig_acc_better", "sig_mae_better"]].sum()
    worse = g[["p_acc", "d_acc", "p_mae", "mae_red"]].apply(lambda x: int(((x.p_acc < ALPHA) & (x.d_acc < 0)).sum() + ((x.p_mae < ALPHA) & (x.mae_red < 0)).sum()))
    bold = lambda v, best: rf"\textbf{{{v}}}" if best else v
    l1 = []
    for name in m.index:
        r = m.loc[name]
        ab, aa = f"{r.acc_base:.1f}", f"{r.acc_sl:.1f}"
        mb, ma = f"{r.mae_base * 100:.2f}", f"{r.mae_sl * 100:.2f}"
        l1.append(f"{name:<13}& {bold(ab, r.acc_base > r.acc_sl)} & {bold(aa, r.acc_sl >= r.acc_base)} & "
                  f"{int(k.loc[name].sig_acc_better)}/5 & {bold(mb, r.mae_base < r.mae_sl)} & "
                  f"{bold(ma, r.mae_sl <= r.mae_base)} & {int(k.loc[name].sig_mae_better)}/5 \\\\")
    l2 = []
    for name in m.index:
        r = m.loc[name]
        l2.append(f"{name:<13}& {r.acc_sl:.1f} & {bold(f'{r.acc_hard:.1f}', True)} & {r.gain_hard:+.1f} & "
                  f"{r.viol_sl:.2f} & {r.viol_hard:.2f} & {r.fallback:.1f} \\\\")
    return "\n".join(l1), "\n".join(l2), worse


def main():
    D = load()
    for fn, name in ((fig_rq1, "rq1_summary"), (fig_rq2, "rq2_summary"), (fig_summary, "rq_summary")):
        fig = fn(D)
        for ext in ("pdf", "png"):
            fig.savefig(HERE / f"{name}.{ext}", dpi=300)
        plt.close(fig)
    t1, t2, worse = tables(D)
    tm = table_merged(D)
    (HERE / "results_tables.tex").write_text("% merged RQ1 + RQ2 rows (tab:rq)\n" + tm + "\n\n% RQ1 rows (old tab:rq1)\n"
                                             + t1 + "\n\n% RQ2 rows (old tab:rq2)\n" + t2 + "\n")
    print(tm + "\n")
    D.to_csv(HERE / "results_rq.csv", index=False, float_format="%.6g")
    pd.set_option("display.width", 250)
    print(D[["log", "test", "d_acc", "p_acc", "mae_red", "p_mae", "gain_hard", "gain_hard_std"]].round(4).to_string())
    print("\nsignificantly worse cells per log:\n", worse.to_string())
    print("\n" + t1 + "\n\n" + t2)


if __name__ == "__main__":
    main()
