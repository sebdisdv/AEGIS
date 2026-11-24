#!/usr/bin/env python3
"""
Two figures per dataset, every test type (even, odd, window, ...), network trained with the semantic loss (tag sl),
against the baseline in BASELINE.json:
    <dataset>_activity_sl_hard_<tag>   accuracy on the masked activities: argmax + SL, argmax + SL + hard constraints, baseline
    <dataset>_timestamp_mae_<tag>      MAE on the masked timestamps: HGNN + SL, baseline (lower is better). The hard
                                       constraints change only the activity labels, so the timestamps are the same with them.

Reads <results_dir>/<dataset>/<test_type>_RESULTS_<tag>.csv (one row per run) and writes to <out_dir> the figures
(png, pdf; suffix _dark with --dark) and a CSV with every value behind each figure.

Significance of argmax + SL vs the baseline (accuracy and MAE): Welch two-sample t-test from summary statistics (the
baseline has only mean and std, over --baseline-runs runs; its std is taken as a sample std), two-sided, Holm-corrected
over the test types of one (dataset, metric). Written to <dataset>_significance_<tag>.csv and significance_<tag>.csv
(all datasets), and shown as stars on the figures (*** p<0.001, ** p<0.01, * p<0.05, ns). The variance is over training
runs on a fixed test set, not test-set sampling variance.

Usage:
    python plot_sl_hard.py
    python plot_sl_hard.py --dark --datasets bpi_2012_CZ_AT_only --formats png
    python plot_sl_hard.py --results-dir results/repair --tag sl     # your own runs
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
TEST_ORDER = ["even", "odd", "window", "random", "attr_level"]

# same palette and slots as plot_next_activity.py (Raw slot 0, Hard slot 3); baseline slot 4
THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", text2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                  series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]),
    "dark": dict(surface="#1a1a19", text="#f0efec", text2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
                 series=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"]),
}
RAW, HARD, BASE = "Argmax + semantic loss", "Argmax + semantic loss + hard constraints", "Baseline"
TS_MODEL = "HGNN + semantic loss (same with hard constraints)"
SLOT = {RAW: 0, HARD: 3, BASE: 4, TS_MODEL: 0}
NICE = {"bpi_2012": "BPI 2012", "bpi_2013": "BPI 2013", "small_log": "Small log", "large_log": "Large log",
        "sp2020": "SP 2020", "BPI20_RequestForPayment": "BPI20 RfP"}


def nice_name(folder):
    for k, v in NICE.items():
        if folder.startswith(k):
            return v
    return folder


def load_dataset(ds_dir: Path, tag: str, baseline: dict):
    """Returns (test_types, act, ts, runs, n_runs) or None; n_runs = runs per test type. act / ts = {label: (mean[list], std[list])};
    act also has 'gain' = paired hard - raw per run (mean, std), ts has 'diff' = model mean - baseline mean."""
    files = {t: ds_dir / f"{t}_RESULTS_{tag}.csv" for t in TEST_ORDER}
    tts = [t for t, f in files.items() if f.exists()]
    if not tts:
        return None
    tab = {t: pd.read_csv(files[t]) for t in tts}
    ms = lambda col: ([tab[t][col].mean() for t in tts], [tab[t][col].std(ddof=1) for t in tts])
    base = lambda key: ([baseline.get(t, {}).get(key, {}).get("mean", np.nan) for t in tts],
                        [baseline.get(t, {}).get(key, {}).get("std", np.nan) for t in tts])
    diffs = [tab[t]["acc_sl_hard"] - tab[t]["acc_sl"] for t in tts]        # same models, paired by run
    act = {RAW: ms("acc_sl"), HARD: ms("acc_sl_hard"), BASE: base("Activity_acc"),
           "gain": ([d.mean() for d in diffs], [d.std(ddof=1) for d in diffs])}
    ts = {TS_MODEL: ms("time:timestamp_mae"), BASE: base("time:timestamp_mae")}
    n_runs = [len(tab[t]) for t in tts]
    return tts, act, ts, int(min(n_runs)), n_runs


def holm(p):
    """Holm step-down adjusted p values; NaN entries are left out of the family."""
    p = np.asarray(p, dtype=float); out = np.full_like(p, np.nan)
    idx = np.where(np.isfinite(p))[0]
    order = idx[np.argsort(p[idx])]; m = len(order); run = 0.0
    for i, j in enumerate(order):
        run = max(run, (m - i) * p[j]); out[j] = min(run, 1.0)
    return out


def stars(p):
    return "" if not np.isfinite(p) else "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


def significance(model, base, n_model, n_base):
    """Welch t-test per test type of model (mean[list], std[list]) vs base (mean[list], std[list]), Holm over types."""
    rows = []
    for m1, s1, n1, m2, s2 in zip(*model, n_model, *base):
        if not (np.isfinite(m2) and np.isfinite(s2)):
            rows.append(dict(t=np.nan, df=np.nan, p=np.nan, hedges_g=np.nan)); continue
        if s1 == 0 and s2 == 0 and m1 == m2:                           # identical constants (e.g. both 100%): no difference
            rows.append(dict(t=0.0, df=np.nan, p=1.0, hedges_g=0.0)); continue
        t, p = stats.ttest_ind_from_stats(m1, s1, n1, m2, s2, n_base, equal_var=False)
        v1, v2 = s1 ** 2 / n1, s2 ** 2 / n_base
        df = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n_base - 1))    # Welch-Satterthwaite
        sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n_base - 1) * s2 ** 2) / (n1 + n_base - 2))
        g = (m1 - m2) / sp * (1 - 3 / (4 * (n1 + n_base) - 9)) if sp > 0 else np.nan
        rows.append(dict(t=t, df=df, p=p, hedges_g=g))
    p_holm = holm([r["p"] for r in rows])
    for r, ph in zip(rows, p_holm):
        r.update(p_holm=ph, stars=stars(ph))
    return rows


def write_significance(name, tag, tts, act, ts, n_runs, n_base, alpha, out_dir: Path):
    """Per dataset CSV of the tests; returns (path, DataFrame, {metric: stars per test type})."""
    rows, st = [], {}
    for metric, model, base in (("activity_acc", act[RAW], act[BASE]), ("timestamp_mae", ts[TS_MODEL], ts[BASE])):
        res = significance(model, base, n_runs, n_base)
        st[metric] = [r["stars"] for r in res]
        for i, (t, r) in enumerate(zip(tts, res)):
            rows.append(dict(dataset=name, metric=metric, test_type=t, model_mean=model[0][i], model_std=model[1][i],
                             n_model=n_runs[i], base_mean=base[0][i], base_std=base[1][i], n_base=n_base,
                             diff=model[0][i] - base[0][i], **{k: r[k] for k in ("t", "df", "p", "p_holm", "hedges_g")},
                             significant=bool(r["p_holm"] < alpha) if np.isfinite(r["p_holm"]) else np.nan,
                             stars=r["stars"]))
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}_significance_{tag}.csv"
    df.to_csv(path, index=False, float_format="%.6g")
    return path, df, st


def write_tables(name, tag, tts, act, ts, out_dir: Path):
    rows = []
    for lab in (RAW, HARD, BASE):
        for i, t in enumerate(tts):
            rows.append({"test_type": t, "series": lab, "acc_mean_pct": act[lab][0][i] * 100,
                         "acc_std_pct": act[lab][1][i] * 100,
                         "gain_hard_over_raw_pts": act["gain"][0][i] * 100 if lab == HARD else np.nan,
                         "gain_paired_std_pts": act["gain"][1][i] * 100 if lab == HARD else np.nan,
                         "diff_vs_baseline_pts": (act[lab][0][i] - act[BASE][0][i]) * 100 if lab != BASE else np.nan})
    out_dir.mkdir(parents=True, exist_ok=True)
    p1 = out_dir / f"{name}_activity_sl_hard_{tag}.csv"
    pd.DataFrame(rows).to_csv(p1, index=False, float_format="%.4f")
    rows = []
    for lab in (TS_MODEL, BASE):
        for i, t in enumerate(tts):
            rows.append({"test_type": t, "series": lab, "mae_mean": ts[lab][0][i], "mae_std": ts[lab][1][i],
                         "diff_vs_baseline": ts[lab][0][i] - ts[BASE][0][i] if lab != BASE else np.nan})
    p2 = out_dir / f"{name}_timestamp_mae_{tag}.csv"
    pd.DataFrame(rows).to_csv(p2, index=False, float_format="%.6g")
    return p1, p2


def _bars(ax, th, x, labels, series):
    """Grouped bars with error bars; returns {label: (xs, top of the error bar)}."""
    n = len(labels); width = 0.8 / n; out = {}
    for k, lab in enumerate(labels):
        mean, std = (np.array(v, dtype=float) for v in series[lab])
        xs = x - 0.4 + width * (k + 0.5)
        ax.bar(xs, mean, width * 0.9, color=th["series"][SLOT[lab]], label=lab, zorder=3, linewidth=0)
        ax.errorbar(xs, mean, yerr=np.nan_to_num(std), fmt="none", ecolor=th["text2"], elinewidth=1, capsize=2.5,
                    capthick=1, zorder=4)
        out[lab] = (xs, mean + np.nan_to_num(std))
    return out


def _finish(fig, ax, th, tts, title, subtitle, ncol):
    ax.set_xticks(np.arange(len(tts)), [t.replace("_", " ") for t in tts], color=th["text"], fontsize=10)
    ax.set_xlim(-0.55, len(tts) - 0.45)
    ax.set_xlabel("Test set (type of missing data)", color=th["text2"], fontsize=10, labelpad=8)
    ax.yaxis.grid(True, color=th["grid"], linewidth=0.8, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(th["axis"]); ax.tick_params(axis="both", length=0, colors=th["text2"])
    fig.suptitle(title, x=0.05, y=0.985, ha="left", fontsize=14, fontweight="bold", color=th["text"])
    fig.text(0.05, 0.935, subtitle, ha="left", va="top", fontsize=9, color=th["text2"])
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=ncol, frameon=False, fontsize=9)
    for t in leg.get_texts(): t.set_color(th["text"])
    fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.2)


def _save(fig, out_dir, stem, theme, formats, th):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        p = out_dir / f"{stem}{'_dark' if theme == 'dark' else ''}.{fmt}"
        fig.savefig(p, dpi=200, facecolor=th["surface"]); paths.append(p)
    plt.close(fig)
    return paths


def _avg_lines(ax, th, avg, gap, fmt):
    """Dashed line at the average of each series over the test sets, labelled at the right edge (labels pushed
    at least `gap` apart in data units)."""
    for lab, v in avg.items():
        ax.axhline(v, color=th["series"][SLOT[lab]], linestyle=(0, (5, 3)), linewidth=1.3, zorder=2.5)
    ys = {}
    for lab in sorted(avg, key=avg.get):
        ys[lab] = max(avg[lab], max(ys.values(), default=-np.inf) + gap)
    for lab, y in ys.items():
        ax.text(1.005, y, "avg " + fmt.format(avg[lab]), transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=8, color=th["series"][SLOT[lab]], fontweight="bold")


def plot_activity(name, tag, tts, act, runs, out_dir, theme, formats, sig):
    th = THEMES[theme]
    fig, ax = plt.subplots(figsize=(12, 5.2), facecolor=th["surface"]); ax.set_facecolor(th["surface"])
    pct = {lab: (np.array(act[lab][0]) * 100, np.array(act[lab][1]) * 100) for lab in (RAW, HARD, BASE)}
    pos = _bars(ax, th, np.arange(len(tts)), [RAW, HARD, BASE], pct)
    xs, top = pos[HARD]                                            # gain of hard over raw, muted when within noise
    for xi, t, g, s in zip(xs, top, *act["gain"]):
        ax.text(xi, t + 0.8, f"{round(g * 100, 1) + 0.0:+.1f}", ha="center", va="bottom", fontsize=7.5,
                color=th["muted"] if abs(g) <= s else th["text"], zorder=5,
                bbox=dict(facecolor=th["surface"], edgecolor="none", pad=0.6))  # readable over the average lines
    xs, top = pos[RAW]                                             # argmax + SL vs baseline
    for xi, t, st in zip(xs, top, sig):
        if st:
            ax.text(xi, t + 0.8, st, ha="center", va="bottom", fontsize=7.5, zorder=5,
                    color=th["muted"] if st == "ns" else th["text"],
                    bbox=dict(facecolor=th["surface"], edgecolor="none", pad=0.6))
    ax.set_ylim(0, 108); ax.set_yticks(range(0, 101, 10))
    ax.set_yticklabels([f"{v}%" for v in range(0, 101, 10)], fontsize=9)
    _avg_lines(ax, th, {lab: float(np.nanmean(pct[lab][0])) for lab in (RAW, HARD, BASE)}, 3.6, "{:.1f}%")
    _finish(fig, ax, th, tts, f"{nice_name(name)}: activity repair",
            f"Accuracy on masked activities, mean ± std over {runs} runs (tag {tag}); dashed lines: average of each series "
            "over all test sets.\nNumbers: gain of hard constraints over argmax in percentage points, same models; "
            "grey = within noise (|gain| ≤ std of the paired differences)." + "\nStars: Welch t-test of argmax + semantic loss vs baseline, Holm-corrected over the test sets (*** p<0.001, ** p<0.01, * p<0.05, ns = not significant).", 3)
    fig.subplots_adjust(right=0.92)                                   # room for the average labels
    return _save(fig, out_dir, f"{name}_activity_sl_hard_{tag}", theme, formats, th)


def plot_timestamp(name, tag, tts, ts, runs, out_dir, theme, formats, sig):
    th = THEMES[theme]
    fig, ax = plt.subplots(figsize=(12, 5.2), facecolor=th["surface"]); ax.set_facecolor(th["surface"])
    pos = _bars(ax, th, np.arange(len(tts)), [TS_MODEL, BASE], ts)
    ymax = np.nanmax(np.concatenate([pos[TS_MODEL][1], pos[BASE][1]]))
    xs, top = pos[TS_MODEL]                                        # model - baseline, relative
    for xi, t, m, b, st in zip(xs, top, ts[TS_MODEL][0], ts[BASE][0], sig):
        if np.isfinite(b) and b > 0:
            ax.text(xi, t + ymax * 0.015, f"{(m - b) / b * 100:+.0f}% {st}".strip(), ha="center", va="bottom",
                    fontsize=7.5, color=th["muted"] if st == "ns" else th["text"], zorder=5,
                    bbox=dict(facecolor=th["surface"], edgecolor="none", pad=0.6))
    ax.set_ylim(0, ymax * 1.12)
    ax.tick_params(axis="y", labelsize=9)
    avg = {lab: float(np.nanmean(ts[lab][0])) for lab in (TS_MODEL, BASE) if np.isfinite(ts[lab][0]).any()}
    _avg_lines(ax, th, avg, ymax * 1.12 * 0.033, "{:.3g}")
    _finish(fig, ax, th, tts, f"{nice_name(name)}: timestamp reconstruction",
            f"MAE on masked timestamps (normalised scale, lower is better), mean ± std over {runs} runs (tag {tag}); "
            "dashed lines: average of each series over all test sets.\n"
            "Numbers: relative difference of the model MAE from the baseline MAE." + "\nStars: Welch t-test of argmax + semantic loss vs baseline, Holm-corrected over the test sets (*** p<0.001, ** p<0.01, * p<0.05, ns = not significant).", 2)
    fig.subplots_adjust(right=0.92)                                   # room for the average labels
    return _save(fig, out_dir, f"{name}_timestamp_mae_{tag}", theme, formats, th)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=str(HERE / "results_reference/repair"),
                    help="results of main.py (results/repair) or the published ones (default)")
    ap.add_argument("--out-dir", default=str(HERE / "plots/sl_hard"))
    ap.add_argument("--baseline-json", default=str(HERE / "BASELINE.json"))
    ap.add_argument("--tag", default="sl", help="tag of the semantic-loss models (main.py --tag)")
    ap.add_argument("--datasets", nargs="*", help="dataset folder names (default: all found)")
    ap.add_argument("--formats", nargs="*", default=["png", "pdf"])
    ap.add_argument("--dark", action="store_true", help="dark theme")
    ap.add_argument("--baseline-runs", type=int, default=10, help="runs behind each baseline mean and std")
    ap.add_argument("--alpha", type=float, default=0.05, help="threshold on the Holm-adjusted p for 'significant'")
    a = ap.parse_args()
    root, out_dir, theme = Path(a.results_dir), Path(a.out_dir), "dark" if a.dark else "light"
    baselines = json.load(open(a.baseline_json))
    dirs = [root / d for d in a.datasets] if a.datasets else sorted(p for p in root.iterdir() if p.is_dir())
    done, sig_all = 0, []
    for d in dirs:
        if d.name not in baselines:
            print(f"warning {d.name}: no baseline in {a.baseline_json}")
        data = load_dataset(d, a.tag, baselines.get(d.name, {})) if d.is_dir() else None
        if data is None:
            print(f"skip {d.name}: no *_RESULTS_{a.tag}.csv"); continue
        tts, act, ts, runs, n_runs = data
        for p in write_tables(d.name, a.tag, tts, act, ts, out_dir): print("wrote", p)
        p, df, sig = write_significance(d.name, a.tag, tts, act, ts, n_runs, a.baseline_runs, a.alpha, out_dir)
        print("wrote", p); sig_all.append(df)
        for p in plot_activity(d.name, a.tag, tts, act, runs, out_dir, theme, a.formats, sig["activity_acc"]):
            print("wrote", p)
        for p in plot_timestamp(d.name, a.tag, tts, ts, runs, out_dir, theme, a.formats, sig["timestamp_mae"]):
            print("wrote", p)
        done += 1
    if sig_all and not a.datasets:                                    # combined table only for a full run
        p = out_dir / f"significance_{a.tag}.csv"
        pd.concat(sig_all).to_csv(p, index=False, float_format="%.6g"); print("wrote", p)
    print(f"{done} dataset(s) in {out_dir}/")


if __name__ == "__main__":
    main()
