#!/usr/bin/env python3
"""
Plots for the next-activity test (next_activity.py): raw argmax vs live-state hard constraints.

Reads <results_dir>/<dataset>/next_act_{RESULTS,predictions}_<tag>.csv and writes to <out_dir>:
    summary_next_activity_<tag>[_dark].{png,pdf}   all datasets, accuracy mean ± std over runs, gain of hard over raw
    <dataset>_next_activity_<tag>[_dark].{png,pdf} one figure per dataset: accuracy, raw rule-violation rate and
                                                   number of prefixes, by prefix length k (lengths grouped
                                                   in buckets when there are more than --max-points of them)
    plus a CSV with every value behind each figure.

Usage:
    python plot_next_activity.py
    python plot_next_activity.py --dark --formats png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# same palette and slots as plot_sl_hard.py, so Raw and Hard keep their colours across figures
THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", text2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                  series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]),
    "dark": dict(surface="#1a1a19", text="#f0efec", text2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
                 series=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"]),
}
SERIES = {"Raw (argmax)": ("acc_raw", "ok_raw", 0), "Hard constraints": ("acc_hard", "ok_hard", 3)}
NICE = {"bpi_2012": "BPI 2012", "bpi_2013": "BPI 2013", "small_log": "Small log", "large_log": "Large log", "sp2020": "SP 2020",
        "BPI20_RequestForPayment": "BPI20 RfP"}


def nice_name(folder):
    for k, v in NICE.items():
        if folder.startswith(k):
            return v
    return folder


def style_axis(ax, th, pct=True):
    ax.set_facecolor(th["surface"])
    ax.yaxis.grid(True, color=th["grid"], linewidth=0.8, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(th["axis"]); ax.tick_params(axis="both", length=0, colors=th["text2"], labelsize=9)
    if pct:
        ax.set_ylim(0, 105); ax.set_yticks(range(0, 101, 20))
        ax.set_yticklabels([f"{v}%" for v in range(0, 101, 20)])


def save(fig, out_dir, stem, theme, formats, th):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        p = out_dir / f"{stem}{'_dark' if theme == 'dark' else ''}.{fmt}"
        fig.savefig(p, dpi=200, facecolor=th["surface"]); paths.append(p)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------- summary over datasets
def load_summary(dirs, tag):
    rows = []
    for d in dirs:
        f = d / f"next_act_RESULTS_{tag}.csv"
        if not f.exists():
            continue
        r = pd.read_csv(f)
        diff = r["acc_hard"] - r["acc_raw"]                            # same models, paired by run
        rows.append(dict(dataset=d.name, name=nice_name(d.name), runs=len(r), n_prefixes=int(r["n_prefixes"].iloc[0]),
                         raw_mean=r["acc_raw"].mean() * 100, raw_std=r["acc_raw"].std(ddof=1) * 100,
                         hard_mean=r["acc_hard"].mean() * 100, hard_std=r["acc_hard"].std(ddof=1) * 100,
                         gain_pts=diff.mean() * 100, gain_paired_std_pts=diff.std(ddof=1) * 100,
                         raw_violation_pct=r["raw_violation_rate"].mean() * 100,
                         truth_allowed_pct=r["truth_allowed"].mean() * 100, mean_allowed=r["mean_allowed"].mean(),
                         fallback_pct=r["fallback"].mean() * 100))
    return pd.DataFrame(rows)


def plot_summary(df, tag, out_dir, theme, formats):
    th = THEMES[theme]
    fig, ax = plt.subplots(figsize=(11, 5.2), facecolor=th["surface"])
    style_axis(ax, th)
    x = np.arange(len(df)); width = 0.36
    for k, (lab, (col, _, slot)) in enumerate(SERIES.items()):
        pre = col.replace("acc_", "")
        xs = x + (k - 0.5) * (width + 0.02)
        ax.bar(xs, df[f"{pre}_mean"], width, color=th["series"][slot], label=lab, zorder=3, linewidth=0)
        ax.errorbar(xs, df[f"{pre}_mean"], yerr=df[f"{pre}_std"], fmt="none", ecolor=th["text2"], elinewidth=1,
                    capsize=2.5, capthick=1, zorder=4)
        if lab == "Hard constraints":                                  # gain over raw, muted when within noise
            for xi, top, g, s in zip(xs, df["hard_mean"] + df["hard_std"], df["gain_pts"], df["gain_paired_std_pts"]):
                ax.text(xi, top + 1.2, f"{round(g, 1) + 0.0:+.1f}", ha="center", va="bottom", fontsize=8.5,
                        color=th["muted"] if abs(g) <= s else th["text"], zorder=5)
    ax.set_xticks(x, [f"{n}\n{p:,} prefixes" for n, p in zip(df["name"], df["n_prefixes"])], fontsize=9.5)
    for t in ax.get_xticklabels(): t.set_color(th["text"])
    ax.set_xlim(-0.6, len(df) - 0.4)
    ax.set_ylim(0, 108)
    fig.suptitle("Next-activity prediction with the repair models", x=0.06, y=0.985, ha="left", fontsize=14,
                 fontweight="bold", color=th["text"])
    fig.text(0.06, 0.935, f"Accuracy on the event after each prefix, mean ± std over {int(df['runs'].min())} runs "
                          f"(tag {tag}).\nNumbers: gain of hard over raw in points, same models; "
                          "grey = within noise (|gain| ≤ std of paired differences).",
             ha="left", va="top", fontsize=9, color=th["text2"])
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False, fontsize=9)
    for t in leg.get_texts(): t.set_color(th["text"])
    fig.subplots_adjust(left=0.06, right=0.99, top=0.84, bottom=0.22)
    return save(fig, out_dir, f"summary_next_activity_{tag}", theme, formats, th)


# ---------------------------------------------------------------- per dataset, by prefix length
def by_position(pred, key):
    """Per-run accuracy at each value of `key`, then mean and std over runs; plus counts and raw violation rate."""
    g = pred.groupby(["run", key])
    per_run = g[["ok_raw", "ok_hard", "raw_violates"]].mean().reset_index()
    agg = per_run.groupby(key).agg(raw_mean=("ok_raw", "mean"), raw_std=("ok_raw", "std"),
                                   hard_mean=("ok_hard", "mean"), hard_std=("ok_hard", "std"),
                                   viol_mean=("raw_violates", "mean"), viol_std=("raw_violates", "std"))
    agg[agg.columns] = agg[agg.columns] * 100
    agg["n_per_run"] = pred[pred["run"] == pred["run"].iloc[0]].groupby(key).size()
    return agg.reset_index()


def k_buckets(pred, min_n, max_points, max_buckets=14, cover=0.98):
    """Bucket label for every prefix length k, or None when at most `max_points` lengths have >= min_n prefixes per run.
    Buckets have a round width (1, 2, 5, 10, ...) such that the lengths holding `cover` of the prefixes fit in
    `max_buckets` buckets; the longer prefixes are pooled into one last bucket '>= K'."""
    cnt = pred[pred["run"] == pred["run"].iloc[0]]["k"].value_counts().sort_index()
    if (cnt >= min_n).sum() <= max_points:
        return None
    k_cov = int(cnt.index[np.argmax((cnt.cumsum() / cnt.sum()).to_numpy() >= cover)])
    width = next(w for m in (1, 10, 100, 1000) for w in (m, 2 * m, 5 * m) if -(-k_cov // w) <= max_buckets)
    tail = -(-k_cov // width) * width + 1                             # first k of the pooled bucket
    starts = list(range(1, tail, width))
    labels = {k: (f"{k}" if width == 1 else f"{k}–{k + width - 1}") for k in starts}
    lab = lambda k: f"≥ {tail}" if k >= tail else labels[1 + (k - 1) // width * width]
    order = [labels[k] for k in starts] + ([f"≥ {tail}"] if cnt.index.max() >= tail else [])
    return {k: lab(k) for k in cnt.index}, order, width


def plot_dataset(name, pred, tag, out_dir, theme, formats, min_n=20, max_points=20):
    th = THEMES[theme]
    runs = pred["run"].nunique()
    full = by_position(pred, "k")                                       # every length, always in the CSV
    bk = k_buckets(pred, min_n, max_points)
    if bk is None:
        t = full[full["n_per_run"] >= min_n].assign(label=lambda d: d["k"].astype(str))   # too few prefixes: noise
        xlab, note = "Prefix length k (visible events)", (f"Lengths with fewer than {min_n} prefixes per run are "
                                                         "left out (all are in the CSV).")
    else:
        mapping, order, width = bk
        t = by_position(pred.assign(label=pred["k"].map(mapping)), "label")
        t = t.set_index("label").loc[order].reset_index()
        t = t[t["n_per_run"] >= min_n]
        xlab = f"Prefix length k (visible events), grouped in buckets of {width}"
        note = (f"Prefix lengths grouped in buckets of {width}; the last bucket pools all longer prefixes. "
                "Per-length values are in the CSV.")
    x = np.arange(len(t)); labels = t["label"].tolist()
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.4), facecolor=th["surface"], sharex=True,
                             gridspec_kw=dict(height_ratios=[3, 1.6, 1.3], hspace=0.28))
    ax = axes[0]; style_axis(ax, th); avg = {}
    for lab, (_, _, slot) in SERIES.items():
        pre = "raw" if slot == 0 else "hard"
        col = th["series"][slot]
        m, s = t[f"{pre}_mean"].to_numpy(), t[f"{pre}_std"].fillna(0).to_numpy()
        ax.fill_between(x, np.clip(m - s, 0, 100), np.clip(m + s, 0, 100), color=col, alpha=0.18, linewidth=0, zorder=2)
        ax.plot(x, m, color=col, linewidth=2, marker="o", markersize=5, label=lab, zorder=3,
                markeredgecolor=th["surface"], markeredgewidth=1)
    for lab, (_, ok, slot) in SERIES.items():                      # accuracy over all prefixes, mean over runs
        v = pred.groupby("run")[ok].mean().mean() * 100
        ax.axhline(v, color=th["series"][slot], linestyle=(0, (5, 3)), linewidth=1.3, zorder=2.5)
        avg[lab] = v
    ys = {}                                                            # labels at the right edge, pushed apart
    for lab in sorted(avg, key=avg.get):
        ys[lab] = max(avg[lab], max(ys.values(), default=-np.inf) + 5)
    for lab, y in ys.items():
        ax.text(1.005, y, f"avg {avg[lab]:.1f}%", transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=8, color=th["series"][SERIES[lab][2]], fontweight="bold")
    ax.set_title("Accuracy by prefix length (dashed: accuracy over all prefixes)", loc="left", fontsize=10,
                 color=th["text"])
    ax = axes[1]; style_axis(ax, th)
    col = th["series"][0]
    m, s = t["viol_mean"].to_numpy(), t["viol_std"].fillna(0).to_numpy()
    ax.fill_between(x, np.clip(m - s, 0, 100), np.clip(m + s, 0, 100), color=col, alpha=0.18, linewidth=0, zorder=2)
    ax.plot(x, m, color=col, linewidth=2, marker="o", markersize=5, zorder=3,
            markeredgecolor=th["surface"], markeredgewidth=1)
    ax.set_yticks(range(0, 101, 50)); ax.set_yticklabels(["0%", "50%", "100%"])
    ax.set_title("Raw prediction breaks a live rule (share of prefixes)", loc="left", fontsize=10, color=th["text"])
    ax = axes[2]; style_axis(ax, th, pct=False)
    ax.bar(x, t["n_per_run"], 0.7, color=th["muted"], zorder=3, linewidth=0)
    ax.set_title("Prefixes per run", loc="left", fontsize=10, color=th["text"])
    ax.set_xlabel(xlab, color=th["text2"], fontsize=10, labelpad=6)
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_xlim(-0.5, len(x) - 0.5)
    fig.suptitle(f"{nice_name(name)}: next-activity prediction", x=0.06, y=0.985, ha="left", fontsize=14,
                 fontweight="bold", color=th["text"])
    fig.text(0.06, 0.955, f"Per-run accuracy, mean (line) ± std (band) over {runs} runs, tag {tag}.\n{note}",
             ha="left", va="top", fontsize=9, color=th["text2"])
    leg = fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.055, 0.915), ncol=2,
                     frameon=False, fontsize=9)
    for tx in leg.get_texts(): tx.set_color(th["text"])
    fig.subplots_adjust(left=0.06, right=0.93, top=0.85, bottom=0.07)  # right: room for the average labels
    rows = [full.rename(columns={"k": "position"}).assign(axis="prefix_len_k")]
    if bk is not None:
        rows.append(t.drop(columns="label").assign(position=labels, axis=f"prefix_len_bucket{width}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / f"{name}_next_activity_{tag}.csv"
    pd.concat(rows).to_csv(csv, index=False, float_format="%.4f")
    return [csv] + save(fig, out_dir, f"{name}_next_activity_{tag}", theme, formats, th)


ORDER = ["bpi_2013", "small_log", "large_log", "sp2020", "BPI20_RequestForPayment"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=str(HERE / "results_reference/next_activity"),
                    help="results of next_activity.py (results/next_activity) or the published ones (default)")
    ap.add_argument("--out-dir", default=str(HERE / "plots/next_activity"))
    ap.add_argument("--tag", default="sl")
    ap.add_argument("--formats", nargs="*", default=["png", "pdf"])
    ap.add_argument("--dark", action="store_true", help="dark theme")
    ap.add_argument("--min-n", type=int, default=20, help="hide positions with fewer prefixes per run in the figures")
    ap.add_argument("--max-points", type=int, default=20,
                    help="bucket the prefix lengths when more than this many have >= min-n prefixes per run")
    a = ap.parse_args()
    root, out_dir = Path(a.results_dir), Path(a.out_dir)
    theme = "dark" if a.dark else "light"
    rank = lambda p: next((i for i, k in enumerate(ORDER) if p.name.startswith(k)), len(ORDER))
    dirs = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: (rank(p), p.name))
    df = load_summary(dirs, a.tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"summary_next_activity_{a.tag}.csv", index=False, float_format="%.4f")
    for p in plot_summary(df, a.tag, out_dir, theme, a.formats):
        print("wrote", p)
    for d in dirs:
        f = d / f"next_act_predictions_{a.tag}.csv"
        if not f.exists():
            print(f"skip {d.name}: no {f.name}"); continue
        for p in plot_dataset(d.name, pd.read_csv(f), a.tag, out_dir, theme, a.formats, a.min_n, a.max_points):
            print("wrote", p)


if __name__ == "__main__":
    main()
