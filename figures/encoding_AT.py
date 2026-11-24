"""Heterogeneous graph encoding of the trace in Table 2 (tab:trace_example_missing), activity + timestamp only.

BPI 2012, trace 173688 ('-' = missing):
    e1: activity -,                  timestamp 2011-09-30 22:38:44.546
    e2: activity A_partlysubmitted,  timestamp -
    e3: activity -,                  timestamp -
    e4: activity A_accepted,         timestamp 2011-10-01 09:42:43.308
    e5: activity -,                  timestamp 2011-10-01 09:45:09.243
Timestamps are shown to the second. Drawn at text width (4.8 in, LNCS): include with width=\\linewidth.
Output: encoding_AT.pdf (vector) and encoding_AT.png (preview), next to this script.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 7.5})

OUT = Path(__file__).resolve().parent
ACT = [None, "A_partlysubmitted", None, "A_accepted", None]                  # None = missing
TS = ["2011-09-30\n22:38:44", None, None, "2011-10-01\n09:42:43", "2011-10-01\n09:45:09"]

# geometry in inches (axes data units = inches, aspect equal)
COLW = [0.72, 1.06, 0.42, 0.72, 0.72]          # column widths (fit the widest label of each column)
GAP, H = 0.27, 0.36                            # horizontal gap between boxes, box height
Y_ACT, Y_TS = 0.80, 0.0
C_AA, C_TT, C_AT, C_TA = "#0072B2", "#E69F00", "#009E73", "#CC79A7"          # Okabe-Ito
FILL = {"act": "#D7E7F5", "ts": "#FCEBC9"}
DASH = (0, (3, 2))

xs, x = [], 0.0
for w in COLW:
    xs.append(x + w / 2)
    x += w + GAP
WIDTH = x - GAP


def node(ax, xc, y, w, val, kind, name, above):
    miss = val is None
    ax.add_patch(FancyBboxPatch((xc - w / 2, y - H / 2), w, H, boxstyle="round,pad=0,rounding_size=0.06",
                                facecolor="white" if miss else FILL[kind], edgecolor="#333333",
                                linestyle=DASH if miss else "-", linewidth=0.9, zorder=3))
    if miss:
        ax.text(xc, y, "?", ha="center", va="center", color="#777777", fontsize=10, zorder=4)
    else:
        ax.text(xc, y, val, ha="center", va="center", zorder=4, linespacing=1.1,
                family="sans-serif" if kind == "act" else "serif")
    dy = H / 2 + 0.11
    ax.text(xc, y + dy if above else y - dy, f"${name}$", ha="center", va="center", fontsize=8, color="#333333")


def arrow(ax, p, q, color, both=False):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="<|-|>" if both else "-|>", mutation_scale=7,
                                 color=color, linewidth=1.0, zorder=2, shrinkA=1.5, shrinkB=1.5))


Y0, Y1 = Y_TS - H / 2 - 0.2, Y_ACT + H / 2 + 0.2          # vertical data extent (inches)
X0, X1 = -0.03, WIDTH + 0.03
LEG = 0.42                                                 # room for the legend below the graph (inches)
FW, FH = X1 - X0, (Y1 - Y0) + LEG
fig = plt.figure(figsize=(FW, FH))
ax = fig.add_axes([0, LEG / FH, 1, (Y1 - Y0) / FH])        # 1 data unit = 1 inch
for i, (xc, w) in enumerate(zip(xs, COLW)):
    node(ax, xc, Y_ACT, w, ACT[i], "act", f"v_{i + 1}^{{act}}", above=True)
    node(ax, xc, Y_TS, w, TS[i], "ts", f"v_{i + 1}^{{ts}}", above=False)
    off = 0.07                                                     # act->ts and ts->act side by side
    arrow(ax, (xc - off, Y_ACT - H / 2), (xc - off, Y_TS + H / 2), C_AT)
    arrow(ax, (xc + off, Y_TS + H / 2), (xc + off, Y_ACT - H / 2), C_TA)
    if i < len(xs) - 1:
        x0, x1 = xc + w / 2, xs[i + 1] - COLW[i + 1] / 2
        arrow(ax, (x0, Y_ACT), (x1, Y_ACT), C_AA, both=True)
        arrow(ax, (x0, Y_TS), (x1, Y_TS), C_TT, both=True)

handles = [
    Line2D([], [], color=C_AA, lw=1.2, label=r"$v_i^{act} \leftrightarrow v_{i+1}^{act}$"),
    Line2D([], [], color=C_TT, lw=1.2, label=r"$v_i^{ts} \leftrightarrow v_{i+1}^{ts}$"),
    Line2D([], [], color=C_AT, lw=1.2, label=r"$v_i^{act} \rightarrow v_i^{ts}$"),
    Line2D([], [], color=C_TA, lw=1.2, label=r"$v_i^{ts} \rightarrow v_i^{act}$"),
    Rectangle((0, 0), 1, 1, facecolor=FILL["act"], edgecolor="#333333", lw=0.9, label="activity node"),
    Rectangle((0, 0), 1, 1, facecolor=FILL["ts"], edgecolor="#333333", lw=0.9, label="timestamp node"),
    Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#333333", lw=0.9, linestyle=DASH, label="missing value"),
]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, frameon=False,
          handlelength=1.6, columnspacing=1.2, fontsize=7.5)
ax.set_xlim(X0, X1)
ax.set_ylim(Y0, Y1)
ax.set_aspect("equal")
ax.axis("off")
r = fig.canvas.get_renderer()                             # report label widths vs box widths
for t in ax.texts:
    if t.get_text() not in ("?",) and not t.get_text().startswith("$"):
        print(f"{t.get_text()!r:28} {t.get_window_extent(r).width / fig.dpi:.2f} in")
fig.savefig(OUT / "encoding_AT.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT / "encoding_AT.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
print("saved", OUT / "encoding_AT.pdf")
