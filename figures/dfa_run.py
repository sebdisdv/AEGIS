"""DFA run on distributions (Eq. eq:dfa_run) for response(a,b) on the trace <a, ?, ?, c>.

A = {a, b, c}; DFA of response(a,b): q0 (accepting, no pending a), q1 (an a waits for a b);
a: q0 -> q1, b: q1 -> q0, other activities keep the state.
Predicted distributions of the missing events: p2 = (a .2, b .5, c .3), p3 = (a .1, b .4, c .5).
State distributions: s0 = (1, 0), s1 = (0, 1), s2 = (.5, .5), s3 = (.65, .35), s4 = (.65, .35)  ->  P = s4(q0) = 0.65
(brute force over the 9 completions gives the same value).
Drawn at text width (4.8 in, LNCS): include with width=\\linewidth.
Output: dfa_run.pdf (vector) and dfa_run.png (preview), next to this script.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 7.5})
OUT = Path(__file__).resolve().parent

ACTS = ["a", "b", "c"]
TRACE = ["a", None, None, "c"]                       # None = missing
P = [None, [.2, .5, .3], [.1, .4, .5], None]          # predicted distributions of the missing events
DELTA = {(0, "a"): 1, (0, "b"): 0, (0, "c"): 0, (1, "a"): 1, (1, "b"): 0, (1, "c"): 1}
ACC = [True, False]

# distributions p_i (one-hot for visible events) and state distributions s_i
dist = [dict(zip(ACTS, p)) if p is not None else {x: float(x == a) for x in ACTS} for a, p in zip(TRACE, P)]
S = [np.array([1.0, 0.0])]
flows = []                                            # per step: {(q, q'): (mass, [activities])}
for p in dist:
    f, s = {}, np.zeros(2)
    for q in (0, 1):
        for x in ACTS:
            m = S[-1][q] * p[x]
            if m > 0:
                k = (q, DELTA[(q, x)])
                mass, xs = f.get(k, (0.0, []))
                f[k] = (mass + m, xs + [x])
                s[DELTA[(q, x)]] += m
    flows.append(f)
    S.append(s)
assert abs(S[-1][0] - 0.65) < 1e-9

C_EDGE, C_ACC, C_HL = "#555555", "#0072B2", "#D55E00"   # Okabe-Ito
FILL_ACT = "#D7E7F5"
DASH = (0, (3, 2))
R, RD = 0.15, 0.11                                    # state radius in the run and in the DFA (inches)

Y_LO, Y_HI = -0.27, 1.82                              # vertical extent (inches): s_i labels to trace boxes
fig = plt.figure(figsize=(4.8, Y_HI - Y_LO))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 4.8)
ax.set_ylim(Y_LO, Y_HI)
ax.set_aspect("equal")
ax.axis("off")


def state(xc, yc, label, accepting, color="#333333", lw=0.9, fs=7.5, face="white", r=R):
    ax.add_patch(Circle((xc, yc), r, facecolor=face, edgecolor=color, lw=lw, zorder=3))
    if accepting:
        ax.add_patch(Circle((xc, yc), r - 0.025, facecolor="none", edgecolor=color, lw=lw * 0.8, zorder=4))
    ax.text(xc, yc, label, ha="center", va="center", fontsize=fs, zorder=5)


def arrow(p, q, rad=0.0, lw=0.9, color=C_EDGE, shrink=R * 72 + 1, style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, connectionstyle=f"arc3,rad={rad}", arrowstyle=style, mutation_scale=6,
                                 color=color, lw=lw, shrinkA=shrink, shrinkB=shrink, zorder=2))


# ---------------- left: the DFA of G(a -> F b) = response(a,b) (compact) ----------------
DX0, DX1, DY = 0.24, 0.80, 0.37
ax.text((DX0 + DX1) / 2, DY + 0.66, r"DFA of $\mathbf{G}(a \rightarrow \mathbf{F}\, b)$", ha="center", va="center", fontsize=7)
state(DX0, DY, "$q_0$", True, fs=7, r=RD)
state(DX1, DY, "$q_1$", False, fs=7, r=RD)
ax.add_patch(FancyArrowPatch((DX0 - RD - 0.17, DY), (DX0 - RD, DY), arrowstyle="-|>", mutation_scale=5,
                             color=C_EDGE, lw=0.8, zorder=2))
arrow((DX0, DY), (DX1, DY), rad=-0.45, lw=0.8, shrink=RD * 72 + 1)
arrow((DX1, DY), (DX0, DY), rad=-0.45, lw=0.8, shrink=RD * 72 + 1)
ax.text((DX0 + DX1) / 2, DY + 0.20, "$a$", ha="center", va="center", fontsize=7)
ax.text((DX0 + DX1) / 2, DY - 0.20, "$b$", ha="center", va="center", fontsize=7)
for xc, lab in ((DX0, "$b, c$"), (DX1, "$a, c$")):            # self-loops above each state
    ax.add_patch(FancyArrowPatch((xc - 0.055, DY + RD - 0.01), (xc + 0.055, DY + RD - 0.01),
                                 connectionstyle="arc3,rad=-1.9", arrowstyle="-|>", mutation_scale=5,
                                 color=C_EDGE, lw=0.8, zorder=2))
    ax.text(xc, DY + RD + 0.22, lab, ha="center", va="center", fontsize=7)

# ---------------- right: the run on distributions ----------------
X0, D = 1.38, 0.80                                   # x of s_0, spacing of the columns s_0..s_4
YQ = {0: 0.62, 1: 0.12}                              # y of the q0 and q1 rows
YT, BH, BW = 1.62, 0.26, 0.30                        # trace row: y, box height, box width
cx = [X0 + i * D for i in range(5)]

for q, lab in ((0, "$q_0$"), (1, "$q_1$")):
    ax.text(X0 - 0.27, YQ[q], lab, ha="center", va="center", color="#333333")
for i in range(5):
    ax.text(cx[i], YQ[1] - R - 0.09, f"$s_{i}$", ha="center", va="center", color="#333333")

for i, (a, p) in enumerate(zip(TRACE, P)):           # trace boxes between columns i and i+1
    xm = (cx[i] + cx[i + 1]) / 2
    miss = a is None
    ax.add_patch(FancyBboxPatch((xm - BW / 2, YT - BH / 2), BW, BH, boxstyle="round,pad=0,rounding_size=0.05",
                                facecolor="white" if miss else FILL_ACT, edgecolor="#333333",
                                linestyle=DASH if miss else "-", lw=0.9, zorder=3))
    ax.text(xm, YT, "?" if miss else f"${a}$", ha="center", va="center", fontsize=9 if miss else 8,
            color="#777777" if miss else "black", zorder=4)
    ax.text(xm - BW / 2 - 0.04, YT + BH / 2 - 0.01, f"$e_{i + 1}$", ha="right", va="top", fontsize=6.5,
            color="#333333")
    if miss:                                          # mini bar chart of p_i under the box
        base, hmax, bw = YT - BH / 2 - 0.36, 0.42, 0.075
        for j, (x, v) in enumerate(zip(ACTS, p)):
            xb = xm + (j - 1) * 0.105
            ax.add_patch(Rectangle((xb - bw / 2, base), bw, v * hmax, facecolor=C_ACC, alpha=0.75, lw=0))
            ax.text(xb, base - 0.055, f"${x}$", ha="center", va="center", fontsize=6)
            ax.text(xb, base + v * hmax + 0.045, f"{v:.1f}".lstrip("0"), ha="center", va="center", fontsize=5.5)
        ax.text(xm + 0.21, base + 0.20, f"$p_{i + 1}$", ha="left", va="center", fontsize=7)

for i, f in enumerate(flows):                         # probability mass moved at step i+1
    for (q, q2), (m, xs) in f.items():
        p0, p1 = (cx[i], YQ[q]), (cx[i + 1], YQ[q2])
        last = i == len(flows) - 1 and q2 == 0
        arrow(p0, p1, lw=0.5 + 2.2 * m, color=C_HL if last else C_EDGE)
        lab = f"${','.join(xs)}$"                     # activities moving the mass (mass = thickness, see the states)
        if q == q2:                                   # horizontal edges: label above (q0) or below (q1)
            xl, yl = (p0[0] + p1[0]) / 2, YQ[q] + (0.075 if q == 0 else -0.075)
        else:                                         # diagonal edges: label near the arrowhead, on its right side
            t = 0.75
            xl = p0[0] + t * (p1[0] - p0[0]) + 0.05
            yl = p0[1] + t * (p1[1] - p0[1]) + (0.06 if q == 0 else -0.06)
        ax.text(xl, yl, lab, ha="center", va="center", fontsize=6.5, color="#333333", zorder=6)

for i, s in enumerate(S):                             # state nodes with their probabilities
    for q in (0, 1):
        hl = i == len(S) - 1 and q == 0
        state(cx[i], YQ[q], f"{s[q]:.2f}".replace("0.", ".") if s[q] not in (0, 1) else f"{s[q]:.0f}",
              ACC[q], color=C_HL if hl else ("#333333" if s[q] > 0 else "#AAAAAA"), lw=1.3 if hl else 0.9,
              fs=6.5)

ax.text(cx[4] + 0.04, YQ[0] + R + 0.16, r"$P_\sigma(\varphi) = s_4(q_0) = 0.65$", ha="right", va="center",
        fontsize=7.5, color=C_HL)

fig.savefig(OUT / "dfa_run.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT / "dfa_run.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
print("states:", [s.round(3).tolist() for s in S])
print("saved", OUT / "dfa_run.pdf")
