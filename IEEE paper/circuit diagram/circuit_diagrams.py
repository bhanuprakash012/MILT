"""
Circuit diagrams for two hardware-efficient ansatze and the XXZ Hamiltonian variational ansatz in PennyLane's drawing style.

Uses PennyLane's low-level ``MPLDrawer`` (the same engine behind ``qml.draw_mpl``)
so the gates look exactly like PennyLane's, but with control over layout that
``qml.draw_mpl`` does not give:
  * periodic (N-1, 0) two-qubit gates are drawn as two linked boxes on the
    qubits they act on, instead of one tall box that seems to cover every wire;
  * the repeated layer is framed and labelled "x L";
  * the MIPT-style mid-circuit measurements are part of every layer and drawn
    with dashed meters: each qubit is measured with probability p.

Outputs (vector PDF + PNG):
    circuit_diagrams.(pdf|png)   (a) and (b) side by side, (c) underneath; sized for a two-column figure*
    hea1.(pdf|png), hea2.(pdf|png), xxz_hva.(pdf|png)   individual panels
"""

import logging

import matplotlib.pyplot as plt
from matplotlib import patches
from pennylane.drawer import MPLDrawer

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# ----------------------------------------------------------------------------
# Palette (PennyLane's blue/pink, plus two neutral tints)
# ----------------------------------------------------------------------------
BLUE = "#D5F0FD"     # PennyLane default gate colour: variational gates
GREEN = "#DDF5E3"    # second sublayer in the HVA (even bonds)
GREY = "#EDEDED"     # fixed (non-variational) state-preparation gates
FRAME = "#F7F4FF"    # repeated-layer background
MEAS = "#FFE9D6"     # probabilistic mid-circuit measurement
INK = "#151515"
MUTED = "#6B6B6B"

# Font sizes are in points at the drawing's native scale (1 grid unit = 1 inch).
# The combined figure is ~23 in wide, so at a two-column text width (~7 in)
# everything shrinks by ~0.3 (see the size report printed by draw_all).
GATE_FS = 26       # gate labels
LABEL_FS = 26      # wire labels  q_i |0>
NOTE_FS = 23.5       # subtitles, brackets, PBC notes
TITLE_FS = 31      # panel titles
PRINT_WIDTH_IN = 7.0   # two-column paper text width, used only to report final sizes

# Compact mode for a paper: the one-line descriptions and "PBC" notes go into the
# LaTeX caption instead (see CAPTION below). Set both True for a self-explaining
# stand-alone figure; it is ~30% taller.
SHOW_DESCRIPTIONS = False
SHOW_PBC_NOTES = False

Y_PAD_BOTTOM = 1.2 if SHOW_PBC_NOTES else 0.72   # room below the last wire


def _y_top(layout):
    """Room above wire 0: more when there are bracket labels to fit."""
    top = -1.4 if any(g[2] == "span" for g in layout.groups) else -0.7   # brackets
    top -= 0.55                                                          # title
    return top - (1.2 if SHOW_DESCRIPTIONS else 0.0)


def _panel_height(layout):
    return layout.n - _y_top(layout) + Y_PAD_BOTTOM - 1

# PennyLane's "pennylane" style (same values as qml.drawer.use_style("pennylane")),
# applied locally through rc_context so global matplotlib settings are untouched.
PL_STYLE = {
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.facecolor": "#FFB5F1",      # CNOT target fill
    "patch.facecolor": BLUE,
    "patch.edgecolor": INK,
    "patch.linewidth": 2.0,
    "patch.force_edgecolor": True,
    "lines.color": INK,
    "text.color": INK,
    "font.weight": "bold",
    "mathtext.fontset": "dejavusans",
}
WIRE_OPTS = {"color": INK, "linewidth": 1.6}


# ----------------------------------------------------------------------------
# A tiny layout layer on top of MPLDrawer
# ----------------------------------------------------------------------------
class Layout:
    """Collects columns of operations, then places them left-to-right."""

    def __init__(self, n_qubits):
        self.n = n_qubits
        self.cols = []      # dicts: {"ops": [...], "w": pitch}
        self.groups = []    # (first_col, last_col, kind, label)

    def col(self, *ops, w=1.0):
        self.cols.append({"ops": list(ops), "w": w})
        return len(self.cols) - 1

    def gap(self, w=0.35):
        return self.col(w=w)

    def group(self, first, last, kind, label=""):
        self.groups.append((first, last, kind, label))

    def stretch(self, total):
        """Spread extra horizontal room evenly between columns (gate sizes unchanged)."""
        cur = self.xs()[1]
        if total <= cur:
            return
        extra = (total - cur) / len(self.cols)
        for c in self.cols:
            c.setdefault("bw", c["w"])
            c["w"] += extra

    # x positions of column centres
    def xs(self):
        out, x = [], 0.0
        for c in self.cols:
            out.append(x + c["w"] / 2)
            x += c["w"]
        return out, x


# Operation constructors -----------------------------------------------------
def gate(wires, text, color=BLUE):
    return ("gate", wires, text, color)


def wrap(a, b, text, color=BLUE):
    """Two-qubit gate on non-neighbouring wires (periodic bond)."""
    return ("wrap", (a, b), text, color)


def cnot(ctrl, targ):
    return ("cnot", ctrl, targ)


def meas(q):
    return ("meas", q)


CAPTION = r"""
Circuits for (a) hardware-efficient ansatz 1, (b) hardware-efficient ansatz 2 and
(c) the XXZ Hamiltonian variational ansatz, shown for $N=4$ qubits. Grey gates prepare
the initial state: $R_y(\pi/4)$ on every qubit in (a), $H$ on every qubit in (b), and
singlets $(|01\rangle-|10\rangle)/\sqrt{2}$ on the even bonds in (c). The shaded block is
one layer and is repeated $L$ times. In (a), $R_P\in\{R_x,R_y,R_z\}$ is a randomly chosen
Pauli rotation; in (b), all qubits share $\theta$ and $\phi$; in (c), the odd bonds use
$(\theta,\phi)$ and the even bonds use $(\beta,\gamma)$. Gates linking qubits $N-1$ and $0$
implement periodic boundary conditions. At the end of every layer, each qubit is
measured with probability $p$ (dashed meters).
"""


def add_meas_column(L):
    """Probabilistic measurement of every qubit at the end of a layer."""
    c = L.col(*[meas(q) for q in range(L.n)], w=1.7)
    L.group(c, c, "span", r"w/ prob. $p$")
    return c


def _pbc_note(ax, x, n):
    if not SHOW_PBC_NOTES:
        return
    ax.text(x, n - 1 + 0.88, "PBC", ha="center", va="center", fontsize=NOTE_FS,
            color=MUTED, fontweight="normal")


def render(layout, title, subtitle, fig=None, x_end=None):
    n = layout.n
    xs, width = layout.xs()
    x_end = width if x_end is None else x_end
    y_top = _y_top(layout)

    drawer = MPLDrawer(n_layers=width, wire_map={q: q for q in range(n)}, fig=fig,
                       wire_options=WIRE_OPTS, figsize=(x_end + 2.2, _panel_height(layout)))
    ax = drawer.ax
    ax.set_xlim(-2.0, x_end + 0.2)
    ax.set_ylim(n - 1 + Y_PAD_BOTTOM, y_top)

    drawer.label([rf"$q_{q}\;|0\rangle$" for q in range(n)],
                 text_options={"ha": "center", "va": "center", "fontsize": LABEL_FS})

    for c, x in zip(layout.cols, xs):
        extra = max(0.0, c.get("bw", c["w"]) - 1.0)
        for op in c["ops"]:
            kind = op[0]
            if kind == "gate":
                _, wires, text, color = op
                drawer.box_gate(x, wires, text, box_options={"facecolor": color},
                                text_options={"fontsize": GATE_FS}, extra_width=extra,
                                autosize=False)
            elif kind == "wrap":
                _, (a, b), text, color = op
                ax.plot([x, x], [a, b], color=INK, lw=2, zorder=1.5,
                        solid_capstyle="round")
                for q in (a, b):
                    drawer.box_gate(x, q, text, box_options={"facecolor": color},
                                    text_options={"fontsize": GATE_FS}, extra_width=extra,
                                    autosize=False)
                _pbc_note(ax, x, n)
            elif kind == "cnot":
                _, ctrl, targ = op
                drawer.CNOT(x, [ctrl, targ], options={"color": INK, "linewidth": 2})
                if abs(ctrl - targ) == n - 1 and n > 2:
                    _pbc_note(ax, x, n)
            elif kind == "meas":
                # dashed meter = measured with probability p (MIPT-style)
                drawer.measure(x, op[1],
                               box_options={"facecolor": MEAS, "linestyle": (0, (2.5, 1.8))})

    # Group annotations
    for first, last, kind, label in layout.groups:
        x0 = xs[first] - layout.cols[first]["w"] / 2
        x1 = xs[last] + layout.cols[last]["w"] / 2
        if kind == "repeat":
            box = patches.FancyBboxPatch(
                (x0 + 0.08, -0.6), x1 - x0 - 0.16, n - 1 + 1.2,
                boxstyle="round,pad=0,rounding_size=0.18",
                facecolor=FRAME, edgecolor=MUTED, lw=1.3, ls=(0, (4, 3)), zorder=0)
            ax.add_patch(box)
            ax.text(x1 + 0.02, -0.6, r"$\times\,L$", ha="left", va="bottom",
                    fontsize=TITLE_FS - 2, color=INK)
        elif kind == "span":
            y = -0.92
            xa, xb = x0 + 0.15, x1 - 0.15
            ax.plot([xa, xa, xb, xb], [y + 0.1, y, y, y + 0.1], color=MUTED, lw=1.1)
            ax.text((x0 + x1) / 2, y - 0.07, label, ha="center", va="bottom",
                    fontsize=NOTE_FS, color=MUTED, fontweight="normal", style="italic")

    ax.text(-1.95, y_top + 0.05, title, ha="left", va="top", fontsize=TITLE_FS, color=INK)
    if SHOW_DESCRIPTIONS:
        ax.text(-1.95, y_top + 0.66, subtitle, ha="left", va="top", fontsize=NOTE_FS,
                linespacing=1.35, color=MUTED, fontweight="normal")
    return drawer


# ----------------------------------------------------------------------------
# The three ansätze (one variational layer shown, framed as "x L")
# ----------------------------------------------------------------------------
def hea1(n):
    L = Layout(n)
    L.col(*[gate(q, r"$R_y(\pi/4)$", GREY) for q in range(n)], w=1.75)
    L.gap()
    a = L.col(*[gate(q, rf"$R_{{P}}(\theta_{{{q + 1}}})$") for q in range(n)], w=1.75)
    L.col(*[cnot(q, q + 1) for q in range(0, n - 1, 2)])
    L.col(*[gate(q, rf"$R_{{P}}(\theta_{{{n + q + 1}}})$") for q in range(n)], w=1.75)
    L.col(*[cnot(q, q + 1) for q in range(1, n - 1, 2)])
    L.col(cnot(n - 1, 0))
    b = add_meas_column(L)
    L.group(a, b, "repeat")
    L.gap(0.85)
    return L, "(a)  Hardware-Efficient Ansatz 1", (
        r"$R_y(\pi/4)$ init  ·  random Pauli rotations $R_P\in\{R_x,R_y,R_z\}$" "\n"
        r"even/odd periodic CNOTs  ·  each qubit measured with probability $p$")


def hea2(n):
    L = Layout(n)
    L.col(*[gate(q, r"$H$", GREY) for q in range(n)])
    L.gap()
    a = L.col(*[gate(q, r"$R_y(\theta)$") for q in range(n)], w=1.45)
    for q in range(n - 1):
        L.col(cnot(q, q + 1), w=0.8)
    L.col(cnot(n - 1, 0), w=0.8)
    L.col(*[gate(q, r"$R_x(\phi)$") for q in range(n)], w=1.45)
    b = add_meas_column(L)
    L.group(a, b, "repeat")
    L.gap(0.85)
    return L, "(b)  Hardware-Efficient Ansatz 2", (
        r"$H$ init  ·  shared $R_y(\theta)$  ·  periodic CNOT ladder  ·  shared $R_x(\phi)$" "\n"
        r"each qubit measured with probability $p$")


def xxz_hva(n):
    assert n % 2 == 0 and n >= 4, "XXZ-HVA needs an even number of qubits (>= 4)"
    L = Layout(n)
    even = [(q, q + 1) for q in range(0, n, 2)]
    odd_bulk = [(q, q + 1) for q in range(1, n - 1, 2)]

    # Singlet prep on even bonds: X⊗X, H on first, CNOT -> (|01> - |10>)/sqrt2
    s0 = L.col(*[gate(q, r"$X$", GREY) for q in range(n)])
    L.col(*[gate(a, r"$H$", GREY) for a, _ in even])
    s1 = L.col(*[cnot(a, b) for a, b in even])
    L.group(s0, s1, "span", "even-bond singlets")
    L.gap()

    first = None
    for name, p in (("ZZ", r"\theta"), ("YY", r"\phi"), ("XX", r"\phi")):
        c = L.col(*[gate([a, b], rf"$\mathrm{{{name}}}({p})$") for a, b in odd_bulk], w=1.45)
        L.col(wrap(n - 1, 0, rf"$\mathrm{{{name}}}({p})$"), w=1.45)
        first = c if first is None else first
    odd_end = len(L.cols) - 1
    L.group(first, odd_end, "span", r"odd bonds  $(\theta,\phi)$")

    e0 = None
    for name, p in (("ZZ", r"\beta"), ("YY", r"\gamma"), ("XX", r"\gamma")):
        c = L.col(*[gate([a, b], rf"$\mathrm{{{name}}}({p})$", GREEN) for a, b in even], w=1.45)
        e0 = c if e0 is None else e0
    L.group(e0, len(L.cols) - 1, "span", r"even bonds  $(\beta,\gamma)$")
    m = add_meas_column(L)
    L.group(first, m, "repeat", "")
    L.gap(0.85)
    return L, "(c)  XXZ Hamiltonian Variational Ansatz", (
        r"singlet init  ·  odd-bond $ZZ/YY/XX$ $(\theta,\phi)$  ·  even-bond $ZZ/YY/XX$ $(\beta,\gamma)$" "\n"
        r"each qubit measured with probability $p$")


# ----------------------------------------------------------------------------
def _as_drawer_target(subfig):
    # MPLDrawer resizes the figure it is given; a SubFigure has a fixed size,
    # so make those calls no-ops and let MPLDrawer draw into the sub-panel.
    subfig.set_figwidth = subfig.set_figheight = lambda *a, **k: None
    return subfig


def draw_all(n_qubits=4, prefix=""):
    with plt.rc_context(PL_STYLE):
        return _draw_all(n_qubits, prefix)


def _draw_all(n_qubits, prefix):
    builders = (("hea1", hea1), ("hea2", hea2), ("xxz_hva", xxz_hva))
    panels = [(name, *fn(n_qubits)) for name, fn in builders]

    # individual panels
    for name, layout, title, sub in panels:
        d = render(layout, title, sub)
        for ext in ("pdf", "png"):
            d.fig.savefig(f"{prefix}{name}.{ext}", dpi=300)
        plt.close(d.fig)

    # combined figure: (a) | (b) on the top row, (c) on the bottom row.
    # Every panel is drawn at 1 data unit = 1 inch, so gates are the same size everywhere.
    (_, La, ta, sa), (_, Lb, tb, sb), (_, Lc, tc, sc) = panels
    wa, wb, wc = (Lx.xs()[1] + 2.2 for Lx in (La, Lb, Lc))   # panel widths incl. wire labels
    W = max(wa + wb, wc)
    h_top = max(_panel_height(La), _panel_height(Lb))
    h_bot = _panel_height(Lc)

    fig = plt.figure(figsize=(W, h_top + h_bot))
    top, bot = fig.subfigures(2, 1, height_ratios=[h_top, h_bot])
    sf_a, sf_b = top.subfigures(1, 2, width_ratios=[wa, W - wa])
    render(La, ta, sa, fig=_as_drawer_target(sf_a))
    render(Lb, tb, sb, fig=_as_drawer_target(sf_b), x_end=W - wa - 2.2)
    Lc.stretch(W - 2.2)          # fill the bottom row instead of leaving a blank strip
    render(Lc, tc, sc, fig=_as_drawer_target(bot))
    for ext in ("pdf", "png"):
        fig.savefig(f"{prefix}circuit_diagrams.{ext}", dpi=300)
    k = PRINT_WIDTH_IN / W
    print(f"Combined figure {W:.1f} in wide. At {PRINT_WIDTH_IN} in print width: "
          f"gate labels {GATE_FS * k:.1f} pt, notes {NOTE_FS * k:.1f} pt, titles {TITLE_FS * k:.1f} pt")
    return fig


if __name__ == "__main__":
    draw_all(n_qubits=4)
    plt.show()
