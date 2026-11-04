import matplotlib.pyplot as plt
import matplotlib.patches as patches

# =============================================================================
# MATPLOTLIB CIRCUIT DIAGRAM DRAWER (MATCHING PAPER FIGURE 2)
# =============================================================================

def draw_ansatz_mpl(n_qubits=4):
    """
    Draws the exact circuit diagrams of HEA1, HEA2, and XXZ-HVA as implemented
    in our Pure-JAX code (Initial state prep + 1 Variational Layer + Measurements)
    so you can visually compare directly against Figure 2(a, b, c) of the paper.
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 11))

    def setup_wires(ax, title, max_x):
        ax.set_xlim(-0.8, max_x + 0.8)
        ax.set_ylim(-0.6, n_qubits - 0.3)
        ax.invert_yaxis()
        ax.axis("off")
        ax.set_title(title, fontsize=12.5, fontweight="bold", loc="left", pad=8)
        for q in range(n_qubits):
            ax.plot([0, max_x], [q, q], color="black", lw=1.3, zorder=1)
            ax.text(-0.45, q, f"$|0\\rangle$ (q{q})", ha="center", va="center", fontsize=10.5, fontweight="bold")

    def draw_1q_box(ax, x, q, label, facecolor="#e8f0fe", edgecolor="#1a73e8"):
        w, h = 0.68, 0.52
        rect = patches.FancyBboxPatch(
            (x - w / 2, q - h / 2), w, h,
            boxstyle="round,pad=0.04", facecolor=facecolor, edgecolor=edgecolor, lw=1.4, zorder=3
        )
        ax.add_patch(rect)
        ax.text(x, q, label, ha="center", va="center", fontsize=9.5, zorder=4)

    def draw_cnot(ax, x, ctrl, targ):
        ax.plot([x, x], [ctrl, targ], color="black", lw=1.6, zorder=2)
        ax.scatter([x], [ctrl], color="black", s=45, zorder=4)
        circle = patches.Circle((x, targ), 0.17, facecolor="white", edgecolor="black", lw=1.5, zorder=3)
        ax.add_patch(circle)
        ax.plot([x - 0.17, x + 0.17], [targ, targ], color="black", lw=1.3, zorder=4)
        ax.plot([x, x], [targ - 0.17, targ + 0.17], color="black", lw=1.3, zorder=4)

    def draw_2q_gate(ax, x, q1, q2, label, facecolor="#fef7e0", edgecolor="#d93025"):
        w = 0.62
        if abs(q1 - q2) == 1:
            top, bot = min(q1, q2), max(q1, q2)
            rect = patches.FancyBboxPatch(
                (x - w / 2, top - 0.25), w, (bot - top) + 0.50,
                boxstyle="round,pad=0.04", facecolor=facecolor, edgecolor=edgecolor, lw=1.4, zorder=3
            )
            ax.add_patch(rect)
            ax.text(x, 0.5 * (top + bot), label, ha="center", va="center", fontsize=9.5, zorder=4)
        else:
            # Periodic wrap-around bond (N-1, 0): draw connected boxes on q1 and q2
            ax.plot([x, x], [min(q1, q2), max(q1, q2)], color=edgecolor, lw=1.8, linestyle="--", zorder=2)
            draw_1q_box(ax, x, q1, label, facecolor=facecolor, edgecolor=edgecolor)
            draw_1q_box(ax, x, q2, label, facecolor=facecolor, edgecolor=edgecolor)

    def draw_meas(ax, x, q):
        circle = patches.Circle((x, q), 0.23, facecolor="#fad2cf", edgecolor="#a50e0e", lw=1.4, zorder=3)
        ax.add_patch(circle)
        ax.text(x, q, "?", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#a50e0e", zorder=4)

    def draw_layer_box(ax, x_start, x_end, label):
        rect = patches.FancyBboxPatch(
            (x_start, -0.45), x_end - x_start, n_qubits - 0.1,
            boxstyle="round,pad=0.08", facecolor="#fff8e1", edgecolor="black", lw=1.2, alpha=0.45, zorder=0
        )
        ax.add_patch(rect)
        ax.text(0.5 * (x_start + x_end), -0.48, label, ha="center", va="bottom", fontsize=9.5, style="italic")

    # --- Panel (a): HEA1 ---
    ax0 = axes[0]
    setup_wires(ax0, "(a) HEA1: Initial Ry(pi) + Repeated Layer [Random Pauli Rotations + Even/Odd Periodic CNOTs + Meas]", max_x=8.2)
    for q in range(n_qubits):
        draw_1q_box(ax0, 0.8, q, "$R_y(\\pi)$", facecolor="#f1f3f4", edgecolor="#5f6368")
    draw_layer_box(ax0, 1.5, 7.8, "Repeated Layer l = 1 .. L")
    for q in range(n_qubits):
        draw_1q_box(ax0, 2.1, q, f"$R_{{P_{q+1}}}(\\theta_{q+1})$")
    for q in range(0, n_qubits - 1, 2):
        draw_cnot(ax0, 3.1, q, q + 1)
    for q in range(n_qubits):
        draw_1q_box(ax0, 4.1, q, f"$R_{{P_{q+5}}}(\\theta_{q+5})$")
    for q in range(1, n_qubits - 1, 2):
        draw_cnot(ax0, 5.1, q, q + 1)
    draw_cnot(ax0, 6.0, n_qubits - 1, 0)  # Periodic wrap-around CNOT (N-1 -> 0)
    for q in range(n_qubits):
        draw_meas(ax0, 7.1, q)

    # --- Panel (b): HEA2 ---
    ax1 = axes[1]
    setup_wires(ax1, "(b) HEA2: Initial Hadamard + Repeated Layer [Shared Ry(theta) + Periodic CNOT Ladder + Shared Rx(phi) + Meas]", max_x=8.2)
    for q in range(n_qubits):
        draw_1q_box(ax1, 0.8, q, "$H$", facecolor="#f1f3f4", edgecolor="#5f6368")
    draw_layer_box(ax1, 1.5, 7.8, "Repeated Layer l = 1 .. L")
    for q in range(n_qubits):
        draw_1q_box(ax1, 2.1, q, "$R_y(\\theta)$")
    for idx, q in enumerate(range(n_qubits - 1)):
        draw_cnot(ax1, 3.1 + idx * 0.7, q, q + 1)
    draw_cnot(ax1, 3.1 + (n_qubits - 1) * 0.7, n_qubits - 1, 0)  # Periodic CNOT (N-1 -> 0)
    for q in range(n_qubits):
        draw_1q_box(ax1, 6.1, q, "$R_x(\\phi)$")
    for q in range(n_qubits):
        draw_meas(ax1, 7.1, q)

    # --- Panel (c): XXZ-HVA ---
    ax2 = axes[2]
    setup_wires(ax2, "(c) XXZ-HVA: Initial Singlet Bell Prep (X, H, CNOT) + Repeated Layer [Odd ZZ/YY/XX (theta, phi) + Even ZZ/YY/XX (beta, gamma) + Meas]", max_x=11.8)
    for q in range(n_qubits):
        draw_1q_box(ax2, 0.7, q, "$X$", facecolor="#f1f3f4", edgecolor="#5f6368")
    for q in range(0, n_qubits - 1, 2):
        draw_1q_box(ax2, 1.6, q, "$H$", facecolor="#f1f3f4", edgecolor="#5f6368")
        draw_cnot(ax2, 2.4, q, q + 1)

    draw_layer_box(ax2, 3.0, 11.4, "Repeated Layer l = 1 .. L")
    # Odd bonds: (1,2) and periodic (3,0)
    draw_2q_gate(ax2, 3.6, n_qubits - 1, 0, "$ZZ(\\theta)$")
    draw_2q_gate(ax2, 4.5, 1, 2, "$ZZ(\\theta)$")
    draw_2q_gate(ax2, 5.5, n_qubits - 1, 0, "$YY(\\phi)$")
    draw_2q_gate(ax2, 6.4, 1, 2, "$YY(\\phi)$")
    draw_2q_gate(ax2, 7.4, n_qubits - 1, 0, "$XX(\\phi)$")
    draw_2q_gate(ax2, 8.3, 1, 2, "$XX(\\phi)$")
    # Even bonds: (0,1) and (2,3)
    for q in range(0, n_qubits - 1, 2):
        draw_2q_gate(ax2, 9.2, q, q + 1, "$ZZ(\\beta)$", facecolor="#e6f4ea", edgecolor="#137333")
        draw_2q_gate(ax2, 10.0, q, q + 1, "$YY(\\gamma)$", facecolor="#e6f4ea", edgecolor="#137333")
        draw_2q_gate(ax2, 10.8, q, q + 1, "$XX(\\gamma)$", facecolor="#e6f4ea", edgecolor="#137333")
    for q in range(n_qubits):
        draw_meas(ax2, 11.6, q)

    plt.tight_layout()
    plt.savefig("circuit_diagrams.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    draw_ansatz_mpl(n_qubits=4)