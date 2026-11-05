"""
Two-Stage Pure-JAX MIPT Scaling Study (HEA1, HEA2, XXZ-HVA with Periodic BC).

Stage 1 (--mode s2_coarse):
  - Fast sweep of Half-Chain Renyi-2 Entropy (S2_half) across all even N
    and a coarse p-grid to identify the approximate critical point p_c via data collapse.

Stage 2 (--mode tmi_fine):
  - High-resolution sweep of Tripartite Mutual Information (I_3 / TMI) and S2_half
    for N divisible by 4 in a narrow window around p_c for scale-invariant crossing.
"""

import os
import gc
import time
import pickle
import datetime
import argparse
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# 0. PRECISION & ENVIRONMENT SETUP
# =============================================================================

DTYPE_REAL = jnp.float64
DTYPE_COMPLEX = jnp.complex128

def configure_precision(use_32bit=False):
    global DTYPE_REAL, DTYPE_COMPLEX
    if use_32bit:
        jax.config.update("jax_enable_x64", False)
        DTYPE_REAL = jnp.float32
        DTYPE_COMPLEX = jnp.complex64
    else:
        jax.config.update("jax_enable_x64", True)
        DTYPE_REAL = jnp.float64
        DTYPE_COMPLEX = jnp.complex128


# =============================================================================
# 1. FAST 3-AXIS BIT-SHIFT QUANTUM KERNELS (PERIODIC BC)
# =============================================================================

def initial_state_zero(n_qubits):
    dim = 1 << n_qubits
    return jnp.zeros(dim, dtype=DTYPE_COMPLEX).at[0].set(1.0)


def hadamard_matrix():
    inv_sqrt2 = jnp.sqrt(jnp.array(0.5, dtype=DTYPE_REAL))
    return jnp.array([[inv_sqrt2, inv_sqrt2],
                      [inv_sqrt2, -inv_sqrt2]], dtype=DTYPE_COMPLEX)


def pauli_x_matrix():
    return jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=DTYPE_COMPLEX)


def rot_1q_matrix(theta, axis):
    c = jnp.cos(theta / 2.0)
    s = jnp.sin(theta / 2.0)
    zero = jnp.zeros_like(c, dtype=DTYPE_COMPLEX)

    Ux = jnp.stack([jnp.stack([c, -1j * s]), jnp.stack([-1j * s, c])])
    Uy = jnp.stack([jnp.stack([c, -s]), jnp.stack([s, c])])
    Uz = jnp.stack([jnp.stack([c - 1j * s, zero]), jnp.stack([zero, c + 1j * s])])
    return jnp.where(axis == 1, Ux, jnp.where(axis == 2, Uy, Uz)).astype(DTYPE_COMPLEX)


vmap_rot_1q = jax.vmap(rot_1q_matrix, in_axes=(0, 0))


def apply_1q_gate_psi(U, q, psi, n_qubits):
    dim_L = 1 << q
    dim_R = 1 << (n_qubits - q - 1)
    psi_r = psi.reshape(dim_L, 2, dim_R)
    new0 = U[0, 0] * psi_r[:, 0, :] + U[0, 1] * psi_r[:, 1, :]
    new1 = U[1, 0] * psi_r[:, 0, :] + U[1, 1] * psi_r[:, 1, :]
    return jnp.stack([new0, new1], axis=1).reshape(-1)


def apply_cnot_adjacent_psi(q, psi, n_qubits):
    dim_L = 1 << q
    dim_R = 1 << (n_qubits - q - 2)
    psi_r = psi.reshape(dim_L, 4, dim_R)
    return psi_r[:, (0, 1, 3, 2), :].reshape(-1)


def apply_cnot_edges_psi(psi, n_qubits):
    """Periodic CNOT with control = N-1 and target = 0."""
    dim_mid = 1 << (n_qubits - 2)
    psi_r = psi.reshape(2, dim_mid, 2)
    part_0 = psi_r[:, :, 0]
    part_1 = psi_r[::-1, :, 1]
    return jnp.stack([part_0, part_1], axis=-1).reshape(-1)


def apply_rzz_adjacent_psi(theta, q, psi, n_qubits):
    dim_L = 1 << q
    dim_R = 1 << (n_qubits - q - 2)
    psi_r = psi.reshape(dim_L, 4, dim_R)
    e_m = jnp.exp(-0.5j * theta).astype(DTYPE_COMPLEX)
    e_p = jnp.exp(0.5j * theta).astype(DTYPE_COMPLEX)
    phases = jnp.array([e_m, e_p, e_p, e_m], dtype=DTYPE_COMPLEX).reshape(1, 4, 1)
    return (psi_r * phases).reshape(-1)


def apply_rzz_edges_psi(theta, psi, n_qubits):
    dim_mid = 1 << (n_qubits - 2)
    psi_r = psi.reshape(2, dim_mid, 2)
    e_m = jnp.exp(-0.5j * theta).astype(DTYPE_COMPLEX)
    e_p = jnp.exp(0.5j * theta).astype(DTYPE_COMPLEX)
    phases = jnp.array([[e_m, e_p], [e_p, e_m]], dtype=DTYPE_COMPLEX).reshape(2, 1, 2)
    return (psi_r * phases).reshape(-1)


def apply_rxx_ryy_adjacent_psi(theta, axis, q, psi, n_qubits):
    dim_L = 1 << q
    dim_R = 1 << (n_qubits - q - 2)
    psi_r = psi.reshape(dim_L, 4, dim_R)
    c = jnp.cos(theta / 2.0).astype(DTYPE_COMPLEX)
    is_ = (-1j * jnp.sin(theta / 2.0)).astype(DTYPE_COMPLEX)
    off_03 = jnp.where(axis == 1, is_, -is_)

    s0, s1, s2, s3 = psi_r[:, 0, :], psi_r[:, 1, :], psi_r[:, 2, :], psi_r[:, 3, :]
    n0 = c * s0 + off_03 * s3
    n1 = c * s1 + is_ * s2
    n2 = is_ * s1 + c * s2
    n3 = off_03 * s0 + c * s3
    return jnp.stack([n0, n1, n2, n3], axis=1).reshape(-1)


def apply_rxx_ryy_edges_psi(theta, axis, psi, n_qubits):
    dim_mid = 1 << (n_qubits - 2)
    psi_r = psi.reshape(2, dim_mid, 2)
    c = jnp.cos(theta / 2.0).astype(DTYPE_COMPLEX)
    is_ = (-1j * jnp.sin(theta / 2.0)).astype(DTYPE_COMPLEX)
    off_03 = jnp.where(axis == 1, is_, -is_)

    s00, s01, s10, s11 = psi_r[0, :, 0], psi_r[0, :, 1], psi_r[1, :, 0], psi_r[1, :, 1]
    n00 = c * s00 + off_03 * s11
    n01 = c * s01 + is_ * s10
    n10 = is_ * s01 + c * s10
    n11 = off_03 * s00 + c * s11
    p0 = jnp.stack([n00, n01], axis=-1)
    p1 = jnp.stack([n10, n11], axis=-1)
    return jnp.stack([p0, p1], axis=0).reshape(-1)


# =============================================================================
# 2. PERIODIC ANSATZ LAYERS & WEAK / PROJECTIVE MEASUREMENT STEP
# =============================================================================

def prepare_initial_psi(n_qubits, ansatz):
    psi = initial_state_zero(n_qubits)
    H = hadamard_matrix()
    X = pauli_x_matrix()

    if ansatz == "HEA1":
        Uy_pi = rot_1q_matrix(jnp.array(jnp.pi, dtype=DTYPE_REAL), 2)
        for q in range(n_qubits):
            psi = apply_1q_gate_psi(Uy_pi, q, psi, n_qubits)
    elif ansatz == "HEA2":
        for q in range(n_qubits):
            psi = apply_1q_gate_psi(H, q, psi, n_qubits)
    elif ansatz == "HVA":
        for q in range(n_qubits):
            psi = apply_1q_gate_psi(X, q, psi, n_qubits)
        for q in range(0, n_qubits - 1, 2):
            psi = apply_1q_gate_psi(H, q, psi, n_qubits)
            psi = apply_cnot_adjacent_psi(q, psi, n_qubits)
    return psi


def apply_ansatz_layer_psi(psi, layer_params, layer_rots, n_qubits, ansatz):
    if ansatz == "HEA1":
        U_blk0 = vmap_rot_1q(layer_params[0], layer_rots[0])
        for q in range(n_qubits):
            psi = apply_1q_gate_psi(U_blk0[q], q, psi, n_qubits)
        for q in range(0, n_qubits - 1, 2):
            psi = apply_cnot_adjacent_psi(q, psi, n_qubits)

        U_blk1 = vmap_rot_1q(layer_params[1], layer_rots[1])
        for q in range(n_qubits):
            psi = apply_1q_gate_psi(U_blk1[q], q, psi, n_qubits)
        for q in range(1, n_qubits - 1, 2):
            psi = apply_cnot_adjacent_psi(q, psi, n_qubits)
        psi = apply_cnot_edges_psi(psi, n_qubits)

    elif ansatz == "HEA2":
        Uy = rot_1q_matrix(layer_params[0, 0], 2)
        for q in range(n_qubits):
            psi = apply_1q_gate_psi(Uy, q, psi, n_qubits)
        for q in range(n_qubits - 1):
            psi = apply_cnot_adjacent_psi(q, psi, n_qubits)
        psi = apply_cnot_edges_psi(psi, n_qubits)

        Ux = rot_1q_matrix(layer_params[1, 0], 1)
        for q in range(n_qubits):
            psi = apply_1q_gate_psi(Ux, q, psi, n_qubits)

    elif ansatz == "HVA":
        th_odd_zz = layer_params[0, 0]
        th_odd_xy = layer_params[1, 0]
        th_even_zz = layer_params[2, 0]
        th_even_xy = layer_params[3, 0]

        for q in range(1, n_qubits - 1, 2):
            psi = apply_rzz_adjacent_psi(th_odd_zz, q, psi, n_qubits)
        psi = apply_rzz_edges_psi(th_odd_zz, psi, n_qubits)

        for q in range(1, n_qubits - 1, 2):
            psi = apply_rxx_ryy_adjacent_psi(th_odd_xy, 2, q, psi, n_qubits)
        psi = apply_rxx_ryy_edges_psi(th_odd_xy, 2, psi, n_qubits)

        for q in range(1, n_qubits - 1, 2):
            psi = apply_rxx_ryy_adjacent_psi(th_odd_xy, 1, q, psi, n_qubits)
        psi = apply_rxx_ryy_edges_psi(th_odd_xy, 1, psi, n_qubits)

        for q in range(0, n_qubits - 1, 2):
            psi = apply_rzz_adjacent_psi(th_even_zz, q, psi, n_qubits)
        for q in range(0, n_qubits - 1, 2):
            psi = apply_rxx_ryy_adjacent_psi(th_even_xy, 2, q, psi, n_qubits)
        for q in range(0, n_qubits - 1, 2):
            psi = apply_rxx_ryy_adjacent_psi(th_even_xy, 1, q, psi, n_qubits)

    return psi


def generalized_measure_step(psi, q, mask_val, rand_u, s_strength, n_qubits):
    """Supports both projective (s=1.0) and weak Kraus measurements (0 < s <= 1)."""
    denom = jnp.sqrt(2.0 * (1.0 + s_strength ** 2))
    P_plus_diag = jnp.array([(1.0 + s_strength) / denom, (1.0 - s_strength) / denom], dtype=DTYPE_COMPLEX)
    P_minus_diag = jnp.array([(1.0 - s_strength) / denom, (1.0 + s_strength) / denom], dtype=DTYPE_COMPLEX)
    P_plus_sq = jnp.real(P_plus_diag) ** 2

    dim_L = 1 << q
    dim_R = 1 << (n_qubits - q - 1)
    psi_r = psi.reshape((dim_L, 2, dim_R))

    prob_0 = jnp.sum(jnp.abs(psi_r[:, 0, :]) ** 2)
    prob_1 = jnp.sum(jnp.abs(psi_r[:, 1, :]) ** 2)
    total_p = jnp.maximum(prob_0 + prob_1, 1e-30)

    p_plus = (prob_0 * P_plus_sq[0] + prob_1 * P_plus_sq[1]) / total_p
    outcome_is_plus = rand_u < p_plus

    chosen_P_diag = jnp.where(outcome_is_plus, P_plus_diag, P_minus_diag)
    p_chosen = jnp.where(outcome_is_plus, p_plus, 1.0 - p_plus) * total_p
    p_chosen_safe = jnp.maximum(p_chosen, 1e-30)

    scaled_diag = chosen_P_diag / jnp.sqrt(p_chosen_safe)
    ident_diag = jnp.ones(2, dtype=DTYPE_COMPLEX)
    applied_diag = jnp.where(mask_val, scaled_diag, ident_diag)

    psi_out = jnp.stack([psi_r[:, 0, :] * applied_diag[0],
                         psi_r[:, 1, :] * applied_diag[1]], axis=1).reshape(-1)
    return jnp.where(mask_val, psi_out, psi)


# =============================================================================
# 3. OPTIMIZED ENTROPY & TMI CALCULATORS
# =============================================================================

def purity_to_s2(rho):
    purity = jnp.sum(jnp.abs(rho) ** 2)
    return -jnp.log2(jnp.clip(purity, a_min=1e-15))


def compute_half_cut_s2(psi, n_qubits):
    """Fast contiguous bipartite cut for Stage 1 (no tensor transpose needed)."""
    n_a = n_qubits // 2
    dim_a = 1 << n_a
    dim_b = 1 << (n_qubits - n_a)
    M = psi.reshape(dim_a, dim_b)
    rho_A = jnp.matmul(M, jnp.conj(M.T))
    return purity_to_s2(rho_A)


def compute_s2_and_tmi_fast(psi, n_qubits):
    """
    Fast Stage 2 TMI + S2 calculator for N divisible by 4.
    Exploits pure-state complementarity (S_ABC = S_D) and contiguous reshapes
    to eliminate redundant 2^(N/2) x 2^(N/2) matrix multiplications.
    """
    q_quarter = n_qubits // 4
    dim_q = 1 << q_quarter        # 2^(N/4)
    dim_h = 1 << (2 * q_quarter)  # 2^(N/2)

    # Reshape state into 4 equal subsystem blocks: (A, B, C, D)
    psi_4 = psi.reshape(dim_q, dim_q, dim_q, dim_q)

    # 1. Contiguous 2-block cuts of size N/2: AB and BC
    # S_AB is also the exact half-chain entanglement entropy S_half!
    M_AB = psi_4.reshape(dim_h, dim_h)
    S_AB = purity_to_s2(jnp.matmul(M_AB, jnp.conj(M_AB.T)))

    # Roll axes so (B, C) are leading, (D, A) are trailing
    M_BC = jnp.transpose(psi_4, (1, 2, 3, 0)).reshape(dim_h, dim_h)
    S_BC = purity_to_s2(jnp.matmul(M_BC, jnp.conj(M_BC.T)))

    # Non-contiguous 2-block cut: AC vs BD
    M_AC = jnp.transpose(psi_4, (0, 2, 1, 3)).reshape(dim_h, dim_h)
    S_AC = purity_to_s2(jnp.matmul(M_AC, jnp.conj(M_AC.T)))

    # 2. Single-block cuts of size N/4: A, B, C, and D (where S_ABC = S_D)
    M_A = psi_4.reshape(dim_q, -1)
    S_A = purity_to_s2(jnp.matmul(M_A, jnp.conj(M_A.T)))

    M_B = jnp.transpose(psi_4, (1, 0, 2, 3)).reshape(dim_q, -1)
    S_B = purity_to_s2(jnp.matmul(M_B, jnp.conj(M_B.T)))

    M_C = jnp.transpose(psi_4, (2, 0, 1, 3)).reshape(dim_q, -1)
    S_C = purity_to_s2(jnp.matmul(M_C, jnp.conj(M_C.T)))

    M_D = jnp.transpose(psi_4, (3, 0, 1, 2)).reshape(dim_q, -1)
    S_ABC = purity_to_s2(jnp.matmul(M_D, jnp.conj(M_D.T)))

    tmi = S_A + S_B + S_C - S_AB - S_AC - S_BC + S_ABC
    return S_AB, tmi


# =============================================================================
# 4. SINGLE-TRAJECTORY MIPT EVALUATOR
# =============================================================================

def sample_circuit_inputs(key, n_qubits, n_layers, p_meas, ansatz):
    k_param, k_rot, k_mask, k_u = jax.random.split(key, 4)
    rots = jax.random.randint(k_rot, (n_layers, 2, n_qubits), minval=1, maxval=4)

    if ansatz == "HEA1":
        base_params = jax.random.uniform(
            k_param, (n_layers, 2, n_qubits), dtype=DTYPE_REAL, minval=-4 * jnp.pi, maxval=4 * jnp.pi
        )
    elif ansatz == "HEA2":
        shared = jax.random.uniform(
            k_param, (n_layers, 2, 1), dtype=DTYPE_REAL, minval=-4 * jnp.pi, maxval=4 * jnp.pi
        )
        base_params = jnp.broadcast_to(shared, (n_layers, 2, n_qubits))
    elif ansatz == "HVA":
        shared = jax.random.uniform(
            k_param, (n_layers, 4, 1), dtype=DTYPE_REAL, minval=-4 * jnp.pi, maxval=4 * jnp.pi
        )
        base_params = jnp.broadcast_to(shared, (n_layers, 4, n_qubits))

    # Measure after every layer except the final layer
    raw_mask = jax.random.uniform(k_mask, (n_layers - 1, n_qubits), dtype=DTYPE_REAL) < p_meas
    meas_mask = jnp.pad(raw_mask, ((0, 1), (0, 0)), constant_values=False)
    rand_u = jax.random.uniform(k_u, (n_layers, n_qubits), dtype=DTYPE_REAL)
    return base_params, rots, meas_mask, rand_u


def eval_mipt_sample(key, p_meas, s_strength, n_qubits, n_layers, ansatz, mode):
    params, rots, meas_mask, rand_u = sample_circuit_inputs(
        key, n_qubits, n_layers, p_meas, ansatz
    )
    psi = prepare_initial_psi(n_qubits, ansatz)

    def layer_body(p_state, xs):
        l_params, l_rots, m_mask, r_u = xs
        p_state = apply_ansatz_layer_psi(p_state, l_params, l_rots, n_qubits, ansatz)
        for q in range(n_qubits):
            p_state = generalized_measure_step(p_state, q, m_mask[q], r_u[q], s_strength, n_qubits)
        return p_state, None

    psi_final, _ = jax.lax.scan(layer_body, psi, (params, rots, meas_mask, rand_u))

    if mode == "tmi_fine" and (n_qubits % 4 == 0):
        s_half, tmi = compute_s2_and_tmi_fast(psi_final, n_qubits)
    else:
        s_half = compute_half_cut_s2(psi_final, n_qubits)
        tmi = jnp.array(jnp.nan, dtype=DTYPE_REAL)

    return jnp.stack([s_half, tmi])


# =============================================================================
# 5. MULTI-GPU BATCHING & MEMORY MANAGEMENT
# =============================================================================

def choose_batch_size(n_qubits, mode, n_devices=1):
    # TMI requires 3 matrices of size 2^(N/2) x 2^(N/2); S2_coarse requires only 1
    scale = 1 if mode == "tmi_fine" else 2
    if n_qubits <= 10: per_dev = 512 * scale
    elif n_qubits <= 12: per_dev = 256 * scale
    elif n_qubits <= 14: per_dev = 64 * scale
    elif n_qubits <= 16: per_dev = 16 * scale
    elif n_qubits <= 18: per_dev = 4 * scale
    else: per_dev = max(1, 1 * scale)
    return per_dev * max(n_devices, 1)


def build_multi_gpu_batch_fn(n_qubits, n_layers, ansatz, mode):
    n_devices = jax.local_device_count()
    single_fn = partial(
        eval_mipt_sample,
        n_qubits=n_qubits,
        n_layers=n_layers,
        ansatz=ansatz,
        mode=mode
    )
    vmapped = jax.vmap(single_fn, in_axes=(0, None, None))

    if n_devices > 1:
        pmapped = jax.pmap(vmapped, in_axes=(0, None, None))
        def batch_fn(keys, p_val, s_val):
            bs = keys.shape[0]
            per_dev = bs // n_devices
            keys_r = keys.reshape(n_devices, per_dev, 2)
            return pmapped(keys_r, DTYPE_REAL(p_val), DTYPE_REAL(s_val)).reshape(bs, 2)
        return batch_fn, n_devices
    else:
        jitted = jax.jit(vmapped)
        def batch_fn(keys, p_val, s_val):
            return jitted(keys, DTYPE_REAL(p_val), DTYPE_REAL(s_val))
        return batch_fn, 1


# =============================================================================
# 6. PLOTTING & METADATA UTILITIES
# =============================================================================

def write_config_txt(output_dir, args):
    config_path = os.path.join(output_dir, f"config_{args.mode}.txt")
    devices = jax.devices()
    precision_str = "float32 / complex64" if args.precision == 32 else "float64 / complex128"

    with open(config_path, "w") as f:
        f.write("====================================================================\n")
        f.write(f"   MIPT ENTANGLEMENT & TMI STUDY ({args.mode.upper()}) - CONFIG\n")
        f.write("====================================================================\n")
        f.write(f"Timestamp               : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Sweep Mode              : {args.mode}\n")
        f.write(f"Output Directory        : {output_dir}\n")
        f.write(f"Detected JAX Devices    : {devices}\n")
        f.write(f"Numerical Precision     : {precision_str}\n")
        f.write("--------------------------------------------------------------------\n")
        f.write("CIRCUIT & PHYSICAL HYPERPARAMETERS:\n")
        f.write(f"  Ansatzes Evaluated    : {args.ansatz_list}\n")
        f.write("  Boundary Conditions   : Periodic (Ring Topology)\n")
        f.write(f"  System Sizes (N)      : {args.n_list}\n")
        f.write(f"  Circuit Depth (L)     : L = {args.depth_mult}N\n")
        f.write(f"  Measurement Strengths : s = {args.s_list}\n")
        f.write(f"  Measurement Rates (p) : {args.p_list}\n")
        f.write(f"  Monte Carlo Samples   : {args.samples} per (Ansatz, N, s, p)\n")
        f.write(f"  Random Seed           : {args.seed}\n")
        f.write("====================================================================\n")


def generate_plots(payload, output_dir):
    N_list = np.asarray(payload["N_list"])
    p_list = np.asarray(payload["p_list"])
    ansatz_list = payload["ansatz_list"]
    s_list = payload.get("s_list", [1.0])
    mode = payload["mode"]
    s2_mean = payload["s2_mean"]
    s2_sem = payload["s2_sem"]
    tmi_mean = payload["tmi_mean"]
    tmi_sem = payload["tmi_sem"]

    n_cols = len(ansatz_list)
    cmap_N = plt.cm.viridis(np.linspace(0.1, 0.9, len(N_list)))

    for s_val in s_list:
        # 1. Plot Half-Chain Entropy S2 vs p
        fig1, axes1 = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5.2), squeeze=False)
        for col_idx, ansatz in enumerate(ansatz_list):
            ax = axes1[0, col_idx]
            for n_idx, N in enumerate(N_list):
                y = np.asarray([s2_mean[ansatz][s_val][float(p)][int(N)] for p in p_list])
                err = np.asarray([s2_sem[ansatz][s_val][float(p)][int(N)] for p in p_list])
                ax.errorbar(p_list, y, yerr=err, fmt="o-", color=cmap_N[n_idx],
                            lw=2.0, markersize=5, capsize=3, label=f"N = {N}")
            ax.set_title(f"{ansatz}: $S_2(N/2)$ vs. $p$ (s={s_val})", fontweight="bold")
            ax.set_xlabel("Measurement Rate $p$")
            if col_idx == 0:
                ax.set_ylabel("Half-Chain Entropy $\\langle S_2 \\rangle$")
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(loc="best", title="System Size N")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"mipt_{mode}_S2_vs_p_s{s_val}.png"), dpi=300)
        plt.close(fig1)

        # 2. Plot TMI vs p (if tmi_fine mode)
        if mode == "tmi_fine":
            fig2, axes2 = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5.2), squeeze=False)
            for col_idx, ansatz in enumerate(ansatz_list):
                ax = axes2[0, col_idx]
                for n_idx, N in enumerate(N_list):
                    if N % 4 != 0:
                        continue
                    y = np.asarray([tmi_mean[ansatz][s_val][float(p)][int(N)] for p in p_list])
                    err = np.asarray([tmi_sem[ansatz][s_val][float(p)][int(N)] for p in p_list])
                    ax.errorbar(p_list, y, yerr=err, fmt="s-", color=cmap_N[n_idx],
                                lw=2.0, markersize=5, capsize=3, label=f"N = {N}")
                ax.set_title(f"{ansatz}: Tripartite Mutual Info $I_3$ vs. $p$ (s={s_val})", fontweight="bold")
                ax.set_xlabel("Measurement Rate $p$")
                if col_idx == 0:
                    ax.set_ylabel("$\\langle I_3(A:B:C) \\rangle$")
                ax.grid(True, linestyle="--", alpha=0.6)
                ax.legend(loc="best", title="System Size N")

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"mipt_{mode}_TMI_vs_p_s{s_val}.png"), dpi=300)
            plt.close(fig2)


# =============================================================================
# 7. BENCHMARK & MAIN CLUSTER SWEEP
# =============================================================================

def run_benchmark(args):
    N_list = args.n_list
    p_list = args.p_list
    s_list = args.s_list
    ansatz_list = args.ansatz_list
    n_devices = jax.local_device_count()
    p_test = 0.15
    s_test = s_list[0]

    print("=" * 96)
    print(f"MIPT BENCHMARK | Mode: {args.mode} | Depth: L={args.depth_mult}N | s_list={s_list} | Samples={args.samples}")
    print(f"Devices ({n_devices}): {jax.devices()} | Precision: {DTYPE_REAL.__name__}")
    print("-" * 96)
    total_pts = len(p_list) * len(s_list)
    print(f"{'N':<4} | {'Ansatz':<6} | {'Batch':<6} | {'Compile (s)':<12} | {'ms / Sample':<13} | "
          f"{'Samples / s':<13} | {'Est. Time (' + str(total_pts) + ' pts)':<22}")
    print("-" * 96)

    base_key = jax.random.PRNGKey(42)
    total_estimated_sec = 0.0

    for N in N_list:
        if args.mode == "tmi_fine" and N % 4 != 0:
            print(f"[Skip] N={N} is not divisible by 4 (required for equal-quarter TMI).")
            continue

        n_layers = args.depth_mult * N
        bs = min(choose_batch_size(N, args.mode, n_devices), args.samples)
        if n_devices > 1 and bs % n_devices != 0:
            bs = max(n_devices, (bs // n_devices) * n_devices)

        for ansatz in ansatz_list:
            batch_fn, _ = build_multi_gpu_batch_fn(N, n_layers, ansatz, args.mode)
            keys = jax.random.split(base_key, bs)

            t_c0 = time.perf_counter()
            _ = batch_fn(keys, p_test, s_test).block_until_ready()
            compile_sec = time.perf_counter() - t_c0

            iters = 3
            t_e0 = time.perf_counter()
            for _ in range(iters):
                _ = batch_fn(keys, p_test, s_test).block_until_ready()
            exec_sec = time.perf_counter() - t_e0

            total_bench_samples = bs * iters
            ms_per_sample = (exec_sec / total_bench_samples) * 1000.0
            samples_per_sec = total_bench_samples / exec_sec

            est_sec = compile_sec + total_pts * (args.samples / samples_per_sec)
            total_estimated_sec += est_sec

            print(f"{N:<4} | {ansatz:<6} | {bs:<6} | {compile_sec:<12.2f} | {ms_per_sample:<13.4f} | "
                  f"{samples_per_sec:<13,.1f} | {est_sec / 60.0:6.2f} min ({est_sec:.0f}s)", flush=True)

            if hasattr(jax, "clear_caches"):
                jax.clear_caches()
            gc.collect()

    print("-" * 96)
    print(f"TOTAL ESTIMATED COMPLETION TIME: {total_estimated_sec / 60.0:.1f} minutes ({total_estimated_sec / 3600.0:.2f} hours)")
    print("=" * 96)


def run_cluster_sweep(args):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    write_config_txt(output_dir, args)

    ansatz_list = args.ansatz_list
    N_list = [int(n) for n in args.n_list if not (args.mode == "tmi_fine" and int(n) % 4 != 0)]
    p_list = [float(np.round(p, 5)) for p in args.p_list]
    s_list = [float(s) for s in args.s_list]

    raw_s2 = {a: {s: {p: {} for p in p_list} for s in s_list} for a in ansatz_list}
    s2_mean = {a: {s: {p: {} for p in p_list} for s in s_list} for a in ansatz_list}
    s2_sem = {a: {s: {p: {} for p in p_list} for s in s_list} for a in ansatz_list}

    raw_tmi = {a: {s: {p: {} for p in p_list} for s in s_list} for a in ansatz_list}
    tmi_mean = {a: {s: {p: {} for p in p_list} for s in s_list} for a in ansatz_list}
    tmi_sem = {a: {s: {p: {} for p in p_list} for s in s_list} for a in ansatz_list}

    master_key = jax.random.PRNGKey(args.seed)
    pkl_path = os.path.join(output_dir, f"mipt_{args.mode}_results.pkl")
    temp_pkl_path = os.path.join(output_dir, f"mipt_{args.mode}_results_TEMP.pkl")
    n_devices = jax.local_device_count()

    for n_idx, N in enumerate(N_list):
        t_start_N = time.perf_counter()
        n_layers = args.depth_mult * N
        bs = choose_batch_size(N, args.mode, n_devices=n_devices)

        for a_idx, ansatz in enumerate(ansatz_list):
            batch_fn, active_devs = build_multi_gpu_batch_fn(N, n_layers, ansatz, args.mode)

            for s_idx, s_val in enumerate(s_list):
                for p_idx, p_val in enumerate(p_list):
                    grid_key = jax.random.fold_in(
                        jax.random.fold_in(
                            jax.random.fold_in(
                                jax.random.fold_in(master_key, n_idx), a_idx
                            ), p_idx
                        ), s_idx
                    )

                    all_batches = []
                    rem, chunk = args.samples, 0
                    while rem > 0:
                        cur_bs = min(bs, rem)
                        if active_devs > 1 and cur_bs % active_devs != 0:
                            cur_bs = max(active_devs, ((cur_bs + active_devs - 1) // active_devs) * active_devs)

                        keys = jax.random.split(jax.random.fold_in(grid_key, chunk), cur_bs)
                        out = np.asarray(batch_fn(keys, p_val, s_val))
                        all_batches.append(out)
                        rem -= cur_bs
                        chunk += 1

                    arr = np.concatenate(all_batches, axis=0)[:args.samples]
                    s2_samples = arr[:, 0]
                    tmi_samples = arr[:, 1]

                    raw_s2[ansatz][s_val][p_val][N] = s2_samples
                    s2_mean[ansatz][s_val][p_val][N] = float(np.mean(s2_samples))
                    s2_sem[ansatz][s_val][p_val][N] = float(np.std(s2_samples, ddof=1) / np.sqrt(len(s2_samples)))

                    if args.mode == "tmi_fine":
                        raw_tmi[ansatz][s_val][p_val][N] = tmi_samples
                        tmi_mean[ansatz][s_val][p_val][N] = float(np.mean(tmi_samples))
                        tmi_sem[ansatz][s_val][p_val][N] = float(np.std(tmi_samples, ddof=1) / np.sqrt(len(tmi_samples)))

            if hasattr(jax, "clear_caches"):
                jax.clear_caches()
            gc.collect()

        payload = {
            "mode": args.mode,
            "N_list": N_list,
            "p_list": p_list,
            "s_list": s_list,
            "ansatz_list": ansatz_list,
            "depth_mult": args.depth_mult,
            "samples": args.samples,
            "seed": args.seed,
            "s2_mean": s2_mean,
            "s2_sem": s2_sem,
            "tmi_mean": tmi_mean,
            "tmi_sem": tmi_sem,
            "raw_s2": raw_s2,
            "raw_tmi": raw_tmi if args.mode == "tmi_fine" else None,
        }
        with open(temp_pkl_path, "wb") as f:
            pickle.dump(payload, f)
        os.replace(temp_pkl_path, pkl_path)

        dt_N = time.perf_counter() - t_start_N
        print(f"[Done] N={N} (L={n_layers}, mode={args.mode}) completed in {dt_N:.1f}s -> saved to {pkl_path}", flush=True)

    generate_plots(payload, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["s2_coarse", "tmi_fine"], default="s2_coarse",
                        help="Stage 1: 's2_coarse' (S2 only, all even N). Stage 2: 'tmi_fine' (S2 + TMI, N % 4 == 0).")
    parser.add_argument("--output-dir", type=str, default="data_mipt")
    parser.add_argument("--ansatz-list", type=str, nargs="+", default=["HEA1", "HEA2", "HVA"])
    parser.add_argument("--n-list", type=int, nargs="+", default=[8, 10, 12, 14, 16, 18, 20])
    parser.add_argument("--depth-mult", type=int, default=4, help="Circuit depth L = depth_mult * N (2 or 4 recommended for MIPT)")
    parser.add_argument("--s-list", type=float, nargs="+", default=[1.0, 0.5], help="List of measurement strengths (1.0 = projective collapse)")
    parser.add_argument(
        "--p-list", type=float, nargs="+",
        default=np.round(np.arange(0.0, 0.96, 0.025), 3).tolist()
    )
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--precision", type=int, choices=[32, 64], default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benchmark", action="store_true", help="Run speed benchmark & estimate total runtime")
    parser.add_argument("--plot-only", type=str, default="", help="Path to saved .pkl to regenerate plots")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
    configure_precision(use_32bit=(args.precision == 32))

    if args.benchmark:
        run_benchmark(args)
    elif args.plot_only:
        with open(args.plot_only, "rb") as f:
            saved_payload = pickle.load(f)
        generate_plots(saved_payload, args.output_dir)
    else:
        run_cluster_sweep(args)