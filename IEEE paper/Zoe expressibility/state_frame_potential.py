"""
Cluster-Ready Pure-JAX Weak-Measurement (s in [0.2, 1.0]) Expressibility Study
for Monitored Quantum Circuits (HEA1, HEA2, HVA).

Features:
  - Generalized weak-measurement operators P_+(s), P_-(s) via SPMM
  - Exact U-statistic Gram-matrix evaluation of Frame Potential F_U^(rho) = E[|<psi_1|psi_2>|^4]
  - Tiled GPU Gram-matrix contraction to prevent VRAM OOM at large N (N=16, 18)
  - Direct in-code dictionary `S_TO_P_SCHEDULE` supporting per-ansatz (p_min, p_max, p_step) bounds
  - Pre-import single-GPU masking (default) with --multi-gpu toggle
  - Resumable / appendable checkpointing in data_milt/expressibility_weak_s_results.pkl
"""

import os
import sys
import gc
import time
import pickle
import datetime
import argparse
from functools import partial

# --- MUST RUN BEFORE `import jax` ---
if "--multi-gpu" not in sys.argv:
    gpu_id = "0"
    if "--gpu-id" in sys.argv:
        idx = sys.argv.index("--gpu-id")
        if idx + 1 < len(sys.argv):
            gpu_id = sys.argv[idx + 1]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
import jax
import jax.numpy as jnp


# =============================================================================
# 0. PRECISION & ENVIRONMENT SETUP
# =============================================================================

DTYPE_REAL = jnp.float64
DTYPE_COMPLEX = jnp.complex128
NP_COMPLEX = np.complex128

def configure_precision(use_32bit=False):
    global DTYPE_REAL, DTYPE_COMPLEX, NP_COMPLEX
    if use_32bit:
        jax.config.update("jax_enable_x64", False)
        DTYPE_REAL = jnp.float32
        DTYPE_COMPLEX = jnp.complex64
        NP_COMPLEX = np.complex64
    else:
        jax.config.update("jax_enable_x64", True)
        DTYPE_REAL = jnp.float64
        DTYPE_COMPLEX = jnp.complex128
        NP_COMPLEX = np.complex128


# =============================================================================
# 1. FAST 3-AXIS BIT-SHIFT QUANTUM KERNELS (ZERO TENSORDOT / MOVEAXIS)
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
# 2. PERIODIC ANSATZ LAYERS & WEAK MEASUREMENT STEP (GENERAL s)
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


def spmm_measure_step_normalized(psi, q, mask_val, rand_u, s_strength, n_qubits):
    denom = jnp.sqrt(2.0 * (1.0 + s_strength ** 2))
    P_plus_diag = jnp.array([(1.0 + s_strength) / denom, (1.0 - s_strength) / denom], dtype=DTYPE_COMPLEX)
    P_minus_diag = jnp.array([(1.0 - s_strength) / denom, (1.0 + s_strength) / denom], dtype=DTYPE_COMPLEX)

    P_plus_sq = jnp.real(P_plus_diag) ** 2
    P_minus_sq = jnp.real(P_minus_diag) ** 2

    dim_L = 1 << q
    dim_R = 1 << (n_qubits - q - 1)
    psi_r = psi.reshape((dim_L, 2, dim_R))

    prob_0 = jnp.sum(jnp.abs(psi_r[:, 0, :]) ** 2)
    prob_1 = jnp.sum(jnp.abs(psi_r[:, 1, :]) ** 2)

    p_plus = prob_0 * P_plus_sq[0] + prob_1 * P_plus_sq[1]
    p_minus = prob_0 * P_minus_sq[0] + prob_1 * P_minus_sq[1]
    total_p = jnp.maximum(p_plus + p_minus, 1e-30)

    outcome_is_plus = rand_u < (p_plus / total_p)
    chosen_P_diag = jnp.where(outcome_is_plus, P_plus_diag, P_minus_diag)
    p_chosen = jnp.where(outcome_is_plus, p_plus, p_minus)
    p_chosen_safe = jnp.maximum(p_chosen, 1e-30)

    scaled_diag = chosen_P_diag / jnp.sqrt(p_chosen_safe)
    ident_diag = jnp.ones(2, dtype=DTYPE_COMPLEX)
    applied_diag = jnp.where(mask_val, scaled_diag, ident_diag)

    psi_out = jnp.stack([psi_r[:, 0, :] * applied_diag[0],
                         psi_r[:, 1, :] * applied_diag[1]], axis=1).reshape(-1)
    return jnp.where(mask_val, psi_out, psi)


# =============================================================================
# 3. TRAJECTORY SAMPLER & TILED GRAM-MATRIX EXPRESSIBILITY ENGINE
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

    raw_mask = jax.random.uniform(k_mask, (n_layers - 1, n_qubits), dtype=DTYPE_REAL) < p_meas
    meas_mask = jnp.pad(raw_mask, ((0, 1), (0, 0)), constant_values=False)
    rand_u = jax.random.uniform(k_u, (n_layers, n_qubits), dtype=DTYPE_REAL)
    return base_params, rots, meas_mask, rand_u


def eval_final_psi_sample(key, s_strength, p_meas, n_qubits, n_layers, ansatz):
    params, rots, meas_mask, rand_u = sample_circuit_inputs(
        key, n_qubits, n_layers, p_meas, ansatz
    )
    psi = prepare_initial_psi(n_qubits, ansatz)

    def layer_body(p_state, xs):
        l_params, l_rots, m_mask, r_u = xs
        p_state = apply_ansatz_layer_psi(p_state, l_params, l_rots, n_qubits, ansatz)
        for q in range(n_qubits):
            p_state = spmm_measure_step_normalized(
                p_state, q, m_mask[q], r_u[q], s_strength, n_qubits
            )
        return p_state, None

    psi_final, _ = jax.lax.scan(layer_body, psi, (params, rots, meas_mask, rand_u))
    # Final safety normalization against float32 drift across deep circuits
    norm = jnp.sqrt(jnp.maximum(jnp.real(jnp.vdot(psi_final, psi_final)), 1e-30))
    return psi_final / norm


def haar_frame_potential_pure(n_qubits):
    """Exact Haar frame potential F_Haar^(rho) = 1 / ((2^N + 1) * 2^(N - 1)) for pure states."""
    d = float(1 << n_qubits)
    return 2.0 / (d * (d + 1.0))


@partial(jax.jit, donate_argnums=())
def _gram_diag_tile_sum(block_i):
    """Sums |<psi_a | psi_b>|^4 over all a != b within a single diagonal block."""
    G = jnp.matmul(jnp.conj(block_i), jnp.transpose(block_i))
    G4 = (jnp.real(G) ** 2 + jnp.imag(G) ** 2) ** 2
    n = block_i.shape[0]
    mask = 1.0 - jnp.eye(n, dtype=G4.dtype)
    return jnp.sum(G4 * mask, dtype=DTYPE_REAL)


@partial(jax.jit, donate_argnums=())
def _gram_offdiag_tile_sum(block_i, block_j):
    """Sums 2 * |<psi_a | psi_b>|^4 over all pairs between two disjoint blocks."""
    G = jnp.matmul(jnp.conj(block_i), jnp.transpose(block_j))
    G4 = (jnp.real(G) ** 2 + jnp.imag(G) ** 2) ** 2
    return 2.0 * jnp.sum(G4, dtype=DTYPE_REAL)


def compute_frame_potential_gram_tiled(psi_host_matrix, n_qubits):
    """
    Computes exact U-statistic estimator of F_U^(rho) = (2 / (K(K-1))) sum_{i < j} |<psi_i | psi_j>|^4
    using GPU-accelerated tiles so large N (e.g. N=18, dim=262144) never triggers VRAM OOM.
    """
    K = psi_host_matrix.shape[0]
    if K < 2:
        raise ValueError("At least 2 samples are required to compute the frame potential.")

    if n_qubits <= 12:
        tile_size = K
    elif n_qubits <= 14:
        tile_size = min(K, 1024)
    elif n_qubits <= 16:
        tile_size = min(K, 512)
    else:
        tile_size = min(K, 256)

    dev0 = jax.local_devices()[0]
    total_sum = 0.0

    for i in range(0, K, tile_size):
        b_i = jax.device_put(psi_host_matrix[i:i + tile_size], dev0)
        total_sum += float(_gram_diag_tile_sum(b_i))
        for j in range(i + tile_size, K, tile_size):
            b_j = jax.device_put(psi_host_matrix[j:j + tile_size], dev0)
            total_sum += float(_gram_offdiag_tile_sum(b_i, b_j))

    f_u = total_sum / (float(K) * float(K - 1))
    f_haar = haar_frame_potential_pure(n_qubits)
    delta_f = f_u - f_haar
    epsilon_rho = float(np.sqrt(max(0.0, delta_f)))
    return float(f_u), float(f_haar), float(delta_f), epsilon_rho


# =============================================================================
# 4. SINGLE/MULTI-GPU BATCHING & CONFIG WRITER
# =============================================================================

def get_circuit_depth(N, args):
    if args.depth_mode == "fixed":
        return int(args.fixed_l)
    else:
        return int(args.linear_mult * N)


def choose_batch_size(n_qubits, n_devices=1):
    # Without JVP tangents, memory usage per sample is halved compared to gradient runs
    if n_qubits <= 10: per_dev = 1024
    elif n_qubits <= 12: per_dev = 512
    elif n_qubits <= 14: per_dev = 128
    elif n_qubits <= 16: per_dev = 32
    else: per_dev = 8
    return per_dev * max(n_devices, 1)


def build_multi_gpu_state_batch_fn(n_qubits, n_layers, ansatz, use_multi_gpu=False):
    n_devices = jax.local_device_count() if use_multi_gpu else 1
    dim = 1 << n_qubits
    single_fn = partial(
        eval_final_psi_sample,
        n_qubits=n_qubits,
        n_layers=n_layers,
        ansatz=ansatz
    )
    vmapped = jax.vmap(single_fn, in_axes=(0, None, None))

    if use_multi_gpu and n_devices > 1:
        pmapped = jax.pmap(vmapped, in_axes=(0, None, None))
        def batch_fn(keys, s_val, p_val):
            bs = keys.shape[0]
            per_dev = bs // n_devices
            keys_r = keys.reshape(n_devices, per_dev, 2)
            return pmapped(keys_r, DTYPE_REAL(s_val), DTYPE_REAL(p_val)).reshape(bs, dim)
        return batch_fn, n_devices
    else:
        dev0 = jax.local_devices()[0]
        jitted = jax.jit(vmapped, device=dev0)
        def batch_fn(keys, s_val, p_val):
            return jitted(keys, DTYPE_REAL(s_val), DTYPE_REAL(p_val))
        return batch_fn, 1


def resolve_p_list_for_s(s_val, s_to_p_schedule, ansatz="HEA1"):
    spec = s_to_p_schedule[s_val]
    if isinstance(spec, dict):
        spec = spec.get(ansatz, next(iter(spec.values())))
    if isinstance(spec, tuple) and len(spec) == 3:
        p_min, p_max, p_step = spec
        n_steps = int(round((p_max - p_min) / p_step))
        p_arr = [float(np.round(p_min + k * p_step, 5)) for k in range(n_steps + 1)]
        return [p for p in p_arr if p <= p_max + 1e-7]
    return [float(np.round(p, 5)) for p in spec]


def write_config_txt(output_dir, args, s_list, s_to_p_schedule):
    config_path = os.path.join(output_dir, "config_expressibility_weak_s.txt")
    devices = jax.devices()
    precision_str = "float32 / complex64" if args.precision == 32 else "float64 / complex128"
    depth_desc = (
        f"Fixed L = {args.fixed_l}" if args.depth_mode == "fixed"
        else f"Linear L = {args.linear_mult}*N"
    )
    gpu_mode_str = "Single-GPU" if args.single_gpu else "Multi-GPU (pmap)"

    with open(config_path, "w") as f:
        f.write("====================================================================\n")
        f.write("   WEAK-MEASUREMENT (s, p) EXPRESSIBILITY SCALING STUDY - CONFIG    \n")
        f.write("====================================================================\n")
        f.write(f"Timestamp               : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output Directory        : {output_dir}\n")
        f.write(f"Detected JAX Devices    : {devices} ({gpu_mode_str})\n")
        f.write(f"Numerical Precision     : {precision_str}\n")
        f.write("--------------------------------------------------------------------\n")
        f.write("CIRCUIT & PHYSICAL HYPERPARAMETERS:\n")
        f.write("  Metric Evaluated      : Pure-State Frame Potential F_U^(rho) & Expressibility eps_U^(rho)\n")
        f.write("  Estimator             : Unbiased Off-Diagonal Gram Matrix U-Statistic (K*(K-1)/2 pairs)\n")
        f.write("  Ansatzes Evaluated    : " + str(args.ansatz_list) + "\n")
        f.write("  Boundary Conditions   : Periodic (Ring Topology)\n")
        f.write("  System Sizes (N)      : " + str(args.n_list) + "\n")
        f.write(f"  Circuit Depth (L)     : {depth_desc}\n")
        f.write(f"  Monte Carlo Traj (K)  : {args.samples} states ({args.samples * (args.samples - 1) // 2} pairs) per (s, Ansatz, N, p)\n")
        f.write(f"  Random Seed           : {args.seed}\n")
        f.write("--------------------------------------------------------------------\n")
        f.write("MEASUREMENT STRENGTH (s) TO PROBABILITY (p) SCHEDULE:\n")
        for s_val in s_list:
            f.write(f"  s = {s_val:.2f}:\n")
            for a in args.ansatz_list:
                p_vals = resolve_p_list_for_s(s_val, s_to_p_schedule, a)
                f.write(f"    {a} ({len(p_vals):2d} pts) : {p_vals}\n")
        f.write("====================================================================\n")


# =============================================================================
# 5. RESUMABLE / APPENDABLE MAIN CLUSTER SWEEP
# =============================================================================

def run_cluster_sweep(args, s_list, s_to_p_schedule):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    write_config_txt(output_dir, args, s_list, s_to_p_schedule)

    target_ansatzes = args.ansatz_list
    N_list_req = [int(n) for n in args.n_list]
    s_list_req = [float(np.round(s, 4)) for s in s_list]
    depth_desc = f"Fixed L={args.fixed_l}" if args.depth_mode == "fixed" else f"Linear L={args.linear_mult}N"

    pkl_path = os.path.join(output_dir, "expressibility_weak_s_results.pkl")
    temp_pkl_path = os.path.join(output_dir, "expressibility_weak_s_results_TEMP.pkl")

    all_ansatzes = ["HEA1", "HEA2", "HVA"]
    if os.path.exists(pkl_path) and not args.overwrite:
        with open(pkl_path, "rb") as f:
            existing = pickle.load(f)
        frame_potentials = existing.get("frame_potentials", {})
        delta_frame_potentials = existing.get("delta_frame_potentials", {})
        expressibilities = existing.get("expressibilities", {})
        N_list_all = sorted(list(set(existing.get("N_list", [])) | set(N_list_req)))
        s_list_all = sorted(list(set(existing.get("s_list", [])) | set(s_list_req)), reverse=True)
        print(f"[Resume/Append] Loaded existing checkpoint from {pkl_path}", flush=True)
    else:
        frame_potentials = {}
        delta_frame_potentials = {}
        expressibilities = {}
        N_list_all = sorted(N_list_req)
        s_list_all = sorted(s_list_req, reverse=True)

    for s_val in s_list_all:
        for store in (frame_potentials, delta_frame_potentials, expressibilities):
            if s_val not in store:
                store[s_val] = {}
            for a in all_ansatzes:
                if a not in store[s_val]:
                    store[s_val][a] = {}

    master_key = jax.random.PRNGKey(args.seed)
    use_multi_gpu = not args.single_gpu
    n_devices = jax.local_device_count() if use_multi_gpu else 1

    for N in N_list_req:
        t_start_N = time.perf_counter()
        n_layers = get_circuit_depth(N, args)
        dim = 1 << N
        bs = choose_batch_size(N, n_devices=n_devices)
        computed_any_for_N = False

        # Pre-allocate reusable host buffer for K state vectors of dimension 2^N
        psi_host_buf = np.empty((args.samples, dim), dtype=NP_COMPLEX)

        for ansatz in target_ansatzes:
            a_idx = all_ansatzes.index(ansatz)

            tasks_sp = []
            for s_val in s_list_req:
                p_list_s = resolve_p_list_for_s(s_val, s_to_p_schedule, ansatz)
                for p_val in p_list_s:
                    for store in (frame_potentials, delta_frame_potentials, expressibilities):
                        if p_val not in store[s_val][ansatz]:
                            store[s_val][ansatz][p_val] = {}
                    if (N not in expressibilities[s_val][ansatz][p_val]) or args.overwrite:
                        tasks_sp.append((s_val, p_val))

            if not tasks_sp:
                continue

            computed_any_for_N = True
            batch_fn, active_devs = build_multi_gpu_state_batch_fn(
                N, n_layers, ansatz, use_multi_gpu=use_multi_gpu
            )

            for s_val, p_val in tasks_sp:
                s_int = int(round(s_val * 1000))
                p_int = int(round(p_val * 10000))
                grid_key = jax.random.fold_in(
                    jax.random.fold_in(
                        jax.random.fold_in(jax.random.fold_in(master_key, N), a_idx),
                        s_int
                    ),
                    p_int
                )

                filled, chunk = 0, 0
                while filled < args.samples:
                    rem = args.samples - filled
                    cur_bs = min(bs, rem)
                    if active_devs > 1 and cur_bs % active_devs != 0:
                        cur_bs = max(active_devs, ((cur_bs + active_devs - 1) // active_devs) * active_devs)

                    keys = jax.random.split(jax.random.fold_in(grid_key, chunk), cur_bs)
                    out = np.asarray(batch_fn(keys, s_val, p_val))
                    take = min(cur_bs, rem)
                    psi_host_buf[filled:filled + take] = out[:take]
                    filled += take
                    chunk += 1

                f_u, _, delta_f, eps_rho = compute_frame_potential_gram_tiled(psi_host_buf, N)
                frame_potentials[s_val][ansatz][p_val][N] = f_u
                delta_frame_potentials[s_val][ansatz][p_val][N] = delta_f
                expressibilities[s_val][ansatz][p_val][N] = eps_rho

            if hasattr(jax, "clear_caches"):
                jax.clear_caches()
            gc.collect()

        del psi_host_buf
        gc.collect()

        payload = {
            "N_list": N_list_all,
            "s_list": s_list_all,
            "s_to_p_schedule": {
                s: {a: resolve_p_list_for_s(s, s_to_p_schedule, a) for a in all_ansatzes}
                for s in s_list_req
            },
            "ansatz_list": all_ansatzes,
            "depth_mode": args.depth_mode,
            "depth_desc": depth_desc,
            "fixed_l": args.fixed_l,
            "linear_mult": args.linear_mult,
            "samples": args.samples,
            "seed": args.seed,
            "haar_frame_potentials": {n: haar_frame_potential_pure(n) for n in N_list_all},
            "frame_potentials": frame_potentials,
            "delta_frame_potentials": delta_frame_potentials,
            "expressibilities": expressibilities,
        }
        with open(temp_pkl_path, "wb") as f:
            pickle.dump(payload, f)
        os.replace(temp_pkl_path, pkl_path)

        dt_N = time.perf_counter() - t_start_N
        status_str = "computed & merged" if computed_any_for_N else "already complete (skipped)"
        print(f"[Done] N={N} (L={n_layers}) {status_str} in {dt_N:.1f}s -> saved {pkl_path}", flush=True)


if __name__ == "__main__":
    # S_LIST = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    S_LIST = [1.0, 0.5]
    P_STEP = 0.05

    S_TO_P_SCHEDULE = {
        1.0: {"HEA1": (0.00, 0.75, P_STEP), "HEA2": (0.00, 0.75, P_STEP), "HVA": (0.00, 0.75, P_STEP)},
        0.5: {"HEA1": (0.00, 0.95, P_STEP), "HEA2": (0.00, 0.95, P_STEP), "HVA": (0.00, 0.95, P_STEP)},
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="data_milt")
    parser.add_argument(
        "--ansatz-list", type=str, nargs="+", default=["HEA1", "HEA2", "HVA"],
        choices=["HEA1", "HEA2", "HVA"]
    )
    parser.add_argument("--n-list", type=int, nargs="+", default=[6, 8, 10, 12, 14, 16, 18])
    parser.add_argument("--depth-mode", type=str, choices=["fixed", "linear"], default="fixed")
    parser.add_argument("--fixed-l", type=int, default=18)
    parser.add_argument("--linear-mult", type=int, default=2)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--precision", type=int, choices=[32, 64], default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--single-gpu", action="store_true", default=True)
    parser.add_argument("--multi-gpu", action="store_false", dest="single_gpu")
    parser.add_argument("--gpu-id", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    configure_precision(use_32bit=(args.precision == 32))
    run_cluster_sweep(args, S_LIST, S_TO_P_SCHEDULE)