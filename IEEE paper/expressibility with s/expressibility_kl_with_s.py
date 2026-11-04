"""
Step 1: Production Expressibility Data Generation & Raw Curve Plotting
- Fused HVA bonds & JAX-native parallel pairing (K=8)[cite: 2].
- Full SPMM generalized measurement support (arbitrary s_strength in [0, 1])[cite: 2].
- Dense p-grid for smooth transition curves[cite: 2].
- Incremental per-N checkpointing/resume support.
- Generates D_KL vs p and log10(D_KL) vs p plots[cite: 2].
"""

import os
import sys
import gc
import time
import pickle
import datetime
import argparse
import warnings
from functools import partial
warnings.filterwarnings("ignore")

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DTYPE_REAL = jnp.float32
DTYPE_COMPLEX = jnp.complex64

def configure_precision(use_32bit=True):
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
# QUANTUM KERNELS & ANSATZES
# =============================================================================

def initial_state_zero(n_qubits):
    dim = 1 << n_qubits
    return jnp.zeros(dim, dtype=DTYPE_COMPLEX).at[0].set(1.0)

def hadamard_matrix():
    inv_sqrt2 = jnp.sqrt(jnp.array(0.5, dtype=DTYPE_REAL))
    return jnp.array([[inv_sqrt2, inv_sqrt2], [inv_sqrt2, -inv_sqrt2]], dtype=DTYPE_COMPLEX)

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

def apply_xxz_bond_adjacent_psi(th_zz, th_xy, q, psi, n_qubits):
    dim_L = 1 << q
    dim_R = 1 << (n_qubits - q - 2)
    psi_r = psi.reshape(dim_L, 4, dim_R)
    e_m = jnp.exp(-0.5j * th_zz).astype(DTYPE_COMPLEX)
    e_p = jnp.exp(0.5j * th_zz).astype(DTYPE_COMPLEX)
    c = (e_p * jnp.cos(th_xy)).astype(DTYPE_COMPLEX)
    is_ = (-1j * e_p * jnp.sin(th_xy)).astype(DTYPE_COMPLEX)
    s0, s1, s2, s3 = psi_r[:, 0, :], psi_r[:, 1, :], psi_r[:, 2, :], psi_r[:, 3, :]
    n0 = e_m * s0
    n1 = c * s1 + is_ * s2
    n2 = is_ * s1 + c * s2
    n3 = e_m * s3
    return jnp.stack([n0, n1, n2, n3], axis=1).reshape(-1)

def apply_xxz_bond_edges_psi(th_zz, th_xy, psi, n_qubits):
    dim_mid = 1 << (n_qubits - 2)
    psi_r = psi.reshape(2, dim_mid, 2)
    e_m = jnp.exp(-0.5j * th_zz).astype(DTYPE_COMPLEX)
    e_p = jnp.exp(0.5j * th_zz).astype(DTYPE_COMPLEX)
    c = (e_p * jnp.cos(th_xy)).astype(DTYPE_COMPLEX)
    is_ = (-1j * e_p * jnp.sin(th_xy)).astype(DTYPE_COMPLEX)
    s00, s01, s10, s11 = psi_r[0, :, 0], psi_r[0, :, 1], psi_r[1, :, 0], psi_r[1, :, 1]
    n00 = e_m * s00
    n01 = c * s01 + is_ * s10
    n10 = is_ * s01 + c * s10
    n11 = e_m * s11
    p0 = jnp.stack([n00, n01], axis=-1)
    p1 = jnp.stack([n10, n11], axis=-1)
    return jnp.stack([p0, p1], axis=0).reshape(-1)

def prepare_initial_psi(n_qubits, ansatz):
    dim = 1 << n_qubits
    if ansatz == "HEA1":
        return jnp.zeros(dim, dtype=DTYPE_COMPLEX).at[dim - 1].set(1.0)
    elif ansatz == "HEA2":
        val = jnp.power(jnp.array(0.5, dtype=DTYPE_REAL), 0.5 * n_qubits).astype(DTYPE_COMPLEX)
        return jnp.full(dim, val, dtype=DTYPE_COMPLEX)
    elif ansatz == "HVA":
        psi = initial_state_zero(n_qubits)
        H = hadamard_matrix()
        X = pauli_x_matrix()
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
            psi = apply_xxz_bond_adjacent_psi(th_odd_zz, th_odd_xy, q, psi, n_qubits)
        psi = apply_xxz_bond_edges_psi(th_odd_zz, th_odd_xy, psi, n_qubits)
        for q in range(0, n_qubits - 1, 2):
            psi = apply_xxz_bond_adjacent_psi(th_even_zz, th_even_xy, q, psi, n_qubits)
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

    return jnp.stack([psi_r[:, 0, :] * applied_diag[0],
                      psi_r[:, 1, :] * applied_diag[1]], axis=1).reshape(-1)

def sample_circuit_inputs(key, n_qubits, n_layers, p_meas, ansatz):
    k_param, k_rot, k_mask, k_u = jax.random.split(key, 4)
    rots = jax.random.randint(k_rot, (n_layers, 2, n_qubits), minval=1, maxval=4)
    if ansatz == "HEA1":
        base_params = jax.random.uniform(k_param, (n_layers, 2, n_qubits), dtype=DTYPE_REAL, minval=-4 * jnp.pi, maxval=4 * jnp.pi)
    elif ansatz == "HEA2":
        shared = jax.random.uniform(k_param, (n_layers, 2, 1), dtype=DTYPE_REAL, minval=-4 * jnp.pi, maxval=4 * jnp.pi)
        base_params = jnp.broadcast_to(shared, (n_layers, 2, n_qubits))
    elif ansatz == "HVA":
        shared = jax.random.uniform(k_param, (n_layers, 4, 1), dtype=DTYPE_REAL, minval=-4 * jnp.pi, maxval=4 * jnp.pi)
        base_params = jnp.broadcast_to(shared, (n_layers, 4, n_qubits))
    raw_mask = jax.random.uniform(k_mask, (n_layers - 1, n_qubits), dtype=DTYPE_REAL) < p_meas
    meas_mask = jnp.pad(raw_mask, ((0, 1), (0, 0)), constant_values=False)
    rand_u = jax.random.uniform(k_u, (n_layers, n_qubits), dtype=DTYPE_REAL)
    return base_params, rots, meas_mask, rand_u

def generate_single_state(key, p_meas, s_strength, n_qubits, n_layers, ansatz):
    params, rots, meas_mask, rand_u = sample_circuit_inputs(key, n_qubits, n_layers, p_meas, ansatz)
    psi = prepare_initial_psi(n_qubits, ansatz)
    def layer_body(p_state, xs):
        l_params, l_rots, m_mask, r_u = xs
        p_state = apply_ansatz_layer_psi(p_state, l_params, l_rots, n_qubits, ansatz)
        for q in range(n_qubits):
            p_state = spmm_measure_step_normalized(p_state, q, m_mask[q], r_u[q], s_strength, n_qubits)
        return p_state, None
    psi_final, _ = jax.lax.scan(layer_body, psi, (params, rots, meas_mask, rand_u))
    return psi_final

def eval_batch_fidelities_paired(keys, p_meas, s_strength, n_qubits, n_layers, ansatz, pairing_factor=8):
    states = jax.vmap(generate_single_state, in_axes=(0, None, None, None, None, None))(
        keys, p_meas, s_strength, n_qubits, n_layers, ansatz
    )
    half = states.shape[0] // 2
    eff_k = min(pairing_factor, half)
    states_A = states[:half]
    states_B = states[half:2 * half]
    if eff_k == half:
        overlaps = jnp.matmul(states_A, jnp.conj(states_B.T))
        return (jnp.abs(overlaps) ** 2).astype(jnp.float32).reshape(-1)
    else:
        fids_list = []
        for k in range(eff_k):
            shifted_B = jnp.roll(states_B, shift=k, axis=0)
            overlaps = jnp.sum(jnp.conj(states_A) * shifted_B, axis=-1)
            fids_list.append((jnp.abs(overlaps) ** 2).astype(jnp.float32))
        return jnp.concatenate(fids_list, axis=0)

_COMPILED_CACHE = {}

def get_circuit_depth(N, args):
    return int(args.fixed_l) if args.depth_mode == "fixed" else int(args.linear_mult * N)

def choose_state_batch_size(n_qubits):
    if n_qubits <= 8: return 4096
    elif n_qubits <= 12: return 1024
    elif n_qubits <= 16: return 256
    else: return 64

def build_batch_fidelity_fn(n_qubits, n_layers, ansatz, pairing_factor=8):
    cache_key = (n_qubits, n_layers, ansatz, pairing_factor)
    if cache_key in _COMPILED_CACHE:
        return _COMPILED_CACHE[cache_key]
    core_fn = partial(eval_batch_fidelities_paired, n_qubits=n_qubits, n_layers=n_layers, ansatz=ansatz, pairing_factor=pairing_factor)
    jitted = jax.jit(core_fn)
    _COMPILED_CACHE[cache_key] = jitted
    return jitted

def sample_fidelities_fast(n_qubits, n_layers, ansatz, p_val, s_val, target_pairs, base_key, pairing_factor=8):
    batch_fn = build_batch_fidelity_fn(n_qubits, n_layers, ansatz, pairing_factor=pairing_factor)
    state_bs = choose_state_batch_size(n_qubits)
    per_dev_half = state_bs // 2
    eff_k = min(pairing_factor, per_dev_half)
    pairs_per_batch = (state_bs // 2) * eff_k
    all_fids = []
    rem_pairs = target_pairs
    chunk = 0
    while rem_pairs > 0:
        keys = jax.random.split(jax.random.fold_in(base_key, chunk), state_bs)
        fids = np.asarray(batch_fn(keys, DTYPE_REAL(p_val), DTYPE_REAL(s_val)))
        all_fids.append(fids)
        rem_pairs -= pairs_per_batch
        chunk += 1
    return np.concatenate(all_fids, axis=0)[:target_pairs]

UNIVERSAL_LOG_BINS = np.insert(np.logspace(-8, 0, 150), 0, 0.0)

def compute_kl_divergence_fair(fidelities, n_qubits, bin_edges=UNIVERSAL_LOG_BINS):
    d = 2.0 ** n_qubits
    n_samples = len(fidelities)
    p_pqc, _ = np.histogram(fidelities, bins=bin_edges)
    p_pqc = p_pqc / float(n_samples)
    a = bin_edges[:-1]
    b = bin_edges[1:]
    p_haar = np.exp((d - 1.0) * np.log1p(-a)) - np.exp((d - 1.0) * np.log1p(-np.minimum(b, 1.0 - 1e-16)))
    p_haar = np.maximum(p_haar, 0.0)
    eps = 1e-12
    p_pqc = np.clip(p_pqc, eps, None)
    p_haar = np.clip(p_haar, eps, None)
    p_pqc /= np.sum(p_pqc)
    p_haar /= np.sum(p_haar)
    return float(np.sum(p_pqc * np.log(p_pqc / p_haar)))

# =============================================================================
# PRODUCTION SWEEP & RAW CURVE PLOTTING
# =============================================================================

def run_production_sweep(args):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "raw_curves")
    os.makedirs(plots_dir, exist_ok=True)

    target_ansatzes = args.ansatz_list
    N_list = args.n_list
    p_list = args.p_list
    s_list = args.s_list
    depth_desc = f"Fixed L={args.fixed_l}" if args.depth_mode == "fixed" else f"Linear L={args.linear_mult}N"

    pkl_path = os.path.join(output_dir, "expressibility_sweep_data.pkl")

    # Checkpoint/Resume support
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            saved_payload = pickle.load(f)
        dkl_values_by_s = saved_payload.get("dkl_values_by_s", {})
        print(f"[Resume] Loaded existing progress from {pkl_path}", flush=True)
    else:
        dkl_values_by_s = {s: {a: {p: {} for p in p_list} for a in target_ansatzes} for s in s_list}

    master_key = jax.random.PRNGKey(args.seed)

    print("=" * 90)
    print(f"PRODUCTION EXPRESSIBILITY SWEEP (S={args.samples} pairs, K={args.pairing_factor})")
    print(f"System Sizes N: {N_list} | Ansatzes: {target_ansatzes} | Strengths s: {s_list}")
    print("=" * 90)

    for idx_n, N in enumerate(N_list):
        n_layers = get_circuit_depth(N, args)
        computed_any = False

        for ansatz_idx, ansatz in enumerate(target_ansatzes):
            for s_val in s_list:
                for p_val in p_list:
                    # Skip if already computed in checkpoint
                    if N in dkl_values_by_s.get(s_val, {}).get(ansatz, {}).get(p_val, {}):
                        continue

                    computed_any = True
                    t0 = time.perf_counter()
                    base_grid_key = jax.random.fold_in(
                        jax.random.fold_in(jax.random.fold_in(master_key, N), ansatz_idx), int(round(p_val * 10000))
                    )
                    grid_key = base_grid_key if abs(s_val - 1.0) < 1e-9 else jax.random.fold_in(base_grid_key, int(round(s_val * 10000)))

                    fids = sample_fidelities_fast(
                        N, n_layers, ansatz, p_val, s_val, target_pairs=args.samples,
                        base_key=grid_key, pairing_factor=args.pairing_factor
                    )
                    dkl = compute_kl_divergence_fair(fids, N, UNIVERSAL_LOG_BINS)
                    dkl_values_by_s[s_val][ansatz][p_val][N] = dkl
                    dt = time.perf_counter() - t0
                    print(f"[{ansatz} | N={N:2d} | s={s_val:.2f} | p={p_val:.3f}] D_KL={dkl:.4f} (log10={np.log10(max(dkl, 1e-15)):.3f}) [{dt:.1f}s]", flush=True)

        _COMPILED_CACHE.clear()
        if hasattr(jax, "clear_caches"):
            jax.clear_caches()
        gc.collect()

        # Incremental per-N atomic checkpoint save
        payload = {
            "N_list": N_list,
            "p_list": p_list,
            "s_list": s_list,
            "ansatz_list": target_ansatzes,
            "depth_desc": depth_desc,
            "samples": args.samples,
            "dkl_values": dkl_values_by_s[s_list[0]],
            "dkl_values_by_s": dkl_values_by_s,
        }
        temp_pkl = pkl_path + ".tmp"
        with open(temp_pkl, "wb") as f:
            pickle.dump(payload, f)
        os.replace(temp_pkl, pkl_path)
        if computed_any:
            print(f"[Checkpoint] Saved progress up to N={N} to {pkl_path}", flush=True)

    print(f"\nSweep complete! Data saved to {pkl_path}")

    # Generate raw D_KL vs p and log10(D_KL) vs p curves for each s_val
    for s_val in s_list:
        s_tag = "" if len(s_list) == 1 and abs(s_val - 1.0) < 1e-9 else f"_s{s_val:.2f}"
        for ansatz in target_ansatzes:
            fig_linear, ax_lin = plt.subplots(figsize=(9, 6))
            fig_log, ax_log = plt.subplots(figsize=(9, 6))
            cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(N_list)))

            for idx_n, N in enumerate(N_list):
                p_vals_n, dkl_vals_n = [], []
                for p_val in p_list:
                    if N in dkl_values_by_s[s_val][ansatz][p_val]:
                        p_vals_n.append(p_val)
                        dkl_vals_n.append(dkl_values_by_s[s_val][ansatz][p_val][N])

                if len(p_vals_n) > 0:
                    p_arr = np.asarray(p_vals_n)
                    dkl_arr = np.asarray(dkl_vals_n)
                    log_dkl_arr = np.log10(np.maximum(dkl_arr, 1e-15))

                    ax_lin.plot(p_arr, dkl_arr, "o-", color=cmap[idx_n], lw=2, markersize=5.5, label=f"N = {N}")
                    ax_log.plot(p_arr, log_dkl_arr, "o-", color=cmap[idx_n], lw=2, markersize=5.5, label=f"N = {N}")

            ax_lin.set_title(f"{ansatz} ({depth_desc}, s={s_val:.2f}): $D_{{KL}}$ vs. $p$", fontsize=13)
            ax_lin.set_xlabel("Measurement Rate ($p$)", fontsize=11.5)
            ax_lin.set_ylabel(r"$D_{KL}$", fontsize=11.5)
            ax_lin.grid(True, linestyle="--", alpha=0.6)
            ax_lin.legend(fontsize=9.5)
            fig_linear.tight_layout()
            fig_linear.savefig(os.path.join(plots_dir, f"{ansatz}{s_tag}_DKL_vs_p_linear.png"), dpi=300)
            plt.close(fig_linear)

            ax_log.set_title(f"{ansatz} ({depth_desc}, s={s_val:.2f}): $\\log_{{10}}(D_{{KL}})$ vs. $p$", fontsize=13)
            ax_log.set_xlabel("Measurement Rate ($p$)", fontsize=11.5)
            ax_log.set_ylabel(r"$\log_{10}(D_{KL})$", fontsize=11.5)
            ax_log.grid(True, linestyle="--", alpha=0.6)
            ax_log.legend(fontsize=9.5)
            fig_log.tight_layout()
            fig_log.savefig(os.path.join(plots_dir, f"{ansatz}{s_tag}_DKL_vs_p_log10.png"), dpi=300)
            plt.close(fig_log)

    print(f"Raw curves saved in {plots_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="data_exp")
    parser.add_argument("--ansatz-list", type=str, nargs="+", default=["HEA1", "HEA2", "HVA"], choices=["HEA1", "HEA2", "HVA"])
    parser.add_argument("--n-list", type=int, nargs="+", default=[6, 8, 10, 12, 14, 16, 18])
    # Dense 23-point p-grid for smooth curves
    parser.add_argument("--p-list", type=float, nargs="+", default=[
        0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 
        0.20, 0.22, 0.24, 0.27, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70
    ])
    parser.add_argument("--s-list", type=float, nargs="+", default=[1.0, 0.5], help="SPMM measurement strengths s in [0, 1]")
    parser.add_argument("--depth-mode", type=str, choices=["fixed", "linear"], default="fixed")
    parser.add_argument("--fixed-l", type=int, default=18)
    parser.add_argument("--linear-mult", type=int, default=1)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--pairing-factor", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    configure_precision(use_32bit=True)
    run_production_sweep(args)