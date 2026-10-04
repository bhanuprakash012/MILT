"""
GPU-accelerated (JAX) HEA1 raw gradient sweep.
Optimized for XLA compilation speed and precision toggling.
"""
import gc
import os
import time
import pickle
import datetime
import argparse

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

# =============================================================================
# 0. PRECISION & DEVICE CONFIGURATION
# =============================================================================

# Default to 64-bit to preserve original math, but togglable via CLI
DTYPE_REAL = jnp.float64
DTYPE_COMPLEX = jnp.complex128

def configure_precision(use_32bit):
    global DTYPE_REAL, DTYPE_COMPLEX
    if use_32bit:
        jax.config.update("jax_enable_x64", False)
        DTYPE_REAL = jnp.float32
        DTYPE_COMPLEX = jnp.complex64
    else:
        jax.config.update("jax_enable_x64", True)
        DTYPE_REAL = jnp.float64
        DTYPE_COMPLEX = jnp.complex128

def get_allocated_cpu_cores():
    if "SLURM_CPUS_PER_TASK" in os.environ: return int(os.environ["SLURM_CPUS_PER_TASK"])
    if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
        try: return int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
        except ValueError: pass
    try: return len(os.sched_getaffinity(0))
    except AttributeError: pass
    return os.cpu_count() or 4

def describe_devices():
    devices = jax.devices()
    kinds = sorted(set(d.platform for d in devices))
    return devices, kinds

# =============================================================================
# 1. SINGLE-SAMPLE QUANTUM CORE
# =============================================================================

def rot_matrix_jax(theta, axis, grad):
    c = jnp.cos(theta / 2.0)
    s = jnp.sin(theta / 2.0)
    zero = jnp.zeros_like(c, dtype=DTYPE_COMPLEX)
    
    if grad == 0:
        Ux = jnp.stack([jnp.stack([c, -1j * s]), jnp.stack([-1j * s, c])])
        Uy = jnp.stack([jnp.stack([c, -s]), jnp.stack([s, c])])
        Uz = jnp.stack([jnp.stack([c - 1j * s, zero]), jnp.stack([zero, c + 1j * s])])
    else:
        Ux = jnp.stack([jnp.stack([-0.5 * s, -0.5j * c]), jnp.stack([-0.5j * c, -0.5 * s])])
        Uy = jnp.stack([jnp.stack([-0.5 * s, -0.5 * c]), jnp.stack([0.5 * c, -0.5 * s])])
        Uz = jnp.stack([jnp.stack([-0.5 * s - 0.5j * c, zero]),
                        jnp.stack([zero, -0.5 * s + 0.5j * c])])
                        
    U = jnp.where(axis == 1, Ux, jnp.where(axis == 2, Uy, Uz))
    return U.astype(DTYPE_COMPLEX)

# Vectorized rotation generator: creates an entire layer's matrices in one XLA op
vmap_rot_matrix = jax.vmap(rot_matrix_jax, in_axes=(0, 0, None))

def initial_state_flat(n_qubits):
    dim = 1 << n_qubits
    psi = jnp.zeros(dim, dtype=DTYPE_COMPLEX)
    psi = psi.at[dim - 1].set(1.0)
    return psi

def apply_1q_gate(U, q, psi, n_qubits):
    dim_L = 1 << q
    dim_R = 1 << (n_qubits - q - 1)
    psi_r = psi.reshape(dim_L, 2, dim_R)
    new0 = U[0, 0] * psi_r[:, 0, :] + U[0, 1] * psi_r[:, 1, :]
    new1 = U[1, 0] * psi_r[:, 0, :] + U[1, 1] * psi_r[:, 1, :]
    return jnp.stack([new0, new1], axis=1).reshape(-1)

def apply_cnot_adjacent(q, psi, n_qubits):
    dim_L = 1 << q
    dim_R = 1 << (n_qubits - q - 2)
    psi_r = psi.reshape(dim_L, 4, dim_R)
    return psi_r[:, (0, 1, 3, 2), :].reshape(-1)

def apply_cnot_edges(psi, n_qubits):
    dim_mid = 1 << (n_qubits - 2)
    psi_r = psi.reshape(2, dim_mid, 2)
    part_0 = psi_r[:, :, 0]
    part_1 = psi_r[::-1, :, 1]
    return jnp.stack([part_0, part_1], axis=-1).reshape(-1)

def spmm_measure_step(psi0, psi1, q, mask_val, outcome_rand, s_strength, n_qubits):
    """
    Softened projective measurement model (SPMM).
    MUST be evaluated sequentially to preserve Born rule chain probabilities.
    """
    denom = jnp.sqrt(2.0 * (1.0 + s_strength**2))
    
    P_plus_diag = jnp.array([(1.0 + s_strength) / denom, (1.0 - s_strength) / denom], dtype=psi0.dtype)
    P_minus_diag = jnp.array([(1.0 - s_strength) / denom, (1.0 + s_strength) / denom], dtype=psi0.dtype)
    P_plus_sq = jnp.real(P_plus_diag)**2

    dim_L = 1 << q
    dim_R = 1 << (n_qubits - q - 1)
    psi0_r = psi0.reshape((dim_L, 2, dim_R))

    prob_0 = jnp.sum(jnp.abs(psi0_r[:, 0, :])**2)
    prob_1 = jnp.sum(jnp.abs(psi0_r[:, 1, :])**2)
    p_plus = prob_0 * P_plus_sq[0] + prob_1 * P_plus_sq[1]

    outcome_is_plus = outcome_rand < p_plus
    chosen_P_diag = jnp.where(outcome_is_plus, P_plus_diag, P_minus_diag)
    p_chosen = jnp.where(outcome_is_plus, p_plus, 1.0 - p_plus)

    safe_p = jnp.where(p_chosen == 0.0, 1.0, p_chosen)
    scaled_diag = chosen_P_diag / jnp.sqrt(safe_p)

    identity_diag = jnp.ones(2, dtype=psi0.dtype)
    applied_diag = jnp.where(mask_val, scaled_diag, identity_diag)

    def apply_gate(psi):
        psi_r = psi.reshape((dim_L, 2, dim_R))
        psi_out_0 = psi_r[:, 0, :] * applied_diag[0]
        psi_out_1 = psi_r[:, 1, :] * applied_diag[1]
        
        psi_out_r = jnp.stack([psi_out_0, psi_out_1], axis=1)
        psi_new = psi_out_r.reshape(-1)
        return jnp.where(mask_val, psi_new, psi)

    psi0_new = apply_gate(psi0)
    psi1_new = apply_gate(psi1)

    return psi0_new, psi1_new

def compute_cost(psi0, psi1, n_qubits):
    rest = 1 << (n_qubits - 2)
    psi0_r = psi0.reshape(4, rest)
    psi1_r = psi1.reshape(4, rest)
    signs = jnp.array([1, -1, -1, 1], dtype=psi0.dtype).reshape(4, 1)
    cost_psi = signs * psi0_r
    return 2.0 * jnp.vdot(psi1_r, cost_psi).real

def _hea1_single_sample(key, s_strength, p_meas, n_qubits):
    n_layers = n_qubits * 3
    mid_grad_idx = n_qubits // 2
    
    k_rot, k_param, k_mask, k_outcome = jax.random.split(key, 4)
    
    rotations = jax.random.randint(k_rot, (n_layers, 2, n_qubits), minval=1, maxval=4)
    parameters = jax.random.uniform(k_param, (n_layers, 2, n_qubits), minval=-4 * jnp.pi, maxval=4 * jnp.pi)
    
    raw_mask = jax.random.uniform(k_mask, (n_layers - 1, n_qubits)) < p_meas
    meas_mask = jnp.pad(raw_mask, ((0, 1), (0, 0)), constant_values=False)
    
    raw_outcome = jax.random.uniform(k_outcome, (n_layers - 1, n_qubits))
    outcome_rand = jnp.pad(raw_outcome, ((0, 1), (0, 0)), constant_values=0.0)
    
    base_indices = jnp.arange(n_layers) * (2 * n_qubits)
    psi0 = initial_state_flat(n_qubits)
    psi1 = initial_state_flat(n_qubits)

    def layer_step(carry, xs):
        p0, p1 = carry
        rots, params, m_mask, m_out, base_idx = xs
        
        # O(1) Graph Compression: Generate all U matrices for Block 1 simultaneously
        U0_all_block1 = vmap_rot_matrix(params[0], rots[0], 0)
        U1_all_block1 = vmap_rot_matrix(params[0], rots[0], 1)
        
        for q in range(n_qubits):
            idx = base_idx + q
            U0 = U0_all_block1[q]
            U1 = U1_all_block1[q]
            U_applied1 = jnp.where(idx == mid_grad_idx, U1, U0)
            
            p0 = apply_1q_gate(U0, q, p0, n_qubits)
            p1 = apply_1q_gate(U_applied1, q, p1, n_qubits)
            
        for q in range(0, n_qubits - 1, 2):
            p0 = apply_cnot_adjacent(q, p0, n_qubits)
            p1 = apply_cnot_adjacent(q, p1, n_qubits)

        # O(1) Graph Compression: Generate all U matrices for Block 2 simultaneously
        U0_all_block2 = vmap_rot_matrix(params[1], rots[1], 0)
        U1_all_block2 = vmap_rot_matrix(params[1], rots[1], 1)

        for q in range(n_qubits):
            idx = base_idx + n_qubits + q
            U0 = U0_all_block2[q]
            U1 = U1_all_block2[q]
            U_applied1 = jnp.where(idx == mid_grad_idx, U1, U0)
            
            p0 = apply_1q_gate(U0, q, p0, n_qubits)
            p1 = apply_1q_gate(U_applied1, q, p1, n_qubits)
            
        for q in range(1, n_qubits - 1, 2):
            p0 = apply_cnot_adjacent(q, p0, n_qubits)
            p1 = apply_cnot_adjacent(q, p1, n_qubits)

        p0 = apply_cnot_edges(p0, n_qubits)
        p1 = apply_cnot_edges(p1, n_qubits)

        for q in range(n_qubits):
            p0, p1 = spmm_measure_step(p0, p1, q, m_mask[q], m_out[q], s_strength, n_qubits)

        true_norm = jnp.linalg.norm(p0)
        safe_true_norm = jnp.where(true_norm == 0.0, 1.0, true_norm)
        p0 = p0 / safe_true_norm
        p1 = p1 / safe_true_norm

        return (p0, p1), None
        
    (psi0, psi1), _ = jax.lax.scan(
        layer_step, 
        (psi0, psi1), 
        (rotations, parameters, meas_mask, outcome_rand, base_indices)
    )

    return compute_cost(psi0, psi1, n_qubits)

# =============================================================================
# 2. BATCHING & MEMORY DYNAMICS
# =============================================================================

_compiled_cache = {}

def get_batched_fn(n_qubits):
    n_devices = jax.local_device_count()
    cache_key = (n_qubits, n_devices, str(DTYPE_COMPLEX))
    if cache_key in _compiled_cache:
        return _compiled_cache[cache_key], n_devices

    single_fn = partial(_hea1_single_sample, n_qubits=n_qubits)
    vmapped = jax.vmap(single_fn, in_axes=(0, None, None))

    if n_devices > 1:
        fn = jax.pmap(vmapped, in_axes=(0, None, None))
        def call(keys, s_strength, p_meas):
            batch = keys.shape[0]
            per_device = batch // n_devices
            keys_r = keys.reshape(n_devices, per_device, 2)
            out = fn(keys_r, DTYPE_REAL(s_strength), DTYPE_REAL(p_meas))
            return out.reshape(batch)
        _compiled_cache[cache_key] = call
    else:
        fn = jax.jit(vmapped)
        def call(keys, s_strength, p_meas):
            return fn(keys, DTYPE_REAL(s_strength), DTYPE_REAL(p_meas))
        _compiled_cache[cache_key] = call

    return _compiled_cache[cache_key], n_devices

def choose_batch_size(n_qubits, target_bytes_per_chunk=2.0e9, n_devices=1, cap=8192):
    dim = 1 << n_qubits
    bytes_per_complex = jnp.zeros(1, dtype=DTYPE_COMPLEX).nbytes
    base_bytes = dim * bytes_per_complex * 2  
    
    if n_qubits <= 10: overhead = 8.0 
    elif n_qubits <= 14: overhead = 4.0
    else: overhead = 2.5 
        
    bytes_per_sample = base_bytes * overhead
    bs = max(1, int(target_bytes_per_chunk // bytes_per_sample))
    
    bs = min(bs, cap)
    if bs > 1: bs = 1 << int(np.log2(bs))
    if n_devices > 1: bs = max(n_devices, (bs // n_devices) * n_devices)
        
    return bs

def compute_flat_sampling_gradients_jax(n_qubits, p_meas, s_strength, n_samples,
                                        base_key, batch_size=None, verbose=False):
    batch_fn, n_devices = get_batched_fn(n_qubits)
    if batch_size is None:
        batch_size = choose_batch_size(n_qubits, n_devices=n_devices)

    all_grads = []
    remaining = n_samples
    chunk_idx = 0
    while remaining > 0:
        this_bs = min(batch_size, remaining)
        if n_devices > 1 and this_bs % n_devices != 0:
            this_bs = max(n_devices, (this_bs // n_devices) * n_devices)

        chunk_key = jax.random.fold_in(base_key, chunk_idx)
        keys = jax.random.split(chunk_key, this_bs)

        try:
            grads = batch_fn(keys, s_strength, p_meas)
        except Exception as e:
            msg = str(e).lower()
            if ("resource_exhausted" in msg or "out of memory" in msg) and this_bs > max(n_devices, 1):
                new_bs = max(n_devices, this_bs // 2)
                if verbose: print(f"    [warn] OOM at batch_size={this_bs}, retrying with {new_bs}")
                batch_size = new_bs
                continue
            raise

        all_grads.append(np.asarray(grads))
        remaining -= this_bs
        chunk_idx += 1

    return np.concatenate(all_grads)[:n_samples]

# =============================================================================
# 3. CLUSTER SWEEP
# =============================================================================
def run_cluster_sweep(output_dir="data_mipt", seed=0, target_mem_bytes=12.0e9, override_samples=None):
    devices, kinds = describe_devices()
    n_devices = jax.local_device_count()
    cpu_cores = get_allocated_cpu_cores()
    
    os.makedirs(output_dir, exist_ok=True)

    p_range = [0.0, 0.02,  0.04,  0.06,  0.08, 0.1,  0.12, 0.14, 0.16]
    s_strengths = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    system_sizes = list(range(4, 17))  

    if override_samples is not None:
        sample_schedule = {N: override_samples for N in system_sizes}
    else:
        sample_schedule = {
            4: 10000, 5: 10000,
            6: 10000, 7: 10000, 8: 10000, 9: 10000, 10: 10000,
            11: 10000, 12: 10000, 13: 10000,
            14: 10000, 15: 10000, 16: 10000
        }

    # THE FIX: Define the label here
    precision_label = "32bit" if DTYPE_REAL == jnp.float32 else "64bit"

    metadata_path = os.path.join(output_dir, "sweep_config_metadata_5N_2.txt")
    with open(metadata_path, "w") as f:
        f.write("=== CLUSTER SWEEP CONFIGURATION (JAX/GPU backend) ===\n")
        f.write(f"Run Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"JAX devices: {devices}\n")
        f.write(f"Local device count: {n_devices}\n")
        f.write(f"Allocated CPU cores: {cpu_cores}\n")
        f.write(f"Precision: {precision_label}\n") # THE FIX: Write it while the file is open
        f.write(f"Circuit Depth (n_layers): 3N\n")
        f.write("-" * 40 + "\n")
        f.write(f"Measurement Rates (p): {p_range}\n")
        f.write(f"Measurement Strengths (s): {s_strengths}\n")
        f.write(f"System Sizes (N): {system_sizes}\n")
        f.write(f"Samples Schedule: {sample_schedule}\n")
        f.write("-" * 40 + "\n")

    print(f"=== COMPUTE ALLOCATION ===\nJAX devices: {devices}\n"
          f"Precision: {precision_label}\n"
          f"Platforms: {kinds} | local_device_count: {n_devices}\n===========================")

    master_key = jax.random.PRNGKey(seed + 1024)
    sweep_results = {s: {} for s in s_strengths}

    for s_idx, s in enumerate(s_strengths):
        for N in system_sizes:
            current_samples = sample_schedule[N]
            batch_size = choose_batch_size(N, target_bytes_per_chunk=target_mem_bytes, n_devices=max(n_devices, 1))

            for p_idx, p in enumerate(p_range):
                print(f"Computing data -> s: {s:.1f}, N: {N}, p: {p:.2f} [{current_samples:,} samples] ...")
                
                point_key = jax.random.fold_in(
                    jax.random.fold_in(jax.random.fold_in(master_key, s_idx), p_idx), N)

                raw_grads = compute_flat_sampling_gradients_jax(
                    N, p, s, current_samples, point_key, batch_size=batch_size, verbose=False)

                if p not in sweep_results[s]:
                    sweep_results[s][p] = {'system_sizes': [], 'raw_gradients': []}
                    
                sweep_results[s][p]['system_sizes'].append(N)
                sweep_results[s][p]['raw_gradients'].append(raw_grads)

            # Purge the compiled gradient graph from VRAM
            _compiled_cache.clear()
            if hasattr(jax, 'clear_caches'):
                jax.clear_caches()
            gc.collect()
            
            # THE FIX: Keep only the dynamic file naming down here
            temp_filepath = os.path.join(output_dir, f"sweep_results_N4_16_{precision_label}_TEMP.pkl")
            final_filepath = os.path.join(output_dir, f"sweep_results_N4_16_{precision_label}.pkl")
            
            with open(temp_filepath, "wb") as f:
                pickle.dump(sweep_results, f)
            os.replace(temp_filepath, final_filepath)

    print("Sweep complete. Data successfully saved.")

# =============================================================================
# 4. RUNTIME / BENCHMARKING
# =============================================================================

def run_hardware_benchmark(target_mem_bytes=12.0e9):
    n_devices = jax.local_device_count()
    print("=== JAX HEA1 HARDWARE BENCHMARK ===")
    print(f"Detected {n_devices} GPU(s).")
    print(f"Active Precision: {DTYPE_REAL.__name__} / {DTYPE_COMPLEX.__name__}")
    print(f"{'N':<5} | {'Batch Size':<11} | {'Compile (s)':<12} | {'Time/Sample (ms)':<17} | {'Samples/Second':<15}")
    print("-" * 75)
    
    s_test, p_test = 1.0, 0.1 
    base_key = jax.random.PRNGKey(42)
    
    for N in range(4, 17):
        batch_size = choose_batch_size(N, target_bytes_per_chunk=target_mem_bytes, n_devices=n_devices)
        batch_fn, _ = get_batched_fn(N)
        
        while True:
            try:
                keys = jax.random.split(base_key, batch_size)
                
                compile_start = time.perf_counter()
                _ = batch_fn(keys, s_test, p_test).block_until_ready()
                compile_time = time.perf_counter() - compile_start
                
                iters = 2 
                exec_start = time.perf_counter()
                for _ in range(iters):
                    _ = batch_fn(keys, s_test, p_test).block_until_ready()
                exec_time = time.perf_counter() - exec_start
                
                total_samples = batch_size * iters
                time_per_sample_ms = (exec_time / total_samples) * 1000
                samples_per_sec = total_samples / exec_time
                
                print(f"{N:<5} | {batch_size:<11} | {compile_time:<12.2f} | {time_per_sample_ms:<17.5f} | {samples_per_sec:,.0f}")
                break  
                
            except Exception as e:
                msg = str(e).lower()
                if ("resource_exhausted" in msg or "out of memory" in msg) and batch_size > max(n_devices, 1):
                    batch_size = max(n_devices, batch_size // 2)
                else:
                    raise
                    
        _compiled_cache.clear()
        if hasattr(jax, 'clear_caches'): jax.clear_caches()
        gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data_mipt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-mem-gb", type=float, default=12.0)
    parser.add_argument("--override-samples", type=int, default=None, help="Force a flat number of samples for all N")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--precision", type=int, choices=[32, 64], default=32, 
                        help="Select 32 for float32/complex64 (faster, lower memory) or 64 for float64/complex128 (exact math).")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
    
    # Configure JAX precision before any arrays are initialized
    configure_precision(use_32bit=(args.precision == 32))

    if args.benchmark:
        run_hardware_benchmark(target_mem_bytes=args.target_mem_gb * 1e9)
    else:
        run_cluster_sweep(
            output_dir=args.output_dir, 
            seed=args.seed, 
            target_mem_bytes=args.target_mem_gb * 1e9,
            override_samples=args.override_samples
        )