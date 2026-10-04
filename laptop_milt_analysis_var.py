import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import dask
from tqdm.auto import tqdm
import pickle  # Imported for data backup

# =============================================================================
# 1. QUANTUM CORE & SPMM WEAK MEASUREMENT
# =============================================================================

def ApplyGate(U, qubits, psi):
    """Multiplies tensor psi by gate U acting on targeted qubits."""
    indices = "".join([chr(97 + q) for q in qubits])
    indices += "".join([chr(65 + q) for q in qubits])
    indices += ","
    indices += "".join([chr(97 + i - 32 * qubits.count(i)) for i in range(len(psi.shape))])
    return np.einsum(indices, U, psi)

def Inner(psi_1, psi_2):
    """Calculates <psi_1|psi_2>."""
    indices = "".join([chr(97 + q) for q in range(len(psi_1.shape))])
    indices += ","
    indices += "".join([chr(97 + q) for q in range(len(psi_2.shape))])
    return np.einsum(indices, psi_1.conj(), psi_2)

def initial_state(n_qubits):
    """Initializes computational ground state |0...0>."""
    zero = np.array([1, 0], dtype=np.complex64)
    psi = zero.copy()
    for _ in range(n_qubits - 1):
        psi = np.kron(psi, zero)
    return psi.reshape((2,) * n_qubits)

def pauli(i):
    """Pauli matrices: 0: I, 1: X, 2: Y, 3: Z."""
    if i == 0: return np.eye(2, dtype=np.complex64)
    elif i == 1: return np.array([[0, 1], [1, 0]], dtype=np.complex64)
    elif i == 2: return np.array([[0, -1j], [1j, 0]], dtype=np.complex64)
    elif i == 3: return np.array([[1, 0], [0, -1]], dtype=np.complex64)

CNOT = np.reshape(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex64), (2, 2, 2, 2))

def rot(theta, i, grad=0):
    """Rotation gate R_i(theta). i = 1 for X, 2 for Y, 3 for Z."""
    if not grad:
        return np.cos(theta / 2) * np.eye(2, dtype=np.complex64) - 1j * np.sin(theta / 2) * pauli(i)
    else:
        return -0.5 * np.sin(theta / 2) * np.eye(2, dtype=np.complex64) - 0.5j * np.cos(theta / 2) * pauli(i)

def spmm_kraus(outcome, s_strength):
    """SPMM weak measurement Kraus operator. outcome: 0 (+) or 1 (-). s in [0, 1]"""
    denom = np.sqrt(2.0 * (1.0 + s_strength**2))
    if outcome == 0:
        return np.array([[1.0 + s_strength, 0], [0, 1.0 - s_strength]], dtype=np.complex64) / denom
    else:
        return np.array([[1.0 - s_strength, 0], [0, 1.0 + s_strength]], dtype=np.complex64) / denom

def apply_spmm_measure(psi_list, measurements, s_strength):
    """Applies SPMM weak measurement differentiably to state and derivative vectors."""
    for q in measurements:
        qubit = q[1] if isinstance(q, (list, tuple, np.ndarray)) else q

        # Born probability for outcome '+' on primary state
        M_plus = spmm_kraus(0, s_strength)
        psi_plus_test = ApplyGate(M_plus, [qubit], psi_list[0])
        p0 = np.real(Inner(psi_plus_test, psi_plus_test))
        p0 = np.clip(p0, 0.0, 1.0)
        p1 = 1.0 - p0

        # Sample outcome
        outcome = np.random.choice([0, 1], p=[p0, p1])
        p_i = p0 if outcome == 0 else p1
        M_chosen = spmm_kraus(outcome, s_strength)

        # Update state and derivative vectors
        for i in range(len(psi_list)):
            psi_list[i] = ApplyGate(M_chosen, [qubit], psi_list[i]) / np.sqrt(p_i + 1e-12)

# =============================================================================
# 2. HEA1 CIRCUIT & UNAWARE GRADIENTS
# =============================================================================

def num_parameters(n_qubits, n_layers): return n_qubits * n_layers * 2
def random_parameters(num): return np.random.uniform(low=-4 * np.pi, high=4 * np.pi, size=num)
def random_rotations(num): return np.random.randint(low=1, high=4, size=num)

def random_measurements_prob(n_layers, n_qubits, p_meas):
    measure_list = []
    for depth in range(n_layers - 1):
        for q in range(n_qubits):
            if p_meas > np.random.rand():
                measure_list.append([depth, q])
    return measure_list if len(measure_list) > 0 else None

def ApplyHam_Z0Z1(psi):
    """Applies the local Z0 Z1 observable to compute the energy."""
    zz = np.kron(pauli(3), pauli(3)).reshape(2, 2, 2, 2)
    return ApplyGate(zz, [0, 1], psi)

def HEA1_unaware_gradient(n_qubits, n_layers, parameters, rotations, s_strength, gradient_index, measurements):
    """Executes HEA1 ansatz and outputs the exact unaware gradient."""
    psi_list = [initial_state(n_qubits), initial_state(n_qubits)]
    current_param = 0

    # Initial layer: Y(pi) rotations
    for q in range(n_qubits):
        for i in range(2):
            psi_list[i] = ApplyGate(rot(np.pi, 2), [q], psi_list[i])

    for l in range(n_layers):
        # 1. Rotation stage 1
        for q in range(n_qubits):
            param, axis = parameters[current_param], rotations[current_param]
            for i in range(2):
                take_grad = 1 if (current_param == gradient_index and i == 1) else 0
                psi_list[i] = ApplyGate(rot(param, axis, take_grad), [q], psi_list[i])
            current_param += 1

        # 2. Even CNOT ladder
        for q in range(0, n_qubits - 1, 2):
            for i in range(2):
                psi_list[i] = ApplyGate(CNOT, [q, q + 1], psi_list[i])

        # 3. Rotation stage 2
        for q in range(n_qubits):
            param, axis = parameters[current_param], rotations[current_param]
            for i in range(2):
                take_grad = 1 if (current_param == gradient_index and i == 1) else 0
                psi_list[i] = ApplyGate(rot(param, axis, take_grad), [q], psi_list[i])
            current_param += 1

        # 4. Odd CNOT ladder
        for q in range(1, n_qubits - 1, 2):
            for i in range(2):
                psi_list[i] = ApplyGate(CNOT, [q, q + 1], psi_list[i])

        # 5. Weak measurement layer
        if l < n_layers - 1 and measurements is not None:
            m_layer = [m for m in measurements if m[0] == l]
            if len(m_layer) > 0:
                apply_spmm_measure(psi_list, m_layer, s_strength)

    # Local Z0Z1 Cost Function
    cost_psi = ApplyHam_Z0Z1(psi_list[0])
    unaware_grad = 2.0 * Inner(psi_list[1], cost_psi).real
    return unaware_grad

# =============================================================================
# 3. RAPPAPORT FLAT SAMPLING & DECAY ANALYSIS
# =============================================================================

def compute_variance_flat_sampling(n_qubits, p_meas, s_strength, n_samples=200):
    """
    Calculates gradient variance using Rappaport's flat sampling method:
    New parameters AND new measurement layout for every single shot.
    """
    n_layers = n_qubits  # Depth scales with system size
    num_param = num_parameters(n_qubits, n_layers)
    mid_grad_idx = num_param // 2
    
    tasks = []
    
    # FLAT LOOP: Matches Rappaport repository logic exactly
    for _ in range(n_samples):
        measurements = random_measurements_prob(n_layers, n_qubits, p_meas)
        rotations = random_rotations(num_param)
        parameters = random_parameters(num_param)

        task = dask.delayed(HEA1_unaware_gradient)(
            n_qubits=n_qubits,
            n_layers=n_layers,
            parameters=parameters,
            rotations=rotations,
            s_strength=s_strength,
            gradient_index=mid_grad_idx,
            measurements=measurements
        )
        tasks.append(task)

    # Compute all tasks and take flat variance across all samples
    sample_grads = np.array(dask.compute(*tasks))
    return np.var(sample_grads)

def extract_bp_decay_rate(p_meas, s_strength, system_sizes, n_samples=200):
    log2_vars = []
    variances = []
    for N in system_sizes:
        var_N = compute_variance_flat_sampling(N, p_meas, s_strength, n_samples)
        var_N = max(var_N, 1e-25)  # Prevent log(0)
        variances.append(var_N)
        log2_vars.append(np.log2(var_N))

    # Linear regression to find the slope
    res = linregress(system_sizes, log2_vars)
    
    # Decay rate gamma = -slope
    gamma = -res.slope
    stderr = res.stderr
    intercept = res.intercept
    
    return gamma, stderr, intercept, np.array(variances)

def run_laptop_sweep():
    # 1. Increased step size to reduce clutter in variance plots
    p_range = [0.0, 0.10, 0.20, 0.30, 0.40]
    
    # 2. Targeted weak measurement strengths
    s_strengths = [1.0, 0.8]              
    
    system_sizes = [8, 10, 12, 14]          # System sizes N
    n_samples = 500                         # Fast flat sampling

    # Dictionary to store all our generated data for clean plotting
    sweep_results = {s: {'gammas': [], 'stderrs': [], 'p_vals': [], 'fits': []} for s in s_strengths}

    for s in s_strengths:
        print(f"\n--- Running Sweep for Strength s = {s} ---")
        for p in tqdm(p_range, desc=f"Probability Sweep (s={s})"):
            
            gamma, stderr, intercept, variances = extract_bp_decay_rate(p, s, system_sizes, n_samples)
            
            # Calculate expected variance points from the log2 regression fit
            expected_log2_vars = intercept - gamma * np.array(system_sizes)
            expected_vars = 2 ** expected_log2_vars
            
            # Compute standard Chi-Square goodness-of-fit: sum( (Observed - Expected)^2 / Expected )
            chi_sq = np.sum((variances - expected_vars)**2 / expected_vars)
            
            # Save data
            sweep_results[s]['gammas'].append(gamma)
            sweep_results[s]['stderrs'].append(stderr)
            sweep_results[s]['p_vals'].append(p)
            sweep_results[s]['fits'].append({
                'p': p, 'gamma': gamma, 'intercept': intercept, 
                'variances': variances, 'expected_vars': expected_vars, 'chi_sq': chi_sq
            })

    # =================================================
    # DATA BACKUP (Protects against plotting crashes)
    # =================================================
    with open("sweep_results_backup.pkl", "wb") as f:
        pickle.dump(sweep_results, f)
    print("\nData successfully backed up to 'sweep_results_backup.pkl'")

    # =================================================
    # PLOT 1: Gamma Decay Rate vs Probability p
    # =================================================
    plt.figure(figsize=(10, 6))
    for s in s_strengths:
        plt.errorbar(
            sweep_results[s]['p_vals'], sweep_results[s]['gammas'], 
            yerr=sweep_results[s]['stderrs'], fmt="-o", capsize=4,
            linewidth=2, label=f"HEA1 Mixed (s = {s})"
        )

    plt.axhline(0, color="gray", linestyle="--", alpha=0.7, label=r"Transition Horizon ($\gamma = 0$)")
    plt.xlabel(r"Measurement Probability $p$", fontsize=12)
    plt.ylabel(r"Decay Rate $\gamma(p)$", fontsize=12)
    plt.title(r"Laptop Flat-Sample Sweep: HEA1 BP Decay Rate vs $p$", fontsize=13)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    plt.savefig("laptop_gamma_vs_p.png", dpi=300)
    plt.close()

    # =================================================
    # PLOT 2: Variance vs Qubits (Scatter + Exp Fits)
    # =================================================
    fig, axes = plt.subplots(len(s_strengths), 1, figsize=(10, 6 * len(s_strengths)))
    if len(s_strengths) == 1: axes = [axes] # Handle indexing if list length is 1

    for idx, s in enumerate(s_strengths):
        ax = axes[idx]
        for fit_data in sweep_results[s]['fits']:
            p = fit_data['p']
            vars_exact = fit_data['variances']
            gamma = fit_data['gamma']
            intercept = fit_data['intercept']
            chi_sq = fit_data['chi_sq']

            # Generate smooth line for the fitted exponential curve
            N_smooth = np.linspace(min(system_sizes), max(system_sizes), 100)
            vars_fit_smooth = 2 ** (intercept - gamma * N_smooth)
            
            # Plot the exact points and grab the auto-assigned color
            scatter_plot = ax.plot(system_sizes, vars_exact, 'o')
            color = scatter_plot[0].get_color()
            
            # Plot the fitted line using the exact same color
            label_str = rf"$p={p:.1f} \rightarrow \gamma={gamma:.2f}, \chi^2={chi_sq:.1e}$"
            ax.plot(N_smooth, vars_fit_smooth, '--', color=color, label=label_str)

        ax.set_yscale('log', base=2)
        ax.set_xlabel("Number of Qubits (N)", fontsize=12)
        ax.set_ylabel(r"Variance of Gradient $\mathrm{Var}(\partial \theta)$", fontsize=12)
        ax.set_title(f"Variance Exponential Decay Fits (Strength s = {s})", fontsize=13)
        ax.grid(True, linestyle=":", alpha=0.6)
        
        # Move legend outside the main plot area so it doesn't cover data points
        ax.legend(fontsize=10, bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig("laptop_variance_fits.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nExperiments completed!")
    print("1. Decay rates plot saved to 'laptop_gamma_vs_p.png'")
    print("2. Variance vs Qubits fits saved to 'laptop_variance_fits.png'")

if __name__ == "__main__":
    run_laptop_sweep()