import pickle
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import warnings
import multiprocessing as mp
from functools import partial

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Set random seed to guarantee identical random initialization across runs
np.random.seed(42)

# ==========================================
# 1. PAPER'S COST FUNCTION WITH SAFETY NET
# ==========================================
def compute_cost_function_safe(params, p_vals, N_vals, S_data):
    p_c, nu = params
    scaled_curves = []
    
    for i, N in enumerate(N_vals):
        S_N = S_data[i, :]
        interp_func = interp1d(p_vals, S_N, kind='linear', fill_value="extrapolate")
        S_pc_N = interp_func(p_c)
        
        x = (p_vals - p_c) * (N ** (1.0 / nu))
        y_N = S_N - S_pc_N
        scaled_curves.append((x, y_N))
        
    min_x = max([np.min(c[0]) for c in scaled_curves])
    max_x = min([np.max(c[0]) for c in scaled_curves])
    
    if min_x >= max_x:
        return 1e6 + (min_x - max_x)**2
        
    common_x = np.linspace(min_x, max_x, 200)
    
    y_interps = []
    for x, y_N in scaled_curves:
        f = interp1d(x, y_N, kind='linear', bounds_error=False)
        y_interps.append(f(common_x))
        
    y_bar = np.nanmean(np.array(y_interps), axis=0)
    y_bar_func = interp1d(common_x, y_bar, kind='linear', fill_value="extrapolate")
    
    R = 0.0
    for x, y_N in scaled_curves:
        y_bar_xi = y_bar_func(x)
        R += np.sum((y_N - y_bar_xi) ** 2)
        
    return R

def run_fss_optimization(p_vals, N_vals, S_data, p_c_guess=0.48, nu_guess=1.33):
    bounds = [(0.0, 1.0), (0.5, 3.0)]
    res = minimize(
        compute_cost_function_safe, 
        [p_c_guess, nu_guess], 
        args=(p_vals, N_vals, S_data), 
        method='L-BFGS-B', 
        bounds=bounds
    )
    
    best_pc, best_nu = res.x
    estimates_pc = [best_pc]
    estimates_nu = [best_nu]
    
    for _ in range(4):
        rand_pc = best_pc + np.random.uniform(-0.02, 0.02)
        rand_nu = best_nu + np.random.uniform(-0.1, 0.1)
        rand_res = minimize(
            compute_cost_function_safe, [rand_pc, rand_nu], args=(p_vals, N_vals, S_data), 
            method='L-BFGS-B', bounds=bounds
        )
        estimates_pc.append(rand_res.x[0])
        estimates_nu.append(rand_res.x[1])
        
    mean_pc = np.mean(estimates_pc)
    mean_nu = np.mean(estimates_nu)
    std_pc = np.std(estimates_pc)
    std_nu = np.std(estimates_nu)
    
    final_cost = compute_cost_function_safe([mean_pc, mean_nu], p_vals, N_vals, S_data)
    
    return mean_pc, std_pc, mean_nu, std_nu, final_cost


# ==========================================
# 2. MULTIPROCESSING WORKER FUNCTION
# ==========================================
def process_window(window_task, L_array, S2_mean, sorted_p):
    idx_start, idx_end, p_min, p_max = window_task
    
    rho_cut = sorted_p[idx_start:idx_end + 1]
    S2_cut = S2_mean[:, idx_start:idx_end + 1]
    
    pc_opt, pc_std, nu_opt, nu_std, cost = run_fss_optimization(rho_cut, L_array, S2_cut)
    
    return {
        "len": p_max - p_min,
        "p_min": p_min,
        "p_max": p_max,
        "num_pts": len(rho_cut),
        "pc_opt": pc_opt,
        "pc_std": pc_std,
        "nu_opt": nu_opt,
        "nu_std": nu_std,
        "cost": cost
    }


# ==========================================
# 3. MAIN EXECUTION BLOCK (Required for MP)
# ==========================================
if __name__ == "__main__":
    
    # LOAD AND PREPARE DATA
    filepath = "data_mipt1/mipt_s2_coarse_results.pkl"

    with open(filepath, "rb") as f:
        payload = pickle.load(f)

    L_array = np.array(payload["N_list"], dtype=float)
    ansatz_list = payload["ansatz_list"]
    s2_mean_dict = payload["s2_mean"]

    s_list = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]

    valid_p_keys = list(payload["p_list"])
    valid_p_vals = [float(p) for p in valid_p_keys]
    rho_array = np.array(valid_p_vals)
    
    filter_tiers = [
        {"name": "STRICT",  "nu_min": 1.20, "nu_max": 1.60, "pc_std_max": 0.01, "nu_std_max": 0.05, "margin": 0.15},
        {"name": "RELAXED", "nu_min": 1.15, "nu_max": 1.65, "pc_std_max": 0.02, "nu_std_max": 0.08, "margin": 0.10},
        {"name": "LENIENT", "nu_min": 1.10, "nu_max": 1.70, "pc_std_max": 0.05, "nu_std_max": 0.15, "margin": 0.05}
    ]

    summary_results = {ansatz: {} for ansatz in ansatz_list}
    
    # Configure multiprocessing pool (Uses 40 if available, else max CPU cores)
    num_cores = mp.cpu_count()
    print(f"[*] Initializing multiprocessing with {num_cores} cores.\n")

    for ansatz in ansatz_list:
        for s_target in s_list:
            
            if s_target not in s2_mean_dict.get(ansatz, {}):
                continue

            print(f"\n{'='*85}")
            print(f"--- ANALYZING: {ansatz} | s = {s_target} ---")
            print(f"{'='*85}")

            S2_mean = np.zeros((len(L_array), len(rho_array)))
            for i, N in enumerate(L_array):
                for j, p_key in enumerate(valid_p_keys):
                    S2_mean[i, j] = s2_mean_dict[ansatz][s_target][float(p_key)][int(N)]

            sorted_p = np.sort(rho_array)
            search_space_idx = np.where((sorted_p >= 0.20) & (sorted_p <= 0.95))[0]
            
            # Generate all window tasks for multiprocessing
            tasks = []
            for i in range(len(search_space_idx)):
                for j in range(i + 3, len(search_space_idx)):
                    idx_start, idx_end = search_space_idx[i], search_space_idx[j]
                    p_min, p_max = sorted_p[idx_start], sorted_p[idx_end]
                    tasks.append((idx_start, idx_end, p_min, p_max))
            
            # Use partial to pass fixed arguments to the worker function
            worker_func = partial(process_window, L_array=L_array, S2_mean=S2_mean, sorted_p=sorted_p)
            
            # Execute tasks in parallel across the cluster CPU node
            with mp.Pool(processes=num_cores) as pool:
                all_results = pool.map(worker_func, tasks)
                    
            print(f"STEP 0: Total valid discrete windows evaluated: {len(all_results)}")

            final_candidates = []
            for tier in filter_tiers:
                print(f"\nApplying {tier['name']} Filters:")
                
                f1 = [r for r in all_results if tier["nu_min"] <= r["nu_opt"] <= tier["nu_max"]]
                print(f" -> STEP 1 (Physics Bounds {tier['nu_min']}-{tier['nu_max']}): Kept {len(f1)}")
                if not f1: continue
                
                f2 = [r for r in f1 if r["pc_std"] <= tier["pc_std_max"] and r["nu_std"] <= tier["nu_std_max"]]
                print(f" -> STEP 2 (Robustness pc_std<={tier['pc_std_max']}, nu_std<={tier['nu_std_max']}): Kept {len(f2)}")
                if not f2: continue
                
                f3 = []
                for r in f2:
                    margin = tier["margin"] * r["len"]
                    if (r["p_min"] + margin) <= r["pc_opt"] <= (r["p_max"] - margin):
                        f3.append(r)
                print(f" -> STEP 3 (Centering Margin >= {int(tier['margin']*100)}%): Kept {len(f3)}")
                
                if f3:
                    final_candidates = f3
                    print(f"\n[+] Candidates found under {tier['name']} limits. Halting relaxation.")
                    break 

            if not final_candidates:
                print("\n -> HALTING: No candidates survived even the most LENIENT filters.")
                continue

            final_candidates.sort(key=lambda x: x["cost"])
            best = final_candidates[0]
            
            summary_results[ansatz][s_target] = best
            
            print(f"\n*** BEST EXTRACTED PARAMETERS FOR {ansatz} (s={s_target}) ***")
            print(f"Optimal Window : [{best['p_min']:.3f}, {best['p_max']:.3f}] (Length: {best['len']:.3f}, Pts: {best['num_pts']})")
            print(f"Critical p_c   : {best['pc_opt']:.4f} ± {best['pc_std']:.4f}")
            print(f"Exponent nu    : {best['nu_opt']:.4f} ± {best['nu_std']:.4f}")
            print(f"Fit Cost (R)   : {best['cost']:.6f}")
            
            if len(final_candidates) > 1:
                runner_up = final_candidates[1]
                print(f"[Runner-Up Window: [{runner_up['p_min']:.3f}, {runner_up['p_max']:.3f}] -> p_c = {runner_up['pc_opt']:.4f}, nu = {runner_up['nu_opt']:.4f}, Cost = {runner_up['cost']:.6f}]")

    # ==========================================
    # 4. PRINT FINAL SUMMARY
    # ==========================================
    print("\n" + "="*85)
    print("FINAL SUMMARY: Extracted Critical Probabilities (p_c)")
    print("="*85)

    for ansatz in ansatz_list:
        if not summary_results[ansatz]:
            continue
        print(f"\nAnsatz: {ansatz}")
        for s_val, best_res in sorted(summary_results[ansatz].items(), reverse=True):
            print(f"  s = {s_val:.2f}  -->  p_c = {best_res['pc_opt']:.4f} ± {best_res['pc_std']:.4f}")
    print("\n" + "="*85)