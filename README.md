# Variance of Gradients using MILT and SPMM

This repository calculates the variance of gradients in quantum circuits, building upon the [Measurement-Induced Local Trajectories (MILT) framework](https://arxiv.org/abs/2312.09135). It adapts the original implementation from [rappaport-dev/Barren-Plateau](https://github.com/rappaport-dev/Barren-Plateau) by replacing standard projective measurements with the [Softened Projective Measurement Model (SPMM)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.110.064301). 

## Theoretical Background

### Analytical Gradients and the Mixed Cost Function
In the MILT framework, a quantum circuit consists of layers of unitary operations followed by measurements, producing a specific trajectory of measurement outcomes denoted as $\mathbf{M}$. The unnormalized state vector conditioned on this trajectory is written as $|\tilde{\psi}_{\mathbf{M}}(\boldsymbol{\theta})\rangle$. 

When calculating the analytical gradient of a projective cost function for a single trajectory, the expression yields two terms: 

$$
\partial_j C_{\mathbf{M}}(\boldsymbol{\theta}) = 2\text{Re}\left[ \frac{\langle \partial_j \tilde{\psi}_{\mathbf{M}} | O | \tilde{\psi}_{\mathbf{M}} \rangle}{p_{\mathbf{M}}} - \frac{\langle \tilde{O} \rangle_{\mathbf{M}}}{p_{\mathbf{M}}^2} \langle \partial_j \tilde{\psi}_{\mathbf{M}} | \tilde{\psi}_{\mathbf{M}} \rangle \right]
$$
    
The first term captures the change in the unnormalized state, while the second term corrects for the drift in the trajectory probability itself. 

However, our implementation focuses on the mixed cost function, which averages the cost over all possible measurement trajectories. For the mixed cost function, the derivatives of the normalization factors cancel out. Therefore, when analytically calculating the gradients in our code, we entirely skip the second term. The gradient simplifies to just the sum of the unnormalized drifts:

$$
\partial_j C(\boldsymbol{\theta}) = \sum_{\mathbf{M}} 2\text{Re} \left[ \langle \partial_j \tilde{\psi}_{\mathbf{M}}(\boldsymbol{\theta}) | O | \tilde{\psi}_{\mathbf{M}}(\boldsymbol{\theta}) \rangle \right]
$$

### Softened Projective Measurement Model (SPMM)
Instead of the standard projective measurements used in the original MILT paper, we incorporate SPMM. Functionally, SPMM acts just like a standard projective measurement with two possible outcomes, but it applies a tunable scaling parameter $\Lambda \in [0,1]$ to control the measurement strength. 

The weak measurement operators at a given site $j$ are defined as:

$$
\hat{P}_{\pm}^{(j)} = \frac{1 \pm \Lambda \sigma_z^{(j)}}{\sqrt{2(1+\Lambda^2)}}
$$

To perform the measurement in the simulation, we calculate the Born probabilities for the two outcomes based on the current system wave function $|\psi\rangle$:

$$
p_{\pm}(\Lambda) = \frac{1}{2(1+\Lambda^2)} \left( 1 + \Lambda^2 \pm 2\Lambda \langle \psi | \sigma_z^{(j)} | \psi \rangle \right)
$$

Based on these probabilities, we measure one of the two outcomes and update the state vector exactly as we would in a standard projection:

$$
|\psi\rangle \to \frac{\hat{P}_{\pm}^{(j)}|\psi\rangle}{||\hat{P}_{\pm}^{(j)}|\psi\rangle||}
$$

## Code Implementation

The simulation is implemented using JAX for high-performance GPU acceleration and XLA compilation. To ensure scalability and maintain memory efficiency, the simulator avoids constructing full density matrices. Instead, it employs a tensor-network-style state vector approach. 

### State Evolution and Gradient Tracking
At the core of the algorithm, we evaluate the quantum circuit for a fixed measurement probability ($p$), number of qubits ($N$), and SPMM softening strength ($s$). For each sample (or shot), we randomly generate the rotation parameters, measurement configurations, and quantum jump outcomes. 

To calculate the analytical gradient described in the MILT framework, the code simultaneously tracks and updates two unnormalized state vectors throughout the circuit:
* **`psi0`**: The standard, unperturbed state vector corresponding to $\vert{}\tilde{\psi}_{\mathbf{M}}\rangle$.
* **`psi1`**: The derivative state vector corresponding to $\vert{}\partial_j \tilde{\psi}_{\mathbf{M}}\rangle$, which applies the differentiated unitary generator at the specific target parameter index. 

By propagating both states through the exact same measurement trajectory and evaluating them at the end of the circuit using the `compute_cost` function, we directly compute the unnormalized drift term required for the mixed cost function gradient.

### Efficient Gate Operations
Applying full $2^N \times 2^N$ unitary matrices to the state vector would be computationally prohibitive. Instead, single-qubit and two-qubit operations (`apply_1q_gate`, `apply_cnot_adjacent`) are implemented via tensor reshaping. 

By reshaping the 1D state array into a 3D tensor of the form `(dim_left, gate_dim, dim_right)`, we isolate the target qubit's subspace. The gate operations are then applied locally as vectorized linear combinations of the tensor slices. This strictly bounds memory usage and significantly accelerates execution. Furthermore, entire blocks of parameterized rotation matrices are generated simultaneously using `jax.vmap` (O(1) graph compression), taking full advantage of JAX's accelerated linear algebra compiler.

### Optimized SPMM Probability Calculation
The implementation of the Softened Projective Measurement Model requires calculating the Born probabilities for the two possible outcomes. The theoretical definition requires computing the expectation value $\langle \psi \vert{} \sigma_z \vert{} \psi \rangle$. 
    
In our optimized `spmm_measure_step` function, this calculation is highly streamlined. We know that the expectation of $\sigma_z$ can be expressed as the difference in probabilities of the computational basis states ($p_0 - p_1$), and that the total probability in that local subspace is $p_0 + p_1 = 1$. By substituting these relationships into the theoretical SPMM probability equation, we can calculate the probability of the "+" outcome (`p_plus`) directly using the squared diagonal elements of the measurement operator: `prob_0 * P_plus_sq[0] + prob_1 * P_plus_sq[1]`. 

This mathematically equivalent substitution allows us to compute the required trajectory probabilities using purely scalar arithmetic, completely bypassing the need for costly matrix multiplications during the sequential measurement steps.

## Repository Structure

* **`cluster_gpu1.py`**: The JAX implementation used for cluster simulations. This is the script detailed in the "Code Implementation" section above, featuring tensor reshaping and XLA compilation.
* **`laptop_milt_analysis_var.py`**: The initial, foundational codebase.
* **`data_anlysis/`**: This directory contains the raw output data generated from the cluster runs, along with the subsequent evaluations.
    * **`analysis1.ipynb`**: The primary Jupyter Notebook containing the data processing code and the final plots presented in our recent meeting. 
    * **`sweep_results_N4_16_32bit_3N.pkl`** & **`sweep_config_metadata_3N_5k.txt`**: The raw output data files and metadata configurations produced by the cluster sweeps.
