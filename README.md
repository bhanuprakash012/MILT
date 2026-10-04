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
*(Content pending code snippet)*
