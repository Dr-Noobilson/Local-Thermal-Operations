# Local Thermal Operations (LTO) & Ergotropy Optimization
### Numerical Methods 2026 Project | Harish-Chandra Research Institute (HRI)

This repository contains the numerical suite developed for the investigation of ergotropy monotonicity and advantage under Local Thermal Operations (LTO). The project explores various bipartite system-bath configurations, ranging from non-degenerate qubit-qubit to harmonic-harmonic oscillators,to determine the physical limits and advantages in work extraction from shared quantum correlations following local thermal operations.

## Project Goals
* **Monotonicity Testing:** Numerically verify if $\Delta \mathcal{R} \le 0$ holds across diverse Hilbert space dimensions.
* **Optimization:** Utilize Nelder-Mead and L-BFGS-B algorithms to search for ergotropic advantages in the unitary manifold.
* **Correlation Mapping:** Analyze the trade-offs between ergotropy and various correlation measures like Negativity, Mutual Information, and Quantum Discord.

---

## Directory Structure
The repository is organized by physical architecture. While each folder represents a different system-bath setup, they share a standardized modular structure.

### Standard Module Overview (Example: `qubit-ham/`)
Each directory contains the following core components:

* **`state_gen.py`**: State Sampling for different state classes (Pure Entangled, Separable, Werner, etc.).
* **`energy.py`**: Handles Hamiltonian construction and various energy/work related calculations.
* **`lin_alg.py`**: Linear Algebra utility functions as well calculation of correlation measures (Negativity, Mutual Information, etc.).
* **`unitary.py`**: Parameterizes energy-conserving unitaries using the exponential map for the optimization routines.
* **`local_to.py`**: The primary LTO engine; simulates the interaction between subsystems and their respective thermal reservoirs.
* **`optimize.py`**: Script for the automated search of the optimal LTO unitary for maximizing ergotropy.

### Analysis Notebooks (`.ipynb`)
* **`optim.ipynb`**: Execution of Nelder-Mead and L-BFGS-B optimization restarts to identify the maximum possible $\Delta \mathcal{R}$ for a given state.
* **`vs_neg.ipynb`**: Visualization of Ergotropy Change ($\Delta \mathcal{R}$) vs. Initial Correlation Measure.
* **`vs_neg_diff.ipynb`**: Advanced analysis of the Ergotropic Change vs. correlation dissipation.
* **`vs_beta.ipynb`** (Exclusive to `ham-ham/`): Generates phase diagrams tracking ergotropy change against temperature and frequency gradients.

---

## Architectures Investigated
1.  **`qubit-nondeg`**: Bipartite qubit system and qubit baths with local system-bath Hamiltonian being non-degenerate.
2.  **`qubit-deg`**: Investigation of degenerate subspaces as a potential "unlock" for ergotropic gain.
3.  **`qubit-ham`**: Mixed architecture coupling discrete qubits to high-dimensional harmonic baths ($d=10$).
4.  **`ham-ham`**: Purely harmonic systems focusing on the large-dimension limit and resonance effects.

## Usage
To replicate the results for a specific architecture (e.g., Qubit-Harmonic):
1. Navigate to the directory: `cd qubit-ham`
2. Use `vs_neg.ipynb` to simulate the LTO process across a large ensemble of initial states and a fixed/random/optimized unitary, generating the $\Delta \mathcal{R}$ vs. Initial Correlation plots.
3. Use `vs_neg_diff.ipynb` to process the resulting data and generate the correlation-work trade-off plots.
4. Run the optimization notebook `optim.ipynb` to generate the Master-Tracker Optimization logs.

## Installation & Requirements

### Dependencies
The project requires `Python 3.8+` and the following libraries:
* **`NumPy`**: Core numerical operations and array handling.
* **`SciPy`**: Optimization routines (Nelder-Mead, L-BFGS-B) and linear algebra.
* **`Matplotlib`**: Plotting the ergotropy heatmaps and correlation scatter plots.
* **`tqdm`**: Progress bars for the multi-restart optimization sweeps.
* **`Jupyter Notebook`**: Environment for running the `.ipynb` analysis files.

### Installation
1. Clone the repository:
```bash
git clone git@github.com:Dr-Noobilson/Local-Thermal-Operations.git
cd Local-Thermal-Operations
```
2. Create a virtual environment and install dependencies:
```bash
python -m venv lto_env
source lto_env/bin/activate  
pip install -r requirements.txt
```