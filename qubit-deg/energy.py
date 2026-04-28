# energy.py - functions to compute energy, free energy, passive energies, and ergotropies for two-qubit states under local Hamiltonians
import numpy as np
import scipy.linalg as la
from lin_alg import partial_trace, vn_entropy

# Convention: sigma_z = diag(-1, +1), so |0> is ground state (energy -w) and |1> is excited state (energy +w)
sigma_z = np.array([[-1, 0], [0, 1]], dtype=complex)
identity = np.eye(2, dtype=complex)

# ------------------------------
# Helper functions for numerical stability and cleaning small values
# ------------------------------
def clean(x, tol=1e-12, precision=12):
    x = float(np.real(x))
    if abs(x) < tol: return 0.0
    return float(np.round(x, precision))

def clean_spectrum(evals):
    evals = np.clip(evals, 0, None)
    return evals / evals.sum()

# --------------------------------------
# Functions to compute system Hamiltonina and energies
# --------------------------------------
def system_hamiltonian(w1=1.0, w2=2.0):
    return w1 * np.kron(sigma_z, identity) + w2 * np.kron(identity, sigma_z)

def system_energies(w1, w2):
    H = system_hamiltonian(w1, w2)
    return np.sort(la.eigvalsh(H))

def energy(rho, w1=1.0, w2=2.0):
    H = system_hamiltonian(w1, w2)
    return clean(np.trace(rho @ H))

def free_energy(rho, H, beta):
    E = np.trace(H @ rho).real
    S = vn_entropy(rho, base='e')  # nats, consistent with 1/beta
    return E - (S / beta)

# --------------------------------------
# Functions to compute passive energies (g: global, l: local) and ergotropies
# --------------------------------------

def passive_energy_g(rho, w1=1.0, w2=2.0):
    # Populations descending paired with energies ascending
    pops = np.sort(clean_spectrum(la.eigvalsh(rho)))[::-1]
    energies = np.sort(system_energies(w1, w2))          # ascending
    return clean(np.dot(pops, energies))

def passive_energy_l(rho, w1=1.0, w2=2.0):
    rho1 = partial_trace(rho, sys=1)   # reduced state of qubit 1
    rho2 = partial_trace(rho, sys=0)   # reduced state of qubit 2

    pops1 = np.sort(clean_spectrum(la.eigvalsh(rho1)))[::-1]  # descending
    pops2 = np.sort(clean_spectrum(la.eigvalsh(rho2)))[::-1]

    E1 = np.sort([-w1, w1])   # ascending: [-w1, +w1]
    E2 = np.sort([-w2, w2])   # ascending: [-w2, +w2]

    return clean(np.dot(pops1, E1) + np.dot(pops2, E2))

def global_ergo(rho, w1=1.0, w2=2.0):
    return clean(energy(rho, w1, w2) - passive_energy_g(rho, w1, w2))

def local_ergo(rho, w1=1.0, w2=2.0):
    return clean(energy(rho, w1, w2) - passive_energy_l(rho, w1, w2))

def ergotropy_gap(rho, w1=1.0, w2=2.0):
    return clean(global_ergo(rho, w1, w2) - local_ergo(rho, w1, w2))