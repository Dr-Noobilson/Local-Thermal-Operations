# energy.py - functions to compute energy, free energy, passive energies, and ergotropies for harmonic states
import numpy as np
import scipy.linalg as la
from lin_alg import partial_trace, vn_entropy, passive_state


# ---------------------------------------------------------------------
# Helper functions for numerical stability and cleaning small values
# ----------------------------------------------------------------------
def clean(x, tol=1e-12, precision=12):
    x = float(np.real(x))
    if abs(x) < tol: return 0.0
    return float(np.round(x, precision))

def clean_spectrum(evals, precision=12):
    evals = np.clip(evals.real, 0, None)
    evals /= evals.sum()
    return np.round(evals, precision)


# ------------------------------------------------------------------
# Function to compute harmonic oscillator Hamiltonian and energies
# ------------------------------------------------------------------

def number_operator(d):
    return np.diag(np.arange(d, dtype=complex))

# def single_hamiltonian(d, w=1.0):
#     """Single harmonic oscillator Hamiltonian. Energies: 0, w, 2w, ...(d-1)w"""
#     return w * number_operator(d)

def system_hamiltonian(dA, dB, w1=1.0, w2=1.0):
    NA = number_operator(dA)
    NB = number_operator(dB)
    IA = np.eye(dA)
    IB = np.eye(dB)
    return w1 * np.kron(NA, IB) + w2 * np.kron(IA, NB)

def system_energies(dA, dB, w1, w2):
    H = system_hamiltonian(dA, dB, w1, w2)
    return np.sort(la.eigvalsh(H))

# --------------------------
# Energy expectation
# --------------------------

def energy(rho, dA, dB, w1=1.0, w2=1.0):
    H = system_hamiltonian(dA, dB, w1, w2)
    return clean(np.trace(rho @ H))

def free_energy(rho, H, beta):
    """F = Tr(Hρ) - S(ρ)/β  in nats."""
    E = np.real(np.trace(H @ rho))
    S = vn_entropy(rho, base='e')
    return E - (S / beta)


# -----------------------------------------------------------------------------
# Functions to compute passive energies (g: global, l: local) and ergotropies
# -----------------------------------------------------------------------------

def passive_energy_g(rho, dA, dB, w1=1.0, w2=1.0):
    pops    = np.sort(clean_spectrum(la.eigvalsh(rho)))[::-1]   # descending
    energies = system_energies(dA, dB, w1, w2)                  # ascending
    return clean(np.dot(pops, energies))

def passive_energy_l(rho, dA, dB, w1=1.0, w2=1.0):
    rho1 = partial_trace(rho, dA, dB, sys=1)
    rho2 = partial_trace(rho, dA, dB, sys=0)

    pops1 = np.sort(clean_spectrum(la.eigvalsh(rho1)))[::-1]
    pops2 = np.sort(clean_spectrum(la.eigvalsh(rho2)))[::-1]

    EA = np.sort(w1 * np.arange(dA))   # ascending: 0, w1, 2w1, ...
    EB = np.sort(w2 * np.arange(dB))   # ascending: 0, w2, 2w2, ...

    return clean(np.dot(pops1, EA) + np.dot(pops2, EB))

def passive_energy(rho, H):
    """Passive energy for a single subsystem."""
    pass_rho = passive_state(rho)
    return np.real(np.trace(H @ pass_rho))

# --------------------------
# Ergotropies
# --------------------------

def global_ergo(rho, dA, dB, w1=1.0, w2=1.0):
    return clean(energy(rho, dA, dB, w1, w2) - passive_energy_g(rho, dA, dB, w1, w2))

def local_ergo(rho, dA, dB, w1=1.0, w2=1.0):
    return clean(energy(rho, dA, dB, w1, w2) - passive_energy_l(rho, dA, dB, w1, w2))

def ergotropy_gap(rho, dA, dB, w1=1.0, w2=2.0):
    return clean(global_ergo(rho, dA, dB, w1, w2) - local_ergo(rho, dA, dB, w1, w2))


