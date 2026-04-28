# lin_alg.py - functions for various matrix operations and measures used in energy and state generation calculations
import numpy as np
import scipy.linalg as la

# Partial transpose function for 2-qubit states
def partial_transpose(rho, sys=1):
    rho = rho.reshape(2,2,2,2)   # rho[i1, i2, j1, j2]
    if sys == 0:
        rho_pt = rho.transpose(2,1,0,3)  # swap i1 <-> j1
    else:
        rho_pt = rho.transpose(0,3,2,1)  # swap i2 <-> j2
    return rho_pt.reshape(4,4)

# Partial trace function for 2-qubit states
def partial_trace(rho, sys=1):
    # sys=0: trace out subsystem 1, keep subsystem 2
    # sys=1: trace out subsystem 2, keep subsystem 1
    rho = rho.reshape(2,2,2,2)
    return np.einsum('ijik->jk', rho) if sys == 0 else np.einsum('ijkj->ik', rho)

# Negativity measure of entanglement
def negativity(rho, tol=1e-12):
    evals = la.eigvalsh(partial_transpose(rho))
    neg = np.sum(np.abs(evals[evals < -tol]))
    return 0.0 if neg < tol else neg

# Von Neumann entropy calculator with base option
def vn_entropy(rho, base='e'):
    evals = np.linalg.eigvalsh(rho)
    evals = np.clip(evals, 0, 1)
    nonzero = evals > 1e-12

    if base == 2:
        log_fn = np.log2
    elif base == 'e' or base == np.e:
        log_fn = np.log
    elif base == 10:
        log_fn = np.log10
    else:
        raise ValueError(f"vn_entropy: unsupported base={base}. Use 2, 'e', or 10.")

    return -np.sum(evals[nonzero] * log_fn(evals[nonzero]))

# Function to compute relative entropy 
def rel_entropy(rho, sigma, tol=1e-12):
    """
    S(rho || sigma) = -S(rho) - Tr(rho ln sigma)
    Convention: ln(0) = 0 for zero eigenvalues of sigma.
    """
    evals_sigma, U_sigma = la.eigh(sigma)
    nonzero = evals_sigma > tol
    log_evals = np.where(nonzero, np.log(np.where(nonzero, evals_sigma, 1.0)), 0.0)
    log_sigma = U_sigma @ np.diag(log_evals) @ U_sigma.conj().T
    return -vn_entropy(rho) - np.real(np.trace(rho @ log_sigma))

# Function to compute the passive state of a given state
def passive_state(rho):
    pops = np.sort(np.linalg.eigvalsh(rho).real)[::-1]
    return np.diag(pops)
