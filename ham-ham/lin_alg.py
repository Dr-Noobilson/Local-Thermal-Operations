# lin_alg.py: Linear algebra utilities for quantum states and correlations
import numpy as np
import scipy.linalg as la


# Function to compute the partial trace of a density matrix of arbitrary dimensions
def partial_trace(rho, dA, dB, sys=1):
    """
    sys=0: trace out A, keep B
    sys=1: trace out B, keep A
    """
    r = rho.reshape(dA, dB, dA, dB)
    if sys == 0: return np.einsum('ijik->jk', r)
    else:  return np.einsum('ijkj->ik', r)

# Function to compute the partial transpose of a density matrix of arbitrary dimensions
def partial_transpose(rho, dA, dB, sys=1):
    r = rho.reshape(dA, dB, dA, dB)
    if sys == 0: rho_pt = r.transpose(2, 1, 0, 3)
    else: rho_pt = r.transpose(0, 3, 2, 1)
    return rho_pt.reshape(dA*dB, dA*dB)

# Function to compute the von Neumann entropy of a density matrix with specified logarithm base
def vn_entropy(rho, base='e'):
    evals = np.linalg.eigvalsh(rho)
    evals = np.clip(evals, 0, None)
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


# Function to compute the relative entropy between two density matrices
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

# Function to compute the passive state of a given density matrix
def passive_state(rho):
    pops = np.sort(np.linalg.eigvalsh(rho).real)[::-1]
    return np.diag(pops)

# Function to compute the mutual information of a bipartite state
def mutual_info(rho, dA, dB):
    rho_A = partial_trace(rho, dA, dB, sys=1)
    rho_B = partial_trace(rho, dA, dB, sys=0)
    return vn_entropy(rho_A) + vn_entropy(rho_B) - vn_entropy(rho)


# Function to compute the purity of a density matrix
def purity(rho):
    return np.real(np.trace(rho @ rho))
