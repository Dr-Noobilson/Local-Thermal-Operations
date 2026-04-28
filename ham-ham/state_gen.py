# state_gen.py - utility functions for generating random density matrices
import numpy as np
import scipy.linalg as la


def clean_matrix(rho, precision=12):
    # enforce hermiticity
    rho = (rho + rho.conj().T) / 2

    # round real and imaginary parts
    rho = np.round(rho.real, precision) + 1j * np.round(rho.imag, precision)

    # clip negative eigenvalues introduced by rounding
    evals, evecs = la.eigh(rho)
    evals = np.clip(evals, 0, None)
    rho = evecs @ np.diag(evals) @ evecs.conj().T

    # enforce trace 1
    rho /= np.trace(rho)

    return rho


def generate_state(d, seed=None):
    """
    Generate Haar-uniform random density matrix of dimension d.
    Method: Haar-random pure state on d^2 then partial trace over d.
    This gives Hilbert-Schmidt uniform mixed states. arXiv:1010.3570
    """
    rng = np.random.default_rng(seed)
    
    # Haar-random pure state on d x d environment
    X = rng.standard_normal((d * d, d)) + 1j * rng.standard_normal((d * d, d))
    Q, _ = np.linalg.qr(X)          # Q is Haar-random unitary, take first d columns
    psi = Q[:, 0]                    # Haar-random pure state on d^2
    
    # partial trace over environment (reshape to d x d, trace out second)
    psi_matrix = psi.reshape(d, d)
    rho = psi_matrix @ psi_matrix.conj().T
    rho /= np.trace(rho)
    
    return clean_matrix(rho)