# unitary.py: Functions to generate random unitaries, and energy-conserving unitaries for given system and bath Hamiltonians.
import numpy as np
import scipy.linalg as la

def random_unitary(n):
    """Haar-random unitary via QR decomposition."""
    X = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    Q, R = np.linalg.qr(X)
    d = np.diag(R)
    Q *= d / np.abs(d) # Multiply Q with normalized columns of R to ensure uniform distribution
    return Q


def deg_unitary(Hs, Hb, tol=1e-10, verify=True):
    """
    Energy-conserving unitary: Haar-random within each degenerate energy block of H_total = Hs * I + I * Hb.
    Assumes Hs and Hb are diagonal in the computational basis (i.e. we are working in the energy eigenbasis throughout).
    """
    dS = Hs.shape[0]
    dB = Hb.shape[0]
    dim = dS * dB
    degen_blocks = []

    Htot = np.kron(Hs, np.eye(dB)) + np.kron(np.eye(dS), Hb)
    energies = np.diag(Htot).real

    U = np.zeros((dim, dim), dtype=complex)
    used = np.zeros(dim, dtype=bool)

    # Finding degenerate energy blocks and filling each block with a random unitary
    for i, E in enumerate(energies):
        if used[i]: continue

        block = np.where(np.abs(energies - E) < tol)[0]
        used[block] = True
        k = len(block)

        if k > 1: degen_blocks.append(f"E={E:.4f}: {list(block)}")

        U[np.ix_(block, block)] = random_unitary(k)
    
    if degen_blocks: print(f"Degenerate blocks: {', '.join(degen_blocks)}")

    if verify:
        # Check unitarity
        err_unitary = np.max(np.abs(U @ U.conj().T - np.eye(dim)))
        assert err_unitary < 1e-10, f"U is not unitary, max error={err_unitary:.2e}"

        # Check energy conservation [U, H_total] = 0
        comm = U @ Htot - Htot @ U
        err_comm = np.max(np.abs(comm))
        assert err_comm < 1e-10, f"[U, H_total] ≠ 0, max error={err_comm:.2e}"

    return U