# lin_alg.py: linear algebra utilities for two-qubit states.
import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize
from tqdm import tqdm


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

# Function to compute mutual information of a two-qubit state
def mutual_info(rho):
    rho1 = partial_trace(rho, sys=1)
    rho2 = partial_trace(rho, sys=0)
    return vn_entropy(rho1) + vn_entropy(rho2) - vn_entropy(rho)

# Function to compute purity of a state
def purity(rho):
    return np.real(np.trace(rho @ rho))


# Function to compute classical correlations and quantum discord for a two-qubit state
def correlations(rho, n_trials=200):
    """
    Compute classical correlations C and quantum discord Q for a two-qubit state rho via optimization over measurements on B.
    C(A:B) = max over projective measurements on B of: S(rho_A) - sum_b p_b S(rho_A|b)
    Q(A:B) = I(A:B) - C(A:B)

    Args:
        rho:      (4,4) two-qubit density matrix
        n_trials: random restarts for optimization

    Returns: C, Q  (classical correlations, quantum discord)
    """
    

    rho_A = partial_trace(rho, sys=1)   # trace out B
    rho_B = partial_trace(rho, sys=0)   # trace out A
    SA    = vn_entropy(rho_A)
    MI    = mutual_info(rho)     

    def conditional_entropy_A(params):
        """
        S(A|B measurement) for projective measurement on B
        parameterized by (theta, phi) on Bloch sphere.
        """
        theta, phi = params

        # projector |b0><b0| and |b1><b1| on B
        b0 = np.array([np.cos(theta/2), np.exp(1j*phi) * np.sin(theta/2)])
        b1 = np.array([-np.exp(-1j*phi) * np.sin(theta/2), np.cos(theta/2)])

        Pi0 = np.outer(b0, b0.conj())   # (2,2)
        Pi1 = np.outer(b1, b1.conj())   # (2,2)

        cond_S = 0.0
        for Pi in [Pi0, Pi1]:
            # extend to full space: I_A ⊗ Pi_B
            M   = np.kron(np.eye(2), Pi)

            # post-measurement (unnormalized) state of A
            rho_b     = M @ rho @ M.conj().T       # (4,4)
            p_b       = np.real(np.trace(rho_b))   # probability

            if p_b < 1e-12: continue

            # reduced state of A given outcome b
            rho_A_b   = partial_trace(rho_b, sys=1) / p_b
            cond_S   += p_b * vn_entropy(rho_A_b)

        return cond_S   # we want to minimize this

    # ── optimize over (theta, phi) ────────────────────────────────
    best_C = -np.inf

    for _ in tqdm(range(n_trials), desc="Optimizing classical correlations", unit="trial"):
        # random initial angles
        theta0 = np.random.uniform(0, np.pi)
        phi0   = np.random.uniform(0, 2*np.pi)

        # Using L-BFGS-B optimization method
        res = minimize(conditional_entropy_A,
                       x0     = [theta0, phi0],
                       method = 'L-BFGS-B',
                       bounds = [(0, np.pi), (0, 2*np.pi)],
                       options = {'ftol': 1e-12, 'gtol': 1e-8})

        # C = S(A) - min_b S(A|b)
        C_candidate = SA - res.fun
        if C_candidate > best_C:
            best_C = C_candidate

    C = np.clip(best_C, 0, None)   # numerical safety
    Q = np.clip(MI - C,  0, None)

    return C, Q