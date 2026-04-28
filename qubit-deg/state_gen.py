# state_gen.py - functions to generate random two-qubit states of various types for testing and analysis
import numpy as np
import scipy.linalg as la
from lin_alg import negativity

MAX_RETRIES = 1000

# --------------------------------------
# Function to generate product states
# --------------------------------------

def random_product_state(pure=True):
    if pure:
        psi1 = np.random.randn(2) + 1j*np.random.randn(2)
        psi1 /= la.norm(psi1)
        psi2 = np.random.randn(2) + 1j*np.random.randn(2)
        psi2 /= la.norm(psi2)
        rho1 = np.outer(psi1, psi1.conj())
        rho2 = np.outer(psi2, psi2.conj())
    else:
        A = np.random.randn(2,2) + 1j*np.random.randn(2,2)
        rho1 = A @ A.conj().T
        rho1 /= np.trace(rho1)
        B = np.random.randn(2,2) + 1j*np.random.randn(2,2)
        rho2 = B @ B.conj().T
        rho2 /= np.trace(rho2)

    return np.kron(rho1, rho2)


# --------------------------------------
# Function to generate separable states
# --------------------------------------

def random_separable_state(n_terms=3):
    probs = np.random.rand(n_terms)
    probs /= probs.sum()
    rho = np.zeros((4,4), dtype=complex)
    for p in probs: 
        rho += p * random_product_state(pure=False)  # Convex combination of product states
    return rho


# ----------------------------------------------
# Function to generate pure entangled states
# ----------------------------------------------

def random_pure_entangled_state():
    for _ in range(MAX_RETRIES):
        psi = np.random.randn(4) + 1j*np.random.randn(4)
        psi /= la.norm(psi)
        rho = np.outer(psi, psi.conj())
        if negativity(rho) > 1e-6: 
            return rho   # Check for entanglement using negativity
    raise RuntimeError("random_pure_entangled_state: failed to find entangled state")


# ------------------------------------------------------------------------------
# Function to generate Schmidt entangled states \sqrt(a)|00> + \sqrt(1-a)|11>
# ------------------------------------------------------------------------------

def schmidt_ent_state(a=0.5):
    if not (0 < a < 1):
        raise ValueError(f"schmidt_ent_state: parameter a={a} must be strictly in (0,1)")
    psi = np.array([np.sqrt(a), 0, 0, np.sqrt(1 - a)], dtype=complex)
    return np.outer(psi, psi.conj())


# --------------------------------------
# Function to generate Werner states
# --------------------------------------
        # BUG FIX: use purity instead of rank for mixedness check
def random_werner_state(p=None):
    # Default to entangled regime p > 1/3
    if p is None:
        p = np.random.uniform(1/3, 1)
    psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    bell = np.outer(psi, psi.conj())
    noise = np.eye(4) / 4
    return p * bell + (1 - p) * noise


# ----------------------------------------------
# Function to generate mixed entangled states
# ----------------------------------------------

def random_mixed_entangled_state(min_purity_gap=0.01):
    for _ in range(MAX_RETRIES):
        L = np.random.randint(2, 8)      # Weights for convex combination of pure states
        weights = np.random.rand(L)
        weights /= weights.sum()

        rho = np.zeros((4,4), dtype=complex)
        for i in range(L):
            rho += weights[i] * random_pure_entangled_state()

        # Check if state is mixed and entangled using purity and negativity
        purity = np.trace(rho @ rho).real
        is_mixed = purity < (1 - min_purity_gap)
        is_entangled = negativity(rho) > 1e-6

        if is_mixed and is_entangled:
            return rho

    raise RuntimeError("random_mixed_entangled_state: failed after max retries")



# -----------------------------------------------------------------------------------------------------
# Unified state generator that can produce various types of two-qubit states based on input parameters
# -----------------------------------------------------------------------------------------------------

def generate_state(kind="random", **kwargs):

    # If kind is "random", randomly select a type of state to generate
    if kind == "random":
        choices = ["product", "separable", "pure_ent", "schmidt_ent", "werner", "mixed_ent"]
        kind = np.random.choice(choices)

    if kind == "product":
        return random_product_state(pure=kwargs.get("pure", True))
    elif kind == "separable":
        return random_separable_state(n_terms=kwargs.get("n_terms", 3))
    elif kind == "pure_ent":
        return random_pure_entangled_state()
    elif kind == "schmidt_ent":
        return schmidt_ent_state(a=kwargs.get("a", 0.5))
    elif kind == "werner":
        return random_werner_state(p=kwargs.get("p", None))
    elif kind == "mixed_ent":
        return random_mixed_entangled_state()
    else:
        raise ValueError(f"Unknown state type: {kind}")


# ----------------------------------------------------------------------------------------
# Function to generate mixed qubit states using extended Hilbert space and partial trace
# ----------------------------------------------------------------------------------------

def random_mixed_qubit_hs():
    # Embed in larger space and partial trace
    psi = np.random.randn(4) + 1j*np.random.randn(4)
    psi /= la.norm(psi)
    rho_full = np.outer(psi, psi.conj()).reshape(2,2,2,2)
    return np.einsum('ijik->jk', rho_full)  # partial trace over ancilla