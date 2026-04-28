# optimize.py - optimization routines for unitaries in LTO protocol
import numpy as np
import scipy.linalg as la
from tqdm import tqdm
from scipy.optimize import minimize

from state_gen import generate_state
from unitary import deg_unitary
from local_to import sho_ham, gibbs_state, LTO_step
from energy import global_ergo, local_ergo, ergotropy_gap


# ------------------------
# Helpers
# ------------------------

def get_blocks(Hs, Hb, tol=1e-10):
    """Identify degenerate energy blocks of H_total = Hs⊗I + I⊗Hb."""
    dS = Hs.shape[0]
    dB = Hb.shape[0]
    Htot     = np.kron(Hs, np.eye(dB)) + np.kron(np.eye(dS), Hb)
    energies = np.diag(Htot).real
    used     = np.zeros(dS * dB, dtype=bool)
    blocks   = []

    for i, E in enumerate(energies):
        if used[i]:
            continue
        block = np.where(np.abs(energies - E) < tol)[0]
        used[block] = True
        blocks.append(block)

    return blocks


def unitary_from_params(theta, k):
    """
    Parameterize a k×k unitary via exp(i*H) where H is Hermitian.
    theta is a real vector of length k^2:
      - first k entries:         real diagonal elements of H
      - remaining k(k-1) entries: pairs (re, im) for upper triangle of H
    """
    H   = np.zeros((k, k), dtype=complex)
    idx = 0

    for i in range(k):
        H[i, i] = theta[idx]
        idx += 1

    for i in range(k):
        for j in range(i + 1, k):
            H[i, j] = theta[idx] + 1j * theta[idx + 1]
            H[j, i] = H[i, j].conj()
            idx += 2

    return la.expm(1j * H)


def deg_unitary_from_params(Hs, Hb, blocks, params_dict):
    """
    Build full energy-conserving unitary from block parameters.
    params_dict = {block_id: theta_vector}
    """
    dS  = Hs.shape[0]
    dB  = Hb.shape[0]
    dim = dS * dB
    U   = np.zeros((dim, dim), dtype=complex)

    for bid, block in enumerate(blocks):
        k     = len(block)
        theta = params_dict.get(bid, np.zeros(k * k))
        U[np.ix_(block, block)] = unitary_from_params(theta, k)

    return U


def params_to_unitaries(x, blocks_a, blocks_b, Hs1, Hb1, Hs2, Hb2):
    """Split flat parameter vector x into Ua and Ub."""
    sizes_a = [len(b) ** 2 for b in blocks_a]
    sizes_b = [len(b) ** 2 for b in blocks_b]

    idx      = 0
    params_a = {}
    for bid, s in enumerate(sizes_a):
        params_a[bid] = x[idx:idx + s]
        idx += s

    params_b = {}
    for bid, s in enumerate(sizes_b):
        params_b[bid] = x[idx:idx + s]
        idx += s

    Ua = deg_unitary_from_params(Hs1, Hb1, blocks_a, params_a)
    Ub = deg_unitary_from_params(Hs2, Hb2, blocks_b, params_b)

    return Ua, Ub


# ---------------------------------
# Random search - fixed state
# ---------------------------------

def random_search(rho12, gamma_Ra, gamma_Rb, Hs1, Hs2, Hb1, Hb2, dA, dB, w1, w2, n_trials=10000):
    """
    Random search over thermal operation unitaries for a fixed state.
    Tracks all three targets independently.
    CHANGE from qubit: dA, dB passed explicitly for energy functions.
    """

    # scores before LTO
    Rg_before  = global_ergo(rho12, dA, dB, w1, w2)
    Rl_before  = local_ergo(rho12,  dA, dB, w1, w2)
    gap_before = ergotropy_gap(rho12, dA, dB, w1, w2)

    best = {
        'global': (-np.inf, None, None, None),
        'local':  (-np.inf, None, None, None),
        'gap':    (-np.inf, None, None, None),
    }

    for _ in tqdm(range(n_trials), desc="Random search", unit="trial", dynamic_ncols=True):

        Ua = deg_unitary(Hs1, Hb1, verify=False)
        Ub = deg_unitary(Hs2, Hb2, verify=False)

        sigma12, _, _ = LTO_step(rho12, gamma_Ra, gamma_Rb, Ua, Ub)

        delta_g   = global_ergo(sigma12,  dA, dB, w1, w2) - Rg_before
        delta_l   = local_ergo(sigma12,   dA, dB, w1, w2) - Rl_before
        delta_gap = ergotropy_gap(sigma12, dA, dB, w1, w2) - gap_before

        if delta_g   > best['global'][0]:
            best['global'] = (delta_g,   sigma12.copy(), Ua.copy(), Ub.copy())
        if delta_l   > best['local'][0]:
            best['local']  = (delta_l,   sigma12.copy(), Ua.copy(), Ub.copy())
        if delta_gap > best['gap'][0]:
            best['gap']    = (delta_gap, sigma12.copy(), Ua.copy(), Ub.copy())

    print("\n" + "="*50)
    print("  RANDOM SEARCH RESULTS")
    print("="*50)
    for target in ['global', 'local', 'gap']:
        delta = best[target][0]
        print(f"  Best Δ({target:6s}) = {delta:.6f}  "
              f"{'✓ > 0' if delta > 1e-8 else '✗ not found'}")
    print("="*50)

    return best


# -------------------------------------------------------------
# Random search - over states and unitaries
# -------------------------------------------------------------

def search_over_states(n_states, n_unitaries, beta_a, beta_b, dA, dB, w1, w2, omega_a, omega_b, bath_dim_a, bath_dim_b, state_kind='random'):
    """
    Double loop random search over both states and unitaries.
    CHANGE from qubit: dA, dB for system dimensions passed explicitly.
    generate_state now takes dimension d=dA=dB.
    """

    # build Hamiltonians once
    Hs1 = sho_ham(dA, w1);            Hs2 = sho_ham(dB, w2)
    Hb1 = sho_ham(bath_dim_a, omega_a); Hb2 = sho_ham(bath_dim_b, omega_b)

    gamma_Ra = gibbs_state(Hb1, beta_a)
    gamma_Rb = gibbs_state(Hb2, beta_b)

    best = {
        'global': (-np.inf, None, None, None, None),
        'local':  (-np.inf, None, None, None, None),
        'gap':    (-np.inf, None, None, None, None),
    }

    for s in tqdm(range(n_states), desc="Searching states", unit="state", dynamic_ncols=True):

        # CHANGE: generate_state now needs dimension dA*dB
        rho12 = generate_state(dA * dB)

        Rg_before  = global_ergo(rho12,  dA, dB, w1, w2)
        Rl_before  = local_ergo(rho12,   dA, dB, w1, w2)
        gap_before = ergotropy_gap(rho12, dA, dB, w1, w2)

        for _ in range(n_unitaries):

            Ua = deg_unitary(Hs1, Hb1, verify=False)
            Ub = deg_unitary(Hs2, Hb2, verify=False)

            sigma12, _, _ = LTO_step(rho12, gamma_Ra, gamma_Rb, Ua, Ub)

            delta_g   = global_ergo(sigma12,  dA, dB, w1, w2) - Rg_before
            delta_l   = local_ergo(sigma12,   dA, dB, w1, w2) - Rl_before
            delta_gap = ergotropy_gap(sigma12, dA, dB, w1, w2) - gap_before

            if delta_g > best['global'][0]:
                best['global'] = (delta_g, rho12.copy(), sigma12.copy(), Ua.copy(), Ub.copy())
            if delta_l > best['local'][0]:
                best['local']  = (delta_l, rho12.copy(), sigma12.copy(), Ua.copy(), Ub.copy())
            if delta_gap > best['gap'][0]:
                best['gap']    = (delta_gap, rho12.copy(), sigma12.copy(), Ua.copy(), Ub.copy())

    print("\n" + "="*50)
    print("  STATE + UNITARY SEARCH RESULTS")
    print("="*50)
    for target in ['global', 'local', 'gap']:
        delta = best[target][0]
        print(f"  Best Δ({target:6s}) = {delta:.6f}  "
              f"{'✓ > 0' if delta > 1e-8 else '✗ not found'}")
    print("="*50)

    return best


# ----------------------------------------------------------
# Gradient-free optimization - Nelder-Mead
# -----------------------------------------------------------

def nelder_mead_search(rho12, gamma_Ra, gamma_Rb, Hs1, Hs2, Hb1, Hb2, dA, dB, w1, w2, n_restarts=20, maxiter=1000):
    """
    Gradient-free optimization via Nelder-Mead.
    CHANGE from qubit: dA, dB passed for energy functions.
    """

    blocks_a = get_blocks(Hs1, Hb1)
    blocks_b = get_blocks(Hs2, Hb2)
    sizes_a  = [len(b) ** 2 for b in blocks_a]
    sizes_b  = [len(b) ** 2 for b in blocks_b]
    n_params = sum(sizes_a) + sum(sizes_b)

    Rg_before  = global_ergo(rho12,  dA, dB, w1, w2)
    Rl_before  = local_ergo(rho12,   dA, dB, w1, w2)
    gap_before = ergotropy_gap(rho12, dA, dB, w1, w2)

    best = {
        'global': (-np.inf, None, None, None),
        'local':  (-np.inf, None, None, None),
        'gap':    (-np.inf, None, None, None),
    }

    def evaluate(x):
        Ua, Ub = params_to_unitaries(x, blocks_a, blocks_b,
                                      Hs1, Hb1, Hs2, Hb2)
        sigma12, _, _ = LTO_step(rho12, gamma_Ra, gamma_Rb, Ua, Ub)
        return (sigma12,
                global_ergo(sigma12,  dA, dB, w1, w2) - Rg_before,
                local_ergo(sigma12,   dA, dB, w1, w2) - Rl_before,
                ergotropy_gap(sigma12, dA, dB, w1, w2) - gap_before)

    for restart in tqdm(range(n_restarts), desc="Nelder-Mead", unit="restart", dynamic_ncols=True):

        x0 = np.random.randn(n_params) * 0.1

        for target, before in [('global', Rg_before), ('local',  Rl_before), ('gap',    gap_before)]:

            score_fn = {
                'global': lambda r: global_ergo(r,  dA, dB, w1, w2),
                'local':  lambda r: local_ergo(r,   dA, dB, w1, w2),
                'gap':    lambda r: ergotropy_gap(r, dA, dB, w1, w2)
            }[target]

            def objective(x):
                Ua, Ub = params_to_unitaries(x, blocks_a, blocks_b, Hs1, Hb1, Hs2, Hb2)
                sigma12, _, _ = LTO_step(rho12, gamma_Ra, gamma_Rb, Ua, Ub)
                return -(score_fn(sigma12) - before)

            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'maxiter': maxiter,'xatol': 1e-9, 'fatol': 1e-9})

            sigma12, dg, dl, dgap = evaluate(res.x)
            delta = {'global': dg, 'local': dl, 'gap': dgap}[target]

            if delta > best[target][0]:
                Ua, Ub = params_to_unitaries(res.x, blocks_a, blocks_b, Hs1, Hb1, Hs2, Hb2)
                best[target] = (delta, sigma12.copy(), Ua.copy(), Ub.copy())

        tqdm.write(f"  restart {restart+1:3d}  "
                   f"Δglobal={best['global'][0]:.6f}  "
                   f"Δlocal={best['local'][0]:.6f}  "
                   f"Δgap={best['gap'][0]:.6f}")

    print("\n" + "="*50)
    print("  NELDER-MEAD RESULTS")
    print("="*50)
    for target in ['global', 'local', 'gap']:
        delta = best[target][0]
        print(f"  Best Δ({target:6s}) = {delta:.6f}  "
              f"{'✓ > 0' if delta > 1e-8 else '✗ not found'}")
    print("="*50)

    return best


# -------------------------------------------------------------
# Gradient-based optimization — L-BFGS-B
# -------------------------------------------------------------

def lbfgs_search(rho12, gamma_Ra, gamma_Rb, Hs1, Hs2, Hb1, Hb2, dA, dB, w1, w2, n_restarts=20, maxiter=1000):
    """
    Gradient-based optimization via L-BFGS-B with finite differences.
    CHANGE from qubit: dA, dB passed for energy functions.
    """

    blocks_a = get_blocks(Hs1, Hb1)
    blocks_b = get_blocks(Hs2, Hb2)
    sizes_a  = [len(b) ** 2 for b in blocks_a]
    sizes_b  = [len(b) ** 2 for b in blocks_b]
    n_params = sum(sizes_a) + sum(sizes_b)

    Rg_before  = global_ergo(rho12,  dA, dB, w1, w2)
    Rl_before  = local_ergo(rho12,   dA, dB, w1, w2)
    gap_before = ergotropy_gap(rho12, dA, dB, w1, w2)

    best = {
        'global': (-np.inf, None, None, None),
        'local':  (-np.inf, None, None, None),
        'gap':    (-np.inf, None, None, None),
    }

    def evaluate(x):
        Ua, Ub = params_to_unitaries(x, blocks_a, blocks_b, Hs1, Hb1, Hs2, Hb2)
        sigma12, _, _ = LTO_step(rho12, gamma_Ra, gamma_Rb, Ua, Ub)
        return (sigma12,
                global_ergo(sigma12,  dA, dB, w1, w2) - Rg_before,
                local_ergo(sigma12,   dA, dB, w1, w2) - Rl_before,
                ergotropy_gap(sigma12, dA, dB, w1, w2) - gap_before)

    for restart in tqdm(range(n_restarts), desc="L-BFGS-B", unit="restart", dynamic_ncols=True):

        x0 = np.random.randn(n_params) * 0.1

        for target, before in [('global', Rg_before), ('local',  Rl_before), ('gap',    gap_before)]:

            score_fn = {
                'global': lambda r: global_ergo(r,  dA, dB, w1, w2),
                'local':  lambda r: local_ergo(r,   dA, dB, w1, w2),
                'gap':    lambda r: ergotropy_gap(r, dA, dB, w1, w2)
            }[target]

            def objective(x):
                Ua, Ub = params_to_unitaries(x, blocks_a, blocks_b, Hs1, Hb1, Hs2, Hb2)
                sigma12, _, _ = LTO_step(rho12, gamma_Ra, gamma_Rb, Ua, Ub)
                return -(score_fn(sigma12) - before)

            res = minimize(objective, x0, method='L-BFGS-B',
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8},
                           jac='2-point')

            sigma12, dg, dl, dgap = evaluate(res.x)
            delta = {'global': dg, 'local': dl, 'gap': dgap}[target]

            if delta > best[target][0]:
                Ua, Ub = params_to_unitaries(res.x, blocks_a, blocks_b, Hs1, Hb1, Hs2, Hb2)
                best[target] = (delta, sigma12.copy(), Ua.copy(), Ub.copy())

        tqdm.write(f"  restart {restart+1:3d}  "
                   f"Δglobal={best['global'][0]:.6f}  "
                   f"Δlocal={best['local'][0]:.6f}  "
                   f"Δgap={best['gap'][0]:.6f}")

    print("\n" + "="*50)
    print("  L-BFGS-B RESULTS")
    print("="*50)
    for target in ['global', 'local', 'gap']:
        delta = best[target][0]
        print(f"  Best Δ({target:6s}) = {delta:.6f}  "
              f"{'✓ > 0' if delta > 1e-8 else '✗ not found'}")
    print("="*50)

    return best