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
    """Identify degenerate energy blocks of H_total = Hs*I + I*Hb."""

    # get dimensions of system and bath
    dS = Hs.shape[0]
    dB = Hb.shape[0]

    # build total Hamiltonian - diagonal since both Hs and Hb are diagonal
    Htot = np.kron(Hs, np.eye(dB)) + np.kron(np.eye(dS), Hb)

    # extract diagonal energies - valid since Htot is diagonal
    energies = np.diag(Htot).real

    # track which indices have already been assigned to a block
    used = np.zeros(dS * dB, dtype=bool)

    blocks = []
    for i, E in enumerate(energies):
        if used[i]: continue
        block = np.where(np.abs(energies - E) < tol)[0]
        used[block] = True
        blocks.append(block)

    return blocks


def unitary_from_params(theta, k):
    """
    Parameterize a k×k unitary via exp(i*H) where H is Hermitian.
    theta is a real vector of length k^2:
      - first k entries:       real diagonal elements of H
      - remaining k(k-1) entries: pairs (re, im) for upper triangle of H
    """

    # initialize Hermitian matrix
    H = np.zeros((k, k), dtype=complex)
    idx = 0

    # fill diagonal with real entries from theta
    for i in range(k):
        H[i, i] = theta[idx]
        idx += 1

    # fill upper triangle with complex entries, lower triangle by conjugate
    for i in range(k):
        for j in range(i + 1, k):
            H[i, j] = theta[idx] + 1j * theta[idx + 1]
            H[j, i] = H[i, j].conj()
            idx += 2

    # exponentiate to get unitary: U = exp(iH)
    return la.expm(1j * H)


def deg_unitary_from_params(Hs, Hb, blocks, params_dict):
    """
    Build full energy-conserving unitary from block parameters.
    Each degenerate block gets its own unitary via unitary_from_params.
    params_dict = {block_id: theta_vector}
    """
    dS  = Hs.shape[0]
    dB  = Hb.shape[0]
    dim = dS * dB

    # initialize full unitary as zeros
    U = np.zeros((dim, dim), dtype=complex)

    for bid, block in enumerate(blocks):
        k = len(block)
        # get parameters for this block, default to zeros (= identity block)
        theta = params_dict.get(bid, np.zeros(k * k))
        U[np.ix_(block, block)] = unitary_from_params(theta, k)

    return U


def params_to_unitaries(x, blocks_a, blocks_b, Hs1, Hb1, Hs2, Hb2):
    """
    Split flat parameter vector x into Ua and Ub.
    First sum(sizes_a) entries go to Ua, rest to Ub.
    """

    # number of parameters per block = k^2
    sizes_a = [len(b) ** 2 for b in blocks_a]
    sizes_b = [len(b) ** 2 for b in blocks_b]

    idx = 0

    # slice out parameters for Ua blocks
    params_a = {}
    for bid, s in enumerate(sizes_a):
        params_a[bid] = x[idx:idx + s]
        idx += s

    # slice out parameters for Ub blocks
    params_b = {}
    for bid, s in enumerate(sizes_b):
        params_b[bid] = x[idx:idx + s]
        idx += s

    # build unitaries from parameters
    Ua = deg_unitary_from_params(Hs1, Hb1, blocks_a, params_a)
    Ub = deg_unitary_from_params(Hs2, Hb2, blocks_b, params_b)

    return Ua, Ub


# ---------------------------------
# Random search - fixed state
# ---------------------------------

def random_search(rho12, gamma_Ra, gamma_Rb, Hs1, Hs2, Hb1, Hb2, w1, w2, n_trials=10000):
    """
    Random search over thermal operation unitaries for a fixed state.
    Tracks all three targets independently: global, local, gap.
    """

    # compute all three scores before LTO — fixed since rho12 is fixed
    Rg_before  = global_ergo(rho12, w1, w2)
    Rl_before  = local_ergo(rho12, w1, w2)
    gap_before = ergotropy_gap(rho12, w1, w2)

    # three independent trackers: (best_delta, sigma12, Ua, Ub)
    best = {
        'global': (-np.inf, None, None, None),
        'local':  (-np.inf, None, None, None),
        'gap':    (-np.inf, None, None, None),
    }

    for _ in tqdm(range(n_trials), desc="Random search", unit="trial", dynamic_ncols=True):

        # sample new random energy-conserving unitaries
        Ua = deg_unitary(Hs1, Hb1, verify=False)
        Ub = deg_unitary(Hs2, Hb2, verify=False)

        # apply LTO
        sigma12, _, _ = LTO_step(rho12, gamma_Ra, gamma_Rb, Ua, Ub)

        # compute deltas for all three targets
        delta_g   = global_ergo(sigma12, w1, w2)  - Rg_before
        delta_l   = local_ergo(sigma12, w1, w2)   - Rl_before
        delta_gap = ergotropy_gap(sigma12, w1, w2) - gap_before

        # update each tracker independently
        if delta_g > best['global'][0]:
            best['global'] = (delta_g, sigma12.copy(), Ua.copy(), Ub.copy())

        if delta_l > best['local'][0]:
            best['local'] = (delta_l, sigma12.copy(), Ua.copy(), Ub.copy())

        if delta_gap > best['gap'][0]:
            best['gap'] = (delta_gap, sigma12.copy(), Ua.copy(), Ub.copy())

    # print summary
    print("\n" + "="*50)
    print("  RANDOM SEARCH RESULTS")
    print("="*50)
    for target in ['global', 'local', 'gap']:
        delta = best[target][0]
        print(f"  Best Δ({target:6s}) = {delta:.6f} for {best[target][2:]} "
              f"{'✓ > 0' if delta > 1e-8 else '✗ not found'}")
    print("="*50)

    return best


# -------------------------------------------------------------
# Random search - over states and unitaries
# -------------------------------------------------------------

def search_over_states(n_states, n_unitaries, beta_a, beta_b, w1, w2, omega_a, omega_b, bath_dim, state_kind='random'):
    """
    Double loop random search over both states and unitaries.
    Tracks all three targets independently.
    """

    # build Hamiltonians once — never change across loops
    Hs1 = sho_ham(2, w1);             Hs2 = sho_ham(2, w2)
    Hb1 = sho_ham(bath_dim, omega_a); Hb2 = sho_ham(bath_dim, omega_b)

    # build bath Gibbs states once — fixed by beta and omega
    gamma_Ra = gibbs_state(Hb1, beta_a)
    gamma_Rb = gibbs_state(Hb2, beta_b)

    # three independent trackers: (best_delta, rho12, sigma12, Ua, Ub)
    best = {
        'global': (-np.inf, None, None, None, None),
        'local':  (-np.inf, None, None, None, None),
        'gap':    (-np.inf, None, None, None, None),
    }

    # outer loop: sample different initial states
    for s in tqdm(range(n_states), desc="Searching states", unit="state", dynamic_ncols=True):

        # sample a new random two-qubit state
        rho12 = generate_state(kind=state_kind)

        # compute scores before LTO - fixed for all unitaries on this state
        Rg_before  = global_ergo(rho12, w1, w2)
        Rl_before  = local_ergo(rho12, w1, w2)
        gap_before = ergotropy_gap(rho12, w1, w2)

        # inner loop: sample different unitaries for this state
        for _ in range(n_unitaries):

            # sample new random energy-conserving unitaries
            Ua = deg_unitary(Hs1, Hb1, verify=False)
            Ub = deg_unitary(Hs2, Hb2, verify=False)

            # apply LTO
            sigma12, _, _ = LTO_step(rho12, gamma_Ra, gamma_Rb, Ua, Ub)

            # compute deltas for all three targets
            delta_g   = global_ergo(sigma12, w1, w2)  - Rg_before
            delta_l   = local_ergo(sigma12, w1, w2)   - Rl_before
            delta_gap = ergotropy_gap(sigma12, w1, w2) - gap_before

            # update each tracker independently — .copy() prevents
            # overwriting stored results in next iteration
            if delta_g > best['global'][0]:
                best['global'] = (delta_g, rho12.copy(), sigma12.copy(), Ua.copy(), Ub.copy())

            if delta_l > best['local'][0]:
                best['local'] = (delta_l, rho12.copy(), sigma12.copy(), Ua.copy(), Ub.copy())

            if delta_gap > best['gap'][0]:
                best['gap'] = (delta_gap, rho12.copy(), sigma12.copy(), Ua.copy(), Ub.copy())

    # print summary
    print("\n" + "="*50)
    print("  STATE + UNITARY SEARCH RESULTS")
    print("="*50)
    for target in ['global', 'local', 'gap']:
        delta = best[target][0]
        print(f"  Best Δ({target:6s}) = {delta:.6f}\n "
              f"{'✓ > 0' if delta > 1e-8 else '✗ not found'}")
    print("="*50)

    return best


# ----------------------------------------------------------
# Gradient-free optimization — Nelder-Mead
# -----------------------------------------------------------

def nelder_mead_search(rho12, gamma_Ra, gamma_Rb, Hs1, Hs2, Hb1, Hb2, w1, w2, n_restarts=20, maxiter=1000):
    """
    Gradient-free optimization over unitary parameters via Nelder-Mead.
    Each degenerate block unitary parameterized as exp(iH).
    Tracks all three targets independently across restarts.
    """

    # identify degenerate blocks once - fixed by Hamiltonians
    blocks_a = get_blocks(Hs1, Hb1)
    blocks_b = get_blocks(Hs2, Hb2)

    # total parameter count = sum of k^2 over all blocks for Ua and Ub
    sizes_a  = [len(b) ** 2 for b in blocks_a]
    sizes_b  = [len(b) ** 2 for b in blocks_b]
    n_params = sum(sizes_a) + sum(sizes_b)

    # compute scores before LTO — fixed since rho12 is fixed
    Rg_before  = global_ergo(rho12, w1, w2)
    Rl_before  = local_ergo(rho12, w1, w2)
    gap_before = ergotropy_gap(rho12, w1, w2)

    # three independent trackers: (best_delta, sigma12, Ua, Ub)
    best = {
        'global': (-np.inf, None, None, None),
        'local':  (-np.inf, None, None, None),
        'gap':    (-np.inf, None, None, None),
    }

    def evaluate(x):
        """
        Given flat parameter vector x, build unitaries,
        apply LTO, return all three deltas.
        """
        Ua, Ub = params_to_unitaries(x, blocks_a, blocks_b, Hs1, Hb1, Hs2, Hb2)
        sigma12, _, _ = LTO_step(rho12, gamma_Ra, gamma_Rb, Ua, Ub)
        return (sigma12,
                global_ergo(sigma12, w1, w2)  - Rg_before,
                local_ergo(sigma12, w1, w2)   - Rl_before,
                ergotropy_gap(sigma12, w1, w2) - gap_before)

    for restart in tqdm(range(n_restarts), desc="Nelder-Mead", unit="restart", dynamic_ncols=True):

        # random initial parameters — small scale to start near identity
        x0 = np.random.randn(n_params) * 0.1

        # optimize each target separately from same x0
        for target, before in [('global', Rg_before), ('local',  Rl_before), ('gap',    gap_before)]:

            score_fn = {
                'global': lambda r: global_ergo(r, w1, w2),
                'local':  lambda r: local_ergo(r, w1, w2),
                'gap':    lambda r: ergotropy_gap(r, w1, w2)
            }[target]

            # objective: minimize negative delta for this target
            def objective(x):
                Ua, Ub = params_to_unitaries(x, blocks_a, blocks_b, Hs1, Hb1, Hs2, Hb2)
                sigma12, _, _ = LTO_step(rho12, gamma_Ra, gamma_Rb, Ua, Ub)
                return -(score_fn(sigma12) - before)

            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'maxiter': maxiter, 'xatol': 1e-9, 'fatol': 1e-9})

            # evaluate all three at the optimum for this target
            sigma12, dg, dl, dgap = evaluate(res.x)
            delta = {'global': dg, 'local': dl, 'gap': dgap}[target]

            # update tracker for this target
            if delta > best[target][0]:
                Ua, Ub = params_to_unitaries(res.x, blocks_a, blocks_b, Hs1, Hb1, Hs2, Hb2)
                best[target] = (delta, sigma12.copy(), Ua.copy(), Ub.copy())

        tqdm.write(f"  restart {restart+1:3d}  "
                   f"Δglobal={best['global'][0]:.6f}  "
                   f"Δlocal={best['local'][0]:.6f}  "
                   f"Δgap={best['gap'][0]:.6f}")

    # print summary
    print("\n" + "="*50)
    print("  NELDER-MEAD RESULTS")
    print("="*50)
    for target in ['global', 'local', 'gap']:
        delta = best[target][0]
        print(f"  Best Δ({target:6s}) = {delta:.6f}\n"
              f"{'✓ > 0' if delta > 1e-8 else '✗ not found'}")
    print("="*50)

    return best


# -------------------------------------------------------------
# Gradient-based optimization — L-BFGS-B
# -------------------------------------------------------------

def lbfgs_search(rho12, gamma_Ra, gamma_Rb, Hs1, Hs2, Hb1, Hb2,
                  w1, w2, n_restarts=20, maxiter=1000):
    """
    Gradient-based optimization via L-BFGS-B with finite difference gradients.
    More efficient than Nelder-Mead for larger parameter spaces.
    Note: ergotropy is not smooth everywhere due to eigenvalue sorting,
    so finite differences ('2-point') are used instead of analytic gradients.
    Tracks all three targets independently across restarts.
    """

    # identify degenerate blocks once
    blocks_a = get_blocks(Hs1, Hb1)
    blocks_b = get_blocks(Hs2, Hb2)

    # total parameter count
    sizes_a  = [len(b) ** 2 for b in blocks_a]
    sizes_b  = [len(b) ** 2 for b in blocks_b]
    n_params = sum(sizes_a) + sum(sizes_b)

    # scores before LTO
    Rg_before  = global_ergo(rho12, w1, w2)
    Rl_before  = local_ergo(rho12, w1, w2)
    gap_before = ergotropy_gap(rho12, w1, w2)

    # three independent trackers: (best_delta, sigma12, Ua, Ub)
    best = {
        'global': (-np.inf, None, None, None),
        'local':  (-np.inf, None, None, None),
        'gap':    (-np.inf, None, None, None),
    }

    def evaluate(x):
        """Build unitaries from x, apply LTO, return sigma and all deltas."""
        Ua, Ub = params_to_unitaries(x, blocks_a, blocks_b, Hs1, Hb1, Hs2, Hb2)
        sigma12, _, _ = LTO_step(rho12, gamma_Ra, gamma_Rb, Ua, Ub)
        return (sigma12,
                global_ergo(sigma12, w1, w2)  - Rg_before,
                local_ergo(sigma12, w1, w2)   - Rl_before,
                ergotropy_gap(sigma12, w1, w2) - gap_before)

    for restart in tqdm(range(n_restarts), desc="L-BFGS-B", unit="restart", dynamic_ncols=True):

        # random initial parameters
        x0 = np.random.randn(n_params) * 0.1

        # optimize each target separately from same x0
        for target, before in [('global', Rg_before), ('local',  Rl_before), ('gap',    gap_before)]:

            score_fn = {
                'global': lambda r: global_ergo(r, w1, w2),
                'local':  lambda r: local_ergo(r, w1, w2),
                'gap':    lambda r: ergotropy_gap(r, w1, w2)
            }[target]

            def objective(x):
                Ua, Ub = params_to_unitaries(x, blocks_a, blocks_b, Hs1, Hb1, Hs2, Hb2)
                sigma12, _, _ = LTO_step(rho12, gamma_Ra, gamma_Rb, Ua, Ub)
                return -(score_fn(sigma12) - before)

            # '2-point' finite differences for gradient estimation
            # needed because ergotropy has kinks from eigenvalue sorting
            res = minimize(objective, x0, method='L-BFGS-B',
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}, 
                           jac='2-point')

            # evaluate all three at optimum
            sigma12, dg, dl, dgap = evaluate(res.x)
            delta = {'global': dg, 'local': dl, 'gap': dgap}[target]

            # update tracker
            if delta > best[target][0]:
                Ua, Ub = params_to_unitaries(res.x, blocks_a, blocks_b, Hs1, Hb1, Hs2, Hb2)
                best[target] = (delta, sigma12.copy(), Ua.copy(), Ub.copy())

        tqdm.write(f"  restart {restart+1:3d}  "
                   f"Δglobal={best['global'][0]:.6f}  "
                   f"Δlocal={best['local'][0]:.6f}  "
                   f"Δgap={best['gap'][0]:.6f}")

    # print summary
    print("\n" + "="*50)
    print("  L-BFGS-B RESULTS")
    print("="*50)
    for target in ['global', 'local', 'gap']:
        delta = best[target][0]
        print(f"  Best Δ({target:6s}) = {delta:.6f}\n "
              f"{'✓ > 0' if delta > 1e-8 else '✗ not found'}")
    print("="*50)

    return best