# local_to.py - Run the LTO simulation steps
import numpy as np
import scipy.linalg as la
from energy import free_energy, system_hamiltonian
from unitary import deg_unitary

# Convention: sigma_z = diag(-1,+1), qubit energies are -w (ground), +w (excited)
# Bath: H_bath = omega * diag(0, 1, 2, ..., dim-1)

# -------------------------------------------------
# Hamiltonians for system and bath
# -------------------------------------------------

def qubit_ham(w):
    """Single qubit Hamiltonian with sigma_z = diag(-1,+1) convention."""
    return np.diag(np.array([-w, w], dtype=complex))

# ------------------------------------------
# Gibbs state for systme and bath
# ------------------------------------------

def gibbs(H, beta):
    """Gibbs state for any Hamiltonian H at inverse temperature beta."""
    expH = la.expm(-beta * H)
    Z = np.trace(expH)
    return expH / Z

def gibbs_states(beta_a=1.0, beta_b=2.0, w1=1.0, w2=2.0):
    """Gibbs states for both qubits via their individual bath temperatures."""
    H1 = qubit_ham(w1)
    H2 = qubit_ham(w2)
    return gibbs(H1, beta_a), gibbs(H2, beta_b)


# ------------------------------------------
# LTO STEP
# ------------------------------------------


def LTO_step(rho12, gamma_Ra, gamma_Rb, Ua, Ub):
    dS = 2
    dB = gamma_Ra.shape[0]

    # (S1,S2,Ra,Rb) -> reorder to (S1,Ra,S2,Rb)
    rho_ext = np.kron(np.kron(rho12, gamma_Ra), gamma_Rb)
    rho_ext = rho_ext.reshape(dS,dS,dB,dB,dS,dS,dB,dB)
    rho_ext = rho_ext.transpose(0,2,1,3,4,6,5,7)
    rho_ext = rho_ext.reshape(dS*dB*dS*dB, dS*dB*dS*dB)

    # Apply Ua on S1⊗Ra, Ub on S2⊗Rb
    U_tot = np.kron(Ua, Ub)
    rho_out = U_tot @ rho_ext @ U_tot.conj().T

    # Reshape to named indices: (s1,ra,s2,rb, s1',ra',s2',rb')
    r = rho_out.reshape(dS,dB,dS,dB,dS,dB,dS,dB)

    # trace Ra: contract axis 1 with axis 5
    sigma = np.einsum('abcdefgh,bf->acdegh', r, np.eye(dB))

    # trace Rb: contract axis 2 with axis 5
    sigma12 = np.einsum('abcdef,cf->abde', sigma, np.eye(dB)).reshape(dS*dS, dS*dS)

    # gamma_a1: trace S1(0=4), S2(2=6), Rb(3=7)
    gamma_a1 = np.einsum('abcdefgh,ae,cg,dh->bf', r, np.eye(dS), np.eye(dS), np.eye(dB))

    # gamma_b1: trace S1(0=4), Ra(1=5), S2(2=6)
    gamma_b1 = np.einsum('abcdefgh,ae,bf,cg->dh', r, np.eye(dS), np.eye(dB), np.eye(dS))

    # Sanity checks
    assert np.allclose(np.trace(sigma12), 1.0, atol=1e-10), "sigma12 not trace-1"
    assert np.allclose(np.trace(gamma_a1), 1.0, atol=1e-10), "gamma_a1 not trace-1"
    assert np.allclose(np.trace(gamma_b1), 1.0, atol=1e-10), "gamma_b1 not trace-1"

    return sigma12, gamma_a1, gamma_b1

#  -------------------------------------------------
#  Sanity Check for Gibbs state fixed point
#  ------------------------------------------------

def sanity_check_gibbs(beta_a=1.0, beta_b=2.0, w1=1.0, w2=2.0, w3=1.5, w4=0.5, tol=1e-8):
    """
    Sanity check: Gibbs state is a fixed point of the thermal operation.
    Sets rho12 = gamma_1 ⊗ gamma_2, applies LTO, checks output == input.
    """
    print("="*50)
    print("SANITY CHECK: Gibbs fixed point")
    print("="*50)

    bath_dim = 2

    # Build system Hamiltonians
    Hs1 = qubit_ham(w1)
    Hs2 = qubit_ham(w2)

    # Build bath Hamiltonians
    Hb1 = qubit_ham(bath_dim, w3)
    Hb2 = qubit_ham(bath_dim, w4)

    # Build Gibbs states
    gamma_1 = gibbs(Hs1, beta_a)
    gamma_2 = gibbs(Hs2, beta_b)
    gamma_Ra = gibbs(Hb1, beta_a)
    gamma_Rb = gibbs(Hb2, beta_b)

    # Initial two-qubit state = product of Gibbs states
    rho12 = np.kron(gamma_1, gamma_2)

    # Build energy-conserving unitaries
    Ua = deg_unitary(Hs1, Hb1)
    Ub = deg_unitary(Hs2, Hb2)

    # Apply LTO
    sigma12, gamma_a1, gamma_b1 = LTO_step(rho12, gamma_Ra, gamma_Rb, Ua, Ub)

    # Check 1: sigma12 == rho12
    err_sys = np.max(np.abs(sigma12 - rho12))
    passed_sys = err_sys < tol
    print(f"  rho12 preserved:   {'PASS' if passed_sys else 'FAIL'}  (max err = {err_sys:.2e})")

    # Check 2: bath states preserved
    err_Ra = np.max(np.abs(gamma_a1 - gamma_Ra))
    err_Rb = np.max(np.abs(gamma_b1 - gamma_Rb))
    passed_Ra = err_Ra < tol
    passed_Rb = err_Rb < tol
    print(f"  gamma_Ra preserved: {'PASS' if passed_Ra else 'FAIL'}  (max err = {err_Ra:.2e})")
    print(f"  gamma_Rb preserved: {'PASS' if passed_Rb else 'FAIL'}  (max err = {err_Rb:.2e})")

    # Check 3: trace preservation
    err_tr = abs(np.trace(sigma12) - 1.0)
    passed_tr = err_tr < tol
    print(f"  trace(sigma12)=1:  {'PASS' if passed_tr else 'FAIL'}  (err = {err_tr:.2e})")

    # Check 4: free energy non-increasing for each qubit
    H_sys = system_hamiltonian(w1, w2)
    F_before = free_energy(rho12, H_sys, beta_a)
    F_after  = free_energy(sigma12, H_sys, beta_a)
    delta_F  = F_after - F_before
    passed_F = delta_F < tol
    print(f"  ΔF ≤ 0:            {'PASS' if passed_F else 'FAIL'}  (ΔF = {delta_F:.2e})")

    print("="*50)
    overall = all([passed_sys, passed_Ra, passed_Rb, passed_tr, passed_F])
    print(f"  OVERALL: {'ALL PASS ✓' if overall else 'SOME CHECKS FAILED ✗'}")
    print("="*50)

    return overall