# local_to.py - Run the LTO simulation steps
import numpy as np
import scipy.linalg as la
from energy import free_energy, system_hamiltonian
from unitary import deg_unitary

# -------------------------------------------------
# Harmonic oscillator Hamiltonian (for baths)
# -------------------------------------------------

def sho_ham(dim, omega=1.0):
    """Harmonic oscillator Hamiltonian. Energies: 0, ω, 2ω, ..., (dim-1)ω."""
    return np.diag(omega * np.arange(dim, dtype=complex))

# -------------------------------------------------
# Gibbs states for harmonic oscillator
# -------------------------------------------------

def gibbs_state(H, beta):
    """Gibbs state for any Hamiltonian H at inverse temperature beta."""
    expH = la.expm(-beta * H)
    Z = np.trace(expH)
    return expH / Z

def gibbs_states(dA, dB, beta_a, beta_b, omega_a=1.0, omega_b=1.0):
    """
    Gibbs states for both subsystems A and B (system or bath).
    Since both are harmonic, same function serves both.
    """
    Ha = sho_ham(dA, omega_a)
    Hb = sho_ham(dB, omega_b)
    return gibbs_state(Ha, beta_a), gibbs_state(Hb, beta_b)


# ------------------------------------------
# LTO STEP (harmonic)
# ------------------------------------------


def LTO_step(rho12, gamma_Ra, gamma_Rb, Ua, Ub):

    dBa = gamma_Ra.shape[0]
    dBb = gamma_Rb.shape[0]
    dA = Ua.shape[0] // dBa
    dB = Ub.shape[0] // dBb

    # (S1,S2,Ra,Rb) -> reorder to (S1,Ra,S2,Rb)
    rho_ext = np.kron(np.kron(rho12, gamma_Ra), gamma_Rb)
    rho_ext = rho_ext.reshape(dA, dB, dBa, dBb, dA, dB, dBa, dBb)
    rho_ext = rho_ext.transpose(0,2,1,3,4,6,5,7)
    rho_ext = rho_ext.reshape(dA*dBa*dB*dBb, dA*dBa*dB*dBb)

    # Apply Ua on S1⊗Ra, Ub on S2⊗Rb
    U_tot = np.kron(Ua, Ub)
    rho_out = U_tot @ rho_ext @ U_tot.conj().T

    # Reshape to named indices: (s1,ra,s2,rb, s1',ra',s2',rb')
    r = rho_out.reshape(dA, dBa, dB, dBb, dA, dBa, dB, dBb)

    # trace Ra: contract axis 1 with axis 5
    sigma = np.einsum('abcdefgh,bf->acdegh', r, np.eye(dBa))

    # trace Rb: contract axis 2 with axis 5
    sigma12 = np.einsum('abcdef,cf->abde', sigma, np.eye(dBb)).reshape(dA*dB, dA*dB)

    # gamma_a1: keep Ra(1=5), trace S1(0=4), S2(2=6), Rb(3=7)
    gamma_a1 = np.einsum('abcdefgh,ae,cg,dh->bf', r, np.eye(dA), np.eye(dB), np.eye(dBb))

    # gamma_b1: keep Rb(3=7), trace S1(0=4), Ra(1=5), S2(2=6)
    gamma_b1 = np.einsum('abcdefgh,ae,bf,cg->dh', r, np.eye(dA), np.eye(dBa), np.eye(dB))

    # Sanity checks
    assert np.allclose(np.trace(sigma12), 1.0, atol=1e-10), "sigma12 not trace-1"
    assert np.allclose(np.trace(gamma_a1), 1.0, atol=1e-10), "gamma_a1 not trace-1"
    assert np.allclose(np.trace(gamma_b1), 1.0, atol=1e-10), "gamma_b1 not trace-1"

    return sigma12, gamma_a1, gamma_b1


#  -------------------------------------------------
#  Sanity Check for Gibbs state fixed point
#  ------------------------------------------------

def sanity_check_gibbs(beta_a=1.0, beta_b=2.0, w1=1.0, w2=1.0, omega_a=1.0, omega_b=1.0, 
                       sys_dim_a=4, sys_dim_b=4, bath_dim_a=4, bath_dim_b=4, tol=1e-8):
    """
    Sanity check: Gibbs state is a fixed point of the thermal operation.
    Sets rho12 = gamma_1 ⊗ gamma_2, applies LTO, checks output == input.
    Both system and bath are harmonic oscillators.
    """
    print("="*50)
    print("SANITY CHECK: Gibbs fixed point")
    print("="*50)

    # Build Hamiltonians
    Hs1 = sho_ham(sys_dim_a, w1)
    Hs2 = sho_ham(sys_dim_b, w2)
    Hb1 = sho_ham(bath_dim_a, omega_a)
    Hb2 = sho_ham(bath_dim_b, omega_b)

    # Build Gibbs states
    gamma_1  = gibbs_state(Hs1, beta_a)
    gamma_2  = gibbs_state(Hs2, beta_b)
    gamma_Ra = gibbs_state(Hb1, beta_a)
    gamma_Rb = gibbs_state(Hb2, beta_b)

    # Initial state = product of Gibbs states
    rho12 = np.kron(gamma_1, gamma_2)

    # Build energy-conserving unitaries
    Ua = deg_unitary(Hs1, Hb1)
    Ub = deg_unitary(Hs2, Hb2)

    # Apply LTO
    sigma12, gamma_a1, gamma_b1 = LTO_step(rho12, gamma_Ra, gamma_Rb, Ua, Ub)

    # Check 1: sigma12 == rho12
    err_sys   = np.max(np.abs(sigma12 - rho12))
    passed_sys = err_sys < tol
    print(f"  rho12 preserved:    {'PASS' if passed_sys  else 'FAIL'}  (max err = {err_sys:.2e})")

    # Check 2: bath states preserved
    err_Ra    = np.max(np.abs(gamma_a1 - gamma_Ra))
    err_Rb    = np.max(np.abs(gamma_b1 - gamma_Rb))
    passed_Ra = err_Ra < tol
    passed_Rb = err_Rb < tol
    print(f"  gamma_Ra preserved: {'PASS' if passed_Ra else 'FAIL'}  (max err = {err_Ra:.2e})")
    print(f"  gamma_Rb preserved: {'PASS' if passed_Rb else 'FAIL'}  (max err = {err_Rb:.2e})")

    # Check 3: trace preservation
    err_tr    = abs(np.trace(sigma12) - 1.0)
    passed_tr = err_tr < tol
    print(f"  trace(sigma12)=1:   {'PASS' if passed_tr else 'FAIL'}  (err = {err_tr:.2e})")

    # Check 4: free energy non-increasing — for Gibbs input should be exactly 0
    H_sys    = system_hamiltonian(sys_dim_a, sys_dim_b, w1, w2)
    F_before = free_energy(rho12,   H_sys, beta_a)
    F_after  = free_energy(sigma12, H_sys, beta_a)
    delta_F  = F_after - F_before
    passed_F = delta_F < tol
    print(f"  ΔF ≤ 0:             {'PASS' if passed_F  else 'FAIL'}  (ΔF = {delta_F:.2e})")

    print("="*50)
    overall = all([passed_sys, passed_Ra, passed_Rb, passed_tr, passed_F])
    print(f"  OVERALL: {'ALL PASS ✓' if overall else 'SOME CHECKS FAILED ✗'}")
    print("="*50)

    return overall