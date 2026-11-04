"""Dipole from paired finite-field AFQMC, with an optional CCSD(T) reference."""

from dataclasses import replace
from functools import partial
from time import perf_counter

from trot import config

config.configure_once(use_gpu=False)

import jax
import jax.numpy as jnp
import numpy as np
from pyscf import cc, gto, lib, scf
from pyscf.data.nist import AU2DEBYE

from trot.core.system import System
from trot.correlated import run_correlated
from trot.ham.chol import HamChol
from trot.meas.rhf import make_rhf_meas_ops
from trot.prop.afqmc import make_prop_ops
from trot.prop.types import QmcParams
from trot.staging import stage
from trot.trial.rhf import RhfTrial, make_rhf_trial_ops

FIELD = 1.0e-5
RUN_CCSD_T = False
REBLOCK = 10
PRINT_EVERY = 20
params = QmcParams(
    n_walkers=400, n_prop_steps=50, n_eql_blocks=100, n_blocks=300,
    dt=0.01, seed=7,
)
if params.n_blocks % REBLOCK or params.n_blocks < 2 * REBLOCK:
    raise ValueError("n_blocks must be divisible by REBLOCK, with at least two groups.")


def paired_stats(energies, weights):
    means = np.sum(energies * weights, axis=0) / np.sum(weights, axis=0)
    estimates = np.r_[means, means[1] - means[0]]
    errors = np.full(3, np.nan)
    groups = len(energies) // REBLOCK
    if groups >= 2:
        n = groups * REBLOCK
        num = (energies[:n] * weights[:n]).reshape(groups, REBLOCK, 2).sum(axis=1)
        den = weights[:n].reshape(groups, REBLOCK, 2).sum(axis=1)
        # Delete the same time group from both ratios to retain covariance.
        loo = (num.sum(axis=0) - num) / (den.sum(axis=0) - den)
        loo = np.column_stack((loo, loo[:, 1] - loo[:, 0]))
        errors = np.sqrt((groups - 1) / groups * np.sum((loo - loo.mean(axis=0)) ** 2, axis=0))
    return estimates, errors


lib.num_threads(12)
# atomstring = """
# C      0.000000    0.531791    0.000000
# O      0.000000   -0.676588    0.000000
# H      0.000000    1.110980    0.934673
# H      0.000000    1.110980   -0.934673
# """
atomstring = """
O        0.000000    0.000000    0.117790
H        0.000000    0.755453   -0.471161
H        0.000000   -0.755453   -0.471161
"""
mol = gto.M(atom=atomstring, basis="aug-ccpvqz", unit="Angstrom", verbose=3)
mf0 = scf.RHF(mol)
mf0.conv_tol = 1.0e-9
mf0.kernel()
if not mf0.converged:
    raise RuntimeError("Zero-field RHF did not converge.")

staged = stage(mf0, norb_frozen_core=0, chol_cut=1.0e-5)
ham0 = HamChol(
    h0=jnp.asarray(staged.ham.h0), h1=jnp.asarray(staged.ham.h1),
    chol=jnp.asarray(staged.ham.chol), basis=staged.ham.basis,
)
sys = System(norb=staged.ham.norb, nelec=staged.ham.nelec, walker_kind="restricted")
trial_ops = make_rhf_trial_ops(sys)
meas_ops = make_rhf_meas_ops(sys)
prop_ops = make_prop_ops(ham0.basis, sys.walker_kind)

with mol.with_common_orig((0.0, 0.0, 0.0)):
    z_ao = mol.intor_symmetric("int1e_r", comp=3)[2]
z_mo = jnp.asarray(mf0.mo_coeff.conj().T @ z_ao @ mf0.mo_coeff)
mu_nuc = np.dot(mol.atom_charges(), mol.atom_coords()[:, 2])
hcore = mf0.get_hcore()
enuc = mol.energy_nuc()
overlap = mf0.get_ovlp()
hams, trials, cc_energies = [], [], []

for field in (-FIELD, FIELD):
    print(f"\nField {field:+.6f} a.u.", flush=True)
    mf = scf.RHF(mol)
    # H(F) = H(0) + F sum_i z_i - F sum_A Z_A R_A,z.
    mf.get_hcore = lambda *args, field=field: hcore + field * z_ao
    mf.energy_nuc = lambda field=field: enuc - field * mu_nuc
    mf.conv_tol = 1.0e-9
    mf.kernel(dm0=mf0.make_rdm1())
    if not mf.converged:
        raise RuntimeError(f"RHF did not converge at field {field}.")
    if RUN_CCSD_T:
        mycc = cc.CCSD(mf)
        mycc.conv_tol = 1.0e-11
        mycc.conv_tol_normt = 1.0e-9
        mycc.max_cycle = 100
        mycc.kernel()
        if not mycc.converged:
            raise RuntimeError(f"CCSD did not converge at field {field}.")
        cc_energies.append(mycc.e_tot + mycc.ccsd_t())
    hams.append(replace(ham0, h0=ham0.h0 - field * mu_nuc, h1=ham0.h1 + field * z_mo))
    coeff = mf0.mo_coeff.conj().T @ overlap @ mf.mo_coeff[:, :sys.nup]
    trials.append(RhfTrial(jnp.asarray(coeff)))

mu_cc = np.nan
if RUN_CCSD_T:
    mu_cc = -(cc_energies[1] - cc_energies[0]) / (2 * FIELD)
    print(f"\nCCSD(T) mu_z: {mu_cc:.8f} a.u.  {mu_cc * AU2DEBYE:.8f} Debye", flush=True)
states = tuple(
    prop_ops.init_prop_state(
        sys=sys, params=params, ham_data=h, trial_data=t,
        trial_ops=trial_ops, meas_ops=meas_ops,
    )
    for h, t in zip(hams, trials)
)
sample = jax.jit(partial(
    run_correlated, sys=sys, ham_data=tuple(hams), trial_data=tuple(trials),
    trial_ops=(trial_ops, trial_ops), meas_ops=(meas_ops, meas_ops),
    prop_ops=(prop_ops, prop_ops),
), static_argnames=("params",))
print("\nE1 = E(-F), E2 = E(+F); energies in Eh, dipoles in Debye.", flush=True)
started = perf_counter()
for phase, n_blocks in (("eql", params.n_eql_blocks), ("prod", params.n_blocks)):
    energy_parts, weight_parts = [], []
    for start in range(0, n_blocks, PRINT_EVERY):
        stop = min(start + PRINT_EVERY, n_blocks)
        states, result = sample(states, params=replace(params, n_blocks=stop - start))
        if not bool(result.alive):
            raise RuntimeError(f"A correlated population became extinct during {phase}.")
        energy_parts.append(np.asarray(result.block_energies))
        weight_parts.append(np.asarray(result.block_weights))
        energies = np.concatenate(energy_parts)
        weights = np.concatenate(weight_parts)
        means, errors = paired_stats(energies, weights)
        mu_afqmc = -means[2] / (2 * FIELD)
        mu_err = errors[2] / (2 * abs(FIELD))
        print(
            f"[{phase} {stop:4d}/{n_blocks}]  "
            f"E1={means[0]:.10f} +/- {errors[0]:.2e}  "
            f"E2={means[1]:.10f} +/- {errors[1]:.2e}  "
            f"dE={means[2]:+.6e} +/- {errors[2]:.2e}  "
            f"mu_z={mu_afqmc * AU2DEBYE:.6f} +/- {mu_err * AU2DEBYE:.2e} D  "
            f"t={perf_counter() - started:.1f}s",
            flush=True,
        )

print(f"\nAFQMC mu_z:   {mu_afqmc:.8f} +/- {mu_err:.8f} a.u.")
print(f"              {mu_afqmc * AU2DEBYE:.8f} +/- {mu_err * AU2DEBYE:.8f} Debye")
if RUN_CCSD_T:
    print(f"CCSD(T) mu_z: {mu_cc:.8f} a.u.  {mu_cc * AU2DEBYE:.8f} Debye")
    print(f"AFQMC - CCSD(T): {(mu_afqmc - mu_cc) * AU2DEBYE:+.8f} Debye")
np.savez(
    "dipole.npz", field=FIELD, run_ccsd_t=RUN_CCSD_T,
    reblock=REBLOCK, dt=params.dt, seed=params.seed,
    mean_energies=means[:2], block_energies=energies,
    block_weights=weights, cc_energies=np.asarray(cc_energies),
    dipole_afqmc=mu_afqmc, dipole_afqmc_error=mu_err, dipole_ccsdt=mu_cc,
)
