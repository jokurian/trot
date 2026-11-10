"""CISD-trial AFQMC dipole via the wrapper, with separate CI/CCSD(T) references."""

from trot import config

config.configure_once(use_gpu=False)

import numpy as np
from pyscf import cc, ci, gto, lib, scf
from pyscf.data.nist import AU2DEBYE

from trot.correlated import run_correlated_dipole
from trot.prop.types import QmcParams

FIELD = 1.0e-5
RUN_CCSD_T = True
FROZEN = np.array([], dtype=int)
basis = "ccpvdz"
PRINT_EVERY = 20
params = QmcParams(
    n_walkers=400, n_prop_steps=20, n_eql_blocks=25, n_blocks=200,
    dt=0.005, seed=7,
)


lib.num_threads(8)
atomstring = """
O        0.000000    0.000000    0.117790
H        0.000000    0.755453   -0.471161
H        0.000000   -0.755453   -0.471161
"""
mol = gto.M(atom=atomstring, basis=basis, unit="Angstrom", verbose=3)
mf0 = scf.RHF(mol)
mf0.conv_tol = 1.0e-9
mf0.kernel()
if not mf0.converged:
    raise RuntimeError("Zero-field RHF did not converge.")

with mol.with_common_orig((0.0, 0.0, 0.0)):
    z_ao = mol.intor_symmetric("int1e_r", comp=3)[2]
mu_nuc = np.dot(mol.atom_charges(), mol.atom_coords()[:, 2])
hcore = mf0.get_hcore()
ci_energies, cc_energies = [], []

for field in (-FIELD, FIELD):
    print(f"\nField {field:+.6f} a.u.", flush=True)
    mf = mf0.copy()
    mf.chkfile = None
    mf.get_hcore = lambda *args, h=hcore + field * z_ao: h
    mf.kernel(dm0=mf0.make_rdm1())
    if not mf.converged:
        raise RuntimeError(f"RHF did not converge at field {field}.")
    myci = ci.CISD(mf, frozen=FROZEN)
    myci.conv_tol = 1.0e-11
    myci.max_cycle = 100
    myci.kernel()
    if not myci.converged:
        raise RuntimeError(f"CISD did not converge at field {field}.")
    ci_energies.append(myci.e_tot)
    if RUN_CCSD_T:
        mycc = cc.CCSD(mf)
        mycc.conv_tol = 1.0e-11
        mycc.conv_tol_normt = 1.0e-9
        mycc.max_cycle = 100
        mycc.kernel()
        if not mycc.converged:
            raise RuntimeError(f"CCSD did not converge at field {field}.")
        cc_energies.append(mycc.e_tot + mycc.ccsd_t())

mu_ci = mu_nuc - (ci_energies[1] - ci_energies[0]) / (2 * FIELD)
print(f"\nCISD mu_z: {mu_ci:.8f} a.u.  {mu_ci * AU2DEBYE:.8f} Debye", flush=True)
mu_cc = np.nan
if RUN_CCSD_T:
    mu_cc = mu_nuc - (cc_energies[1] - cc_energies[0]) / (2 * FIELD)
    print(f"\nCCSD(T) mu_z: {mu_cc:.8f} a.u.  {mu_cc * AU2DEBYE:.8f} Debye", flush=True)

dipole = run_correlated_dipole(
    mf0, trial_type="cisd", field=FIELD, frozen=FROZEN, params=params,
    print_every=PRINT_EVERY, scf_tol=1.0e-9, trial_tol=1.0e-11,
)
result = dipole.afqmc
mu_afqmc, mu_err = dipole.dipole, dipole.dipole_error

print(f"\nAFQMC mu_z:   {mu_afqmc:.8f} +/- {mu_err:.8f} a.u.")
print(f"              {mu_afqmc * AU2DEBYE:.8f} +/- {mu_err * AU2DEBYE:.8f} Debye")
print(f"Final Q: {result.Q[-1]:.6e}")
print(f"CISD mu_z: {mu_ci:.8f} a.u.  {mu_ci * AU2DEBYE:.8f} Debye")
if RUN_CCSD_T:
    print(f"CCSD(T) mu_z: {mu_cc:.8f} a.u.  {mu_cc * AU2DEBYE:.8f} Debye")
    print(f"AFQMC - CCSD(T): {(mu_afqmc - mu_cc) * AU2DEBYE:+.8f} Debye")
np.savez(
    "dipole_cisd.npz", field=FIELD, run_ccsd_t=RUN_CCSD_T,
    frozen=FROZEN, ci_energies=np.asarray(ci_energies), dipole_cisd=mu_ci,
    observable_factor=-1.0 / (2 * FIELD), observable_offset=mu_nuc,
    statistics="blocking_analysis_ratio", dt=params.dt, seed=params.seed,
    mean_energies=result.mean_energies, energy_errors=result.energy_errors,
    delta_energy=result.delta_energy, delta_energy_error=result.delta_energy_error,
    block_energies=result.block_energies, block_weights=result.block_weights,
    cc_energies=np.asarray(cc_energies), Q=result.Q,
    dipole_afqmc=mu_afqmc, dipole_afqmc_error=mu_err, dipole_ccsdt=mu_cc,
)
