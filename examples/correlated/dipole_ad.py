"""Water dipole from the AD-AFQMC 1-RDM, with blocking statistics."""

from trot import config

config.configure_once(use_gpu=False)

import numpy as np
from pyscf import gto, lib, scf
from pyscf.data.nist import AU2DEBYE

from trot.afqmc import Afqmc
from trot.prop.types import QmcParams
from trot.stat_utils import blocking_analysis_ratio

BASIS = "ccpvdz"
N_AD_RUNS = 20
N_BLOCKS_PER_AD_RUN = 10
SR_INTERVAL = 10
ORBITAL_RELAXATION = True
params = QmcParams(
    n_walkers=400, n_prop_steps=50, n_eql_blocks=25, n_blocks=200,
    dt=0.005, seed=7,
)

lib.num_threads(8)
mol = gto.M(
    atom="""
    O  0.000000   0.000000   0.117790
    H  0.000000   0.755453  -0.471161
    H  0.000000  -0.755453  -0.471161
    """,
    basis=BASIS, unit="Angstrom", verbose=3,
)
mf = scf.RHF(mol).run(conv_tol=1.0e-9)
with mol.with_common_orig((0.0, 0.0, 0.0)):
    z_ao = mol.intor_symmetric("int1e_r", comp=3)[2]
z_mo = mf.mo_coeff.T @ z_ao @ mf.mo_coeff
mu_nuc = np.dot(mol.atom_charges(), mol.atom_coords()[:, 2])
print(f"RHF mu_z: {mf.dip_moment(unit='AU', verbose=0)[2]:.8f} a.u.", flush=True)

af = Afqmc(mf, norb_frozen_core=0, chol_cut=1.0e-5)
af.params = params
af.mixed_precision = False
af.kernel_rdm1(
    output="dipole_ad_rdm1.npz", n_ad_runs=N_AD_RUNS,
    n_blocks_per_ad_run=N_BLOCKS_PER_AD_RUN, sr_interval=SR_INTERVAL,
    orbital_relaxation=ORBITAL_RELAXATION,
)
result = af.qmc_result
dipoles = mu_nuc - np.einsum("nij,ji->n", result.block_rdm1, z_mo).real
stats = blocking_analysis_ratio(dipoles, result.block_weights)
mu = stats["mu"]
error = np.nan if stats["se_star"] is None else stats["se_star"]
print(f"AD-AFQMC mu_z: {mu:.8f} +/- {error:.3e} a.u.")
print(f"               {mu * AU2DEBYE:.8f} +/- {error * AU2DEBYE:.3e} Debye")
np.savez(
    "dipole_ad.npz", dipole=mu, dipole_error=error,
    dipole_samples=dipoles, block_weights=result.block_weights,
    mean_energy=result.mean_energy, energy_error=result.stderr_energy,
    block_energies=result.block_energies, basis=BASIS,
    n_blocks_per_ad_run=N_BLOCKS_PER_AD_RUN, n_prop_steps=params.n_prop_steps,
    dt=params.dt, orbital_relaxation=ORBITAL_RELAXATION,
)
