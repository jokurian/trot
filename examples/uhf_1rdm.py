import numpy as np
from pyscf import gto, scf

from trot import config

config.configure_once(use_gpu=False)

from trot.afqmc import Afqmc

mol = gto.M(atom="H 0 0 0; H 0 0 1.2", basis="sto-6g", verbose=3)

mf = scf.UHF(mol)
mf.kernel()

af = Afqmc(mf)
af.n_walkers = 400
af.n_eql_blocks = 50
af.n_prop_steps = 50
af.seed = 7
af.mixed_precision = False

mean, err, rdm1, rdm1_err = af.kernel_rdm1(
    output="h2_ad_1rdm.npz",
    n_ad_runs=20,
    n_blocks_per_ad_run=50,
    sr_interval=10,
    orbital_relaxation=True,
)

print(f"AFQMC energy: {mean:.12f} +/- {err:.3e}")
print("AD 1RDM:")
print(rdm1)
print("AD 1RDM error:")
print(rdm1_err)
