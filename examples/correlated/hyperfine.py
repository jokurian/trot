"""BeH contact hyperfine coupling from AFQMC, with an optional CCSD(T) reference."""

from trot import config

config.configure_once(use_gpu=False)

import numpy as np
from pyscf import cc, gto, lib, scf

from trot.correlated import run_correlated_hyperfine
from trot.correlated.wrapper import HYPERFINE_PREFACTORS
from trot.prop.types import QmcParams

TRIAL_TYPE = "gcisd"  # "gcisd" or "gccsd"
RUN_CCSD_T = True
ATOM = 1  # zero-based: Be=0, H=1
FIELD = 1e-4
BASIS = "ccpvdz" #"ccpvdz"
params = QmcParams(
    n_walkers=400, n_prop_steps=50, n_eql_blocks=40, n_blocks=100,
    dt=0.005, seed=7,
)
atomstring = """
     Be     -1.51975637     1.71736374     0.00000000
     H      -0.17462876     1.71736374     0.00000000
     """

lib.num_threads(8)
mol = gto.M(
    atom=atomstring, unit="Angstrom",
    basis=BASIS, charge=0, spin=1, verbose=3,
)
mf = scf.UHF(mol).run(conv_tol=1e-12, max_cycle=200)
if not mf.converged:
    raise RuntimeError("Zero-field UHF did not converge.")

cc_energies = []
coupling_ccsdt = np.nan
if RUN_CCSD_T:
    ghf = mf.to_ghf()
    ao = mol.eval_gto("GTOval", mol.atom_coord(ATOM)[None, :])[0]
    contact = np.outer(ao.conj(), ao)
    zero = np.zeros_like(contact)
    operator = np.block([[contact, zero], [zero, -contact]])
    hcore, dm0 = ghf.get_hcore(), ghf.make_rdm1()
    for field in (-FIELD, FIELD):
        perturbed = ghf.copy()
        perturbed.chkfile = None
        perturbed.get_hcore = lambda *args, h=hcore + field * operator: h
        perturbed.kernel(dm0=dm0)
        if not perturbed.converged:
            raise RuntimeError(f"GHF did not converge at field {field}.")
        mycc = cc.GCCSD(perturbed)
        mycc.conv_tol = 1e-11
        mycc.conv_tol_normt = 1e-10
        mycc.max_cycle = 100
        mycc.kernel()
        if not mycc.converged:
            raise RuntimeError(f"GCCSD did not converge at field {field}.")
        cc_energies.append(mycc.e_tot + mycc.ccsd_t())
    prefactor = HYPERFINE_PREFACTORS[mol.atom_pure_symbol(ATOM)] / mol.spin
    coupling_ccsdt = prefactor * (cc_energies[1] - cc_energies[0]) / (2 * FIELD)

result = run_correlated_hyperfine(
    mf, atom=ATOM, trial_type=TRIAL_TYPE, field=FIELD,
    params=params, print_every=20,
)
if RUN_CCSD_T:
    print(f"CCSD(T) A_iso = {coupling_ccsdt:.6f} MHz")
    print(f"AFQMC - CCSD(T) = {result.coupling - coupling_ccsdt:.6f} MHz")
np.savez(
    "hyperfine.npz", atom=ATOM, trial_type=TRIAL_TYPE, field=FIELD,
    coupling=result.coupling, coupling_error=result.coupling_error,
    run_ccsd_t=RUN_CCSD_T, cc_energies=np.asarray(cc_energies),
    coupling_ccsdt=coupling_ccsdt,
    block_energies=result.afqmc.block_energies,
    block_weights=result.afqmc.block_weights, Q=result.afqmc.Q,
)
