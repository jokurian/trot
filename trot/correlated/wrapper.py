"""Host-side correlated AFQMC driver and PySCF trial conversion."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from time import perf_counter
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import StatePair, parent_population_control, run_correlated
from ..core.system import System
from ..ham.chol import HamChol
from ..meas.rhf import make_rhf_meas_ops
from ..meas.uhf import make_uhf_meas_ops
from ..meas.cisd import make_cisd_meas_ops
from ..meas.ucisd import make_ucisd_meas_ops
from ..meas.ghf import make_ghf_meas_ops_chol
from ..meas.gcisd import make_gcisd_meas_ops
from ..prop.afqmc import make_prop_ops
from ..prop.types import QmcParams
from ..stat_utils import blocking_analysis_ratio
from ..trial.rhf import make_rhf_trial_data, make_rhf_trial_ops
from ..trial.uhf import make_uhf_trial_data, make_uhf_trial_ops
from ..trial.cisd import make_cisd_trial_data, make_cisd_trial_ops
from ..trial.ucisd import make_ucisd_trial_data, make_ucisd_trial_ops
from ..trial.ghf import make_ghf_trial_data, make_ghf_trial_ops
from ..trial.gcisd import make_gcisd_trial_data, make_gcisd_trial_ops


HYPERFINE_PREFACTORS = {
    "H": 4469.869756, "Be": -628.15, "B": 1434.368268,
    "C": 1124.190028, "N": 323.104008, "O": -606.1916653,
    "F": 4207.428646, "P": 1811.093817, "S": 343.4724192,
    "Cl": 438.4624524,
}


class CorrelatedAfqmcResult(NamedTuple):
    mean_energies: np.ndarray
    energy_errors: np.ndarray
    delta_energy: float
    delta_energy_error: float
    block_energies: np.ndarray
    block_weights: np.ndarray
    Q: np.ndarray
    final_state: StatePair


class CorrelatedDipoleResult(NamedTuple):
    dipole: float
    dipole_error: float
    afqmc: CorrelatedAfqmcResult


class CorrelatedHyperfineResult(NamedTuple):
    coupling: float
    coupling_error: float
    afqmc: CorrelatedAfqmcResult


def _embed(amplitudes, indices, shape):
    full = np.zeros(shape, dtype=np.asarray(amplitudes).dtype)
    full[np.ix_(*indices)] = amplitudes
    return jnp.asarray(full)


def _make_trial(obj, nelec, norb):
    from pyscf.cc.ccsd import CCSD
    from pyscf.cc.uccsd import UCCSD
    from pyscf.cc.gccsd import GCCSD
    from pyscf.ci.cisd import CISD
    from pyscf.ci.ucisd import UCISD
    from pyscf.ci.gcisd import GCISD
    from pyscf.scf.hf import RHF
    from pyscf.scf.uhf import UHF
    from pyscf.scf.ghf import GHF

    if isinstance(obj, (GCCSD, GCISD)):
        kind = "gcisd"
    elif isinstance(obj, (UCCSD, UCISD)):
        kind = "ucisd"
    elif isinstance(obj, (CCSD, CISD)):
        kind = "cisd"
    elif isinstance(obj, UHF):
        kind = "uhf"
    elif isinstance(obj, GHF):
        kind = "ghf"
    elif isinstance(obj, RHF):
        kind = "rhf"
    else:
        raise TypeError("Expected a completed RHF, UHF, GHF, CCSD, UCCSD, GCCSD, CISD, UCISD, or GCISD object.")
    generalized = kind in ("ghf", "gcisd")
    nmo = 2 * norb if generalized else norb
    identity = np.eye(nmo)

    coeff = np.asarray(obj.mo_coeff)
    occ = np.asarray(obj.mo_occ)
    unrestricted = kind in ("uhf", "ucisd")

    sys = System(norb=norb, nelec=nelec,
                 walker_kind="generalized" if generalized else
                             "unrestricted" if unrestricted else "restricted")
    if kind == "ghf":
        return kind, make_ghf_trial_data({"mo": coeff[:, occ > 0]}, sys)
    if kind == "rhf":
        return kind, make_rhf_trial_data({"mo": coeff[:, occ > 0]}, sys)
    if kind == "uhf":
        return kind, make_uhf_trial_data({
            "mo_a": coeff[0][:, occ[0] > 0],
            "mo_b": coeff[1][:, occ[1] > 0],
        }, sys)

    if not generalized:
        coeff = coeff.real
    reference = coeff[0] if unrestricted else coeff

    is_cc = isinstance(obj, (CCSD, UCCSD, GCCSD))
    if is_cc:
        if generalized:
            singles = np.asarray(obj.t1)
            doubles = (np.asarray(obj.t2)
                       + np.einsum("ia,jb->ijab", singles, singles)
                       - np.einsum("ib,ja->ijab", singles, singles))
        elif unrestricted:
            c1a, c1b = map(np.asarray, obj.t1)
            t2aa, t2ab, t2bb = map(np.asarray, obj.t2)
            c2aa = t2aa + 2 * np.einsum("ia,jb->ijab", c1a, c1a)
            c2aa = 0.5 * (c2aa - c2aa.transpose(0, 1, 3, 2))
            c2ab = t2ab + np.einsum("ia,jb->ijab", c1a, c1b)
            c2bb = t2bb + 2 * np.einsum("ia,jb->ijab", c1b, c1b)
            c2bb = 0.5 * (c2bb - c2bb.transpose(0, 1, 3, 2))
            singles, doubles = (c1a, c1b), (c2aa, c2ab, c2bb)
        else:
            singles = np.asarray(obj.t1)
            doubles = np.asarray(obj.t2) + np.einsum("ia,jb->ijab", singles, singles)
    else:
        c0, singles, doubles = obj.cisdvec_to_amplitudes(obj.ci)
        if generalized and abs(c0) < 1e-12:
            raise ValueError("GCISD reference coefficient is too small to normalize the trial.")
        if unrestricted:
            singles = tuple(np.asarray(c) / c0 for c in singles)
            doubles = tuple(np.asarray(c) / c0 for c in doubles)
        else:
            singles, doubles = np.asarray(singles) / c0, np.asarray(doubles) / c0

    masks = obj.get_frozen_mask()
    if not unrestricted:
        nocc = sum(nelec) if generalized else nelec[0]
        nvir = nmo - nocc
        occupied = np.flatnonzero(masks[:nocc])
        virtual = np.flatnonzero(masks[nocc:])
        data = {
            "ci1": _embed(singles, (occupied, virtual), (nocc, nvir)),
            "ci2": _embed(doubles.transpose(0, 2, 1, 3),
                          (occupied, virtual, occupied, virtual), (nocc, nvir, nocc, nvir)),
        }
        if generalized:
            return kind, make_gcisd_trial_data({"mo_coeff": identity, **data}, sys)
        return kind, make_cisd_trial_data(data, sys)
    na, nb = nelec
    va, vb = norb - na, norb - nb
    ia, ib = (np.flatnonzero(mask[:n]) for mask, n in zip(masks, nelec))
    aa, ab = (np.flatnonzero(mask[n:]) for mask, n in zip(masks, nelec))
    c1a, c1b = singles
    c2aa, c2ab, c2bb = doubles
    return kind, make_ucisd_trial_data({
        "mo_coeff_a": identity, "mo_coeff_b": reference.T @ coeff[1],
        "ci1a": _embed(c1a, (ia, aa), (na, va)),
        "ci1b": _embed(c1b, (ib, ab), (nb, vb)),
        "ci2aa": _embed(c2aa.transpose(0, 2, 1, 3), (ia, aa, ia, aa), (na, va, na, va)),
        "ci2ab": _embed(c2ab.transpose(0, 2, 1, 3), (ia, aa, ib, ab), (na, va, nb, vb)),
        "ci2bb": _embed(c2bb.transpose(0, 2, 1, 3), (ib, ab, ib, ab), (nb, vb, nb, vb)),
    }, sys)


def _real_chol(ham):
    if np.any(np.abs(np.asarray(ham.chol).imag) > 1e-12):
        raise ValueError("Generalized correlated AFQMC requires real Cholesky factors; "
                         "shared propagation does not preserve complex Cholesky factors.")
    return replace(ham, chol=jnp.real(ham.chol))


def run_correlated_afqmc(
    ham_data: tuple[HamChol, HamChol],
    *,
    nelec: tuple[int, int],
    trial_objects: tuple[Any, Any],
    params: QmcParams,
    print_every: int = 20,
    observable_factor: float | None = None,
    observable_offset: float = 0.0,
    population_control_fn: Callable = parent_population_control,
) -> CorrelatedAfqmcResult:
    """Sample a pair of Hamiltonians using completed MF, CCSD, or CISD objects.

    Trial mo_coeff arrays must use the supplied common orthonormal MO basis.
    CI/CC reference orbitals may relax independently in each calculation.
    CI/CC runs use their own reference bases; final walkers are returned in the
    input basis. Frozen indices restrict trial excitations only.
    GHF/GCISD/GCCSD require generalized Hamiltonians with real Cholesky factors
    before and after reference rotation (imaginary tolerance 1e-12).
    """

    generalized_ham = ham_data[0].basis == "generalized"
    nmo = ham_data[0].h1.shape[0]
    if generalized_ham and nmo % 2:
        raise ValueError("Generalized Hamiltonians require an even spin-orbital dimension.")
    norb = nmo // 2 if generalized_ham else nmo
    prepared = tuple(_make_trial(obj, nelec, norb) for obj in trial_objects)
    kind = prepared[0][0]
    if prepared[1][0] != kind:
        raise ValueError("Both objects must map to the same trial family.")
    generalized = kind in ("ghf", "gcisd")
    basis = "generalized" if generalized else "restricted"
    if any(h.basis != basis for h in ham_data):
        raise ValueError(f"{kind.upper()} trials require {basis} Hamiltonians.")
    if generalized:
        ham_data = tuple(_real_chol(h) for h in ham_data)

    trial_data = tuple(item[1] for item in prepared)

    walker_kind = ("generalized" if generalized else
                   "unrestricted" if kind in ("uhf", "ucisd") else "restricted")
    sys = System(norb=norb, nelec=nelec, walker_kind=walker_kind)
    factories = {
        "rhf": (make_rhf_trial_ops, make_rhf_meas_ops),
        "uhf": (make_uhf_trial_ops, make_uhf_meas_ops),
        "cisd": (make_cisd_trial_ops, make_cisd_meas_ops),
        "ucisd": (make_ucisd_trial_ops, make_ucisd_meas_ops),
        "ghf": (make_ghf_trial_ops, make_ghf_meas_ops_chol),
        "gcisd": (make_gcisd_trial_ops, make_gcisd_meas_ops),
    }
    trial_factory, meas_factory = factories[kind]
    trial_ops = trial_factory(sys)
    meas_ops = (meas_factory(sys, mixed_precision=False, testing=True)
                if kind in ("cisd", "ucisd") else meas_factory(sys))
    orbitals = None
    if kind in ("cisd", "ucisd", "gcisd"):
        orbitals = tuple(jnp.asarray(obj.mo_coeff[0] if kind == "ucisd" else obj.mo_coeff)
                         for obj in trial_objects)
        if kind != "gcisd":
            orbitals = tuple(u.real for u in orbitals)
        ham_data = tuple(replace(h, h1=u.conj().T @ h.h1 @ u, chol=u.conj().T @ h.chol @ u)
                         for h, u in zip(ham_data, orbitals))
        if generalized:
            ham_data = tuple(_real_chol(h) for h in ham_data)
    prop_ops = make_prop_ops(ham_data[0].basis, sys.walker_kind)
    states = tuple(prop_ops.init_prop_state(
        sys=sys, params=params, ham_data=h, trial_data=t,
        trial_ops=trial_ops, meas_ops=meas_ops,
    ) for h, t in zip(ham_data, trial_data))
    sample = jax.jit(partial(
        run_correlated, sys=sys,
        trial_ops=(trial_ops, trial_ops), meas_ops=(meas_ops, meas_ops),
        prop_ops=(prop_ops, prop_ops), population_control_fn=population_control_fn,
    ), static_argnames=("params",))

    print("\nE1 = parent, E2 = child; dE = paired block difference (Eh).", flush=True)
    started = perf_counter()
    for phase, n_blocks in (("eql", params.n_eql_blocks), ("prod", params.n_blocks)):
        energy_parts, weight_parts, q_parts = [], [], []
        for start in range(0, n_blocks, print_every):
            stop = min(start + print_every, n_blocks)
            states, result = sample(
                states, ham_data=ham_data, trial_data=trial_data,
                params=replace(params, n_blocks=stop - start),
            )
            if not bool(result.alive):
                raise RuntimeError(f"A correlated population became extinct during {phase}.")
            energy_parts.append(np.asarray(result.block_energies))
            weight_parts.append(np.asarray(result.block_weights))
            q_parts.append(np.asarray(result.Q))
            energies = np.concatenate(energy_parts)
            weights = np.concatenate(weight_parts)
            q_values = np.concatenate(q_parts)
            stats = [blocking_analysis_ratio(
                energies[:, i], weights[:, i], print_q=False,
            ) for i in range(2)]
            delta = energies[:, 1] - energies[:, 0]
            stats.append(blocking_analysis_ratio(delta, np.ones(len(delta)), print_q=False))
            means = np.array([s["mu"] for s in stats])
            errors = np.array([s["se_star"] if s["se_star"] is not None else np.nan for s in stats])
            observable_text = ""
            if observable_factor is not None:
                observable = observable_factor * means[2] + observable_offset
                observable_error = abs(observable_factor) * errors[2]
                observable_text = f"obs={observable:.8f} +/- {observable_error:.2e}  "
            print(
                f"[{phase} {stop:4d}/{n_blocks}]  "
                f"E1={means[0]:.10f} +/- {errors[0]:.2e}  "
                f"E2={means[1]:.10f} +/- {errors[1]:.2e}  "
                f"dE={means[2]:+.6e} +/- {errors[2]:.2e}  "
                f"{observable_text}"
                f"Q={q_values[-1]:.6e}  t={perf_counter() - started:.1f}s",
                flush=True,
            )
    if orbitals is not None:
        states = tuple(s._replace(walkers=(tuple(u @ w for w in s.walkers)
                                          if isinstance(s.walkers, tuple) else u @ s.walkers))
                       for s, u in zip(states, orbitals))
    return CorrelatedAfqmcResult(
        means[:2], errors[:2], float(means[2]), float(errors[2]),
        energies, weights, q_values, states,
    )


def run_correlated_dipole(
    mf: Any,
    trial_type: str | None = None,
    field: float = 1e-5,
    *,
    params: QmcParams,
    axis: str = "z",
    frozen: Any = None,
    chol_cut: float = 1e-5,
    scf_tol: float = 1e-10,
    trial_tol: float = 1e-10,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    print_every: int = 20,
    population_control_fn: Callable = parent_population_control,
) -> CorrelatedDipoleResult:
    """Calculate one dipole component (a.u.) from paired fields -field/+field.

    Supply a converged molecular AO-basis RHF, UHF, or GHF object. Trial names
    are rhf/uhf/ghf, cisd/ucisd/gcisd, or ccsd/uccsd/gccsd; None keeps the MF
    family. Generalized trials use field-relaxed UHF converted to GHF.
    Frozen indices restrict trial excitations only. The origin is in
    Bohr. The supplied MF and its checkpoint are not modified.
    """
    from pyscf import cc, ci, scf

    from ..staging import stage

    axis = axis.lower()

    component = ("x", "y", "z").index(axis)
    if trial_type is None:
        trial_type = "ghf" if mf.istype("GHF") else "uhf" if mf.istype("UHF") else "rhf"
    trial_type = trial_type.lower()
    constructors = {
        "rhf": None, "uhf": None, "ghf": None,
        "cisd": ci.CISD, "ucisd": ci.UCISD, "gcisd": ci.GCISD,
        "ccsd": cc.CCSD, "uccsd": cc.UCCSD, "gccsd": cc.GCCSD,
    }
    family = ("ghf" if trial_type in ("ghf", "gcisd", "gccsd") else
              "uhf" if trial_type in ("uhf", "ucisd", "uccsd") else "rhf")
    reference = getattr(mf, f"to_{family}")()

    staged = stage(reference, norb_frozen_core=0, chol_cut=chol_cut)
    ham0 = HamChol(jnp.asarray(staged.ham.h0), jnp.asarray(staged.ham.h1),
                   jnp.asarray(staged.ham.chol), basis=staged.ham.basis)
    if family == "ghf":
        ham0 = _real_chol(ham0)
    coeff = np.asarray(reference.mo_coeff[0] if family == "uhf" else reference.mo_coeff)
    overlap = reference.get_ovlp()
    to_mo = coeff.conj().T @ overlap
    hcore = reference.get_hcore()
    dm0 = reference.make_rdm1()
    mol = reference.mol
    field_reference = reference
    if family == "ghf":
        field_reference = scf.UHF(mol) if mf.istype("GHF") else mf.to_uhf()
        n = mol.nao_nr()
        dm0 = np.asarray([dm0[:n, :n], dm0[n:, n:]])
    with mol.with_common_orig(origin):
        position = mol.intor_symmetric("int1e_r", comp=3)[component]
    if family == "ghf":
        zero = np.zeros_like(position)
        position = np.block([[position, zero], [zero, position]])
    position_mo = coeff.conj().T @ position @ coeff
    nuclear = float(np.dot(mol.atom_charges(), mol.atom_coords()[:, component] - origin[component]))
    hams, trials = [], []
    for strength in (-field, field):
        ham = replace(ham0, h1=ham0.h1 + strength * position_mo)
        perturbed = field_reference.copy()
        perturbed.chkfile = None
        field_hcore = hcore + strength * position
        scf_hcore = (np.asarray([field_hcore[:n, :n], field_hcore[n:, n:]])
                     if family == "ghf" else field_hcore)
        perturbed.get_hcore = lambda *args, h=scf_hcore: h
        perturbed.conv_tol = scf_tol
        perturbed.kernel(dm0=dm0)
        if not perturbed.converged:
            raise RuntimeError(f"SCF did not converge at field {strength:+.6e}.")
        if family == "ghf":
            perturbed = perturbed.to_ghf()
            perturbed.get_hcore = lambda *args, h=field_hcore: h
        constructor = constructors[trial_type]
        trial = perturbed
        if constructor is not None:
            trial = constructor(perturbed, frozen=frozen)
            trial.conv_tol = trial_tol
            trial.max_cycle = 100
            if trial_type.endswith("ccsd"):
                trial.conv_tol_normt = trial_tol
            trial.kernel()
            if not trial.converged:
                raise RuntimeError(f"{trial_type.upper()} did not converge at field {strength:+.6e}.")
        trial = trial.copy()
        trial.mo_coeff = (np.asarray([to_mo @ c for c in trial.mo_coeff]) if family == "uhf"
                          else to_mo @ trial.mo_coeff)
        hams.append(ham)
        trials.append(trial)
    result = run_correlated_afqmc(
        tuple(hams), nelec=staged.ham.nelec, trial_objects=tuple(trials), params=params,
        print_every=print_every, observable_factor=-1.0 / (2 * field),
        observable_offset=nuclear, population_control_fn=population_control_fn,
    )
    dipole = nuclear - result.delta_energy / (2 * field)
    error = result.delta_energy_error / (2 * field)
    print(f"\nDipole {axis}: {dipole:.8f} +/- {error:.2e} a.u.", flush=True)
    return CorrelatedDipoleResult(float(dipole), float(error), result)


def run_correlated_hyperfine(
    mf: Any,
    atom: int,
    trial_type: str = "ghf",
    field: float = 1e-4,
    *,
    params: QmcParams,
    prefactor: float | None = None,
    staged: Any = None,
    frozen: Any = None,
    chol_cut: float = 1e-5,
    scf_tol: float = 1e-12,
    trial_tol: float = 1e-10,
    print_every: int = 20,
    population_control_fn: Callable = parent_population_control,
) -> CorrelatedHyperfineResult:
    """Isotropic Fermi-contact coupling (MHz) for a zero-based atom index.

    Supply a completed molecular AO-basis MF with nonzero mol.spin. Trials are
    ghf, gcisd, or gccsd, using field-relaxed UHF converted to GHF. An isotope
    prefactor is looked up by element unless supplied; it is divided by mol.spin.
    Frozen indices restrict trial excitations only. The input MF is preserved.
    Optional staged inputs must use this MF's zero-field GHF basis, without core removal.
    """
    from pyscf import cc, ci, scf

    from ..staging import stage

    mol = mf.mol
    if mol.spin == 0:
        raise ValueError("Fermi-contact normalization requires nonzero mol.spin.")
    trial_type = trial_type.lower()
    constructor = {"ghf": None, "gcisd": ci.GCISD, "gccsd": cc.GCCSD}[trial_type]
    symbol = mol.atom_pure_symbol(atom)
    if prefactor is None:
        prefactor = HYPERFINE_PREFACTORS[symbol]
    factor = prefactor / (mol.spin * 2 * field)
    reference = mf.to_ghf()
    if staged is None:
        staged = stage(reference, norb_frozen_core=0, chol_cut=chol_cut)
    ham0 = _real_chol(HamChol(jnp.asarray(staged.ham.h0), jnp.asarray(staged.ham.h1),
                             jnp.asarray(staged.ham.chol), basis=staged.ham.basis))
    coeff = np.asarray(reference.mo_coeff)
    to_mo = coeff.conj().T @ reference.get_ovlp()
    ao = mol.eval_gto("GTOval", mol.atom_coord(atom)[None, :])[0]
    contact = np.outer(ao.conj(), ao)
    zero = np.zeros_like(contact)
    operator = np.block([[contact, zero], [zero, -contact]])
    operator_mo = coeff.conj().T @ operator @ coeff
    hcore, dm0 = reference.get_hcore(), reference.make_rdm1()
    unrestricted = scf.UHF(mol) if mf.istype("GHF") else mf.to_uhf()
    n = mol.nao_nr()
    dm0 = np.asarray([dm0[:n, :n], dm0[n:, n:]])
    hams, trials = [], []
    for strength in (-field, field):
        perturbed = unrestricted.copy()
        perturbed.chkfile = None
        field_hcore = hcore + strength * operator
        spin_hcore = np.asarray([field_hcore[:n, :n], field_hcore[n:, n:]])
        perturbed.get_hcore = lambda *args, h=spin_hcore: h
        perturbed.conv_tol = scf_tol
        perturbed.kernel(dm0=dm0)
        if not perturbed.converged:
            raise RuntimeError(f"UHF did not converge for atom {atom + 1}, field {strength:+.6e}.")
        perturbed = perturbed.to_ghf()
        perturbed.get_hcore = lambda *args, h=field_hcore: h
        trial = perturbed
        if constructor is not None:
            trial = constructor(perturbed, frozen=frozen)
            trial.conv_tol = trial_tol
            trial.max_cycle = 100
            if trial_type == "gccsd":
                trial.conv_tol_normt = trial_tol
            if trial_type == "gcisd":
                guess = np.zeros(trial.vector_size(), dtype=trial.mo_coeff.dtype)
                guess[0] = 1
                trial.kernel(ci0=guess)
            else:
                trial.kernel()
            if not trial.converged:
                raise RuntimeError(f"{trial_type.upper()} did not converge for atom {atom + 1}, field {strength:+.6e}.")
        trial = trial.copy()
        trial.mo_coeff = to_mo @ trial.mo_coeff
        trials.append(trial)
        hams.append(replace(ham0, h1=ham0.h1 + strength * operator_mo))
    result = run_correlated_afqmc(
        tuple(hams), nelec=staged.ham.nelec, trial_objects=tuple(trials), params=params,
        print_every=print_every, observable_factor=factor,
        population_control_fn=population_control_fn,
    )
    coupling = factor * result.delta_energy
    error = abs(factor) * result.delta_energy_error
    print(f"\nAtom {atom + 1} {symbol}: A_iso = {coupling:.6f} +/- {error:.2e} MHz", flush=True)
    return CorrelatedHyperfineResult(float(coupling), float(error), result)

