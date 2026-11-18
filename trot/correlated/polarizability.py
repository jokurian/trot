"""Static polarizability from correlated finite-field energy curvatures."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from itertools import product
from time import perf_counter
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from pyscf.lib import param

from . import States, parent_population_control_dynamic, parent_population_control_static, run_correlated
from .wrapper import _make_trial
from ..core.system import System
from ..ham.chol import HamChol
from ..meas.cisd import make_cisd_meas_ops
from ..meas.rhf import make_rhf_meas_ops
from ..meas.ucisd import make_ucisd_meas_ops
from ..meas.uhf import make_uhf_meas_ops
from ..prop.afqmc import make_prop_ops
from ..prop.types import QmcParams
from ..stat_utils import blocking_analysis_ratio, reject_outliers
from ..walkers import init_walkers
from ..trial.cisd import make_cisd_trial_ops
from ..trial.rhf import make_rhf_trial_ops
from ..trial.ucisd import make_ucisd_trial_ops
from ..trial.uhf import make_uhf_trial_ops


class CorrelatedPolarizabilityResult(NamedTuple):
    """Polarizabilities/errors in Angstrom^3, energies in Hartree, fields in a.u."""

    polarizability: np.ndarray
    polarizability_error: np.ndarray
    components: tuple[str, ...]
    field_vectors: np.ndarray
    block_energies: np.ndarray
    block_weights: np.ndarray
    Q: np.ndarray
    final_state: States
    eql_block_energies: np.ndarray
    eql_block_weights: np.ndarray
    eql_Q: np.ndarray
    block_polarizabilities: np.ndarray
    keep_mask: np.ndarray

    @property
    def clean_block_polarizabilities(self):
        return tuple(self.block_polarizabilities[self.keep_mask[:, k], k]
                     for k in range(len(self.components)))


def _field_stencil(components, field):
    if not np.isfinite(field) or field <= 0:
        raise ValueError("field must be positive and finite.")
    if isinstance(components, str):
        components = (components,)
    components = tuple(dict.fromkeys("".join(sorted(c.lower())) for c in components))
    if not components or any(c not in ("xx", "xy", "xz", "yy", "yz", "zz") for c in components):
        raise ValueError("Select components from xx, xy, xz, yy, yz, zz (or their transposes).")
    axes = np.eye(3, dtype=int)
    indices = {(0, 0, 0): 0}
    stencils = []
    for component in components:
        i, j = ("xyz".index(c) for c in component)
        if i == j:
            terms = {(0, 0, 0): 2., tuple(axes[i]): -1., tuple(-axes[i]): -1.}
        else:
            terms = {tuple(a * axes[i] + b * axes[j]): -a * b / 4.
                     for a, b in product((-1, 1), repeat=2)}
        for vector in terms:
            indices.setdefault(vector, len(indices))
        stencils.append(terms)
    coefficients = np.zeros((len(components), len(indices)))
    for row, terms in enumerate(stencils):
        for vector, coefficient in terms.items():
            coefficients[row, indices[vector]] = coefficient / field**2
    return components, field * np.asarray(tuple(indices)), coefficients


def _polarizability_stats(samples, components, keep_mask=None):
    tensor, errors = np.full((3, 3), np.nan), np.full((3, 3), np.nan)
    for k, component in enumerate(components):
        weights = np.ones(len(samples)) if keep_mask is None else keep_mask[:, k].astype(float)
        stats = blocking_analysis_ratio(samples[:, k], weights, print_q=False)
        i, j = ("xyz".index(c) for c in component)
        tensor[i, j] = tensor[j, i] = stats["mu"]
        errors[i, j] = errors[j, i] = stats["se_star"] if stats["se_star"] is not None else np.nan
    return tensor, errors


def run_correlated_polarizability(
    mf: Any,
    trial_type: str | None = None,
    field: float = 1e-3,
    *,
    params: QmcParams,
    components: tuple[str, ...] = ("xx", "yy", "zz"),
    frozen: Any = None,
    chol_cut: float = 1e-5,
    scf_tol: float = 1e-12,
    trial_tol: float = 1e-10,
    origin: tuple[float, float, float] = (0., 0., 0.),
    print_every: int = 20,
    rdm1: Any = None,
    outlier_m: float | None = 10.0,
    correlation_mode: str = "static",
) -> CorrelatedPolarizabilityResult:
    """Sample selected static tensor entries in Angstrom^3 with fixed nuclei.

    Supply a converged AO-basis RHF/UHF object. Trials are rhf/uhf/cisd/ucisd
    or ccsd (alias rccsd)/uccsd, using the SD projection of the CC wavefunction;
    None keeps the MF family. Every field trial is independently relaxed.
    Frozen indices restrict trial excitations only. Unrequested entries are NaN.
    Fields share the zero-field parent. correlation_mode selects "static"
    (parent-weight ancestry) or "dynamic" (mean-weight ancestry across fields).
    The input MF/checkpoint is preserved; final walkers use its common reference MO basis.
    rdm1 initializes all states from the same natural-orbital determinant in
    this zero-field reference MO basis (alpha MO basis for UHF).
    Final statistics reject outliers in each component's per-block
    polarizabilities (Angstrom^3); outlier_m=None disables rejection.
    """
    from pyscf import cc, ci

    from ..staging import stage

    if correlation_mode not in ("static", "dynamic"):
        raise ValueError('correlation_mode must be "static" or "dynamic".')
    population_control_fn = (parent_population_control_static if correlation_mode == "static"
                             else parent_population_control_dynamic)
    components, field_vectors, coefficients = _field_stencil(components, field)
    if outlier_m is not None and (not np.isfinite(outlier_m) or outlier_m <= 0):
        raise ValueError("outlier_m must be positive and finite, or None.")
    if params.n_blocks < 1 or print_every < 1:
        raise ValueError("n_blocks and print_every must be positive.")
    if not mf.converged:
        raise ValueError("The supplied mean-field calculation is not converged.")
    if trial_type is None:
        trial_type = "uhf" if mf.istype("UHF") else "rhf"
    trial_type = trial_type.lower()
    if trial_type == "rccsd":
        trial_type = "ccsd"
    factories = {
        "rhf": (make_rhf_trial_ops, make_rhf_meas_ops),
        "uhf": (make_uhf_trial_ops, make_uhf_meas_ops),
        "cisd": (make_cisd_trial_ops, make_cisd_meas_ops),
        "ucisd": (make_ucisd_trial_ops, make_ucisd_meas_ops),
        "ccsd": (make_cisd_trial_ops, make_cisd_meas_ops),
        "uccsd": (make_ucisd_trial_ops, make_ucisd_meas_ops),
    }
    if trial_type not in factories:
        raise ValueError("trial_type must be rhf, uhf, cisd, ucisd, ccsd, rccsd, or uccsd.")
    unrestricted = trial_type in ("uhf", "ucisd", "uccsd")
    reference = mf.to_uhf() if unrestricted else mf.to_rhf()
    staged = stage(reference, norb_frozen_core=0, chol_cut=chol_cut)
    ham0 = HamChol(jnp.asarray(staged.ham.h0), jnp.asarray(staged.ham.h1),
                   jnp.asarray(staged.ham.chol), basis=staged.ham.basis)
    coeff = np.asarray(reference.mo_coeff[0] if unrestricted else reference.mo_coeff)
    to_mo = coeff.conj().T @ reference.get_ovlp()
    with reference.mol.with_common_orig(origin):
        position = reference.mol.intor_symmetric("int1e_r", comp=3)
    position_mo = coeff.conj().T @ position @ coeff
    hcore, dm0 = reference.get_hcore(), reference.make_rdm1()
    hams, objects = [], []
    constructor = {"cisd": ci.CISD, "ucisd": ci.UCISD,
                   "ccsd": cc.CCSD, "uccsd": cc.UCCSD}.get(trial_type)
    for vector in field_vectors:
        perturbed = reference.copy()
        perturbed.chkfile = None
        field_hcore = hcore + np.einsum("x,xpq->pq", vector, position)
        perturbed.get_hcore = lambda *args, h=field_hcore: h
        perturbed.conv_tol = scf_tol
        perturbed.kernel(dm0=dm0)
        if not perturbed.converged:
            raise RuntimeError(f"SCF did not converge at field {vector}.")
        trial = perturbed
        if constructor is not None:
            trial = constructor(perturbed, frozen=frozen)
            trial.conv_tol = trial_tol
            trial.max_cycle = 100
            if trial_type.endswith("ccsd"):
                trial.conv_tol_normt = trial_tol
            trial.kernel()
            if not trial.converged:
                raise RuntimeError(f"{trial_type.upper()} did not converge at field {vector}.")
        trial = trial.copy()
        trial.mo_coeff = (np.asarray([to_mo @ c for c in trial.mo_coeff]) if unrestricted
                          else to_mo @ trial.mo_coeff)
        objects.append(trial)
        hams.append(replace(ham0, h1=ham0.h1 + np.einsum("x,xpq->pq", vector, position_mo)))

    sys = System(norb=coeff.shape[1], nelec=staged.ham.nelec,
                 walker_kind="unrestricted" if unrestricted else "restricted")
    trial_data = tuple(_make_trial(obj, sys.nelec, sys.norb)[1] for obj in objects)
    trial_factory, meas_factory = factories[trial_type]
    trial_ops = trial_factory(sys)
    meas_ops = (meas_factory(sys, mixed_precision=False, testing=True)
                if constructor is not None else meas_factory(sys))
    orbitals = None
    if constructor is not None:
        orbitals = tuple(jnp.asarray(obj.mo_coeff[0] if unrestricted else obj.mo_coeff).real
                         for obj in objects)
        hams = [replace(h, h1=u.conj().T @ h.h1 @ u, chol=u.conj().T @ h.chol @ u)
                for h, u in zip(hams, orbitals)]
    prop_ops = make_prop_ops(ham0.basis, sys.walker_kind)
    initial_walkers = (None,) * len(hams)
    if rdm1 is not None:
        walkers = init_walkers(sys, jnp.asarray(rdm1), params.n_walkers)
        initial_walkers = (tuple(jax.tree_util.tree_map(lambda w: u.conj().T @ w, walkers)
                                 for u in orbitals) if orbitals is not None else (walkers,) * len(hams))
    states = tuple(prop_ops.init_prop_state(
        sys=sys, params=params, ham_data=h, trial_data=t,
        trial_ops=trial_ops, meas_ops=meas_ops, initial_walkers=w,
    ) for h, t, w in zip(hams, trial_data, initial_walkers))
    n_systems = len(hams)
    sample = jax.jit(partial(
        run_correlated, sys=sys,
        trial_ops=(trial_ops,) * n_systems, meas_ops=(meas_ops,) * n_systems,
        prop_ops=(prop_ops,) * n_systems, population_control_fn=population_control_fn,
    ), static_argnames=("params",))

    print(f"\nPolarizability (Angstrom^3): {', '.join(components)}; "
          f"{n_systems} systems, zero-field parent, {correlation_mode} population control.", flush=True)
    for i, vector in enumerate(field_vectors):
        field_text = ", ".join(f"{v:+.6e}" for v in vector)
        print(f"  E{i}: F=({field_text}) a.u.", flush=True)
    started = perf_counter()
    equilibration = (np.empty((0, n_systems)), np.empty((0, n_systems)), np.empty(0))
    for phase, n_blocks in (("eql", params.n_eql_blocks), ("prod", params.n_blocks)):
        energy_parts, weight_parts, q_parts = [], [], []
        for start in range(0, n_blocks, print_every):
            stop = min(start + print_every, n_blocks)
            states, result = sample(
                states, ham_data=tuple(hams), trial_data=trial_data,
                params=replace(params, n_blocks=stop - start),
            )
            if not bool(result.alive):
                raise RuntimeError(f"A correlated population became extinct during {phase}.")
            energy_parts.append(np.asarray(result.block_energies))
            weight_parts.append(np.asarray(result.block_weights))
            q_parts.append(np.asarray(result.Q))
            energies, weights, q_values = map(np.concatenate, (energy_parts, weight_parts, q_parts))
            samples = ((energies - energies[:, :1]) @ coefficients.T) * param.BOHR**3
            tensor, errors = _polarizability_stats(samples, components)
            values = []
            for component in components:
                i, j = ("xyz".index(c) for c in component)
                values.append(f"alpha_{component}={tensor[i, j]:.8f} +/- {errors[i, j]:.2e}")
            print(f"[{phase} {stop:4d}/{n_blocks}]  {'  '.join(values)}  "
                  f"Q={q_values[-1]:.6e}  t={perf_counter() - started:.1f}s", flush=True)
            for i in range(n_systems):
                stats = blocking_analysis_ratio(energies[:, i], weights[:, i], print_q=False)
                error = stats["se_star"] if stats["se_star"] is not None else np.nan
                print(f"  E{i}={stats['mu']:.10f} +/- {error:.2e} Eh", flush=True)
            if phase == "prod" and stop == n_blocks:
                keep_mask = np.ones(samples.shape, dtype=bool)
                for k, component in enumerate(components):
                    if outlier_m is not None:
                        _, keep_mask[:, k] = reject_outliers(
                            samples, obs=k, m=outlier_m, min_threshold=np.finfo(float).eps,
                        )
                    rejected = int((~keep_mask[:, k]).sum())
                    print(f"alpha_{component}: Rejected {rejected} of {len(samples)} production blocks "
                          f"(outlier_m={outlier_m}).", flush=True)
                    if not keep_mask[:, k].any():
                        raise ValueError(f"Outlier rejection removed all {component} production blocks.")
                if not keep_mask.all():
                    tensor, errors = _polarizability_stats(samples, components, keep_mask)
                values = []
                for component in components:
                    i, j = ("xyz".index(c) for c in component)
                    values.append(f"alpha_{component}={tensor[i, j]:.8f} +/- {errors[i, j]:.2e}")
                print(f"[prod {stop:4d}/{n_blocks} filtered]  {'  '.join(values)}", flush=True)
        if phase == "eql" and energy_parts:
            equilibration = (energies, weights, q_values)
    if orbitals is not None:
        states = tuple(s._replace(walkers=(tuple(u @ w for w in s.walkers)
                                          if unrestricted else u @ s.walkers))
                       for s, u in zip(states, orbitals))
    return CorrelatedPolarizabilityResult(
        tensor, errors, components, field_vectors, energies, weights, q_values, states, *equilibration,
        samples, keep_mask,
    )
