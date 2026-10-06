from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax, tree_util

from ..core.ops import TrialOps
from ..core.system import System


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class RhfTrial:
    mo_coeff: jax.Array  # (norb, nocc)

    @property
    def norb(self) -> int:
        return int(self.mo_coeff.shape[0])

    @property
    def nocc(self) -> int:
        return int(self.mo_coeff.shape[1])

    def tree_flatten(self):
        return (self.mo_coeff,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        (mo_coeff,) = children
        return cls(mo_coeff=mo_coeff)


def _det(m: jax.Array) -> jax.Array:
    return jnp.linalg.det(m)


def get_rdm1(trial_data: RhfTrial) -> jax.Array:
    c = trial_data.mo_coeff
    dm = c @ c.conj().T  # (norb, norb)
    return jnp.stack([dm, dm], axis=0)  # (2, norb, norb)


def _canonicalize_evec_signs(mo_energy: jax.Array, mo_coeff: jax.Array) -> jax.Array:
    idx = jnp.argmax(jnp.abs(jnp.real(mo_coeff)), axis=0)
    signs = jnp.where(jnp.real(mo_coeff[idx, jnp.arange(mo_energy.size)]) < 0, -1, 1)
    return mo_coeff * signs


def optimize(ham_data: Any, trial_data: RhfTrial, n_opt_iter: int = 30) -> RhfTrial:
    h1 = (ham_data.h1 + ham_data.h1.T.conj()) * 0.5
    chol = ham_data.chol
    nocc = trial_data.nocc

    def scanned_fun(dm: jax.Array, _x: Any):
        f = jnp.einsum("gij,ik->gjk", chol, dm, optimize="optimal")
        c = jax.vmap(jnp.trace)(f)
        vj = jnp.einsum("g,gij->ij", c, chol, optimize="optimal")
        vk = jnp.einsum("glj,gjk->lk", f, chol, optimize="optimal")
        fock = h1 + vj - 0.5 * vk
        mo_energy, mo_coeff = jnp.linalg.eigh((fock + fock.T.conj()) * 0.5)
        mo_coeff = _canonicalize_evec_signs(mo_energy, mo_coeff)
        e_idx = jnp.argsort(mo_energy)
        mo_occ = mo_coeff[:, e_idx[:nocc]]
        dm_next = 2.0 * (mo_occ @ mo_occ.T.conj())
        return dm_next, mo_occ

    dm0 = 2.0 * (trial_data.mo_coeff @ trial_data.mo_coeff.T.conj())
    _, mo_occ = lax.scan(scanned_fun, dm0, xs=None, length=n_opt_iter)
    return RhfTrial(mo_coeff=mo_occ[-1])


def overlap_r(walker: jax.Array, trial_data: RhfTrial) -> jax.Array:
    m = trial_data.mo_coeff.conj().T @ walker  # (nocc, nocc)
    return _det(m) ** 2


def overlap_u(walker: tuple[jax.Array, jax.Array], trial_data: RhfTrial) -> jax.Array:
    wu, wd = walker
    cu = trial_data.mo_coeff.conj().T @ wu  # (nocc_a, nocc_a)
    cd = trial_data.mo_coeff.conj().T @ wd  # (nocc_b, nocc_b)
    return _det(cu) * _det(cd)


def overlap_g(walker: jax.Array, trial_data: RhfTrial) -> jax.Array:
    norb = trial_data.norb
    cH = trial_data.mo_coeff.conj().T  # (nocc, norb)
    top = cH @ walker[:norb, :]  # (nocc, 2*nocc)
    bot = cH @ walker[norb:, :]  # (nocc, 2*nocc)
    m = jnp.vstack([top, bot])  # (2*nocc, 2*nocc)
    return _det(m)


def make_rhf_trial_ops(sys: System) -> TrialOps:
    if sys.nup != sys.ndn:
        raise ValueError("RHF requires nelec[0] == nelec[1].")

    wk = sys.walker_kind.lower()

    if wk == "restricted":
        overlap_fn = overlap_r
        get_rdm1_fn = get_rdm1
    elif wk == "unrestricted":
        overlap_fn = overlap_u
        get_rdm1_fn = get_rdm1
    elif wk == "generalized":
        overlap_fn = overlap_g
        get_rdm1_fn = get_rdm1
    else:
        raise ValueError(f"unknown walker_kind: {sys.walker_kind}")

    return TrialOps(
        overlap=overlap_fn,
        get_rdm1=get_rdm1_fn,
        optimize=optimize,
    )


def make_rhf_trial_data(data: dict, sys: System) -> RhfTrial:
    mo = jnp.asarray(data["mo"])
    mo_occ = mo[:, : sys.nup]
    return RhfTrial(mo_occ)
