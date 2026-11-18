"""Differentiable correlated sampling with a shared parent population."""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from .. import walkers as wk
from ..core.ops import MeasOps, TrialOps
from ..core.system import System
from ..ham.chol import HamChol
from ..prop.blocks import BlockObs, block
from ..prop.types import PropOps, PropState, QmcParams

StatePair = tuple[PropState, PropState]
States = tuple[PropState, ...]


class CorrelatedResult(NamedTuple):
    mean_energies: jax.Array
    delta_energy: jax.Array
    block_energies: jax.Array
    block_weights: jax.Array
    alive: jax.Array
    Q: jax.Array | None = None



def parent_population_control(state: States, zeta: jax.Array) -> States:
    return parent_population_control_static(state, zeta)


def parent_population_control_static(state: States, zeta: jax.Array) -> States:
    parent = state[0]
    n = parent.weights.shape[0]
    idx = wk._sr_indices(parent.weights, zeta, n)
    avg_weights = jnp.mean(parent.weights[idx])
    return (
        parent._replace(
            walkers=wk.take_walkers(parent.walkers, idx),
            weights=jnp.full_like(parent.weights, avg_weights),
            overlaps=parent.overlaps[idx],
        ),
    ) + tuple(
        child._replace(
            walkers=wk.take_walkers(child.walkers, idx),
            weights=child.weights[idx] * avg_weights / parent.weights[idx],
            overlaps=child.overlaps[idx],
        )
        for child in state[1:]
    )



def parent_population_control_dynamic(state: States, zeta: jax.Array) -> States:
    parent = state[0]
    n = parent.weights.shape[0]
    avg_weights_across_states = jnp.mean(jnp.stack([s.weights for s in state]), axis=0)
    idx = wk._sr_indices(avg_weights_across_states, zeta, n)
    avg_weights = jnp.mean(parent.weights[idx])
    return tuple(
        s._replace(
            walkers=wk.take_walkers(s.walkers, idx),
            weights=s.weights[idx] * avg_weights / avg_weights_across_states[idx],
            overlaps=s.overlaps[idx],
        )
        for s in state
    )


def correlated_block(
    state: States,
    *,
    sys: System,
    params: QmcParams,
    ham_data: tuple[HamChol, ...],
    trial_data: tuple[Any, ...],
    trial_ops: tuple[TrialOps, ...],
    meas_ops: tuple[MeasOps, ...],
    prop_ops: tuple[PropOps, ...],
    meas_ctx: tuple[Any, ...],
    prop_ctx: tuple[Any, ...],
    population_control_fn: Callable = parent_population_control,
) -> tuple[States, BlockObs]:
    """Propagate with shared Gaussian draws, measure, then resample together."""
    n_systems = len(state)
    if n_systems < 2 or any(len(items) != n_systems for items in (
        ham_data, trial_data, trial_ops, meas_ops, prop_ops, meas_ctx, prop_ctx,
    )):
        raise ValueError("Provide matching tuples for at least two systems.")
    state = tuple(s._replace(rng_key=state[0].rng_key) for s in state)
    outputs = tuple(
        block(
            state[i], sys=sys, params=params, ham_data=ham_data[i],
            trial_data=trial_data[i], trial_ops=trial_ops[i], meas_ops=meas_ops[i],
            prop_ops=prop_ops[i], meas_ctx=meas_ctx[i], prop_ctx=prop_ctx[i],
            sr_fn=wk.no_sr,
        )
        for i in range(n_systems)
    )
    key, subkey = jax.random.split(outputs[0][0].rng_key)
    states = population_control_fn(
        tuple(s for s, _ in outputs), jax.random.uniform(subkey)
    )
    states = tuple(s._replace(rng_key=key) for s in states)
    energies = jnp.stack([obs.scalars["energy"] for _, obs in outputs])
    weights = jnp.stack([obs.scalars["weight"] for _, obs in outputs])
    totals = jnp.stack([jnp.sum(s.weights) for s in states])
    alive = jnp.all(jnp.isfinite(weights) & (weights > 0)) & jnp.all(
        jnp.isfinite(totals) & (totals > 0)
    )
    population_weights = jnp.stack([s.weights for s, _ in outputs])
    Q = jnp.mean(jnp.std(population_weights, axis=0) / jnp.mean(population_weights, axis=0))
    return states, BlockObs(
        scalars={"energy": energies, "weight": weights, "alive": alive, "Q": Q}, observables={}
    )


def run_correlated(
    state: States,
    *,
    sys: System,
    params: QmcParams,
    ham_data: tuple[HamChol, ...],
    trial_data: tuple[Any, ...],
    trial_ops: tuple[TrialOps, ...],
    meas_ops: tuple[MeasOps, ...],
    prop_ops: tuple[PropOps, ...],
    population_control_fn: Callable = parent_population_control,
) -> tuple[States, CorrelatedResult]:
    """Sample params.n_blocks; the caller checks result.alive outside AD/JIT.

    Systems share the orbital and auxiliary-field layout and are ordered
    (parent, child1, ...). Block arrays have shape (n_blocks, n_systems).
    delta_energy is scalar for two systems and a child-minus-parent vector otherwise.
    """
    if params.n_blocks < 1:
        raise ValueError("Correlated sampling requires at least one block.")
    n_systems = len(state)
    if n_systems < 2 or any(len(items) != n_systems for items in (
        ham_data, trial_data, trial_ops, meas_ops, prop_ops,
    )):
        raise ValueError("Provide matching tuples for at least two systems.")
    parent = ham_data[0]
    if any((parent.basis, parent.h1.shape, parent.chol.shape) != (
        child.basis, child.h1.shape, child.chol.shape
    ) for child in ham_data[1:]):
        raise ValueError("All systems must share the orbital and auxiliary-field layout.")
    if any(s.weights.shape != (params.n_walkers,) for s in state):
        raise ValueError("All populations must have params.n_walkers weights.")

    meas_ctx = tuple(meas_ops[i].build_meas_ctx(ham_data[i], trial_data[i]) for i in range(n_systems))
    prop_ctx = tuple(
        prop_ops[i].build_prop_ctx(ham_data[i], trial_ops[i].get_rdm1(trial_data[i]), params)
        for i in range(n_systems)
    )
    state = tuple(
        state[i]._replace(
            overlaps=wk.vmap_chunked(
                meas_ops[i].overlap, n_chunks=params.n_chunks, in_axes=(0, None)
            )(state[i].walkers, trial_data[i]),
            rng_key=state[0].rng_key,
        )
        for i in range(n_systems)
    )

    def one_block(states, _):
        states, obs = correlated_block(
            states, sys=sys, params=params, ham_data=ham_data, trial_data=trial_data,
            trial_ops=trial_ops, meas_ops=meas_ops, prop_ops=prop_ops,
            meas_ctx=meas_ctx, prop_ctx=prop_ctx, population_control_fn=population_control_fn,
        )
        return states, obs.scalars

    state, samples = lax.scan(
        jax.checkpoint(one_block, prevent_cse=False), state, xs=None, length=params.n_blocks
    )
    energies, weights = samples["energy"], samples["weight"]
    totals = jnp.sum(weights, axis=0)
    means = jnp.sum(energies * weights, axis=0) / jnp.where(totals > 0, totals, 1)
    alive = jnp.all(samples["alive"])
    means = jnp.where(alive, means, jnp.nan)
    delta_energy = means[1] - means[0] if n_systems == 2 else means[1:] - means[0]
    return state, CorrelatedResult(means, delta_energy, energies, weights, alive, samples["Q"])


from .wrapper import (
    CorrelatedAfqmcResult as CorrelatedAfqmcResult,
    CorrelatedDipoleResult as CorrelatedDipoleResult,
    CorrelatedHyperfineResult as CorrelatedHyperfineResult,
    run_correlated_afqmc as run_correlated_afqmc,
    run_correlated_dipole as run_correlated_dipole,
    run_correlated_hyperfine as run_correlated_hyperfine,
)
from .polarizability import (
    CorrelatedPolarizabilityResult as CorrelatedPolarizabilityResult,
    run_correlated_polarizability as run_correlated_polarizability,
)
