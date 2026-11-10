"""Differentiable parent-child correlated sampling for one-body changes."""

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


class CorrelatedResult(NamedTuple):
    mean_energies: jax.Array
    delta_energy: jax.Array
    block_energies: jax.Array
    block_weights: jax.Array
    alive: jax.Array
    Q: jax.Array | None = None



def parent_population_control(state: StatePair, zeta: jax.Array) -> StatePair:
    # return parent_population_control_static(state, zeta)
  return parent_population_control_dynamic(state, zeta)

def parent_population_control_static(state: StatePair, zeta: jax.Array) -> StatePair:
    parent, child = state
    n = parent.weights.shape[0]
    idx = wk._sr_indices(parent.weights, zeta, n)
    avg_weights = jnp.mean(parent.weights[idx])
    child_weights = child.weights[idx] * avg_weights/ parent.weights[idx]
    return (
        parent._replace(
            walkers=wk.take_walkers(parent.walkers, idx),
            weights=jnp.full_like(parent.weights, avg_weights),
            overlaps=parent.overlaps[idx],
        ),
        child._replace(
            walkers=wk.take_walkers(child.walkers, idx),
            weights=child_weights,
            overlaps=child.overlaps[idx],
        ),
    )



def parent_population_control_dynamic(state: StatePair, zeta: jax.Array) -> StatePair:
    parent, child = state
    n = parent.weights.shape[0]
    avg_weights_across_states = jnp.mean(jnp.stack([parent.weights, child.weights]), axis=0)
    idx = wk._sr_indices(avg_weights_across_states, zeta, n)
    
    avg_2 = jnp.mean(parent.weights[idx])

    parent_weights = parent.weights[idx]* avg_2 / avg_weights_across_states[idx]
    child_weights = child.weights[idx] * avg_2 / avg_weights_across_states[idx] 
    return (
        parent._replace(
            walkers=wk.take_walkers(parent.walkers, idx),
            weights=parent_weights,
            overlaps=parent.overlaps[idx],
        ),
        child._replace(
            walkers=wk.take_walkers(child.walkers, idx),
            weights=child_weights,
            overlaps=child.overlaps[idx],
        ),
    )


def correlated_block(
    state: StatePair,
    *,
    sys: System,
    params: QmcParams,
    ham_data: tuple[HamChol, HamChol],
    trial_data: tuple[Any, Any],
    trial_ops: tuple[TrialOps, TrialOps],
    meas_ops: tuple[MeasOps, MeasOps],
    prop_ops: tuple[PropOps, PropOps],
    meas_ctx: tuple[Any, Any],
    prop_ctx: tuple[Any, Any],
    population_control_fn: Callable = parent_population_control,
) -> tuple[StatePair, BlockObs]:
    """Propagate with shared Gaussian draws, measure, then resample the pair."""
    state = (state[0], state[1]._replace(rng_key=state[0].rng_key))
    outputs = tuple(
        block(
            state[i], sys=sys, params=params, ham_data=ham_data[i],
            trial_data=trial_data[i], trial_ops=trial_ops[i], meas_ops=meas_ops[i],
            prop_ops=prop_ops[i], meas_ctx=meas_ctx[i], prop_ctx=prop_ctx[i],
            sr_fn=wk.no_sr,
        )
        for i in range(2)
    )
    key, subkey = jax.random.split(outputs[0][0].rng_key)
    states = population_control_fn(
        (outputs[0][0], outputs[1][0]), jax.random.uniform(subkey)
    )
    states = tuple(s._replace(rng_key=key) for s in states)
    energies = jnp.stack([obs.scalars["energy"] for _, obs in outputs])
    weights = jnp.stack([obs.scalars["weight"] for _, obs in outputs])
    totals = jnp.stack([jnp.sum(s.weights) for s in states])
    alive = jnp.all(jnp.isfinite(weights) & (weights > 0)) & jnp.all(
        jnp.isfinite(totals) & (totals > 0)
    )
    parent_weights, child_weights = (s.weights for s, _ in outputs)
    pair_weights = parent_weights + child_weights
    Q = jnp.mean(
        jnp.abs(parent_weights - child_weights)/(pair_weights)
    )
    return states, BlockObs(
        scalars={"energy": energies, "weight": weights, "alive": alive, "Q": Q}, observables={}
    )


def run_correlated(
    state: StatePair,
    *,
    sys: System,
    params: QmcParams,
    ham_data: tuple[HamChol, HamChol],
    trial_data: tuple[Any, Any],
    trial_ops: tuple[TrialOps, TrialOps],
    meas_ops: tuple[MeasOps, MeasOps],
    prop_ops: tuple[PropOps, PropOps],
    population_control_fn: Callable = parent_population_control,
) -> tuple[StatePair, CorrelatedResult]:
    """Sample params.n_blocks; the caller checks result.alive outside AD/JIT.

    Both Hamiltonians must share their Cholesky factors. All pairs are ordered
    (parent, child); returned block arrays have shape (n_blocks, 2).
    """
    if params.n_blocks < 1:
        raise ValueError("Correlated sampling requires at least one block.")
    parent, child = ham_data
    if (parent.basis, parent.h1.shape, parent.chol.shape) != (
        child.basis, child.h1.shape, child.chol.shape
    ):
        raise ValueError("Parent and child must share the orbital and auxiliary-field layout.")
    if any(s.weights.shape != (params.n_walkers,) for s in state):
        raise ValueError("Both populations must have params.n_walkers weights.")

    meas_ctx = tuple(meas_ops[i].build_meas_ctx(ham_data[i], trial_data[i]) for i in range(2))
    prop_ctx = tuple(
        prop_ops[i].build_prop_ctx(ham_data[i], trial_ops[i].get_rdm1(trial_data[i]), params)
        for i in range(2)
    )
    state = tuple(
        state[i]._replace(
            overlaps=wk.vmap_chunked(
                meas_ops[i].overlap, n_chunks=params.n_chunks, in_axes=(0, None)
            )(state[i].walkers, trial_data[i]),
            rng_key=state[0].rng_key,
        )
        for i in range(2)
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
    return state, CorrelatedResult(means, means[1] - means[0], energies, weights, alive, samples["Q"])


from .wrapper import (
    CorrelatedAfqmcResult as CorrelatedAfqmcResult,
    CorrelatedDipoleResult as CorrelatedDipoleResult,
    CorrelatedHyperfineResult as CorrelatedHyperfineResult,
    run_correlated_afqmc as run_correlated_afqmc,
    run_correlated_dipole as run_correlated_dipole,
    run_correlated_hyperfine as run_correlated_hyperfine,
)
