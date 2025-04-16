import math
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import lax, random
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Float, Int, Int32
from nav_msgs.msg import OccupancyGrid
from numpyro.distributions import MixtureGeneral, constraints
from tf_transformations import euler_from_quaternion

from _liblocalization_cpp import distance_2d as _distance_2d
from libracecar.batched import batched, batched_zip
from libracecar.numpyro_utils import (
    mixturesamefamily_,
    normal_,
    numpyro_param,
    numpyro_scope_fn,
    prng_key_,
    trunc_normal_,
    vmap_seperate_seed,
)
from libracecar.plot import plot_ctx, plot_point, plot_style, plotable, plotmethod
from libracecar.specs import position
from libracecar.utils import (
    bval,
    debug_print,
    ensure_not_weak_typed,
    flike,
    fpair,
    fval,
    ival,
    jit,
    pformat_repr,
    round_clip,
    tree_at_,
    tree_select,
)
from libracecar.vector import unitvec, vec

from .map import _trace_ray_res, precomputed_map, trace_ray
from .ros import lidar_obs


class Condition(eqx.Module):
    d_pixels: ArrayLike

    @staticmethod
    def from_traced_ray(ray: _trace_ray_res) -> "Condition":
        d_pixels = lax.select(
            ray.distance_to_nearest < 0.2,
            on_true=ray.dist,
            on_false=lax.stop_gradient(ray.dist),
        )
        d_pixels = lax.select(
            d_pixels < 1.0, on_true=lax.stop_gradient(d_pixels), on_false=d_pixels
        )
        return Condition(d_pixels)


class EmpiricalRayModel_one(eqx.Module):
    """a ray distribution conditioned on ray traced distance"""

    counts: Int[Array, "n"]

    def probs(self, add_uniform: flike = 0.0) -> fval:
        counts = self.counts
        n = len(counts)

        return (counts / jnp.sum(counts)) * (1 - add_uniform) + (
            jnp.ones(n) / n
        ) * add_uniform

    def _as_dist(self) -> dist.Distribution:
        counts = self.counts
        n = len(counts)

        d = (counts / jnp.sum(counts)) * 0.9 + (jnp.ones(n) / n) * 0.1
        return dist.CategoricalProbs(probs=d)

    def _round_obs(self, obs: fval):
        return round_clip(obs / 1.0, 0, len(self.counts))

    def log_prob(self, obs: fval) -> fval:
        obs_ = self._round_obs(obs)

        cached: Array = jax.vmap(self._as_dist().log_prob)(jnp.arange(len(self.counts)))
        return cached.at[obs_].get(mode="promise_in_bounds")


class EmpiricalRayModel(eqx.Module):

    parts: batched[EmpiricalRayModel_one]

    __repr__ = pformat_repr

    def _round_cond(self, c: Condition):
        return round_clip(c.d_pixels / 1.0, 0, len(self.parts))

    def log_prob(self, c: Condition, obs: fval) -> fval:
        cond = self._round_cond(c)
        # print(jax.make_jaxpr(lambda: self.parts.map(lambda p: p.log_prob(obs)))())

        cached = self.parts.map(lambda p: p.log_prob(obs))
        # cached = jax.pure_callback(
        #     lambda v: (print("computed cached values!") or v), cached, cached
        # )
        # cached = jax.pure_callback(lambda v: v, cached, cached)
        return cached.uf.at[cond].get(mode="promise_in_bounds")

    @staticmethod
    def empty(cond_bins: int, measured_bins: int):
        return EmpiricalRayModel(
            batched.create(
                EmpiricalRayModel_one(jnp.zeros(measured_bins, dtype=jnp.int32)),
            ).repeat(cond_bins)
        )

    def push_counts(self, added: batched[tuple[Condition, fval]]):
        clipped = added.tuple_map(
            lambda c, obs: (
                self._round_cond(c),
                self.parts.static_map(lambda p: p._round_obs(obs)),
            )
        )
        cs, obss = clipped.uf
        self = tree_at_(
            lambda me: me.parts.uf.counts,
            self,
            replace_fn=lambda x: x.at[cs, obss].add(1),
        )
        return self


def log_likelyhood(
    model: EmpiricalRayModel,
    map: precomputed_map,
    pos: position,
    observations: batched[lidar_obs],
    n_traces: int = 30,
) -> fval:

    n_traces = 30
    part_len = len(observations) // 30

    def handle_batch(obs: batched[lidar_obs]):
        ang = obs.map(lambda x: x.angle.to_angle()).mean().unwrap()
        ray_ang = pos.rot.mul_unit(unitvec.from_angle(ang))

        ray = trace_ray(map, pos.tran, ray_ang)

        log_probs = obs.map(
            lambda x:
            # print(
            #     "model.log_prob",
            #     jax.make_jaxpr(model.log_prob)(Condition.from_traced_ray(ray), x.dist),
            # )
            # or
            model.log_prob(Condition.from_traced_ray(ray), x.dist)
        ).unflatten()

        return jnp.mean(log_probs)

    ans = jax.vmap(handle_batch)(
        observations[: n_traces * part_len].reshape(n_traces, part_len)
    )
    return jnp.mean(ans)

    # def sample_one():
    #     idx = random.randint(prng_key_(), (), minval=2, maxval=len(observations) - 2)

    #     ray_ang = pos.rot.mul_unit(observations[idx].unwrap().angle)
    #     ray = trace_ray(map, pos.tran, ray_ang)
    #     ray_dist = ray_model(ray, map.res)

    #     near = observations.dynamic_slice((idx - 2,), (5,))
    #     log_probs = jnp.mean(
    #         near.map(
    #             lambda x: ray_dist.log_prob(jnp.clip(x.dist * map.res, 0.0, d_max))
    #         ).unflatten()
    #     )
    #     return log_probs

    # ans = jnp.mean(vmap_seperate_seed(sample_one, axis_size=n_traces)())

    # return ans * len(observations)
