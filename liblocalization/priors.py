import math
import time
from queue import Queue
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import optax
from geometry_msgs.msg import Twist
from jax import lax, random
from jaxtyping import Array, Float
from numpyro.distributions import constraints
from termcolor import colored

from libracecar.batched import batched
from libracecar.jax_utils import divide_x_at_zero
from libracecar.numpyro_utils import (
    batched_dist,
    batched_vmap_with_rng,
    numpyro_param,
    trunc_normal_,
    vmap_seperate_seed,
)
from libracecar.plot import plot_ctx, plot_style, plotable
from libracecar.specs import position
from libracecar.utils import (
    cast_unchecked,
    cast_unchecked_,
    cond_,
    debug_print,
    ensure_not_weak_typed,
    flike,
    fval,
    io_callback_,
    jit,
    lazy,
    pformat_repr,
    safe_select,
    timer,
    tree_at_,
    tree_to_ShapeDtypeStruct,
)
from libracecar.vector import unitvec, vec

from .api import LocalizationBase, localization_params
from .map import _trace_ray_res, precomputed_map, trace_ray
from .motion import deterministic_position, twist_t
from .ros import lidar_obs
from .stats import stats_t


class gaussian_prior(eqx.Module):
    mean: Float[Array, "3"]
    scale_tril: Float[Array, "3 3"]

    def covariance(self):
        return self.scale_tril @ self.scale_tril.T

    @staticmethod
    def from_mean_cov(
        mean: Float[Array, "3"], cov: Float[Array, "3 3"]
    ) -> "gaussian_prior":
        cov_ = jnp.linalg.cholesky(cov)
        return gaussian_prior(mean, cov_)

    def apply_twist(self, twist: twist_t):
        twist_p = deterministic_position(twist)

        def inner(pos_arr: Float[Array, "3"]):
            ans = position.from_arr(pos_arr) + twist_p
            ans_arr = ans.as_arr()
            return ans_arr, ans_arr

        jac, ans_arr = jax.jacfwd(inner, has_aux=True)(self.mean)
        return gaussian_prior.from_mean_cov(ans_arr, jac @ self.covariance() @ jac.T)

    def sample(self) -> position:
        d = dist.MultivariateNormal(
            cast_unchecked_(self.mean), scale_tril=self.scale_tril
        )
        ans = numpyro.sample("noisy_position", d)
        return position.from_arr(ans)

    def sample_batch(self, n_points: int) -> batched[position]:
        return vmap_seperate_seed(
            lambda: batched.create(self.sample()),
            axis_size=n_points,
        )()

    def plot(self, n_points: int, style: plot_style = plot_style()) -> plotable:
        return self.sample_batch(n_points).map(lambda x: x.plot_as_seg(style))


class particles(eqx.Module):
    # (pose, weight)
    points: batched[tuple[position, fval]]

    def sample(self) -> position:
        pos, probs = self.points.split_tuple()
        d = dist.Categorical(probs=probs.unflatten())
        assert isinstance(d, dist.Distribution)
        ans = numpyro.sample("draw_particle", d)
        return pos[ans].unwrap()

    @staticmethod
    def from_samples(s: batched[position]):
        (n,) = s.batch_dims()
        return particles(s.map(lambda x: (x, jnp.array(1 / n))))

    def resample(self, count: int) -> "particles":
        ans = vmap_seperate_seed(
            lambda: batched.create(self.sample()),
            axis_size=count,
        )()
        return particles.from_samples(ans)

    def map(self, fn: Callable[[position], position]) -> "particles":
        ans = batched_vmap_with_rng(lambda x: (fn(x[0]), x[1]), self.points)
        return particles(ans)

    # def max_likelyhood(self) -> position:
    #     pos, probs = self.points.split_tuple()
    #     best = jnp.argmax(probs.unflatten())
    #     return pos[best].unwrap()

    def mean(self) -> position:
        ans = self.points.tuple_map(lambda p, w: p.as_arr() * w).mean()
        return position.from_arr(ans.unwrap())

    def plot(self, n_points: int, style: plot_style = plot_style()) -> plotable:
        return self.resample(n_points).points.tuple_map(
            lambda x, _: x.plot_as_seg(style)
        )
