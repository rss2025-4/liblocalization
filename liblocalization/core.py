import math
import queue
import time
from dataclasses import dataclass
from functools import partial
from queue import Queue
from typing import Callable, final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import optax
from jax import lax, random
from jax import tree_util as jtu
from jaxtyping import Array, ArrayLike, Float
from numpyro.distributions import TruncatedDistribution, constraints
from sensor_msgs.msg import LaserScan
from termcolor import colored

from libracecar.batched import batched
from libracecar.numpyro_utils import (
    batched_dist,
    jit_with_seed,
    normal_,
    numpyro_param,
    trunc_normal_,
    vmap_seperate_seed,
)
from libracecar.plot import plot_ctx, plot_style, plotable
from libracecar.specs import position
from libracecar.utils import (
    cast,
    cast_unchecked,
    cast_unchecked_,
    cond_,
    debug_callback,
    debug_print,
    ensure_not_weak_typed,
    flike,
    fval,
    io_callback_,
    jit,
    pformat_repr,
    tree_at_,
    tree_to_ShapeDtypeStruct,
)
from libracecar.vector import unitvec

from .map import _trace_ray_res, precomputed_map, trace_ray
from .ros import lidar_obs
from .stats import stats_t

a_arr_min = 0.02
# def _vector3_to_np(v: Vector3):
#     return np.array([v.x, v.y, v.z])


# class odom_jax(eqx.Module):
#     linear: Float[ArrayLike, "3"]
#     angular: Float[ArrayLike, "3"]
#     cov: Float[ArrayLike, "6 6"]

#     @staticmethod
#     def create_np(odom: Odometry):
#         twist = odom.twist
#         return odom_jax(
#             linear=_vector3_to_np(twist.twist.linear),
#             angular=_vector3_to_np(twist.twist.angular),
#             cov=twist.covariance,
#         )


# class position_update(eqx.Module):
#     mean: Float[Array, "3"]
#     cov: Float[Array, "3 3"]
#     elapsed: fval

#     @staticmethod
#     def from_odom(odom: odom_jax, elapsed: flike):
#         x, y, _ = jnp.array(odom.linear)
#         _, _, a = jnp.array(odom.angular)
#         cov = jnp.array(odom.cov)[(0, 1, 5), :][:, (0, 1, 5)]
#         return position_update(jnp.array([x, y, a]), cov, jnp.array(elapsed))

#     def sample(self, rng_key: Array | None = None) -> position:
#         d = dist.MultivariateNormal(
#             cast_unchecked_(self.mean), covariance_matrix=self.cov
#         )
#         ans = numpyro.sample("position_update", d, rng_key=rng_key)
#         x, y, a = jnp.array(ans * self.elapsed)
#         return position(jnp.array([x, y]), a)


class hyper_params(eqx.Module):
    a_arr: Float[Array, "a_arr"]
    sigma: fval

    __repr__ = pformat_repr


class noisy_position(eqx.Module):
    mean: Float[Array, "3"]
    scale_tril: Float[Array, "3 3"]

    a_arr_mean: Float[Array, "a_arr"]
    sigma_mean: fval

    def covariance(self):
        return self.scale_tril @ self.scale_tril.T

    def with_mean_cov(
        self, mean: Float[Array, "3"], cov: Float[Array, "3 3"]
    ) -> "noisy_position":
        cov_ = jnp.linalg.cholesky(cov)
        return tree_at_(lambda me: (me.mean, me.scale_tril), self, (mean, cov_))

    def truncated_normal(
        self, name: str, mean: flike, stdev: flike, low: flike
    ) -> fval:
        d = dist.TruncatedNormal(
            loc=cast_unchecked_(mean), scale=cast_unchecked_(stdev), low=low
        )
        ans = numpyro.sample(name, cast_unchecked_(d))
        return jnp.array(ans)

    def sample(self) -> tuple[position, hyper_params]:
        d = dist.MultivariateNormal(
            cast_unchecked_(self.mean), scale_tril=self.scale_tril
        )
        ans = numpyro.sample("noisy_position", d)
        x, y, a = jnp.array(ans)
        ans_pos = position.create((x, y), a)

        hp = hyper_params(
            a_arr=self.truncated_normal(
                "a_arr", self.a_arr_mean, jnp.ones_like(self.a_arr_mean) * 0.1, 0.1
            ),
            sigma=self.truncated_normal("sigma", self.sigma_mean, 5.0, 5.0),
        )

        return ans_pos, hp

    def normalize(self) -> "noisy_position":
        a_arr_mean = self.a_arr_mean / jnp.sum(self.a_arr_mean)
        a_arr_mean = jnp.maximum(a_arr_mean, a_arr_min)
        sigma_mean = jnp.minimum(self.sigma_mean, 40.0)
        sigma_mean = jnp.maximum(self.sigma_mean, 6.0)
        return tree_at_(
            lambda me: (me.a_arr_mean, me.sigma_mean), self, (a_arr_mean, sigma_mean)
        )

    def plot(self, n_points: int, style: plot_style = plot_style()) -> plotable:
        return vmap_seperate_seed(
            lambda: batched.create(self.sample()[0].plot_as_seg(style)),
            axis_size=n_points,
        )()


class _compute_posterior_ret(eqx.Module):
    posterior: noisy_position
    losses: Array

    __repr__ = pformat_repr


def sensor_model(
    ray: _trace_ray_res, hp: hyper_params, d_max: float
) -> dist.Distribution:
    ray_dist = lax.select(
        ray.distance_to_nearest < 0.2,
        on_true=ray.dist,
        on_false=lax.stop_gradient(ray.dist),
    )
    # ray_dist = lax.stop_gradient(ray.dist)

    ray_dist = jnp.minimum(ray_dist, d_max)

    parts: list[tuple[dist.Distribution, flike]] = [
        (trunc_normal_(ray_dist, 10.0, 0.0, d_max), hp.a_arr[0]),
        (trunc_normal_(ray_dist, 50.0, 0.0, d_max), hp.a_arr[1]),
        (trunc_normal_(ray_dist, hp.sigma, 0.0, d_max), hp.a_arr[2]),
        (dist.Uniform(0.0, d_max), hp.a_arr[3]),
    ]
    probs = jnp.array([p for _, p in parts])
    return dist.MixtureGeneral(
        dist.Categorical(probs=probs / jnp.sum(probs)),
        [d for d, _ in parts],
        support=constraints.positive,
    )


def compute_posterior(
    map: precomputed_map,
    prior: noisy_position,
    observed: batched[lidar_obs],
) -> _compute_posterior_ret:

    def guide():
        d = noisy_position(
            numpyro_param("mean", prior.mean),
            numpyro_param("scale_tril", prior.scale_tril, constraints.lower_cholesky),
            a_arr_mean=numpyro_param(
                "a_arr_mean",
                jnp.maximum(prior.a_arr_mean, a_arr_min + 0.01),
                constraints.greater_than(a_arr_min),
            ),
            sigma_mean=numpyro_param(
                "sigma_mean", prior.sigma_mean, constraints.greater_than(5.0)
            ),
        )
        _ = d.sample()

    def model():
        pos, hp = prior.sample()

        def handle_one(obs: lidar_obs):
            ray_ang = pos.rot.mul_unit(obs.angle)
            ans = trace_ray(map, pos.tran, ray_ang)

            d_max = 10.0 / map.grid.res
            return sensor_model(ans, hp, d_max), jnp.minimum(obs.dist, d_max)

        with numpyro.plate("lidar", len(observed), subsample_size=16) as ind:
            db, ob = observed[ind].map(handle_one).split_tuple()
            _ = numpyro.sample("_obs", batched_dist(db), obs=ob.unflatten())

    optimizer = optax.adamw(learning_rate=0.05)
    svi = numpyro.infer.SVI(
        model, guide, optimizer, loss=numpyro.infer.Trace_ELBO(num_particles=128)
    )
    svi_result = svi.run(numpyro.prng_key(), 32, progress_bar=False)

    params = svi_result.params

    return _compute_posterior_ret(
        noisy_position(
            params["mean"],
            params["scale_tril"],
            a_arr_mean=params["a_arr_mean"],
            sigma_mean=params["sigma_mean"],
        ),
        svi_result.losses,
    )


class localization_state(eqx.Module):
    prior: noisy_position
    rng: Array

    stats: stats_t

    __repr__ = pformat_repr

    @staticmethod
    @jit
    def init(map: precomputed_map, start: position):
        debug_print("localization_state.init")
        res = map.grid.res
        prior = noisy_position(
            mean=jnp.array(0),
            scale_tril=jnp.array(0),
            a_arr_mean=jnp.ones(5) / 5,
            sigma_mean=jnp.array(10.0),
        )
        prior = prior.with_mean_cov(
            start.as_arr(),
            jnp.diag(jnp.array([0.2 / res, 0.2 / res, math.pi / 8]) ** 2),
        )
        return localization_state(
            prior, random.PRNGKey(0), stats_t.create(int(10.0 / res))
        ).ensure_not_weak_typed()

    def ensure_not_weak_typed(self):
        return ensure_not_weak_typed(self)


# class localization_params(eqx.Module):
#     state: localization_state
#     obs: Callable[[], batched[lidar_obs]]
#     true_pos_pixels: position | None = None

#     __repr__ = pformat_repr

#     def ensure_not_weak_typed(self):
#         return ensure_not_weak_typed(self)


class request_t(eqx.Module):
    laser: Callable[[], batched[lidar_obs]]
    true_pos_pixels: position | None = None


class response_t(eqx.Module):
    plot: plot_ctx


class localization_ret(eqx.Module):
    state: localization_state
    plot: plot_ctx

    __repr__ = pformat_repr


def compute_localization(
    state: localization_state, req: request_t, map: precomputed_map
) -> localization_ret:
    print("compute_localization: tracing")
    new_key, key = random.split(state.rng, 2)

    with numpyro.handlers.seed(rng_seed=key):
        ans = _compute_localization(state, req, map)

    ans = tree_at_(lambda me: me.state.rng, ans, new_key)
    print("compute_localization: tracing done")
    return ans


def _compute_localization(
    state: localization_state, req: request_t, map: precomputed_map
) -> localization_ret:
    obs = req.laser()
    true_pos = req.true_pos_pixels

    ctx = plot_ctx.create(100)

    res = map.grid.res

    if true_pos is not None:
        is_far = jnp.any(jnp.isnan(state.prior.mean)) | (
            jnp.linalg.norm(state.prior.mean - true_pos.as_arr()) > 100
        )

        def on_too_far():
            debug_print(colored("too far; resetting", "red"))
            ans = localization_state.init(map, true_pos)
            return tree_at_(lambda me: me.prior, state, ans.prior)

        state = cond_(
            is_far,
            true_fun=on_too_far,
            false_fun=lambda: state,
        )

        # update stats
        state = tree_at_(
            lambda me: me.stats, state, state.stats.update(map, true_pos, obs)
        )

        ctx += true_pos.plot_as_seg(plot_style(color=(0, 0, 1)))

        # ctx += obs.map(
        #     lambda o: trace_ray(
        #         map,
        #         true_pos.tran,
        #         true_pos.rot.mul_unit(o.angle),
        #     ).plot(),
        # )

    # TODO: motion model
    prior = state.prior
    prior = prior.with_mean_cov(
        prior.mean,
        prior.covariance()
        + jnp.diag(jnp.array([0.05 / res, 0.05 / res, math.pi / 32]) ** 2),
    )

    ctx += prior.plot(20, plot_style(color=(1, 0, 0)))

    post = compute_posterior(map, prior, obs)
    posterior = post.posterior

    # posterior = prior

    # debug_print("losses", post.losses)
    ctx += posterior.plot(20, plot_style(color=(0, 1, 0)))

    ctx = map.grid.plot_from_pixels_vec(ctx)
    ctx.check()

    new_state = tree_at_(
        lambda me: me.prior,
        state,
        posterior.normalize(),
    )
    return localization_ret(
        state=new_state.ensure_not_weak_typed(),
        plot=ctx,
    )


class main_loop:

    def __init__(self, ex_req: request_t):

        self.req_shapes = tree_to_ShapeDtypeStruct(ex_req)

        self.request_q: Queue[request_t] = Queue()
        self.response_q: Queue[response_t] = Queue()

    def process(self, req: request_t) -> response_t:
        self.request_q.put_nowait(req)
        return self.response_q.get()

    def _request_callback(self) -> request_t:
        return self.request_q.get()

    def _response_callback(self, ctx: response_t) -> None:
        self.response_q.put_nowait(ctx)

    def jit(self, map_ex: precomputed_map, init_pos_pixels_ex: position):
        return self._jit(
            tree_to_ShapeDtypeStruct(map_ex),
            tree_to_ShapeDtypeStruct(init_pos_pixels_ex),
        )

    def _jit(self, map_ex: precomputed_map, init_pos_pixels_ex: position):

        def inner_(map: precomputed_map, init_pos_pixels: position) -> None:

            def loop_fn(state: localization_state) -> localization_state:

                req = io_callback_(self._request_callback, self.req_shapes)()
                assert isinstance(req, request_t)

                ans = compute_localization(state, req, map)

                _ = io_callback_(self._response_callback)(response_t(ans.plot))

                debug_print(
                    "hyperparams:",
                    ans.state.prior.a_arr_mean,
                    ans.state.prior.sigma_mean,
                )
                return ans.state

            lax.while_loop(
                cond_fun=lambda _: True,
                body_fun=loop_fn,
                init_val=localization_state.init(map, init_pos_pixels),
            )

        inner = jit(inner_)
        traced = inner.trace(map_ex, init_pos_pixels_ex)

        # print("traced:")
        # print(traced.jaxpr)

        lowered = traced.lower()

        # print("lowered:")
        # print(lowered.as_text())

        _start = time.time()
        print("compiling:")
        comp = lowered.compile()
        print(f"elapsed (compile): {time.time() - _start:.5f}")
        print("cost_analysis:", comp.cost_analysis())

        # return inner
        return cast_unchecked(inner)(comp)
