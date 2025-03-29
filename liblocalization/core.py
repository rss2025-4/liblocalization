import math

import equinox as eqx
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import optax
from jax import lax, random
from jaxtyping import Array, Float
from numpyro.distributions import constraints
from sensor_msgs.msg import LaserScan

from libracecar.batched import batched
from libracecar.numpyro_utils import batched_dist, numpyro_param, vmap_seperate_seed
from libracecar.plot import plot_ctx, plot_style, plotable
from libracecar.specs import position
from libracecar.utils import (
    cast_unchecked_,
    debug_print,
    flike,
    fval,
    jit,
    pformat_repr,
    tree_at_,
)
from libracecar.vector import unitvec

from .map import _trace_ray_res, precomputed_map, trace_ray

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
    a_arr: Float[Array, "3"]
    sigma: fval

    __repr__ = pformat_repr


class noisy_position(eqx.Module):
    mean: Float[Array, "3"]
    scale_tril: Float[Array, "3 3"]

    a_arr_mean: Float[Array, "2"]
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
        a_arr_mean = jnp.maximum(a_arr_mean, 0.07)
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


class lidar_obs(eqx.Module):
    angle: unitvec
    dist: fval

    __repr__ = pformat_repr

    @staticmethod
    def from_msg_meters(msg: LaserScan, res: float) -> batched["lidar_obs"]:
        am = msg.angle_min
        ai = msg.angle_increment
        range = jnp.array(msg.ranges) / res

        return batched.create(range, (len(range),)).enumerate(
            lambda x, i: lidar_obs(unitvec.from_angle(am + ai * i), x)
        )


class _compute_posterior_ret(eqx.Module):
    posterior: noisy_position
    losses: Array

    __repr__ = pformat_repr


def sensor_model(
    ray: _trace_ray_res, hp: hyper_params, res: float
) -> dist.Distribution:
    ray_dist = lax.select(
        ray.distance_to_nearest < 0.2,
        on_true=ray.dist,
        on_false=lax.stop_gradient(ray.dist),
    )
    # ray_dist = lax.stop_gradient(ray.dist)

    d_max = 10.0 / res
    ray_dist = jnp.minimum(ray_dist, d_max)
    parts: list[tuple[dist.Distribution, flike]] = [
        (
            dist.Normal(
                loc=cast_unchecked_(ray_dist),
                scale=cast_unchecked_(10.0),
            ),
            hp.a_arr[0],
        ),
        (
            dist.Normal(
                loc=cast_unchecked_(ray_dist),
                scale=cast_unchecked_(50.0),
            ),
            hp.a_arr[1],
        ),
        (
            dist.Normal(
                loc=cast_unchecked_(ray_dist),
                scale=cast_unchecked_(hp.sigma),
            ),
            hp.a_arr[2],
        ),
        (dist.Uniform(0.0, d_max + 2), hp.a_arr[3]),
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
                "a_arr_mean", prior.a_arr_mean, constraints.greater_than(0.05)
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

            return sensor_model(ans, hp, map.grid.res), obs.dist

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
    map: precomputed_map
    state: noisy_position
    rng: Array

    __repr__ = pformat_repr

    @staticmethod
    def init(map: precomputed_map, start: position):
        res = map.grid.res
        prior = noisy_position(
            mean=jnp.array(0),
            scale_tril=jnp.array(0),
            a_arr_mean=jnp.array([0.25, 0.25, 0.25, 0.25]),
            sigma_mean=jnp.array(10.0),
        )
        prior = prior.with_mean_cov(
            start.as_arr(),
            jnp.diag(jnp.array([0.2 / res, 0.2 / res, math.pi / 8]) ** 2),
        )
        return localization_state(map, prior, random.PRNGKey(0))


class localization_params(eqx.Module):
    state: localization_state
    obs: batched[lidar_obs]

    __repr__ = pformat_repr


class localization_ret(eqx.Module):
    state: localization_state
    plot: plot_ctx

    __repr__ = pformat_repr


@jit
def compute_localization(args: localization_params) -> localization_ret:
    new_key, key = random.split(args.state.rng, 2)

    with numpyro.handlers.seed(rng_seed=key):
        ans = _compute_localization(args)

    return tree_at_(lambda me: me.state.rng, ans, new_key)


def _compute_localization(args: localization_params) -> localization_ret:

    ctx = plot_ctx.create(100)

    map = args.state.map
    res = map.grid.res

    prior = args.state.state

    prior = prior.with_mean_cov(
        prior.mean,
        prior.covariance()
        + jnp.diag(jnp.array([0.05 / res, 0.05 / res, math.pi / 32]) ** 2),
    )

    ctx += prior.plot(20, plot_style(color=(1, 0, 0)))
    post = compute_posterior(map, prior, args.obs)
    debug_print("losses", post.losses)
    ctx += post.posterior.plot(20, plot_style(color=(0, 1, 0)))

    ctx = map.grid.plot_from_pixels_vec(ctx)

    new_state = tree_at_(
        lambda me: me.state,
        args.state,
        post.posterior.normalize(),
    )
    return localization_ret(
        state=new_state,
        plot=ctx,
    )
