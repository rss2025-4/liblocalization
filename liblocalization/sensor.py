import equinox as eqx
import jax.numpy as jnp
import numpyro.distributions as dist
from jax import lax
from jaxtyping import Array, Float
from numpyro.distributions import constraints

from libracecar.numpyro_utils import (
    trunc_normal_,
)
from libracecar.specs import position
from libracecar.utils import (
    flike,
    fval,
    pformat_repr,
)

from .map import _trace_ray_res, precomputed_map
from .ros import lidar_obs


class hyper_params(eqx.Module):
    a_arr: Float[Array, "a_arr"]
    sigma: fval

    __repr__ = pformat_repr


def ray_model(
    ray: _trace_ray_res,
    hp: hyper_params,
    d_max: float,
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


def model(
    map: precomputed_map,
    pos: position,
    hp: hyper_params,
    obs: lidar_obs,
):
    pass


def log_density(
    map: precomputed_map,
    pos: position,
    hp: hyper_params,
    obs: lidar_obs,
    num_particles: int = 10,
):
    pass
