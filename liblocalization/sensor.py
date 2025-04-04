import equinox as eqx
import jax.numpy as jnp
import numpyro.distributions as dist
from jax import lax, random
from jaxtyping import Array, Float
from numpyro.distributions import constraints

from libracecar.batched import batched
from libracecar.numpyro_utils import (
    prng_key_,
    trunc_normal_,
    vmap_seperate_seed,
)
from libracecar.specs import position
from libracecar.utils import (
    cast_unchecked_,
    flike,
    fval,
    pformat_repr,
)

from .map import _trace_ray_res, precomputed_map, trace_ray
from .ros import lidar_obs

d_max: float = 10.0


def ray_model(ray: _trace_ray_res, res: float) -> dist.Distribution:
    # this in meters!!!
    d = lax.select(
        ray.distance_to_nearest < 0.2,
        on_true=ray.dist,
        on_false=lax.stop_gradient(ray.dist),
    )
    d *= res
    d = jnp.clip(d, 0.0, d_max)

    parts: list[tuple[dist.Distribution, flike]] = [
        (trunc_normal_(d, 0.2, 0.0, d_max), 1.0),
        (dist.Delta(d_max), d / d_max + jnp.maximum(0.0, ((d / d_max) - 0.9) * 20)),
        (trunc_normal_(d, 2.5, 0.0, d_max), 0.2),
        (dist.Uniform(0.0, d_max), 0.2),
    ]
    probs = jnp.array([p for _, p in parts])
    return dist.MixtureGeneral(
        dist.Categorical(probs=probs / jnp.sum(probs)),
        [d for d, _ in parts],
        support=constraints.nonnegative,
    )


def log_likelyhood(
    map: precomputed_map, pos: position, observations: batched[lidar_obs]
) -> fval:

    def sample_one():
        idx = random.randint(prng_key_(), (), minval=2, maxval=len(observations) - 2)

        ray_ang = pos.rot.mul_unit(observations[idx].unwrap().angle)
        ray = trace_ray(map, pos.tran, ray_ang)
        ray_dist = ray_model(ray, map.res)

        near = observations.dynamic_slice((idx - 2,), (5,))
        log_probs = jnp.sum(
            near.map(
                lambda x: ray_dist.log_prob(jnp.clip(x.dist * map.res, 0.0, d_max))
            ).unflatten()
        )
        return log_probs

    return jnp.sum(vmap_seperate_seed(sample_one, axis_size=30)())
