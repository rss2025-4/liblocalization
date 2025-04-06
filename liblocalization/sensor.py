import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax import lax, random
from numpyro.distributions import constraints

from libracecar.batched import batched
from libracecar.numpyro_utils import (
    normal_,
    prng_key_,
    trunc_normal_,
    vmap_seperate_seed,
)
from libracecar.specs import position
from libracecar.utils import (
    flike,
    fval,
)
from libracecar.vector import unitvec

from .map import _trace_ray_res, precomputed_map, trace_ray
from .ros import lidar_obs

d_max: float = 10.0


def ray_model(ray: _trace_ray_res, res: float) -> dist.Distribution:
    """
    contain sensor model parameters (see source).

    this functions is in METERS
    """
    # d_pixels = ray.dist
    d_pixels = lax.select(
        ray.distance_to_nearest < 0.2,
        on_true=ray.dist,
        on_false=lax.stop_gradient(ray.dist),
    )
    # d_pixels = lax.stop_gradient(ray.dist)

    d_pixels = lax.select(
        d_pixels < 1.0, on_true=lax.stop_gradient(d_pixels), on_false=d_pixels
    )

    d = d_pixels * res
    d = jnp.clip(d, 0.0, d_max)

    # sensor is modeled as a mixture of theses distributions
    parts: list[tuple[dist.Distribution, flike]] = [
        (trunc_normal_(d, 0.2, 0.0, d_max), 1.0),
        (
            dist.Delta(d_max),
            0.01 + d / d_max + jnp.maximum(0.0, ((d / d_max) - 0.9) * 20),
        ),
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
        ray_dist = ray_model(ray, map.res)
        log_probs = jnp.sum(
            obs.map(
                lambda x: ray_dist.log_prob(jnp.clip(x.dist * map.res, 0.0, d_max))
            ).unflatten()
        )
        return log_probs

    ans = jax.vmap(handle_batch)(
        observations[: n_traces * part_len].reshape(n_traces, part_len)
    )
    return jnp.sum(ans)

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
