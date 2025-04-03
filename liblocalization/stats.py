import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Int32

from libracecar.batched import batched
from libracecar.numpyro_utils import (
    batched_vmap_with_rng,
    normal_,
    prng_key_,
    vmap_seperate_seed,
)
from libracecar.specs import position
from libracecar.utils import (
    pformat_repr,
    round_clip,
    tree_at_,
)
from libracecar.vector import unitvec, vec

from .map import precomputed_map, trace_ray
from .ros import lidar_obs


class stats_t(eqx.Module):
    # (truth, measured) -> count
    max_d: int = eqx.field(static=True)
    counts_using_truth: Int32[Array, "max_d max_d"]

    __repr__ = pformat_repr

    @staticmethod
    def create(max_d: int):
        return stats_t(max_d, jnp.zeros((max_d, max_d)))

    def update(
        self, map: precomputed_map, true_pos_pixels: position, lidar: batched[lidar_obs]
    ) -> "stats_t":

        def handle_one(obs: lidar_obs):
            res = trace_ray(
                map,
                # 5cm of noise
                vec.create(
                    normal_(true_pos_pixels.tran.x, 0.05 / map.res).sample(prng_key_()),
                    normal_(true_pos_pixels.tran.y, 0.05 / map.res).sample(prng_key_()),
                ),
                true_pos_pixels.rot.mul_unit(obs.angle).mul_unit(
                    unitvec.from_angle(
                        # 5 degrees of noise
                        normal_(0.0, 5.0 / 360 * 2 * math.pi).sample(prng_key_()),
                    )
                ),
            )
            return round_clip(obs.dist, 0, self.max_d), round_clip(
                res.dist, 0, self.max_d
            )

        truth, measured = (
            vmap_seperate_seed(lambda: batched_vmap_with_rng(handle_one, lidar), 8)()
            .reshape(-1)
            .unflatten()
        )

        self = tree_at_(
            lambda me: me.counts_using_truth,
            self,
            self.counts_using_truth.at[truth, measured].add(1),
        )

        return self

    def counts_using_truth_normalized(self):
        def inner(x: Array):
            return x / jnp.sum(x)

        return jax.vmap(inner)(self.counts_using_truth)
