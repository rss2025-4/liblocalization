import math
from typing import Callable, final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import optax
from jax import lax, random
from jax import tree_util as jtu
from jaxtyping import Array, ArrayLike, Float
from numpyro.distributions import constraints
from sensor_msgs.msg import LaserScan
from termcolor import colored

from libracecar.batched import batched
from libracecar.numpyro_utils import batched_dist, numpyro_param, vmap_seperate_seed
from libracecar.plot import plot_ctx, plot_style, plotable
from libracecar.specs import position
from libracecar.utils import (
    cast,
    cast_unchecked_,
    cond_,
    debug_callback,
    debug_print,
    flike,
    fval,
    jit,
    pformat_repr,
    tree_at_,
)
from libracecar.vector import unitvec

from .map import _trace_ray_res, precomputed_map, trace_ray


class lidar_obs(eqx.Module):
    angle: unitvec
    dist: fval

    __repr__ = pformat_repr

    @staticmethod
    def _create(
        am: flike, ai: flike, ranges: ArrayLike, res: flike
    ) -> batched["lidar_obs"]:
        ranges = jnp.array(ranges) / res
        return batched.create(ranges, (len(ranges),)).enumerate(
            lambda x, i: lidar_obs(unitvec.from_angle(am + jnp.array(ai) * i), x)
        )

    @staticmethod
    def from_msg_meters(
        msg: LaserScan, res: float
    ) -> Callable[[], batched["lidar_obs"]]:
        return jtu.Partial(
            lidar_obs._create,
            am=float(msg.angle_min),
            ai=float(msg.angle_increment),
            ranges=np.array(msg.ranges),
            res=res,
        )
