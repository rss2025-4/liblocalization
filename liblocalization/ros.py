from typing import Callable

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import tree_util as jtu
from jaxtyping import ArrayLike
from sensor_msgs.msg import LaserScan

from libracecar.batched import batched
from libracecar.utils import (
    flike,
    fval,
    pformat_repr,
)
from libracecar.vector import unitvec


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
