import math
import time
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from geometry_msgs.msg import Twist
from jax import tree_util as jtu
from jaxtyping import ArrayLike
from sensor_msgs.msg import LaserScan

from libracecar.batched import batched
from libracecar.jax_utils import divide_x_at_zero
from libracecar.numpyro_utils import (
    trunc_normal_,
)
from libracecar.plot import plot_ctx, plotable
from libracecar.specs import lazy, position
from libracecar.utils import (
    debug_print,
    flike,
    fval,
    jit,
    lazy,
    pformat_repr,
    safe_select,
    timer,
)
from libracecar.vector import unitvec, vec


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
    def from_ros(msg: LaserScan, res: float) -> lazy[batched["lidar_obs"]]:
        return lazy(
            lidar_obs._create,
            am=float(msg.angle_min),
            ai=float(msg.angle_increment),
            ranges=np.array(msg.ranges),
            res=res,
        )


class twist_t(eqx.Module):
    # pixel / s
    linear: vec
    # rad / s
    angular: fval
    # s
    time: fval

    @staticmethod
    def _create(linear_x, linear_y, angular, time):
        return twist_t(vec.create(linear_x, linear_y), angular, time)

    @staticmethod
    def zero():
        return twist_t(vec.create(0, 0), jnp.array(0.0), jnp.array(0.0))

    @staticmethod
    def from_ros(msg: Twist, time: float, res: float) -> lazy["twist_t"]:
        assert msg.linear.z == 0.0

        assert msg.angular.x == 0.0
        assert msg.angular.y == 0.0

        return lazy(
            twist_t._create,
            linear_x=float(msg.linear.x) / res,
            linear_y=float(msg.linear.y) / res,
            angular=float(msg.angular.z),
            time=time,
        )
