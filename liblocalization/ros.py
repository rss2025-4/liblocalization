from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from geometry_msgs.msg import Twist
from jax.typing import ArrayLike
from sensor_msgs.msg import LaserScan

from libracecar.batched import batched
from libracecar.jax_utils import divide_x_at_zero
from libracecar.specs import lazy, position
from libracecar.utils import (
    flike,
    fval,
    lazy,
    pformat_repr,
    safe_select,
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

    __repr__ = pformat_repr

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

    def to_position(self) -> position:
        # ang = exp( (angular * t) * i) * linear
        # ang_integal = exp( (angular * t) * i) / (angular * i) * linear
        # ang_integal(0) = 1 / (angular * i) * linear
        # ang_integal(T) = rot / (angular * i) linear

        def n(angular: fval):
            rot = unitvec.from_angle(angular * self.time)
            return rot - unitvec.one

        def d(angular: fval):
            return angular * unitvec.i

        def nonzero_case():
            return n(self.angular) / d(self.angular)

        def zero_case():
            return divide_x_at_zero(n)(self.angular) / divide_x_at_zero(d)(self.angular)

        ans = safe_select(
            jnp.abs(self.angular) <= 1e-5,
            on_false=nonzero_case,
            on_true=zero_case,
        )

        return position(ans * self.linear, unitvec.from_angle(self.angular * self.time))

    def transform(self, transform: position) -> twist_t:

        def inner(t: fval):
            transform_self = twist_t(self.linear, self.angular, t).to_position()
            transform_other = transform.invert_pose() + transform_self + transform
            return transform_other.tran

        _primals_out, tangents_out = jax.jvp(inner, (0.0,), (1.0,))
        assert isinstance(tangents_out, vec)

        return twist_t(tangents_out, self.angular, self.time)
