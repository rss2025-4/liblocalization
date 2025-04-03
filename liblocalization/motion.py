import math
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro
from geometry_msgs.msg import Twist

from liblocalization.api import LocalizationBase, localization_params
from libracecar.numpyro_utils import (
    trunc_normal_,
)
from libracecar.plot import plot_ctx
from libracecar.specs import position
from libracecar.utils import (
    debug_print,
    fval,
    jit,
    lazy,
    safe_select,
)
from libracecar.vector import unitvec, vec


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
    def from_ros(msg: Twist, time: float) -> lazy["twist_t"]:
        assert msg.linear.z == 0.0

        assert msg.angular.x == 0.0
        assert msg.angular.y == 0.0

        return lazy(
            twist_t._create,
            linear_x=float(msg.linear.x),
            linear_y=float(msg.linear.y),
            angular=float(msg.angular.z),
            time=time,
        )

    def deterministic_position(self) -> position:
        assert isinstance(self.linear, vec)

        # ang = exp( (angular * t) * i) * linear
        # ang_integal = exp( (angular * t) * i) / (angular * i) * linear
        # ang_integal(0) = 1 / (angular * i) * linear
        # ang_integal(T) = rot / (angular * i) linear

        def inner(angular: fval):
            rot = unitvec.from_angle(angular * self.time)
            return (rot - unitvec.one), (angular * unitvec.i)

        def nonzero_case():
            debug_print("nonzero_case")
            n, d = inner(self.angular)
            return n / d

        def zero_case():
            _primals_out, (n, d) = jax.jvp(
                inner, primals=(self.angular,), tangents=(1.0,)
            )
            debug_print("zero_case", _primals_out, n, d)
            return n / d

        ans = safe_select(
            jnp.abs(self.angular) <= 1e-5,
            on_false=nonzero_case,
            on_true=zero_case,
        )

        return position(ans * self.linear, unitvec.from_angle(self.angular * self.time))


from ._api import Controller


class _deterministic_motion_tracker(Controller):
    def __init__(self, cfg: localization_params):
        print("deterministic_motion_tracker!")
        super().__init__(cfg)
        self.pos = position.zero()

    def _get_pose(self):
        return self.pos

    def _set_pose(self, pose) -> None:
        self.pos = pose()

    def _twist(self, twist):
        self.pos, ctx = _deterministic_motion_tracker._step(self.pos, twist)
        self._visualize(ctx)

    @staticmethod
    @jit
    def _step(pos: position, twist: lazy[twist_t]):
        ctx = plot_ctx.create(100)
        ans = pos + twist().deterministic_position()
        ctx += ans.plot_as_seg()
        return ans, ctx


def deterministic_motion_tracker(cfg: localization_params) -> LocalizationBase:
    """localization implementation that only uses motion model"""
    return _deterministic_motion_tracker(cfg)


def motion_model(twist: twist_t):

    with numpyro.handlers.scope(prefix="motion"):

        linear = twist.linear * numpyro.sample(
            "linear_noise", trunc_normal_(1.0, 0.1, 0.7, 1.3)
        )

        angle_stdev = 10.0 / 360 * 2 * math.pi

        angular = twist.angular + numpyro.sample(
            "angular_noise",
            trunc_normal_(0.0, angle_stdev, -3 * angle_stdev, 3 * angle_stdev),
        )

        # exp(ax) ==> exp(ax) * a

        # v = rot ** (t / T) * linear
        # integral(t) = T / ln(rot) * ln(rot) ** (t / T)
        # integral(T) = T / ln(rot) * ln(rot) = T

        # ang: 0 .. angular * time
        # v: (cos(ang), sin(ang)) * linear

        x_integral = lambda x: jnp.sin(x)
        x_disp = (x_integral(angular * time) - x_integral(0.0)) * linear

        y_integral = lambda x: -jnp.cos(x)
        y_disp = (y_integral(angular * time) - y_integral(0.0)) * linear
