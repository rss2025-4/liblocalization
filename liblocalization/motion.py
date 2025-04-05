import math

import jax.numpy as jnp
import numpyro

from libracecar.jax_utils import divide_x_at_zero
from libracecar.numpyro_utils import (
    normal_,
    trunc_normal_,
)
from libracecar.specs import position
from libracecar.utils import (
    fval,
    safe_select,
)
from libracecar.vector import unitvec, vec

from .ros import twist_t


def deterministic_position(twist: twist_t) -> position:
    assert isinstance(twist.linear, vec)

    # ang = exp( (angular * t) * i) * linear
    # ang_integal = exp( (angular * t) * i) / (angular * i) * linear
    # ang_integal(0) = 1 / (angular * i) * linear
    # ang_integal(T) = rot / (angular * i) linear

    def n(angular: fval):
        rot = unitvec.from_angle(angular * twist.time)
        return rot - unitvec.one

    def d(angular: fval):
        return angular * unitvec.i

    def nonzero_case():
        return n(twist.angular) / d(twist.angular)

    def zero_case():
        return divide_x_at_zero(n)(twist.angular) / divide_x_at_zero(d)(twist.angular)

    ans = safe_select(
        jnp.abs(twist.angular) <= 1e-5,
        on_false=nonzero_case,
        on_true=zero_case,
    )

    return position(ans * twist.linear, unitvec.from_angle(twist.angular * twist.time))


def motion_model(twist: twist_t) -> position:
    """contain motion model parameters (see source)"""

    with numpyro.handlers.scope(prefix="motion"):

        linear = twist.linear * numpyro.sample(
            "linear_noise", trunc_normal_(1.0, 0.1, 0.7, 1.3)
        )

        angle_stdev = 10.0 / 360 * 2 * math.pi

        angular = twist.angular + numpyro.sample(
            "angular_noise",
            trunc_normal_(0.0, angle_stdev, -3 * angle_stdev, 3 * angle_stdev),
        )

        return deterministic_position(twist_t(linear, angular, twist.time))


def dummy_motion_model(time: fval, res: float, factor: float = 1.0) -> position:
    with numpyro.handlers.scope(prefix="motion"):
        linear_x = numpyro.sample("linear_x", normal_(0.0, 3.0 / res))
        linear_y = numpyro.sample("linear_y", normal_(0.0, 0.5 / res))
        angle_stdev = 45.0 / 360 * 2 * math.pi
        angular = numpyro.sample("angular_noise", normal_(0.0, angle_stdev))

        return deterministic_position(
            twist_t(
                vec.create(linear_x * factor, linear_y * factor),
                jnp.array(angular) * factor,
                time,
            )
        )
