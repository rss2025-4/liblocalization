import math
import time
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro
from geometry_msgs.msg import Twist
from jax import Array, random

from libracecar.batched import batched
from libracecar.jax_utils import (
    dispatch,
    dispatch_spec,
    divide_x_at_zero,
    io_callback_,
    jax_jit_dispatcher,
)
from libracecar.numpyro_utils import (
    prng_key_,
    trunc_normal_,
    vmap_seperate_seed,
)
from libracecar.plot import plot_ctx, plot_style, plotable
from libracecar.specs import position
from libracecar.utils import (
    cast,
    cast_unchecked,
    debug_print,
    fval,
    jit,
    lazy,
    safe_select,
    timer,
    tree_at_,
)
from libracecar.vector import unitvec, vec

from .._api import Controller
from ..api import LocalizationBase, localization_params
from ..map import Grid, precompute, precomputed_map, trace_ray
from ..motion import deterministic_position, twist_t
from ..priors import gaussian
from ..ros import lidar_obs


@dataclass
class deterministic_motion_params:
    """localization implementation that only uses motion model"""

    plot_points_limit: int = 1000

    def __call__(self, cfg: localization_params) -> LocalizationBase:
        return _deterministic_motion_tracker(cfg, self)


class state(eqx.Module):
    params: deterministic_motion_params = eqx.field(static=True)

    map: precomputed_map

    pos: position
    noisy_pos: gaussian

    def get_pose(self):
        return self, self.pos

    def set_pose(self, pose: lazy[position]):
        p = pose()
        self = tree_at_(lambda me: me.pos, self, p)
        self = tree_at_(lambda me: me.noisy_pos.mean, self, p.as_arr())
        return self, None

    def twist(self, twist_: lazy[twist_t]):
        twist = twist_()
        # gt = gt_()

        ctx = plot_ctx.create(1000)
        # ctx += gt

        ctx += position.zero().plot_as_seg(plot_style(color=(0.0, 1.0, 0.0)))

        # ctx += self.grid.plot()

        twist_p = deterministic_position(twist)
        self = tree_at_(lambda s: s.pos, self, self.pos + twist_p)
        self = tree_at_(
            lambda s: s.noisy_pos, self, self.noisy_pos.apply_twist(twist_p)
        )

        ctx += self.pos.plot_as_seg()
        # ctx += self.noisy_pos.plot(20)
        # ctx += position.from_arr(self.noisy_pos.mean).plot_as_seg(
        #     plot_style(color=(1.0, 0.0, 0.0))
        # )
        return self, ctx

    def plot_computed_rays(self, obs: batched[lidar_obs]) -> plotable:
        pos = self.get_pose()[1]

        def plot_one():
            x = obs[random.randint(prng_key_(), (), 0, len(obs))].unwrap()
            return trace_ray(self.map, pos.tran, pos.rot.mul_unit(x.angle)).plot(x.dist)

        return vmap_seperate_seed(plot_one, axis_size=50)()

    def lidar(self, obs: lazy[batched[lidar_obs]]):
        ctx = plot_ctx.create(self.params.plot_points_limit)
        ctx += self.pos.plot_as_seg()
        ctx += self.plot_computed_rays(obs())
        return self, ctx


class _deterministic_motion_tracker(Controller):
    def __init__(self, cfg: localization_params, params: deterministic_motion_params):
        super().__init__(cfg)

        init_state = state(
            params,
            precompute(self.grid),
            position.zero(),
            gaussian.from_mean_cov(
                jnp.zeros(3),
                jnp.diag(jnp.array([0.1 / self._res, 0.1 / self._res, 0.02])),
            ),
        )
        self.dispatcher = jax_jit_dispatcher(
            dispatch(state.get_pose)(),
            dispatch(state.set_pose)(self._lazy_position_ex()),
            dispatch(state.lidar)(self._lazy_lidar_ex()),
            dispatch(state.twist)(self._lazy_twist_ex()),
        )
        self.dispatcher.run(init_state)

    def _get_pose(self):
        return self.dispatcher.process(state.get_pose)

    def _set_pose(self, pose, time) -> None:
        return self.dispatcher.process(state.set_pose, pose)

    def _twist(self, twist, time):
        with timer.create() as t:
            # gt = self._plot_ground_truth()
            ctx = self.dispatcher.process(state.twist, twist)
            # jax.block_until_ready(ctx)
            # self._visualize(ctx)
            # print(f"deterministic_motion_tracker/twist: {t.val} seconds")
            # print()

    def _lidar(self, obs, time) -> None:
        ctx = self.dispatcher.process(state.lidar, obs)
        self._visualize(ctx)


def deterministic_motion_tracker(cfg: localization_params) -> LocalizationBase:
    """localization implementation that only uses motion model"""
    return deterministic_motion_params()(cfg)
