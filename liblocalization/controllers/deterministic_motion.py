import math
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro
from geometry_msgs.msg import Twist
from jax import Array, random

from liblocalization.map import Grid
from liblocalization.ros import lidar_obs
from libracecar.batched import batched
from libracecar.jax_utils import dispatch_spec, divide_x_at_zero, jax_jit_dispatcher
from libracecar.numpyro_utils import (
    trunc_normal_,
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
from ..motion import deterministic_position, twist_t
from ..priors import gaussian_prior


class state(eqx.Module):
    res: float = eqx.field(static=True)
    grid: Grid

    pos: position
    noisy_pos: gaussian_prior
    rng_key: Array = random.PRNGKey(0)

    def get_seed(self):
        new_key, key = random.split(self.rng_key, 2)
        return tree_at_(lambda me: me.rng_key, self, new_key), jnp.array(key)

    def get_pose(self):
        return self, self.pos

    def set_pose(self, pose: lazy[position]):
        p = pose()
        self = tree_at_(lambda me: me.pos, self, p)
        self = tree_at_(lambda me: me.noisy_pos.mean, self, p.as_arr())
        return self, None

    def twist(self, twist_: lazy[twist_t], gt_: lazy[plotable]):
        self, key = self.get_seed()
        with numpyro.handlers.seed(rng_seed=key):
            twist = twist_()
            gt = gt_()

            ctx = plot_ctx.create(1000)
            ctx += gt

            ctx += position.zero().plot_as_seg(plot_style(color=(0.0, 1.0, 0.0)))

            # ctx += self.grid.plot()

            self = tree_at_(
                lambda s: s.pos, self, self.pos + deterministic_position(twist)
            )
            self = tree_at_(
                lambda s: s.noisy_pos, self, self.noisy_pos.apply_twist(twist)
            )

            ctx += self.pos.plot_as_seg()
            # ctx += self.noisy_pos.plot(20)
            # ctx += position.from_arr(self.noisy_pos.mean).plot_as_seg(
            #     plot_style(color=(1.0, 0.0, 0.0))
            # )
            return self, ctx


class _deterministic_motion_tracker(Controller):
    def __init__(self, cfg: localization_params):
        super().__init__(cfg)

        init_state = state(
            self._res,
            self.grid,
            position.zero(),
            gaussian_prior.from_mean_cov(
                jnp.zeros(3),
                jnp.diag(jnp.array([0.1 / self._res, 0.1 / self._res, 0.02])),
            ),
        )
        self.dispatcher = jax_jit_dispatcher(
            dispatch_spec(state.get_pose),
            dispatch_spec(state.set_pose, self._lazy_position_ex()),
            dispatch_spec(
                state.twist, self._lazy_twist_ex(), self._plot_ground_truth()
            ),
        )
        self.dispatcher.run(init_state)

    def _get_pose(self):
        return self.dispatcher.process(state.get_pose)

    def _set_pose(self, pose) -> None:
        return self.dispatcher.process(state.set_pose, pose)

    def _twist(self, twist):
        with timer.create() as t:
            gt = self._plot_ground_truth()
            ctx = self.dispatcher.process(state.twist, twist, gt)
            self._visualize(ctx)
            # print(f"deterministic_motion_tracker/twist: {t.val} seconds")
            # print()

    def _lidar(self, obs: lazy[batched[lidar_obs]]) -> None:
        pass


def deterministic_motion_tracker(cfg: localization_params) -> LocalizationBase:
    """localization implementation that only uses motion model"""
    return _deterministic_motion_tracker(cfg)
