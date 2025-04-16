import itertools
import math
import os
import pickle
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import deprecation
import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from geometry_msgs.msg import Twist
from jax import Array, lax, random
from jaxtyping import Array, Int32
from tf2_ros import TransformStamped

from libracecar.batched import batched, batched_treemap
from libracecar.jax_utils import dispatch_spec, divide_x_at_zero, jax_jit_dispatcher
from libracecar.numpyro_utils import (
    batched_vmap_with_rng,
    normal_,
    prng_key_,
    trunc_normal_,
    vmap_seperate_seed,
)
from libracecar.plot import plot_ctx, plot_style, plotable
from libracecar.specs import position
from libracecar.utils import (
    PropagatingThread,
    bval,
    cast,
    cast_unchecked,
    debug_print,
    flike,
    fval,
    ival,
    jit,
    lazy,
    pformat_repr,
    round_clip,
    safe_select,
    timer,
    tree_at_,
)
from libracecar.vector import unitvec, vec

from .._api import Controller
from ..api import LocalizationBase, localization_params
from ..map import _trace_ray_res, precompute, precomputed_map, trace_ray
from ..motion import dummy_motion_model, motion_model, twist_t
from ..priors import gaussian, particles
from ..ros import lidar_obs
from ..sensor import Condition, EmpiricalRayModel
from ..stats import datapoint, stats_base_dir, stats_state


@dataclass
class stats_params:
    out_dir: Path = stats_base_dir / "default"

    def __call__(self, cfg: localization_params) -> LocalizationBase:
        return _stats(cfg, self)


class _stats(Controller):
    def __init__(self, cfg: localization_params, params: stats_params):
        super().__init__(cfg)
        self.params = params

        self.params.out_dir.mkdir(exist_ok=True)
        self.dispatcher = jax_jit_dispatcher(
            dispatch_spec(
                stats_state.update,
                self._pose_from_ros(TransformStamped()),
                self._lazy_lidar_ex(),
            )
        )
        self.dispatcher.run_with_setup(self._init_state)

    def _init_state(self):
        return stats_state(
            out_dir=self.params.out_dir,
            map=precompute(self.grid),
            data_idx=jnp.array(0),
            data=batched.create(
                datapoint(Condition(0.0), jnp.array(0.0)),
            ).repeat(32 * 100 * 50 * 10),
        )

    def _get_pose(self):
        assert False

    def _get_particles(self):
        assert False

    def _set_pose(self, pose, time):
        pass

    def _twist(self, twist, time):
        pass

    def _lidar(self, obs, time):
        gt = self._ground_truth(time)
        if gt is None:
            return

        with timer.create() as t:
            self.dispatcher.process(stats_state.update, gt, obs)
