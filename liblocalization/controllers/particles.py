import math
import time
from dataclasses import dataclass
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from geometry_msgs.msg import Twist
from jax import Array, random
from tf2_ros import TransformStamped

from liblocalization.map import precompute, precomputed_map
from liblocalization.sensor import log_likelyhood
from libracecar.batched import batched
from libracecar.jax_utils import dispatch_spec, divide_x_at_zero, jax_jit_dispatcher
from libracecar.numpyro_utils import (
    batched_vmap_with_rng,
    prng_key_,
    trunc_normal_,
    vmap_seperate_seed,
)
from libracecar.plot import plot_ctx, plot_style, plotable
from libracecar.specs import position
from libracecar.utils import (
    PropagatingThread,
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
from ..motion import motion_model, twist_t
from ..priors import gaussian_prior, particles
from ..ros import lidar_obs


@dataclass
class particles_params:
    n_particles: int = 1000


class state(eqx.Module):
    res: float = eqx.field(static=True)
    params: particles_params = eqx.field(static=True)

    prior: particles
    map: precomputed_map

    rng_key: Array = random.PRNGKey(0)

    def get_seed(self):
        new_key, key = random.split(self.rng_key, 2)
        return tree_at_(lambda me: me.rng_key, self, new_key), jnp.array(key)

    def get_pose(self):
        return self, self.prior.mean()

    def set_pose(self, pose: lazy[position]):
        p = pose()
        self, key = self.get_seed()
        with numpyro.handlers.seed(rng_seed=key):
            samples = particles.from_samples(
                gaussian_prior.from_mean_cov(
                    p.as_arr(),
                    jnp.diag(jnp.array([0.2, 0.2, 10.0 / 360 * 2 * math.pi])),
                ).sample_batch(self.params.n_particles)
            )
            return tree_at_(lambda me: me.prior, self, samples), None

    def twist(self, twist_: lazy[twist_t], gt_: lazy[plotable]):
        self, key = self.get_seed()
        with numpyro.handlers.seed(rng_seed=key):
            twist = twist_()
            gt = gt_()

            ctx = plot_ctx.create(100)
            ctx += gt

            new_prior = self.prior.map(lambda x: x + motion_model(twist))
            self = tree_at_(lambda s: s.prior, self, new_prior)

            ctx += self.prior.plot(20)
            return self, ctx

    def lidar(self, msg: lazy[batched[lidar_obs]]):
        self, key = self.get_seed()
        with numpyro.handlers.seed(rng_seed=key):
            obs = msg()
            pos, new_weights = self.prior.points.tuple_map(
                lambda p, w: (p, jnp.log(w) + log_likelyhood(self.map, p, obs))
            ).split_tuple()

            # debug_print("new_weights", new_weights)

            def sample_one():
                idx = dist.CategoricalLogits(new_weights.unflatten()).sample(
                    prng_key_()
                )
                return pos[idx]

            ans = vmap_seperate_seed(sample_one, axis_size=self.params.n_particles)()
            self = tree_at_(lambda me: me.prior, self, particles.from_samples(ans))
            return self, None


class _particles_model(Controller):
    def __init__(self, cfg: localization_params, params: particles_params):
        super().__init__(cfg)
        self.params = params
        self.dispatcher = jax_jit_dispatcher(
            dispatch_spec(state.get_pose),
            dispatch_spec(state.set_pose, self._lazy_position_ex()),
            dispatch_spec(
                state.twist, self._lazy_twist_ex(), self._plot_ground_truth()
            ),
            dispatch_spec(state.lidar, self._lazy_lidar_ex()),
        )
        self.dispatcher.run_with_setup(self._init_state)
        # wait for dispatcher to start
        _ = self._get_pose()

    def _init_state(self):
        return state(
            res=self._res,
            params=self.params,
            map=precompute(self.grid),
            prior=particles.from_samples(
                batched.create(position.zero()).repeat(self.params.n_particles)
            ),
        )

    def _get_pose(self):
        return self.dispatcher.process(state.get_pose)

    def _set_pose(self, pose) -> None:
        return self.dispatcher.process(state.set_pose, pose)

    def _twist(self, twist):
        with timer.create() as t:
            gt = self._plot_ground_truth()
            ctx = self.dispatcher.process(state.twist, twist, gt)
            self._visualize(ctx)
            print(f"twist: {t.val} seconds")

    def _lidar(self, obs):
        with timer.create() as t:
            self.dispatcher.process(state.lidar, obs)
            print(f"lidar: {t.val} seconds")


def particles_model(
    params: particles_params,
) -> Callable[[localization_params], LocalizationBase]:
    def inner(cfg: localization_params) -> LocalizationBase:
        return _particles_model(cfg, params)

    return inner
