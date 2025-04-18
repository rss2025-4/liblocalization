from __future__ import annotations

import math
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from geometry_msgs.msg import Twist
from jax import Array, random
from tf2_ros import TransformStamped

from libracecar.batched import batched, batched_zip
from libracecar.jax_utils import dispatch, divide_x_at_zero, jax_jit_dispatcher
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
    flike,
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
from ..map import precompute, precomputed_map, trace_ray
from ..motion import deterministic_position, dummy_motion_model, motion_model, twist_t
from ..priors import gaussian, particles
from ..ros import lidar_obs
from ..sensor import Condition, EmpiricalRayModel, log_likelyhood
from ..stats import (
    apply_noise,
    datapoint,
    ray_model_from_pkl,
    stats_base_dir,
    stats_state,
)


@dataclass
class particles_params:
    n_particles: int = 1000
    n_from_gaus: int = 0

    use_motion_model: bool = True
    # motion_model_noise: float = 0.5

    plot_level: int = 0
    plot_points_limit: int = 1000

    evidence_factor: float = 1.0

    stats_in_dir: Path | list[Path] | None = None
    stats_out_dir: Path | None = stats_base_dir / "default"

    model_path: Path | None = None

    collect_stats: bool = False

    def __call__(self, cfg: localization_params) -> LocalizationBase:
        return _particles_model(cfg, self)


class state(eqx.Module):
    res: float = eqx.field(static=True)
    params: particles_params = eqx.field(static=True)

    prior: particles
    cur_est: position

    map: precomputed_map

    sensor_model: EmpiricalRayModel
    stats: stats_state

    rng_key: Array = random.PRNGKey(0)

    confidence: flike = 0.0

    def get_seed(self):
        new_key, key = random.split(self.rng_key, 2)
        return tree_at_(lambda me: me.rng_key, self, new_key), jnp.array(key)

    def get_pose(self) -> tuple[state, position]:
        # return self, self.prior.mean()
        return self, self.cur_est

    def get_confidence(self):
        return self, jnp.array(self.confidence)

    def get_particles(self):
        ans, _ = self.prior.points.split_tuple()
        return self, ans

    def set_pose(self, pose: lazy[position]):
        p = pose()
        self, key = self.get_seed()
        with numpyro.handlers.seed(rng_seed=key):
            samples = particles.from_samples(
                gaussian.from_mean_cov(
                    p.as_arr(),
                    jnp.diag(
                        jnp.array(
                            [2.0 / self.res, 2.0 / self.res, 45.0 / 360 * 2 * math.pi]
                        )
                        ** 2
                    ),
                ).sample_batch(self.params.n_particles)
            )
            return tree_at_(lambda me: me.prior, self, samples), None

    def twist(self, twist_: lazy[twist_t], gt_: lazy[plotable]):
        self, key = self.get_seed()
        with numpyro.handlers.seed(rng_seed=key):
            twist = twist_()
            gt = gt_()

            ctx = plot_ctx.create(self.params.plot_points_limit)
            # ctx += gt

            self = tree_at_(
                lambda me: me.cur_est,
                self,
                replace_fn=lambda x: x + deterministic_position(twist),
            )

            if self.params.use_motion_model:
                new_prior = self.prior.map(lambda x: x + motion_model(twist))
                # new_prior = new_prior.map(
                #     lambda x: x
                #     + dummy_motion_model(
                #         twist.time, self.res, self.params.motion_model_noise
                #     )
                # )
            else:
                new_prior = self.prior.map(
                    lambda x: x + dummy_motion_model(twist.time, self.res, 2.0)
                )

            self = tree_at_(lambda s: s.prior, self, new_prior)

            ctx += self.prior.plot(20, plot_style(color=(0.9, 0.3, 0.0)))
            ctx.check()
            return self, ctx

    def plot_computed_rays(self, obs: batched[lidar_obs]) -> plotable:
        pos = self.get_pose()[1]

        def plot_one():
            x = obs[random.randint(prng_key_(), (), 0, len(obs))].unwrap()
            return trace_ray(self.map, pos.tran, pos.rot.mul_unit(x.angle)).plot(x.dist)

        return vmap_seperate_seed(plot_one, axis_size=50)()

    def _lidar(self, obs: batched[lidar_obs]):
        ctx = plot_ctx.create(self.params.plot_points_limit)
        ctx += self.prior.plot(20, plot_style(color=(1.0, 0.0, 0.0)))

        def prior_point(p: position, w: fval) -> tuple[tuple[position, fval], fval]:
            l = log_likelyhood(self.sensor_model, self.map, p, obs)
            logit = jnp.log(w) + l * self.params.evidence_factor
            return ((p, logit), l)

        logits, log_likelyhoods = self.prior.points.tuple_map(prior_point).split_tuple()

        self = tree_at_(
            lambda me: me.cur_est,
            self,
            particles.from_logits(logits.tuple_map(lambda p, l: (p, l * 10))).mean(),
        )

        ans1_ = particles.from_logits(logits)

        avg_likelyhood = (
            batched_zip(ans1_.points.split_tuple()[1], log_likelyhoods)
            .tuple_map(lambda w, l: w * l)
            .sum()
            .unwrap()
        )
        debug_print("avg_likelyhood", avg_likelyhood)

        self = tree_at_(lambda me: me.confidence, self, avg_likelyhood)

        ans1 = ans1_.resample(self.params.n_particles - self.params.n_from_gaus)
        ans1 = particles.from_samples(
            batched_vmap_with_rng(lambda x: apply_noise(x[0], self.res), ans1.points)
        )

        if self.params.n_from_gaus == 0:
            ans_conc = ans1
        else:
            gaus = self.prior.fit_to_gaussian()

            def update_guide(guide: gaussian, noise_factor: float):
                added_noise = jnp.array(
                    [2.0 / self.res, 0.2 / self.res, 5.0 / 360 * 2 * math.pi]
                )
                guide_noisy = gaussian.from_mean_cov(
                    gaus.mean,
                    gaus.covariance() + jnp.diag((added_noise * noise_factor) ** 2),
                )

                def log_l(p: position) -> fval:
                    return (
                        -guide_noisy.log_prob(p)
                        + gaus.log_prob(p)
                        + log_likelyhood(self.sensor_model, self.map, p, obs)
                        * self.params.evidence_factor
                    )

                logits = guide_noisy.sample_batch(1000).map(lambda p: (p, log_l(p)))
                return particles.from_logits(logits)

            guide = ans1_.fit_to_gaussian()

            for noise_factor in [1.0, 1.0, 0.5]:
                guide = update_guide(guide, noise_factor).fit_to_gaussian()

            ans2 = update_guide(guide, 0.5).resample(self.params.n_from_gaus)

            ans_conc = particles.from_samples(
                batched.concat(
                    [ans1.points.split_tuple()[0], ans2.points.split_tuple()[0]]
                )
            )

        self = tree_at_(lambda me: me.prior, self, ans_conc)

        ctx += self.prior.plot(20, plot_style(color=(0.0, 1.0, 0.0)))

        if self.params.plot_level >= 1:
            ctx += self.plot_computed_rays(obs)
        ctx.check()
        return self, ctx

    def lidar(self, msg: lazy[batched[lidar_obs]]):
        self, key = self.get_seed()
        with numpyro.handlers.seed(rng_seed=key):
            return self._lidar(msg())

    def update_stats(
        self, true_pos_pixels: lazy[position], lidar: lazy[batched[lidar_obs]]
    ):
        stats, _ = self.stats.update(true_pos_pixels, lidar)
        return tree_at_(lambda me: me.stats, self, stats), None


class _particles_model(Controller):
    def __init__(self, cfg: localization_params, params: particles_params):
        global ct
        super().__init__(cfg)
        self.params = params
        self.dispatcher = jax_jit_dispatcher(
            dispatch(state.get_pose)(),
            dispatch(state.get_particles)(),
            dispatch(state.get_confidence)(),
            dispatch(state.set_pose)(self._lazy_position_ex()),
            dispatch(state.twist)(self._lazy_twist_ex(), self._plot_ground_truth()),
            dispatch(state.lidar)(self._lazy_lidar_ex()),
            dispatch(state.update_stats)(
                self._pose_from_ros(TransformStamped()),
                self._lazy_lidar_ex(),
            ),
        )
        assert (self.params.stats_in_dir is None) != (self.params.model_path is None)

        if self.params.stats_in_dir is not None:
            model = ray_model_from_pkl(self.params.stats_in_dir)
        else:
            assert self.params.model_path is not None
            assert self.params.model_path.exists()
            model = pickle.loads(self.params.model_path.read_bytes())

        self.dispatcher.run_with_setup(self._init_state, model)
        # wait for dispatcher to start
        _ = self._get_pose()

    def _init_state(self, sensor_model: EmpiricalRayModel):
        precomp = precompute(self.grid)
        return state(
            res=self._res,
            params=self.params,
            map=precomp,
            sensor_model=sensor_model,
            stats=stats_state(
                out_dir=self.params.stats_out_dir,
                map=precomp,
                data_idx=jnp.array(0),
                data=batched.create(
                    datapoint(Condition(0.0), jnp.array(0.0)),
                ).repeat(32 * 100 * 50 * 10),
            ),
            prior=particles.from_samples(
                batched.create(position.zero()).repeat(self.params.n_particles)
            ),
            cur_est=position.zero(),
        )

    def _get_pose(self):
        return self.dispatcher.process(state.get_pose)

    def _get_particles(self) -> batched[position]:
        return self.dispatcher.process(state.get_particles)

    def get_confidence(self):
        return float(self.dispatcher.process(state.get_confidence))

    def _set_pose(self, pose, time) -> None:
        print("set_pose!!!")
        return self.dispatcher.process(state.set_pose, pose)

    def _twist(self, twist, time):
        with timer.create() as t:
            gt = self._plot_ground_truth()
            ctx = self.dispatcher.process(state.twist, twist, gt)
            # self._visualize(ctx)
            # print(f"twist: {t.val} seconds")

    def _lidar(self, obs, time):
        with timer.create() as t:
            ctx = self.dispatcher.process(state.lidar, obs)

            if self.params.collect_stats:
                gt = self._ground_truth(time)
                if gt is not None:
                    self.dispatcher.process(state.update_stats, gt, obs)

            jax.block_until_ready(ctx)
            print(f"lidar: {t.val} seconds")
            self._visualize(ctx)


def particles_model(
    params: particles_params,
) -> Callable[[localization_params], LocalizationBase]:
    def inner(cfg: localization_params) -> LocalizationBase:
        return _particles_model(cfg, params)

    return inner
