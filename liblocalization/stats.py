import itertools
import math
import os
import pickle
import tempfile
import time
from dataclasses import dataclass
from functools import partial
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
    cond_,
    debug_print,
    flike,
    fval,
    io_callback_,
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

from .map import _trace_ray_res, precompute, precomputed_map, trace_ray
from .motion import dummy_motion_model, motion_model, twist_t
from .priors import gaussian, particles
from .ros import lidar_obs
from .sensor import Condition, EmpiricalRayModel

stats_base_dir = Path(__file__).parent.parent / "stats"
models_base_dir = Path(__file__).parent.parent / "models"


class datapoint(eqx.Module):
    condition: Condition
    observed_pixels: Array


def apply_noise(p: position, res: float):
    ans_tran = vec.create(
        # 5cm of noise
        normal_(p.tran.x, 0.05 / res).sample(prng_key_()),
        normal_(p.tran.y, 0.05 / res).sample(prng_key_()),
    )
    ans_rot = p.rot.mul_unit(
        unitvec.from_angle(
            # 5 degrees of noise
            normal_(0.0, 5.0 / 360 * 2 * math.pi).sample(prng_key_()),
        )
    )
    return position(ans_tran, ans_rot)


class stats_state(eqx.Module):
    out_dir: Path | None = eqx.field(static=True)

    map: precomputed_map

    data_idx: ival
    data: batched[datapoint]

    rng_key: Array = random.PRNGKey(0)

    __repr__ = pformat_repr

    def get_seed(self):
        new_key, key = random.split(self.rng_key, 2)
        return tree_at_(lambda me: me.rng_key, self, new_key), jnp.array(key)

    @staticmethod
    def _write_data_cb(out_dir: Path, data_idx: ival, data: batched[datapoint]):
        out_dir.mkdir(exist_ok=True, parents = True)
        for i in itertools.count():
            file = out_dir / f"data_{i}.pkl"
            if file.exists():
                continue
            file.write_bytes(pickle.dumps(data[: int(data_idx)]))
            return

    def _do_write(self):
        if self.out_dir is not None:
            io_callback_(partial(stats_state._write_data_cb, self.out_dir))(
                self.data_idx, self.data
            )
        return tree_at_(lambda me: me.data_idx, self, 0)

    def update(self, true_pos_pixels: lazy[position], lidar: lazy[batched[lidar_obs]]):
        self, key = self.get_seed()
        with numpyro.handlers.seed(rng_seed=key):
            return self._update(true_pos_pixels(), lidar())

    def _update(self, true_pos_pixels: position, lidar: batched[lidar_obs]):
        def handle_one(obs: lidar_obs) -> datapoint:
            ans = trace_ray(
                self.map,
                # 5cm of noise
                vec.create(
                    normal_(true_pos_pixels.tran.x, 0.05 / self.map.res).sample(
                        prng_key_()
                    ),
                    normal_(true_pos_pixels.tran.y, 0.05 / self.map.res).sample(
                        prng_key_()
                    ),
                ),
                true_pos_pixels.rot.mul_unit(obs.angle).mul_unit(
                    unitvec.from_angle(
                        # 5 degrees of noise
                        normal_(0.0, 5.0 / 360 * 2 * math.pi).sample(prng_key_()),
                    )
                ),
            )
            return datapoint(Condition.from_traced_ray(ans), obs.dist)

        new_data = vmap_seperate_seed(
            lambda: batched_vmap_with_rng(handle_one, lidar), 16
        )().reshape(-1)

        def update_one(x: Array, y: Array):
            return lax.dynamic_update_slice(
                x, y, (self.data_idx,), allow_negative_indices=False
            )

        self = tree_at_(
            lambda me: (me.data_idx, me.data),
            self,
            (
                self.data_idx + len(new_data),
                batched_treemap(update_one, self.data, new_data),
            ),
        )

        self = cond_(
            self.data_idx + len(new_data) > len(self.data),
            true_fun=lambda: self._do_write(),
            false_fun=lambda: self,
        )
        return self, None


def load_from_pkl_one(file: Path) -> batched[datapoint]:
    obj = pickle.loads(file.read_bytes())
    assert type(obj) is batched
    assert type(obj.uf) is datapoint
    return obj


def _get_files(dir: Path):
    def gen():
        for i in itertools.count():
            file = dir / f"data_{i}.pkl"
            if not file.exists():
                return
            yield file

    return list(gen())


def load_from_pkl(dir: Path) -> batched[datapoint]:
    def gen():
        for i in itertools.count():
            file = dir / f"data_{i}.pkl"
            if not file.exists():
                return
            yield load_from_pkl_one(file)

    parts = list(gen())
    assert len(parts) > 0
    return batched.concat(parts)


@jit
def _push(model: EmpiricalRayModel, data: batched[datapoint]):
    return model.push_counts(data.map(lambda x: (x.condition, x.observed_pixels)))


def ray_model_from_pkl(dir: Path | list[Path]) -> EmpiricalRayModel:
    if isinstance(dir, Path):
        files = _get_files(dir)
    else:

        def gen():
            for x in dir:
                yield from _get_files(x)

        files = list(gen())

    ans = EmpiricalRayModel.empty(400, 400)
    for f in files:
        ans = _push(ans, load_from_pkl_one(f))
    return ans
