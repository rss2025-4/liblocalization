import itertools
import math
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from queue import Queue
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import optax
from geometry_msgs.msg import Twist
from jax import Array, lax, random
from jax import tree_util as jtu
from jaxtyping import Array, Float
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from numpyro.distributions import constraints
from termcolor import colored
from tf2_ros import TransformStamped

from liblocalization.sensor import EmpiricalRayModel
from liblocalization.stats import (
    datapoint,
    load_from_pkl,
    models_base_dir,
    ray_model_from_pkl,
    stats_base_dir,
    stats_state,
)
from libracecar.batched import batched
from libracecar.jax_utils import dispatch_spec, divide_x_at_zero, jax_jit_dispatcher
from libracecar.numpyro_utils import (
    batched_dist,
    batched_vmap_with_rng,
    jit_with_seed,
    numpyro_param,
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
    cast_unchecked_,
    cond_,
    debug_print,
    ensure_not_weak_typed,
    flike,
    fval,
    io_callback_,
    jit,
    lazy,
    pformat_repr,
    safe_select,
    timer,
    tree_at_,
    tree_to_ShapeDtypeStruct,
)
from libracecar.vector import unitvec, vec

jax.config.update("jax_platform_name", "cpu")


# def get_one(dir: Path) -> np.ndarray:
#     model = ray_model_from_pkl(
#         [
#             dir,
#             # stats_base_dir / "rosbag",
#             # stats_base_dir / "sim",
#             #
#         ]
#     )
#     return np.array(model.parts.map(lambda x: x.probs(0.01)).uf).T


def parts():
    return [
        #
        # stats_base_dir / "rosbags_lidar_fixed2",
        stats_base_dir / "sim",
        stats_base_dir / "rosbags_4_16_v2",
        #
    ]


def dump_model():
    out_path = models_base_dir / "model5.pkl"

    model = ray_model_from_pkl(parts())
    out_path.write_bytes(pickle.dumps(model))


def plot_counts():
    # model = ray_model_from_pkl(stats_base_dir / "sim")
    model = ray_model_from_pkl(parts())

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_aspect("equal")

    cax = ax.imshow(
        np.array(model.parts.uf.counts + 1).T,
        cmap="viridis",
        origin="lower",
        norm=LogNorm(),
    )
    fig.colorbar(cax, ax=ax)
    plt.show()

    # fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # norm = LogNorm()

    # def do_one(ax: Axes, data: np.ndarray, title: str):

    #     ax.set_title(title)
    #     ax.set_aspect("equal")
    #     ax.set_xlabel("f=ray_est(*, *) (pixels)")
    #     ax.set_ylabel("lidar observation (O*) (pixels)")

    #     cax = ax.imshow(
    #         data,
    #         cmap="viridis",
    #         origin="lower",
    #         norm=norm,
    #     )
    #     return cax

    # cax = do_one(
    #     axes[0][0],
    #     data=get_one(stats_base_dir / "sim"),
    #     title="likelihood∗ for lidar in sim",
    # )
    # cax = do_one(
    #     axes[0][1],
    #     data=get_one(stats_base_dir / "sim")[:100, :100],
    #     title="likelihood∗ for lidar in sim (zoomed in)",
    # )

    # cax = do_one(
    #     axes[1][0],
    #     data=get_one(stats_base_dir / "rosbag"),
    #     title="likelihood∗ for lidar in real",
    # )
    # cax = do_one(
    #     axes[1][1],
    #     data=get_one(stats_base_dir / "rosbag")[:100, :100],
    #     title="likelihood∗ for lidar in real (zoomed in)",
    # )

    # fig.tight_layout()
    # fig.colorbar(cax, ax=axes, orientation="vertical")

    # fig.savefig("likelihood_sim.png", dpi=300)
    # plt.close(fig)

    # # plt.show()
