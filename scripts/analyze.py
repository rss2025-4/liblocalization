import math
import pickle
from typing import Callable, final

import equinox as eqx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import optax
from jax import lax, random
from jax import tree_util as jtu
from jaxtyping import Array, ArrayLike, Float, Int32
from mpl_toolkits.mplot3d import Axes3D
from numpyro.distributions import constraints
from sensor_msgs.msg import LaserScan
from termcolor import colored

from liblocalization.map import _trace_ray_res, precomputed_map, trace_ray
from liblocalization.ros import lidar_obs
from liblocalization.stats import stats_t
from libracecar.batched import batched
from libracecar.numpyro_utils import (
    batched_dist,
    batched_vmap_with_rng,
    jit_with_seed,
    normal_,
    numpyro_param,
    prng_key_,
    vmap_seperate_seed,
)
from libracecar.plot import plot_ctx, plot_style, plotable
from libracecar.specs import position
from libracecar.utils import (
    cast,
    cast_unchecked_,
    cond_,
    debug_callback,
    debug_print,
    flike,
    fval,
    jit,
    pformat_repr,
    round_clip,
    tree_at_,
)
from libracecar.vector import unitvec, vec


def analyze():
    with open(
        "/home/dockeruser/racecar_ws/src/liblocalization/stats1.pkl", "rb"
    ) as file:
        stats = pickle.load(file)

    assert isinstance(stats, stats_t)

    x = np.arange(stats.max_d)
    y = np.arange(stats.max_d)
    x, y = np.meshgrid(x, y)

    # Create the figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    assert isinstance(ax, Axes3D)

    # Plot the surface
    ax.plot_surface(
        x,
        y,
        np.array(stats.counts_using_truth_normalized()),
        cmap="viridis",
        rcount=20,
        ccount=20,
    )

    # Add labels
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")

    # print(stats.counts_using_truth[150:175, 150:175])
    # ax.set_xlim(150, 175)
    # ax.set_ylim(150, 175)
    ax.set_zlim(0, 0.1)

    # Show the plot
    plt.show()
