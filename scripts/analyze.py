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
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from numpyro.distributions import constraints
from termcolor import colored
from tf2_ros import TransformStamped

from liblocalization.controllers.stats import (
    datapoint,
    default_stats_dir,
    load_from_pkl,
    ray_model_from_pkl,
    stats_state,
)
from liblocalization.sensor import EmpiricalRayModel
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


def plot_counts():
    model = ray_model_from_pkl(default_stats_dir)
    model.parts.uf.counts

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
