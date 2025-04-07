import pickle

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from liblocalization.sensor import *


def analyze():
    x1 = jnp.linspace(0, d_max, 100)
    y1 = jnp.linspace(0, d_max, 100)

    x, y = jnp.meshgrid(x1, y1)

    # Create the figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    assert isinstance(ax, Axes3D)

    def get(d):
        disttrib = ray_model_(d)

        def get2(obs):
            return disttrib.log_prob(obs)

        return jax.vmap(get2)(y1)

    # Plot the surface
    ax.plot_surface(
        x,
        y,
        np.array(jax.vmap(get)(x1)),
        cmap="viridis",
        # rcount=20,
        # ccount=20,
    )

    # Add labels
    ax.set_xlabel("observed")
    ax.set_ylabel("actual")
    ax.set_zlabel("log density")

    # print(stats.counts_using_truth[150:175, 150:175])
    # ax.set_xlim(150, 175)
    # ax.set_ylim(150, 175)
    # ax.set_zlim(0, 0.1)

    # Show the plot
    plt.show()
