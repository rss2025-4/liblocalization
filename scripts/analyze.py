import pickle

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from liblocalization.stats import stats_t


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
