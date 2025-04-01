import os
import pickle
import sys
import threading
import time
import traceback
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import tf2_ros
from jax import tree_util as jtu
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from termcolor import colored
from tf_transformations import euler_from_quaternion
from visualization_msgs.msg import Marker

from libracecar.plot import plot_ctx
from libracecar.specs import position
from libracecar.utils import PropagatingThread, ensure_not_weak_typed

from .core import (
    compute_localization,
    localization_state,
    main_loop,
    request_t,
)
from .map import Grid, precompute
from .ros import lidar_obs

np.set_printoptions(precision=3, suppress=True)

jax.config.update("jax_platform_name", "cpu")

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")


@dataclass
class localization_config:
    map_topic: str = "/map"
    scan_topic: str = "/scan"
    odom_topic: str = "/odom"


class Localization(Node):

    def __init__(self, cfg: localization_config):
        super().__init__("Localization")

        self.cfg = cfg

        self.map_subscriber = self.create_subscription(
            OccupancyGrid, self.cfg.map_topic, self.map_callback, 1
        )
        self.map = None
        self.handler: main_loop | None = None

        self.tfBuffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tfBuffer, self)

        # self._tick_timer = self.create_timer(0.5, self.tick)

        self.visualization_pub = self.create_publisher(Marker, "/visualization", 10)

        self.lidar_sub = self.create_subscription(
            LaserScan, "/scan", self.lidar_callback, 1
        )

        self._i = 0

    def lidar_callback(self, msg: LaserScan):
        if self._i % 50 == 0:
            self.tick(msg)
        self._i += 1

    def map_callback(self, map_msg: OccupancyGrid):
        print("liblocalization: received map")

        grid = Grid.create(map_msg)
        _start = time.time()
        self.map = precompute(grid)
        jax.block_until_ready(self.map)
        print(f"elapsed (precompute): {time.time() - _start:.5f}")

    def _sim_current_pos_meters(self):
        t = self.tfBuffer.lookup_transform("map", "base_link", Time())

        t_rot = t.transform.rotation
        t_rot_ = euler_from_quaternion((t_rot.x, t_rot.y, t_rot.z, t_rot.w))

        return position.create(
            (t.transform.translation.x, t.transform.translation.y),
            t_rot_[2],
        )

    def initialize(self, msg: LaserScan):
        assert self.map is not None

        obs_pixels = lidar_obs.from_msg_meters(msg, self.map.grid.res)

        self.handler = main_loop(
            request_t(laser=obs_pixels, true_pos_pixels=position.zero())
        )

        self.handler_worker = self.handler.jit(self.map, position.zero())

        self.map.grid.to_pixels(self._sim_current_pos_meters())

        start_pos = self.map.grid.to_pixels(self._sim_current_pos_meters())

        PropagatingThread(
            target=self.handler_worker, args=(self.map, start_pos)
        ).start()

    def tick(self, msg: LaserScan):
        if self.map is None:
            return

        if self.handler is None:
            self.initialize(msg)
            return

        obs_pixels = lidar_obs.from_msg_meters(msg, self.map.grid.res)

        truth = self.map.grid.to_pixels(self._sim_current_pos_meters())

        _start = time.time()
        res = self.handler.process(request_t(obs_pixels, truth))
        jax.block_until_ready(res)
        print(f"elapsed: {time.time() - _start:.5f}")

        # print(res.state.stats.counts_using_truth[10:20, 10:20])

        # if self._i % 200 == 0:
        #     with open(
        #         "/home/dockeruser/racecar_ws/src/liblocalization/stats.pkl", "wb"
        #     ) as file:
        #         pickle.dump(res.state.stats, file)

        # self.state = res.state
        self.visualize(res.plot)

    def visualize(self, ctx: plot_ctx):
        m = Marker()
        m.type = Marker.POINTS
        m.header.frame_id = "map"
        m.scale.x = 0.05
        m.scale.y = 0.05
        ctx.execute(m)
        self.visualization_pub.publish(m)
