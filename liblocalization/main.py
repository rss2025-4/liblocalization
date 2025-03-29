import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import tf2_ros
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from termcolor import colored
from tf_transformations import euler_from_quaternion
from visualization_msgs.msg import Marker

from libracecar.plot import plot_ctx
from libracecar.specs import position

from .core import (
    compute_localization,
    lidar_obs,
    localization_params,
    localization_state,
)
from .map import Grid, precompute

np.set_printoptions(precision=3, suppress=True)

jax.config.update("jax_platform_name", "cpu")


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
        self.tmp_map = None
        self.state = None

        self.tfBuffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tfBuffer, self)

        # self._tick_timer = self.create_timer(0.5, self.tick)

        self.visualization_pub = self.create_publisher(Marker, "/visualization", 10)

        self.lidar_sub = self.create_subscription(
            LaserScan, "/scan", self.lidar_callback, 1
        )

        self._i = 0

    def lidar_callback(self, msg: LaserScan):
        if self._i % 20 == 0:
            self.tick(msg)
        self._i += 1

    def map_callback(self, map_msg: OccupancyGrid):
        print("liblocalization: received map")

        grid = Grid.create(map_msg)
        _start = time.time()
        map = precompute(grid)
        self.tmp_map = map
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
        assert self.tmp_map is not None
        obs_pixels = lidar_obs.from_msg_meters(msg, self.tmp_map.grid.res)

        state_ex = localization_state.init(self.tmp_map, position.zero())
        localization_params(state_ex, obs_pixels)

        comp = compute_localization.lower(
            localization_params(state_ex, obs_pixels)
        ).compile()
        print("cost_analysis:", comp.cost_analysis())

        self.state = localization_state.init(
            self.tmp_map, self.tmp_map.grid.to_pixels(self._sim_current_pos_meters())
        )
        self.tmp_map = None

    def tick(self, msg: LaserScan):
        if self.state is None and self.tmp_map is None:
            return

        if self.tmp_map is not None:
            self.initialize(msg)

        assert self.state is not None

        obs_pixels = lidar_obs.from_msg_meters(msg, self.state.map.grid.res)

        true_pos = self.state.map.grid.to_pixels(self._sim_current_pos_meters())

        state = self.state
        if (
            jnp.any(jnp.isnan(state.state.mean))
            or jnp.linalg.norm(state.state.mean - true_pos.as_arr()) > 100
        ):
            print(colored("too far; resetting", "red"))

            state = localization_state.init(
                self.state.map,
                self.state.map.grid.to_pixels(self._sim_current_pos_meters()),
            )

        _start = time.time()
        res = compute_localization(localization_params(state, obs_pixels))
        res.state.state.mean.block_until_ready()
        print(f"elapsed: {time.time() - _start:.5f}")

        # print("res.plot", res.plot.idx, res.plot.points[:20])
        print("hyperparams:", res.state.state.a_arr_mean, res.state.state.sigma_mean)

        self.state = res.state
        self.visualize(res.plot)

    def visualize(self, ctx: plot_ctx):
        m = Marker()
        m.type = Marker.POINTS
        m.header.frame_id = "map"
        m.scale.x = 0.05
        m.scale.y = 0.05
        ctx.execute(m)
        self.visualization_pub.publish(m)


# @jit
# # @checkify_simple
# def testfn(
#     map: precomputed_map,
#     pos: position,
#     obs: batched[lidar_obs],
#     rng_seed=random.PRNGKey(0),
# ):
#     with numpyro.handlers.seed(rng_seed=rng_seed):

#         # def _obs_at_pos(obs: lidar_obs):
#         #     ray_ang = pos.rot.mul_unit(obs.angle)
#         #     ans = trace_ray(map, pos.tran, ray_ang)
#         #     return jnp.array([ans.dist, obs.dist, ans.distance_to_nearest])

#         # obs_at_pos = obs.map(_obs_at_pos).unflatten()
#         # debug_print("obs_at_pos", obs_at_pos)

#         ctx = plot_ctx.create(100)

#         ctx += pos.plot_as_seg(plot_style(color=(0, 0, 1)))

#         res = map.grid.res
#         prior = noisy_position.from_mean_cov(
#             pos.as_arr(), jnp.diag(jnp.array([0.2 / res, 0.2 / res, math.pi / 8]) ** 2)
#         )
#         prior = noisy_position(prior.sample().as_arr(), prior.scale_tril)

#         # angles_ = jnp.linspace(0.0, 2 * math.pi, 65)[:-1]
#         # angles = batched.create(angles_, angles_.shape)
#         # # angles = batched.create(jnp.array(0.5))
#         # lidar_points = angles.map(
#         #     lambda angle: trace_ray(
#         #         map,
#         #         vec.create(pos.coord[0], pos.coord[1]),
#         #         unitvec.from_angle(angle),
#         #     ),
#         #     # sequential=True,
#         # )
#         # ctx += lidar_points.map(lambda x: x.plot())

#         ctx += prior.plot(20, plot_style(color=(1, 0, 0)))
#         post = compute_posterior(map, prior, obs, pos)
#         debug_print("losses", post.losses)
#         ctx += post.posterior.plot(20, plot_style(color=(0, 1, 0)))

#         ctx = map.grid.plot_from_pixels_vec(ctx)
#         return ctx
