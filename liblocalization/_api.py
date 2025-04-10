from abc import abstractmethod
from typing import Any

import numpy as np
import tf_transformations
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Twist
from jax import Array
from jaxtyping import Float
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from termcolor import colored
from tf2_ros import TransformStamped
from visualization_msgs.msg import Marker

from libracecar.batched import batched, blike
from libracecar.plot import plot_ctx, plot_style, plotable, plotfn
from libracecar.ros_utils import float_to_time_msg, time_msg_to_float
from libracecar.specs import position
from libracecar.utils import jit, lazy, tree_select

from .api import LocalizationBase, localization_params
from .map import Grid, GridMeta
from .ros import lidar_obs, twist_t


class Controller(LocalizationBase):

    def __init__(self, cfg: localization_params):
        self.cfg = cfg

        self.grid = Grid.create(cfg.map)

        self._prev_time: float = -1.0
        self.last_twist: Twist | None = None

        self._res = self.grid.meta.res

    @abstractmethod
    def _get_pose(self) -> position: ...

    @abstractmethod
    def _set_pose(self, pose: lazy[position], time: float) -> None: ...

    @abstractmethod
    def _twist(self, twist: lazy[twist_t], time: float) -> None: ...

    @abstractmethod
    def _lidar(self, obs: lazy[batched[lidar_obs]], time: float) -> None: ...

    def _get_particles(self) -> batched[position]:
        return batched.create(self._get_pose()).reshape(1)

    def _lazy_position_ex(self) -> lazy[position]:
        return self._pose_from_ros(TransformStamped())

    def _lazy_twist_ex(self) -> lazy[twist_t]:
        return twist_t.from_ros(Twist(), 0.0, 1.0)

    def _lazy_lidar_ex(self) -> lazy[batched[lidar_obs]]:
        msg = LaserScan()
        msg.ranges = [1.0 for _ in range(self.cfg.n_laser_points)]
        return lidar_obs.from_ros(msg, 0.1)

    def _update_time(self, new_time_: Time, warn_tag: Any) -> float:
        new_time = time_msg_to_float(new_time_)
        ans = new_time - self._prev_time
        if ans < 0.0:
            print(
                colored(
                    f"warning: ({warn_tag}) time went backwards by: {ans:.5f}", "red"
                )
            )
            ans = 0.0
        if ans > 1.0:
            print(colored(f"warning: ({warn_tag}) {ans:.2f}s with no messages", "red"))
            ans = 1.0

        self._prev_time = new_time
        return ans

    def odom_callback(self, msg: Odometry) -> None:
        super().odom_callback(msg)
        duration = self._update_time(msg.header.stamp, self.odom_callback)
        self.last_twist = msg.twist.twist
        return self._twist(
            twist_t.from_ros(msg.twist.twist, duration, self._res),
            time_msg_to_float(msg.header.stamp),
        )

    def lidar_callback(self, msg: LaserScan) -> None:
        super().lidar_callback(msg)
        assert msg.header.frame_id == self.cfg.laser_frame
        duration = self._update_time(msg.header.stamp, self.lidar_callback)
        if self.last_twist is not None:
            self._twist(
                twist_t.from_ros(self.last_twist, duration, self._res),
                time_msg_to_float(msg.header.stamp),
            )

        self._lidar(
            lidar_obs.from_ros(msg, self._res), time_msg_to_float(msg.header.stamp)
        )

    def get_pose(self) -> TransformStamped:
        ans = self._get_pose()
        ans = self.grid.meta.from_pixels(ans)

        x, y, z, w = tf_transformations.quaternion_from_euler(
            0.0, 0.0, float(ans.rot.to_angle())
        )

        msg = TransformStamped()
        msg.header.frame_id = self.cfg.map_frame
        msg.header.stamp = float_to_time_msg(self._prev_time)
        msg.child_frame_id = self.cfg.laser_frame

        msg.transform.translation.x = float(ans.tran.x)
        msg.transform.translation.y = float(ans.tran.y)

        msg.transform.rotation.x = float(x)
        msg.transform.rotation.y = float(y)
        msg.transform.rotation.z = float(z)
        msg.transform.rotation.w = float(w)

        return msg

    @staticmethod
    @jit
    def _get_particles_get_arr(
        particles: batched[position], meta: GridMeta
    ) -> Float[Array, "n 3"]:
        ans = particles.map(meta.from_pixels).map(lambda p: p.as_arr())
        return ans.unflatten()

    def get_particles(self) -> np.ndarray:
        ans = Controller._get_particles_get_arr(self._get_particles(), self.grid.meta)
        return np.array(ans)

    @staticmethod
    def _pose_from_ros_cb(pose: lazy[position], map: GridMeta):
        return map.to_pixels(pose())

    def _pose_from_ros(self, pose: TransformStamped) -> lazy[position]:
        return lazy(
            Controller._pose_from_ros_cb, position.from_ros(pose), self.grid.meta
        )

    def set_pose(self, pose: TransformStamped) -> None:
        super().set_pose(pose)
        _ = self._update_time(pose.header.stamp, self.set_pose)
        self._set_pose(self._pose_from_ros(pose), time_msg_to_float(pose.header.stamp))

    def _visualize(self, ctx: plot_ctx):
        m = Marker()
        m.type = Marker.POINTS
        m.header.frame_id = "map"
        m.scale.x = self._res
        m.scale.y = self._res
        # m.scale.x = self._res / 2
        # m.scale.y = self._res / 2
        ctx = self.grid.meta.plot_from_pixels_vec(ctx)
        ctx.execute(m)
        self.cfg.marker_callback(m)

    def _ground_truth(
        self, reference_time: float | None = None, allowed_offset: float = 0.03
    ) -> lazy[position] | None:
        t = self.cfg.ground_truth_callback()
        if t is None:
            return
        assert t.child_frame_id == self.cfg.laser_frame
        assert t.header.frame_id == self.cfg.map_frame
        if reference_time is not None:
            time_diff = reference_time - time_msg_to_float(t.header.stamp)
            if abs(time_diff) > allowed_offset:
                print(
                    colored(
                        f"ground truth is behind by {time_diff} seconds; discarding.",
                        "red",
                    ),
                )
                return None
        return self._pose_from_ros(t)

    def _plot_ground_truth(self) -> lazy[plotable]:
        pos = self._ground_truth()
        valid = pos is not None
        if pos is None:
            pos = self._lazy_position_ex()
        return lazy(Controller._ground_truth_plot_lazy, pos, valid)

    @staticmethod
    @plotfn
    def _ground_truth_plot_lazy(ctx: plot_ctx, p: lazy[position], valid: blike):
        return tree_select(
            valid, ctx + p().plot_as_seg(plot_style(color=(0.0, 1.0, 0.0))), ctx
        )
