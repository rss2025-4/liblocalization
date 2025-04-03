from abc import abstractmethod

import tf_transformations
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformStamped
from visualization_msgs.msg import Marker

from libracecar.plot import plot_ctx
from libracecar.ros_utils import float_to_time_msg, time_msg_to_float
from libracecar.specs import position
from libracecar.utils import lazy

from .api import LocalizationBase, localization_params


class Controller(LocalizationBase):

    def __init__(self, cfg: localization_params):
        self.cfg = cfg
        self._prev_time: float = -1.0
        self.last_twist: Twist | None = None

    def _update_time(self, new_time_: Time) -> float:
        new_time = time_msg_to_float(new_time_)
        ans = new_time - self._prev_time
        assert ans >= 0, f"negative time: {ans}"
        self._prev_time = new_time
        return ans

    def odom_callback(self, msg: Odometry) -> None:
        super().odom_callback(msg)
        duration = self._update_time(msg.header.stamp)
        self.last_twist = msg.twist.twist
        return self._twist(twist_t.from_ros(msg.twist.twist, duration))

    def lidar_callback(self, msg: LaserScan) -> None:
        super().lidar_callback(msg)
        assert msg.header.frame_id == self.cfg.laser_frame
        duration = self._update_time(msg.header.stamp)
        if self.last_twist is not None:
            self._twist(twist_t.from_ros(self.last_twist, duration))

        # TODO
        return

    def get_pose(self) -> TransformStamped:
        ans = self._get_pose()

        x, y, z, w = tf_transformations.quaternion_from_euler(
            0.0, 0.0, ans.rot.to_angle()
        )

        msg = TransformStamped()
        msg.header.frame_id = self.cfg.map_frame
        msg.header.stamp = float_to_time_msg(self._prev_time)
        msg.child_frame_id = self.cfg.laser_frame

        msg.transform.translation.x = ans.tran.x
        msg.transform.translation.y = ans.tran.y

        msg.transform.rotation.x = x
        msg.transform.rotation.y = y
        msg.transform.rotation.z = z
        msg.transform.rotation.w = w

        return msg

    def set_pose(self, pose: TransformStamped) -> None:
        super().set_pose(pose)
        _ = self._update_time(pose.header.stamp)
        self._set_pose(position.from_ros(pose))

    @abstractmethod
    def _get_pose(self) -> position: ...

    @abstractmethod
    def _set_pose(self, pose: lazy[position]) -> None: ...

    @abstractmethod
    def _twist(self, twist: lazy["twist_t"]) -> None: ...

    # @abstractmethod
    # def _lidar(self, twist: lazy["twist_t"]) -> None: ...

    def _visualize(self, ctx: plot_ctx):
        m = Marker()
        m.type = Marker.POINTS
        m.header.frame_id = "map"
        m.scale.x = 0.05
        m.scale.y = 0.05
        ctx.execute(m)
        self.cfg.marker_callback(m)


# to avoid circular imports
from .motion import twist_t
